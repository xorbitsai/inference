from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Qwen2Config

from ...models.dots_tts.config import ModelConfig
from ...models.dots_tts.core import DotsTtsCore, DotsTtsForwardOutput
from ...modules.backbone.dit_inference import (
    DiTInferenceContext,
    DiTSolver,
    DiTSolverState,
)
from ...modules.backbone.encoder_inference import (
    SemanticEncoderInference,
)
from ...modules.backbone.llm_inference import LLMInference, LLMInferenceState
from ...modules.backbone.scm_inference import SCMDiTSolver
from ...modules.speaker.encoder import SpeakerXVectorFeatures
from ...modules.vocoder.bigvgan import AudioVAE
from ...modules.vocoder.vocoder_inference import VocoderInference
from ...training.losses import LossMasks, LossTerm, LossTerms
from ...utils.logging import categorized_log as logc
from ...utils.profiling import measure_inference
from ...utils.tokenizer import AUDIO_GEN_START_TOKEN, require_token_id
from ...utils.util import get_dtype


@dataclass
class _GenerateState:
    llm_hiddens: torch.Tensor | None = None
    llm_state: LLMInferenceState = field(default_factory=LLMInferenceState)
    patch_encoder_state: Any | None = None
    fm_seq_len: int = 0
    fm_capacity: int = 0
    fm_sequence: torch.Tensor | None = None
    fm_cfg_sequence: torch.Tensor | None = None
    fm_null_g_cond: torch.Tensor | None = None
    fm_dit_state: DiTSolverState = field(default_factory=DiTSolverState)
    end_flag: bool = False


@dataclass(frozen=True)
class _PromptConditioning:
    prompt_patches: torch.Tensor | None = None
    prompt_latents: torch.Tensor | None = None
    g_cond: torch.Tensor | None = None


@dataclass
class _PromptFeatureCacheEntry:
    speaker_embedding: torch.Tensor | None = None
    prompt_latent_distribution: torch.Tensor | None = None


class DotsTtsModel(nn.Module):
    """Full train/infer model assembly around the dots.tts core network."""

    _GENERATE_LENGTH_BUCKETS = (64, 128, 256, 512)
    _optimize_enabled = True
    _PROMPT_FEATURE_CACHE_MAX_ENTRIES = 256
    VOCODER_STREAM_INITIAL_UNMERGED_PATCHES = 2
    DEFAULT_MAX_SEQUENCE_LENGTH = 2048
    CONFIG_FILENAME = "config.json"
    HF_MODEL_TYPE = "dots_tts"
    HF_ARCHITECTURES = ["DotsTTSForConditionalGeneration"]
    LATENT_STATS_FILENAME = "latent_stats.pt"
    LLM_CONFIG_FILENAME = "llm_config.json"
    MODEL_FILENAME = "model.safetensors"
    VOCODER_FILENAME = "vocoder.safetensors"
    SPEAKER_ENCODER_FILENAME = "speaker_encoder.safetensors"
    _ARTIFACT_ALIASES = (("llm.lm_head.weight", "llm.model.embed_tokens.weight"),)
    REQUIRED_ARTIFACT_FILES = (
        CONFIG_FILENAME,
        LATENT_STATS_FILENAME,
        LLM_CONFIG_FILENAME,
        MODEL_FILENAME,
        VOCODER_FILENAME,
        SPEAKER_ENCODER_FILENAME,
    )

    # region Module assembly and checkpoint IO
    def __init__(
        self,
        config: ModelConfig,
        tokenizer,
        latent_stats_path: str | Path,
        llm_config: Qwen2Config,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.latent_stats_path = (
            None if latent_stats_path is None else Path(latent_stats_path)
        )
        self.audio_gen_start_id = require_token_id(
            self.tokenizer, AUDIO_GEN_START_TOKEN
        )

        self.core = DotsTtsCore(
            config,
            llm_config=llm_config,
            tokenizer=tokenizer,
            latent_stats_path=self.latent_stats_path,
        )
        self.vocoder = AudioVAE(config.vocoder).eval()
        self.vocoder.remove_weight_norm()
        self.hop_size = self.vocoder.hop_size
        self.xvector_extractor = SpeakerXVectorFeatures(
            sample_rate=self.vocoder.sample_rate,
            campplus_embedding_size=config.campplus_embedding_size,
            max_audio_seconds=config.xvec_max_audio_seconds,
        ).eval()

        for param in self.vocoder.parameters():
            param.requires_grad = False
        for param in self.xvector_extractor.parameters():
            param.requires_grad = False
        self._optimize_enabled = True
        self._llm_max_sequence_length = self.DEFAULT_MAX_SEQUENCE_LENGTH
        self._static_generate_workspaces: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._prompt_feature_cache: OrderedDict[
            str, _PromptFeatureCacheEntry
        ] = OrderedDict()
        self._fm_dit_solvers: dict[tuple[str, bool], DiTSolver] = {}
        self._llm_inference: LLMInference | None = None
        self._llm_inference_core_id: int | None = None
        self._patch_encoder_inference: SemanticEncoderInference | None = None
        self._patch_encoder_inference_encoder_id: int | None = None
        self._vocoder_inference: VocoderInference | None = None
        self._vocoder_inference_vocoder_id: int | None = None

    def set_optimize(
        self,
        optimize: bool,
        *,
        max_sequence_length: int | None = None,
    ) -> None:
        if max_sequence_length is not None:
            requested_max_sequence_length = int(max_sequence_length)
            if requested_max_sequence_length <= 0:
                raise ValueError("max_sequence_length must be positive.")
            if requested_max_sequence_length != self._llm_max_sequence_length:
                llm_inference = getattr(self, "_llm_inference", None)
                if llm_inference is not None:
                    llm_inference.clear()
            self._llm_max_sequence_length = requested_max_sequence_length
        self._optimize_enabled = bool(optimize)
        if not self._optimize_enabled:
            self._fm_dit_solvers.clear()
            llm_inference = getattr(self, "_llm_inference", None)
            if llm_inference is not None:
                llm_inference.clear()
            patch_encoder_inference = getattr(self, "_patch_encoder_inference", None)
            if patch_encoder_inference is not None:
                patch_encoder_inference.clear()
            vocoder_inference = getattr(self, "_vocoder_inference", None)
            if vocoder_inference is not None:
                vocoder_inference.clear()

    def set_cfg_droprate(
        self,
        cfg_droprate: float | None = None,
        xvec_drop_rate: float | None = None,
    ) -> None:
        if cfg_droprate is not None:
            self.config.cfg_droprate = cfg_droprate
            self.core.config.cfg_droprate = cfg_droprate
            self.core.cfg_droprate = cfg_droprate

        if xvec_drop_rate is not None:
            self.config.xvec_drop_rate = xvec_drop_rate
            self.core.config.xvec_drop_rate = xvec_drop_rate
            self.core.xvec_drop_rate = xvec_drop_rate

    @classmethod
    def _resolve_generate_length_bucket(
        cls,
        max_generate_length: int,
    ) -> int:
        requested = int(max_generate_length)
        if requested <= 0:
            raise ValueError("max_generate_length must be positive.")
        for bucket in cls._GENERATE_LENGTH_BUCKETS:
            if requested <= bucket:
                return bucket
        raise ValueError(
            "max_generate_length exceeds the largest supported compile bucket: "
            f"max_generate_length={requested} "
            f"max_supported={cls._GENERATE_LENGTH_BUCKETS[-1]}."
        )

    def _get_llm_inference(self) -> LLMInference:
        core_id = id(self.core)
        adapter = getattr(self, "_llm_inference", None)
        if adapter is None or getattr(self, "_llm_inference_core_id", None) != core_id:
            adapter = LLMInference(self.core)
            self._llm_inference = adapter
            self._llm_inference_core_id = core_id
        return adapter

    def _get_vocoder_inference(self) -> VocoderInference:
        vocoder_id = id(self.vocoder)
        adapter = getattr(self, "_vocoder_inference", None)
        if (
            adapter is None
            or getattr(self, "_vocoder_inference_vocoder_id", None) != vocoder_id
        ):
            adapter = VocoderInference(self.vocoder)
            self._vocoder_inference = adapter
            self._vocoder_inference_vocoder_id = vocoder_id
        return adapter

    def _allocate_generate_state(
        self,
        *,
        max_audio_patch_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _GenerateState:
        state_dtype = dtype if device.type == "cuda" else torch.float32
        requested_audio_patch_count = int(max_audio_patch_count)
        if requested_audio_patch_count <= 0:
            raise ValueError("max_audio_patch_count must be positive.")
        state_audio_patch_count = (
            self._resolve_generate_length_bucket(requested_audio_patch_count)
            if self._optimize_enabled
            else requested_audio_patch_count
        )
        fm_capacity = state_audio_patch_count * (
            self.core.hidden_patch_size + self.core.latent_patch_size
        )
        workspace_key = (
            state_audio_patch_count,
            str(device),
            state_dtype,
        )
        workspace = self._static_generate_workspaces.get(workspace_key)
        if workspace is None:
            workspace = {
                "fm_sequence": torch.zeros(
                    (1, fm_capacity, self.core.fm_hidden_size),
                    dtype=state_dtype,
                    device=device,
                ),
                "fm_cfg_sequence": torch.zeros(
                    (1, fm_capacity, self.core.fm_hidden_size),
                    dtype=state_dtype,
                    device=device,
                ),
                "fm_null_g_cond": torch.zeros(
                    (1, self.core.fm_hidden_size),
                    dtype=state_dtype,
                    device=device,
                ),
            }
            self._static_generate_workspaces[workspace_key] = workspace
        else:
            workspace["fm_sequence"].zero_()
            workspace["fm_cfg_sequence"].zero_()

        llm_state = self._get_llm_inference().init_state(
            optimize=self._optimize_enabled,
            max_sequence_length=self._llm_max_sequence_length,
            device=device,
            dtype=self.core.llm.get_input_embeddings().weight.dtype,
        )

        return _GenerateState(
            llm_state=llm_state,
            patch_encoder_state=None,
            fm_seq_len=0,
            fm_capacity=fm_capacity,
            fm_sequence=workspace["fm_sequence"],
            fm_cfg_sequence=workspace["fm_cfg_sequence"],
            fm_null_g_cond=workspace["fm_null_g_cond"],
        )

    @staticmethod
    def _tensor_storage_signature(tensor: torch.Tensor) -> tuple:
        return (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.size()),
            tuple(tensor.stride()),
            tensor.dtype,
        )

    @classmethod
    def _build_artifact_state_dict(cls, module) -> dict[str, torch.Tensor]:
        state_dict = module.state_dict()
        skip_keys = set()

        for redundant_key, canonical_key in cls._ARTIFACT_ALIASES:
            redundant_tensor = state_dict.get(redundant_key)
            canonical_tensor = state_dict.get(canonical_key)
            if (
                redundant_tensor is not None
                and canonical_tensor is not None
                and cls._tensor_storage_signature(redundant_tensor)
                == cls._tensor_storage_signature(canonical_tensor)
            ):
                skip_keys.add(redundant_key)

        cleaned_state_dict = {}
        seen_storage = set()
        for key, value in state_dict.items():
            if key in skip_keys:
                continue

            storage_signature = cls._tensor_storage_signature(value)
            if storage_signature in seen_storage:
                continue

            seen_storage.add(storage_signature)
            cleaned_state_dict[key] = value.detach().cpu().contiguous()

        return cleaned_state_dict

    @classmethod
    def _restore_artifact_state_dict(cls, state_dict: dict, module) -> dict:
        restored_state_dict = dict(state_dict)
        for redundant_key, canonical_key in cls._ARTIFACT_ALIASES:
            if (
                canonical_key in restored_state_dict
                and redundant_key not in restored_state_dict
                and redundant_key in module.state_dict()
            ):
                restored_state_dict[redundant_key] = restored_state_dict[canonical_key]
        return restored_state_dict

    @classmethod
    def _save_artifact_module(cls, module, path: Path) -> None:
        save_file(cls._build_artifact_state_dict(module), path)

    @classmethod
    def _load_artifact_module(cls, module, path: Path):
        state_dict = load_file(path, device="cpu")
        restored_state_dict = cls._restore_artifact_state_dict(state_dict, module)
        mismatch = module.load_state_dict(restored_state_dict, strict=False)
        if mismatch.missing_keys or mismatch.unexpected_keys:
            raise RuntimeError(f"Failed to load {path}: {mismatch}")
        return module

    @classmethod
    def _validate_pretrained_directory(
        cls, pretrained_model_name_or_path: str | Path
    ) -> Path:
        pretrained_path = Path(pretrained_model_name_or_path).expanduser().resolve()
        missing_files = [
            name
            for name in cls.REQUIRED_ARTIFACT_FILES
            if not (pretrained_path / name).is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                f"Pretrained path {pretrained_path} is missing required files: {missing_files}"
            )
        return pretrained_path

    @classmethod
    def _load_pretrained_config(cls, pretrained_path: Path) -> ModelConfig:
        return ModelConfig.model_validate(
            json.loads(
                (pretrained_path / cls.CONFIG_FILENAME).read_text(encoding="utf-8")
            )
        )

    @staticmethod
    def _save_llm_config(llm_config: Qwen2Config, path: Path) -> None:
        path.write_text(
            json.dumps(llm_config.to_dict(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _load_llm_config(path: Path) -> Qwen2Config:
        return Qwen2Config.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _tie_llm_weights(self) -> None:
        if hasattr(self.core.llm, "tie_weights"):
            self.core.llm.tie_weights()

    def save_pretrained(self, save_directory: str | Path) -> Path:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        config_payload = self.config.to_declared_dict()
        config_payload["model_type"] = self.HF_MODEL_TYPE
        config_payload["architectures"] = list(self.HF_ARCHITECTURES)
        (save_directory / self.CONFIG_FILENAME).write_text(
            json.dumps(config_payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        self._save_llm_config(
            self.core.llm.config,
            save_directory / self.LLM_CONFIG_FILENAME,
        )
        self.tokenizer.save_pretrained(save_directory)
        shutil.copy2(
            self.latent_stats_path,
            save_directory / self.LATENT_STATS_FILENAME,
        )
        self._save_artifact_module(self.core, save_directory / self.MODEL_FILENAME)
        self._save_artifact_module(self.vocoder, save_directory / self.VOCODER_FILENAME)
        self._save_artifact_module(
            self.xvector_extractor,
            save_directory / self.SPEAKER_ENCODER_FILENAME,
        )
        return save_directory

    def _load_pretrained_artifacts(self, pretrained_path: Path) -> None:
        self.latent_stats_path = pretrained_path / self.LATENT_STATS_FILENAME
        self.core.io_helper = type(self.core.io_helper)(
            latent_stats_path=self.latent_stats_path
        )
        self._load_artifact_module(self.core, pretrained_path / self.MODEL_FILENAME)
        self._tie_llm_weights()
        self._load_artifact_module(
            self.vocoder, pretrained_path / self.VOCODER_FILENAME
        )
        self._load_artifact_module(
            self.xvector_extractor,
            pretrained_path / self.SPEAKER_ENCODER_FILENAME,
        )
        self.core.eval()
        self.vocoder.eval()
        self.xvector_extractor.eval()

    def load_pretrained_weights(
        self, pretrained_model_name_or_path: str | Path
    ) -> None:
        pretrained_path = self._validate_pretrained_directory(
            pretrained_model_name_or_path
        )
        saved_config = self._load_pretrained_config(pretrained_path)
        if saved_config.to_declared_dict() != self.config.to_declared_dict():
            raise ValueError(
                f"Pretrained config at {pretrained_path} does not match the current model."
            )
        saved_llm_config = self._load_llm_config(
            pretrained_path / self.LLM_CONFIG_FILENAME
        )
        if saved_llm_config.to_dict() != self.core.llm.config.to_dict():
            raise ValueError(
                f"Pretrained LLM config at {pretrained_path} does not match the current model."
            )
        self._load_pretrained_artifacts(pretrained_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path):
        logger.debug(
            logc("model", "DotsTtsModel load started: pretrained_path={}"),
            pretrained_model_name_or_path,
        )
        pretrained_model_name_or_path = cls._validate_pretrained_directory(
            pretrained_model_name_or_path
        )
        config = cls._load_pretrained_config(pretrained_model_name_or_path)
        llm_config = cls._load_llm_config(
            pretrained_model_name_or_path / cls.LLM_CONFIG_FILENAME
        )
        logger.debug(
            logc(
                "model",
                "DotsTtsModel config loaded: pretrained_path={} sample_rate={} patch_size={}",
            ),
            pretrained_model_name_or_path,
            config.vocoder.sample_rate,
            config.patch_size,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(pretrained_model_name_or_path),
            local_files_only=True,
        )
        model = cls(
            config,
            tokenizer=tokenizer,
            latent_stats_path=pretrained_model_name_or_path / cls.LATENT_STATS_FILENAME,
            llm_config=llm_config,
        )
        model._load_pretrained_artifacts(pretrained_model_name_or_path)
        logger.info(
            logc("model", "DotsTtsModel load completed: pretrained_path={}"),
            pretrained_model_name_or_path,
        )
        return model.eval()

    # endregion Module assembly and checkpoint IO

    # region Training batch preparation
    @torch.no_grad()
    def prepare_training_inputs(self, data: dict[str, Any]) -> dict[str, Any]:
        self.vocoder.eval()
        self.xvector_extractor.eval()
        processed = dict(data)
        sample: torch.Tensor | None = data.get("sample")
        sample_lengths: torch.Tensor | None = data.get("sample_lengths")

        if sample is not None:
            latents = self.vocoder.extract_latents(sample)
            processed["latents"] = latents
            if sample_lengths is not None:
                processed["latent_lengths"] = sample_lengths // self.hop_size
            else:
                processed["latent_lengths"] = torch.full(
                    (latents.size(0),),
                    latents.size(-1),
                    dtype=torch.long,
                    device=latents.device,
                )
            processed["latents_sampled"] = self.core.io_helper.sample_from_latent(
                latents
            )
            fbank = data.get("fbank")
            fbank_lengths = data.get("fbank_lengths")
            processed["xvector"] = self.xvector_extractor(
                sample,
                audio_lengths=sample_lengths,
                fbank=fbank,
                fbank_lengths=fbank_lengths,
            )
        else:
            processed["latents"] = None
            processed["latent_lengths"] = None

        return processed

    def _build_audio_span_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        span_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for token_id in self.core.audio_span_token_ids:
            span_mask = span_mask | (token_ids == token_id)
        return span_mask

    def _prepare_loss_metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        input_ids: torch.Tensor = data["input_ids"]
        labels: torch.Tensor = data["labels"]
        loss_mask: torch.Tensor = data["loss_mask"]
        input_span_mask = self._build_audio_span_mask(input_ids)
        output_span_mask = self._build_audio_span_mask(labels)
        output_span_mask_float = output_span_mask.to(loss_mask.dtype)
        llm_loss_mask = loss_mask * (1.0 - output_span_mask_float)
        fm_loss_mask = loss_mask * output_span_mask_float
        patch_counts = output_span_mask.sum(dim=1)
        max_patch_count = max(1, int(patch_counts.max().item()))
        fm_patch_mask = loss_mask.new_zeros((loss_mask.size(0), max_patch_count))
        for batch_idx in range(loss_mask.size(0)):
            count = int(patch_counts[batch_idx].item())
            if count <= 0:
                continue
            fm_patch_mask[batch_idx, :count] = fm_loss_mask[batch_idx].masked_select(
                output_span_mask[batch_idx]
            )

        return {
            "input_span_mask": input_span_mask,
            "output_span_mask": output_span_mask,
            "loss_masks": {
                "ce_loss": llm_loss_mask,
                "fm_loss": fm_patch_mask,
                "eos_loss": self._build_eos_loss_mask(fm_loss_mask),
            },
        }

    @staticmethod
    def _build_eos_loss_mask(eos_loss_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = eos_loss_mask.shape
        mask = eos_loss_mask.to(dtype=torch.bool)
        target = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=mask.device)
        mask_counts = mask.sum(dim=1, keepdim=True)
        cumulative = mask.long().cumsum(dim=1)
        target[mask & (cumulative == mask_counts)] = True

        mask_counts_flat = mask_counts.squeeze(1)
        neg_counts = (mask_counts_flat - 1).clamp_min(0).to(eos_loss_mask.dtype)
        pos_weight = torch.where(
            neg_counts > 0,
            torch.full_like(neg_counts, 0.5),
            torch.ones_like(neg_counts),
        ).unsqueeze(1)
        neg_weight = torch.where(
            neg_counts > 0,
            0.5 / neg_counts,
            torch.zeros_like(neg_counts),
        ).unsqueeze(1)

        positive_mask = target & mask
        negative_mask = mask & ~positive_mask
        return torch.where(
            positive_mask,
            pos_weight,
            negative_mask.to(eos_loss_mask.dtype) * neg_weight,
        )
    # endregion Training batch preparation

    # region Training loss assembly and forward
    @staticmethod
    def _compute_ce_loss_term(
        llm_logits: torch.Tensor,
        llm_labels: torch.Tensor,
        llm_loss_mask: torch.Tensor,
    ) -> LossTerm:
        vocab_size = llm_logits.size(-1)
        ce_loss = F.cross_entropy(
            llm_logits.view(-1, vocab_size),
            llm_labels.view(-1),
            reduction="none",
        ).view_as(llm_labels)
        return LossTerm(loss=ce_loss, mask=llm_loss_mask.to(ce_loss.dtype))

    @staticmethod
    def _compute_fm_loss_term(
        pred: torch.Tensor,
        target: torch.Tensor,
        fm_patch_mask: torch.Tensor,
    ) -> LossTerm:
        batch_size, max_patch_count = fm_patch_mask.shape
        fm_loss = (pred - target).pow(2).mean(dim=2).mean(dim=1)
        loss = fm_loss.new_zeros((batch_size, max_patch_count))
        patch_counts = fm_patch_mask.gt(0).sum(dim=1).tolist()
        expected_count = int(sum(patch_counts))
        if expected_count > 0 and int(fm_loss.numel()) != expected_count:
            raise RuntimeError(
                "Flow-matching loss count mismatch: "
                f"expected {expected_count}, got {int(fm_loss.numel())}."
            )

        offset = 0
        for batch_idx, patch_count in enumerate(patch_counts):
            if patch_count <= 0:
                continue
            next_offset = offset + int(patch_count)
            loss[batch_idx, :patch_count] = fm_loss[offset:next_offset]
            offset = next_offset
        return LossTerm(loss=loss, mask=fm_patch_mask.to(loss.dtype))

    @staticmethod
    def _compute_eos_loss_term(
        eos_out: torch.Tensor,
        eos_loss_mask: torch.Tensor,
    ) -> LossTerm:
        batch_size, seq_len, _ = eos_out.shape
        weights = eos_loss_mask.to(device=eos_out.device)
        mask = weights.gt(0)
        target = torch.zeros(
            (batch_size, seq_len),
            dtype=torch.long,
            device=eos_out.device,
        )
        mask_counts = mask.sum(dim=1, keepdim=True)
        cumulative = mask.long().cumsum(dim=1)
        target[mask & (cumulative == mask_counts)] = 1

        logits = rearrange(eos_out, "b n c -> b c n")
        ce_per_token = F.cross_entropy(logits, target, reduction="none")
        return LossTerm(loss=ce_per_token, mask=weights.to(ce_per_token.dtype))

    @staticmethod
    def _compute_eos_loss_stats(
        eos_out: torch.Tensor,
        eos_loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = DotsTtsModel._build_eos_loss_mask(eos_loss_mask)
        term = DotsTtsModel._compute_eos_loss_term(eos_out, weights)
        mask = term.mask.to(device=term.loss.device, dtype=term.loss.dtype)
        eos_loss_sum = (term.loss * mask).sum(dim=1)
        eos_sample_count = eos_loss_mask.to(device=term.loss.device).gt(0).any(
            dim=1
        ).to(term.loss.dtype)
        return eos_loss_sum, eos_sample_count

    def _compute_loss_terms(
        self,
        outputs: DotsTtsForwardOutput,
        *,
        labels: torch.Tensor,
        loss_masks: LossMasks,
    ) -> LossTerms:
        return {
            "ce_loss": self._compute_ce_loss_term(
                outputs.llm_logits,
                labels,
                loss_masks["ce_loss"],
            ),
            "fm_loss": self._compute_fm_loss_term(
                outputs.pred,
                outputs.target,
                loss_masks["fm_loss"],
            ),
            "eos_loss": self._compute_eos_loss_term(
                outputs.eos_out,
                loss_masks["eos_loss"],
            ),
        }

    def prepare_training_batch(self, data: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(data)
        prepared.update(self._prepare_loss_metadata(prepared))
        return prepared

    def forward(self, data: dict[str, Any]) -> LossTerms:
        loss_masks: LossMasks = data["loss_masks"]
        processed = self.prepare_training_inputs(data)
        processed["input_span_mask"] = data["input_span_mask"]
        processed["output_span_mask"] = data["output_span_mask"]
        return self._compute_loss_terms(
            self.core(processed),
            labels=processed["labels"],
            loss_masks=loss_masks,
        )
    # endregion Training loss assembly and forward

    # region Inference helpers
    def _get_patch_encoder_inference(self) -> SemanticEncoderInference:
        encoder = self.core.patch_encoder
        encoder_id = id(encoder)
        adapter = getattr(self, "_patch_encoder_inference", None)
        if (
            adapter is None
            or getattr(self, "_patch_encoder_inference_encoder_id", None) != encoder_id
        ):
            adapter = SemanticEncoderInference(encoder)
            self._patch_encoder_inference = adapter
            self._patch_encoder_inference_encoder_id = encoder_id
        return adapter

    def _get_llm_inference(self) -> LLMInference:
        core_id = id(self.core)
        adapter = getattr(self, "_llm_inference", None)
        if adapter is None or getattr(self, "_llm_inference_core_id", None) != core_id:
            adapter = LLMInference(self.core)
            self._llm_inference = adapter
            self._llm_inference_core_id = core_id
        return adapter

    def _get_dit_solver(self, *, solver_mode: str) -> DiTSolver:
        key = (
            solver_mode,
            bool(self._optimize_enabled),
        )
        solver = self._fm_dit_solvers.get(key)
        if solver is None:
            context = DiTInferenceContext.from_core(self.core)
            if solver_mode == "scm":
                sampling = self.config.sampling
                if sampling is None:
                    raise RuntimeError("sCM solver requires artifact sampling config.")
                solver = SCMDiTSolver(
                    context,
                    optimize=self._optimize_enabled,
                    bucket_resolver=self._resolve_generate_length_bucket,
                    tau_mid=sampling.tau_mid,
                )
            elif solver_mode in {"flow_matching", "meanflow"}:
                solver = DiTSolver(
                    context,
                    optimize=self._optimize_enabled,
                    bucket_resolver=self._resolve_generate_length_bucket,
                    meanflow=solver_mode == "meanflow",
                )
            else:
                raise ValueError(f"Unsupported DiT solver mode: {solver_mode!r}.")
            self._fm_dit_solvers[key] = solver
        return solver

    def _get_vocoder_inference(self) -> VocoderInference:
        vocoder_id = id(self.vocoder)
        adapter = getattr(self, "_vocoder_inference", None)
        if (
            adapter is None
            or getattr(self, "_vocoder_inference_vocoder_id", None) != vocoder_id
        ):
            adapter = VocoderInference(self.vocoder)
            self._vocoder_inference = adapter
            self._vocoder_inference_vocoder_id = vocoder_id
        return adapter

    # endregion Inference helpers

    # region Prompt conditioning and decode state helpers
    def _prepare_prompt_audio_for_conditioning(
        self,
        prompt_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, str]:
        if prompt_audio.ndim == 1:
            prompt_audio = prompt_audio.unsqueeze(0)
        prompt_audio = prompt_audio.detach().cpu().contiguous()

        samples_per_patch = self.config.patch_size * self.hop_size
        target_len = (
            math.ceil(prompt_audio.size(1) / samples_per_patch) * samples_per_patch
        )
        pad_len = target_len - prompt_audio.size(1)
        if pad_len > 0:
            prompt_audio = F.pad(prompt_audio, (0, pad_len))

        digest = hashlib.sha1()
        digest.update(str(tuple(prompt_audio.shape)).encode("ascii"))
        digest.update(str(prompt_audio.dtype).encode("ascii"))
        digest.update(prompt_audio.numpy().tobytes())
        return prompt_audio, digest.hexdigest()

    def _get_prompt_feature_cache_entry(
        self,
        cache_key: str,
    ) -> _PromptFeatureCacheEntry | None:
        entry = self._prompt_feature_cache.get(cache_key)
        if entry is not None:
            self._prompt_feature_cache.move_to_end(cache_key)
        return entry

    def _store_prompt_feature_cache_entry(
        self,
        cache_key: str,
        entry: _PromptFeatureCacheEntry,
    ) -> None:
        if (
            entry.speaker_embedding is None
            and entry.prompt_latent_distribution is None
        ):
            return
        self._prompt_feature_cache[cache_key] = entry
        self._prompt_feature_cache.move_to_end(cache_key)
        while (
            len(self._prompt_feature_cache) > self._PROMPT_FEATURE_CACHE_MAX_ENTRIES
        ):
            self._prompt_feature_cache.popitem(last=False)

    def _can_cache_speaker_embedding(self, prompt_sample_count: int) -> bool:
        max_audio_seconds = self.xvector_extractor.max_audio_seconds
        if max_audio_seconds <= 0:
            return True
        max_input_length = round(
            self.xvector_extractor.sample_rate * max_audio_seconds
        )
        return int(prompt_sample_count) <= max_input_length

    @torch.no_grad()
    def _prepare_prompt_conditioning(
        self,
        prompt_audio: torch.Tensor | None,
        *,
        use_prompt_prefill: bool,
        speaker_scale: float = 1.5,
    ) -> _PromptConditioning:
        if prompt_audio is None:
            logger.debug(
                logc(
                    "conditioning",
                    "Prompt conditioning skipped: no prompt audio provided.",
                )
            )
            return _PromptConditioning()

        self.vocoder.eval()
        self.xvector_extractor.eval()
        device = next(self.core.parameters()).device
        prompt_audio, cache_key = self._prepare_prompt_audio_for_conditioning(
            prompt_audio
        )
        prompt_sample_count = int(prompt_audio.shape[-1])
        cache_entry = self._get_prompt_feature_cache_entry(cache_key)
        if cache_entry is None:
            cache_entry = _PromptFeatureCacheEntry()
        prompt_audio = prompt_audio.to(device=device)

        can_cache_speaker = self._can_cache_speaker_embedding(prompt_sample_count)
        speaker_embedding = (
            cache_entry.speaker_embedding if can_cache_speaker else None
        )
        if speaker_embedding is None:
            with measure_inference("speaker_encoder", phase="prompt_conditioning"):
                speaker_embedding = self.xvector_extractor(prompt_audio[None, :])
            if can_cache_speaker:
                cache_entry.speaker_embedding = speaker_embedding.detach()
        else:
            logger.debug(
                logc(
                    "conditioning",
                    "Prompt speaker cache hit: key={} prompt_samples={}",
                ),
                cache_key[:12],
                prompt_sample_count,
            )
        g_cond = self.core.xvec_proj(speaker_embedding * float(speaker_scale))
        if not use_prompt_prefill:
            self._store_prompt_feature_cache_entry(cache_key, cache_entry)
            logger.debug(
                logc(
                    "conditioning",
                    "Reference-audio-only conditioning prepared: prompt_samples={} speaker_scale={} device={}",
                ),
                prompt_sample_count,
                speaker_scale,
                device,
            )
            return _PromptConditioning(g_cond=g_cond)

        prompt_latents = cache_entry.prompt_latent_distribution
        if prompt_latents is None:
            with measure_inference("latent_encoder", phase="prompt_conditioning"):
                prompt_latents = self._get_vocoder_inference().extract_latents(
                    prompt_audio[None, :]
                )
            cache_entry.prompt_latent_distribution = prompt_latents.detach()
        else:
            logger.debug(
                logc(
                    "conditioning",
                    "Prompt latent cache hit: key={} prompt_samples={}",
                ),
                cache_key[:12],
                prompt_sample_count,
            )
        self._store_prompt_feature_cache_entry(cache_key, cache_entry)
        prompt_latents_sampled = self.core.io_helper.sample_from_latent(prompt_latents)
        prompt_latents_sampled = prompt_latents_sampled[:, : -self.config.patch_size]
        prompt_patches = rearrange(
            self.core.io_helper.normalize(prompt_latents_sampled),
            "b (s p) d -> b s p d",
            p=self.config.patch_size,
        )
        logger.debug(
            logc(
                "conditioning",
                "Prompt conditioning prepared: prompt_samples={} prompt_patch_count={} "
                "speaker_scale={} device={}",
            ),
            prompt_sample_count,
            prompt_patches.size(1),
            speaker_scale,
            device,
        )
        return _PromptConditioning(
            prompt_patches=prompt_patches,
            prompt_latents=prompt_latents_sampled,
            g_cond=g_cond,
        )

    def _prepare_patch_encoder_input(
        self,
        latents: torch.Tensor,
        *,
        already_normalized: bool = False,
    ) -> torch.Tensor:
        if self.core.patch_encoder.expects_normalized_input:
            return (
                latents
                if already_normalized
                else self.core.io_helper.normalize(latents)
            )
        if already_normalized:
            return self.core.io_helper.denormalize(latents).to(
                dtype=self.core.patch_encoder.input_dtype
            )
        return latents

    def _prefill_prompt_latents(
        self,
        prompt_latents: torch.Tensor | None,
        *,
        state: _GenerateState,
    ) -> torch.Tensor | None:
        if prompt_latents is None:
            return None
        if prompt_latents.size(1) == 0:
            return prompt_latents.new_zeros(
                (prompt_latents.size(0), 0, self.core.llm_hidden_size)
            )
        patch_encoder_input = self._prepare_patch_encoder_input(prompt_latents)
        state_dtype = (
            state.fm_sequence.dtype
            if state.fm_sequence is not None
            else patch_encoder_input.dtype
        )
        with measure_inference("patch_encoder", phase="prompt_prefill"):
            prompt_patch_embeddings, state.patch_encoder_state = (
                self._get_patch_encoder_inference().prefill_with_state(
                    patch_encoder_input,
                    state.patch_encoder_state,
                    optimize=self._optimize_enabled,
                    bucket_resolver=self._resolve_generate_length_bucket,
                    dtype=state_dtype,
                )
            )
        return prompt_patch_embeddings

    def _append_to_fm_buffer(
        self,
        buffer: torch.Tensor | None,
        state: _GenerateState,
        chunk: torch.Tensor,
    ) -> tuple[int, int]:
        if buffer is None:
            raise RuntimeError("FM static buffer is not initialized.")
        start = state.fm_seq_len
        end = start + chunk.size(1)
        if end > state.fm_capacity:
            raise RuntimeError(
                "FM StaticBuffer capacity exceeded: "
                f"next_length={end} capacity={state.fm_capacity}."
            )
        buffer[:, start:end].copy_(chunk.to(buffer.dtype))
        return start, end

    def _append_hidden_chunk(
        self, state: _GenerateState, hidden_chunk: torch.Tensor
    ) -> None:
        last_hidden = hidden_chunk[:, -self.core.hidden_patch_size :, :]
        projected = self.core.hidden_proj(last_hidden)
        null_projected = self.core.hidden_proj(torch.zeros_like(last_hidden))
        _start, end = self._append_to_fm_buffer(
            state.fm_sequence,
            state,
            projected,
        )
        cfg_buffer = state.fm_cfg_sequence
        if cfg_buffer is None:
            raise RuntimeError("FM cfg static buffer is not initialized.")
        cfg_buffer[:, state.fm_seq_len : end].copy_(null_projected.to(cfg_buffer.dtype))
        state.fm_seq_len = end

    def _append_history_chunk(
        self, state: _GenerateState, latent_chunk: torch.Tensor
    ) -> None:
        history_latent = self.core.latent_proj(latent_chunk)
        _start, end = self._append_to_fm_buffer(
            state.fm_sequence,
            state,
            history_latent,
        )
        cfg_buffer = state.fm_cfg_sequence
        if cfg_buffer is None:
            raise RuntimeError("FM cfg static buffer is not initialized.")
        cfg_buffer[:, state.fm_seq_len : end].copy_(history_latent.to(cfg_buffer.dtype))
        state.fm_seq_len = end

    def _consume_text_schedule(
        self,
        generation_schedule: torch.Tensor,
        *,
        position: int,
        next_audio_position: int,
        state: _GenerateState,
        profile_step: int | None = None,
    ) -> int:
        with measure_inference("LLM", phase="text_schedule", step=profile_step):
            text_chunk = generation_schedule[:, position:next_audio_position]
            _, state.llm_hiddens, _logits = self._get_llm_inference().step(
                state.llm_state,
                input_ids=text_chunk,
                request_logits=False,
                compile_static=True,
                optimize=self._optimize_enabled,
                max_sequence_length=self._llm_max_sequence_length,
            )
        self._append_hidden_chunk(state, state.llm_hiddens)
        return next_audio_position

    def _locate_prefill_boundary(
        self,
        *,
        span_positions: torch.Tensor,
        prompt_patch_count: int,
    ) -> tuple[int, torch.Tensor]:
        if span_positions.numel() > prompt_patch_count:
            return int(span_positions[prompt_patch_count].item()), span_positions[
                :prompt_patch_count
            ]
        raise RuntimeError(
            "Prefill boundary discovery failed despite prior schedule validation."
        )

    @staticmethod
    def _find_audio_span_positions(
        generation_schedule: torch.Tensor,
        *,
        audio_placeholder_ids: set[int],
    ) -> torch.Tensor:
        schedule = generation_schedule[0]
        placeholder_ids = torch.tensor(
            sorted(audio_placeholder_ids),
            device=schedule.device,
            dtype=schedule.dtype,
        )
        return torch.nonzero(
            torch.isin(schedule, placeholder_ids),
            as_tuple=False,
        ).squeeze(-1)

    @staticmethod
    def _next_token_is_audio_span(
        generation_schedule: torch.Tensor,
        *,
        position: int,
        audio_placeholder_ids: set[int],
    ) -> bool:
        next_position = position + 1
        if next_position >= generation_schedule.size(1):
            return False
        return (
            int(generation_schedule[0, next_position].item()) in audio_placeholder_ids
        )

    def _build_prefill_inputs_embeds(
        self,
        generation_schedule: torch.Tensor,
        *,
        prompt_patch_embeddings: torch.Tensor | None,
        prompt_span_positions: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.core.llm.get_input_embeddings()(
            generation_schedule
        ).clone()
        if prompt_span_positions.numel() > 0:
            if prompt_patch_embeddings is None:
                raise RuntimeError(
                    "Prompt patch embeddings are required when prefill includes prompt audio spans."
                )
            patch_embeddings = prompt_patch_embeddings[
                :, : prompt_span_positions.numel()
            ].to(inputs_embeds.dtype)
            if patch_embeddings.size(1) != prompt_span_positions.numel():
                raise RuntimeError(
                    f"Prompt patch embeddings ({patch_embeddings.size(1)}) do not match prompt span count ({prompt_span_positions.numel()})."
                )
            inputs_embeds[:, prompt_span_positions, :] = patch_embeddings
        return inputs_embeds

    def _prefill(
        self,
        generation_schedule: torch.Tensor,
        *,
        state: _GenerateState,
        span_positions: torch.Tensor,
        prompt_patches: torch.Tensor | None,
        prompt_patch_embeddings: torch.Tensor | None,
        audio_placeholder_ids: set[int],
    ) -> int:
        prompt_patch_count = (
            0 if prompt_patches is None else int(prompt_patches.size(1))
        )
        prefill_end, prompt_span_positions = self._locate_prefill_boundary(
            span_positions=span_positions,
            prompt_patch_count=prompt_patch_count,
        )
        if prefill_end == 0:
            return 0
        inputs_embeds = self._build_prefill_inputs_embeds(
            generation_schedule[:, :prefill_end],
            prompt_patch_embeddings=prompt_patch_embeddings,
            prompt_span_positions=prompt_span_positions,
        )
        with measure_inference("LLM", phase="prefill"):
            _, llm_hiddens, _logits = self._get_llm_inference().step(
                state.llm_state,
                inputs_embeds=inputs_embeds,
                request_logits=False,
                optimize=self._optimize_enabled,
                max_sequence_length=self._llm_max_sequence_length,
            )
        state.llm_hiddens = llm_hiddens[:, -1:, :]

        cursor = 0
        for prompt_index, span_position in enumerate(prompt_span_positions.tolist()):
            if span_position > cursor:
                self._append_hidden_chunk(
                    state, llm_hiddens[:, span_position - 1 : span_position, :]
                )
            self._append_history_chunk(state, prompt_patches[:, prompt_index])
            if self._next_token_is_audio_span(
                generation_schedule,
                position=span_position,
                audio_placeholder_ids=audio_placeholder_ids,
            ):
                self._append_hidden_chunk(
                    state, llm_hiddens[:, span_position : span_position + 1, :]
                )
            cursor = span_position + 1
        if prefill_end > cursor:
            self._append_hidden_chunk(
                state, llm_hiddens[:, prefill_end - 1 : prefill_end, :]
            )
        return prefill_end

    def _decode_next_audio(
        self,
        state: _GenerateState,
        *,
        g_cond: torch.Tensor | None,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
        profile_step: int | None = None,
    ) -> torch.Tensor:
        with measure_inference("FM", phase="decode", step=profile_step):
            sequence = state.fm_sequence
            if sequence is None:
                raise RuntimeError("FM static buffer is not initialized.")
            null_g_cond = state.fm_null_g_cond
            if null_g_cond is None:
                raise RuntimeError("FM null conditioning buffer is not initialized.")
            sampling = self.config.sampling
            if sampling is not None:
                sampling.resolve(
                    ode_method=ode_method,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                )
            solver_mode = sampling.solver if sampling is not None else self.core.mode
            cfg_sequence = state.fm_cfg_sequence
            if solver_mode == "flow_matching" and cfg_sequence is None:
                raise RuntimeError("FM cfg static buffer is not initialized.")
            solver_cfg_sequence = (
                cfg_sequence if solver_mode == "flow_matching" else None
            )
            audio_patch = self._get_dit_solver(solver_mode=solver_mode).decode_next(
                state.fm_dit_state,
                sequence=sequence,
                cfg_sequence=solver_cfg_sequence,
                fm_seq_len=state.fm_seq_len,
                null_g_cond=null_g_cond,
                g_cond=g_cond,
                nfe=num_steps,
                ode_method=ode_method,
                guidance_scale=guidance_scale,
            )
            return audio_patch

    def _consume_audio_patch(
        self,
        state: _GenerateState,
        *,
        audio_patch: torch.Tensor,
        profile_step: int | None = None,
    ) -> None:
        audio_patch_for_llm = self._prepare_patch_encoder_input(
            audio_patch,
            already_normalized=True,
        )
        self._append_history_chunk(state, audio_patch)
        state_dtype = (
            state.fm_sequence.dtype
            if state.fm_sequence is not None
            else audio_patch_for_llm.dtype
        )
        with measure_inference("patch_encoder", phase="decode", step=profile_step):
            llm_embedding, state.patch_encoder_state = (
                self._get_patch_encoder_inference().decode_patch_with_state(
                    audio_patch_for_llm,
                    state.patch_encoder_state,
                    optimize=self._optimize_enabled,
                    bucket_resolver=self._resolve_generate_length_bucket,
                    dtype=state_dtype,
                )
            )
        with measure_inference("LLM", phase="decode", step=profile_step):
            _, state.llm_hiddens, _logits = self._get_llm_inference().step(
                state.llm_state,
                inputs_embeds=llm_embedding,
                request_logits=False,
                compile_static=True,
                optimize=self._optimize_enabled,
                max_sequence_length=self._llm_max_sequence_length,
            )

    def _decode(
        self,
        generation_schedule: torch.Tensor,
        *,
        position: int,
        state: _GenerateState,
        audio_placeholder_ids: set[int],
        span_positions: torch.Tensor,
        g_cond: torch.Tensor | None,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
        eos_threshold: float,
        suppress_first_eos_check: bool = False,
    ) -> Iterator[torch.Tensor]:
        span_cursor = torch.searchsorted(
            span_positions,
            torch.tensor(
                position,
                device=span_positions.device,
                dtype=span_positions.dtype,
            ),
        ).item()
        decoded_audio_count = 0
        while position < generation_schedule.size(1):
            token_id = int(generation_schedule[0, position].item())
            if token_id in audio_placeholder_ids:
                profile_step = decoded_audio_count + 1
                should_check_eos = not (
                    suppress_first_eos_check and decoded_audio_count == 0
                )
                stop_after_current_audio = (
                    self._should_stop_after_current_audio(
                        state,
                        eos_threshold=eos_threshold,
                    )
                    if should_check_eos
                    else False
                )
                audio_patch = self._decode_next_audio(
                    state,
                    g_cond=g_cond,
                    ode_method=ode_method,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    profile_step=profile_step,
                )
                self._consume_audio_patch(
                    state,
                    audio_patch=audio_patch,
                    profile_step=profile_step,
                )
                decoded_audio_count += 1
                if self._next_token_is_audio_span(
                    generation_schedule,
                    position=position,
                    audio_placeholder_ids=audio_placeholder_ids,
                ):
                    self._append_hidden_chunk(state, state.llm_hiddens)
                position += 1
                span_cursor += 1
                yield audio_patch
                if stop_after_current_audio:
                    state.end_flag = True
                    return
                continue
            next_audio_position = (
                int(span_positions[span_cursor].item())
                if span_cursor < span_positions.numel()
                else generation_schedule.size(1)
            )
            position = self._consume_text_schedule(
                generation_schedule,
                position=position,
                next_audio_position=next_audio_position,
                state=state,
                profile_step=decoded_audio_count + 1,
            )

    def _should_stop_after_current_audio(
        self, state: _GenerateState, *, eos_threshold: float
    ) -> bool:
        if state.llm_hiddens is None:
            return False
        eos = (
            self.core.eos_proj(state.llm_hiddens).softmax(dim=-1)[:, -1, 1]
            > eos_threshold
        )
        return state.end_flag or bool(eos.item())

    # endregion Prompt conditioning and decode state helpers

    # region Public generation APIs
    @torch.no_grad()
    def _generate_latents_stream(
        self,
        data: dict[str, Any],
        *,
        precision: str,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
        speaker_scale: float = 1.5,
        eos_threshold: float = 0.8,
    ) -> Iterator[torch.Tensor]:
        dtype = get_dtype(precision)
        device = next(self.core.parameters()).device
        use_amp = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            generation_schedule: torch.Tensor = data["generation_schedule"]
            if generation_schedule.size(0) != 1:
                raise ValueError(
                    "DotsTtsModel.generate expects batch size 1 for generation_schedule."
                )
            if self._optimize_enabled:
                max_sequence_length = int(self._llm_max_sequence_length)
                schedule_length = int(generation_schedule.size(1))
                if schedule_length > max_sequence_length:
                    raise ValueError(
                        "generation_schedule length exceeds max_sequence_length for "
                        "optimized LLM StaticCache inference: "
                        f"schedule_length={schedule_length} "
                        f"max_sequence_length={max_sequence_length}."
                    )

            use_prompt_prefill = data.get("prompt_audio") is not None and bool(
                data.get("prompt_text")
            )
            prompt_conditioning = self._prepare_prompt_conditioning(
                data.get("prompt_audio"),
                use_prompt_prefill=use_prompt_prefill,
                speaker_scale=speaker_scale,
            )
            has_prompt_prefill = prompt_conditioning.prompt_patches is not None
            prompt_patch_count = (
                0
                if not has_prompt_prefill
                else int(prompt_conditioning.prompt_patches.size(1))
            )
            audio_placeholder_ids = set(self.core.audio_span_token_ids)
            span_positions = self._find_audio_span_positions(
                generation_schedule,
                audio_placeholder_ids=audio_placeholder_ids,
            )
            span_count = int(span_positions.numel())
            minimum_required_spans = prompt_patch_count + 1
            if span_count < minimum_required_spans:
                raise ValueError(
                    f"generation_schedule provides {span_count} audio spans, but prompt prefill requires "
                    f"{prompt_patch_count} spans and generation requires at least one additional decode span."
                )
            logger.debug(
                logc(
                    "decode",
                    "Latent generation prepared: schedule_audio_spans={} prompt_patch_count={} "
                    "minimum_required_spans={}",
                ),
                span_count,
                prompt_patch_count,
                minimum_required_spans,
            )

            state = self._allocate_generate_state(
                max_audio_patch_count=span_count,
                device=device,
                dtype=dtype,
            )
            prompt_patch_embeddings = self._prefill_prompt_latents(
                prompt_conditioning.prompt_latents,
                state=state,
            )
            position = self._prefill(
                generation_schedule,
                state=state,
                span_positions=span_positions,
                prompt_patches=prompt_conditioning.prompt_patches,
                prompt_patch_embeddings=prompt_patch_embeddings,
                audio_placeholder_ids=audio_placeholder_ids,
            )

            payload_patch_count = 0
            should_drop_regenerated_prompt_patch = has_prompt_prefill
            for audio_patch in self._decode(
                generation_schedule,
                position=position,
                state=state,
                audio_placeholder_ids=audio_placeholder_ids,
                span_positions=span_positions,
                g_cond=prompt_conditioning.g_cond,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                eos_threshold=eos_threshold,
                suppress_first_eos_check=has_prompt_prefill,
            ):
                if should_drop_regenerated_prompt_patch:
                    should_drop_regenerated_prompt_patch = False
                    continue
                payload_patch_count += 1
                if payload_patch_count == 1 or payload_patch_count % 10 == 0:
                    logger.debug(
                        logc(
                            "decode",
                            "Latent generation progress: payload_audio_patches={}",
                        ),
                        payload_patch_count,
                    )
                yield self.core.io_helper.denormalize(audio_patch)

            if payload_patch_count == 0:
                if has_prompt_prefill:
                    raise RuntimeError(
                        "Generation produced no payload latents after discarding the regenerated prompt-tail patch. "
                        "This usually means EOS triggered immediately after prompt continuation "
                        "or the generation schedule did not provide an effective decode span."
                    )
                raise RuntimeError(
                    "Generation produced no decodable latents. "
                    "This usually means EOS triggered before the first decode patch "
                    "or the generation schedule did not provide an effective decode span."
                )
            logger.debug(
                logc("decode", "Latent generation completed: payload_audio_patches={}"),
                payload_patch_count,
            )

    @torch.no_grad()
    def generate_audio_stream(
        self,
        data: dict[str, Any],
        *,
        precision: str,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
        speaker_scale: float = 1.5,
        eos_threshold: float = 0.8,
        vocoder_merge_steps: int = 1,
    ) -> Iterator[torch.Tensor]:
        merge_steps = vocoder_merge_steps if self._optimize_enabled else 1
        if merge_steps < 1:
            raise ValueError(
                f"vocoder_merge_steps must be >= 1, got {vocoder_merge_steps}."
            )
        vocoder_inference = self._get_vocoder_inference()
        stream_state = vocoder_inference.init_stream_state(
            batch_size=1,
            chunk_size=int(self.core.latent_patch_size) * merge_steps,
        )
        pending_latent_patches: list[torch.Tensor] = []
        pending_start_index = 0

        for latent_patch_index, latent_patch in enumerate(
            self._generate_latents_stream(
                data,
                precision=precision,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                speaker_scale=speaker_scale,
                eos_threshold=eos_threshold,
            ),
            start=1,
        ):
            # Keep the initial cadence unchanged for first-packet latency.
            if (
                merge_steps == 1
                or latent_patch_index <= self.VOCODER_STREAM_INITIAL_UNMERGED_PATCHES
            ):
                audio_chunk = vocoder_inference.stream_step(
                    latent_patch,
                    stream_state=stream_state,
                    optimize=self._optimize_enabled,
                    profile_step=latent_patch_index,
                    use_compiled=True,
                )
            else:
                if not pending_latent_patches:
                    pending_start_index = latent_patch_index
                pending_latent_patches.append(latent_patch)
                if len(pending_latent_patches) < merge_steps:
                    continue
                audio_chunk = vocoder_inference.stream_step(
                    torch.cat(pending_latent_patches, dim=1),
                    stream_state=stream_state,
                    optimize=self._optimize_enabled,
                    profile_step=f"{pending_start_index}-{latent_patch_index}",
                    use_compiled=True,
                )
                pending_latent_patches = []
                pending_start_index = 0
            if audio_chunk.size(-1) > 0:
                yield audio_chunk

        if pending_latent_patches:
            final_index = pending_start_index + len(pending_latent_patches) - 1
            audio_chunk = vocoder_inference.stream_step(
                torch.cat(pending_latent_patches, dim=1),
                stream_state=stream_state,
                optimize=self._optimize_enabled,
                profile_step=f"{pending_start_index}-{final_index}",
                use_compiled=False,
            )
            if audio_chunk.size(-1) > 0:
                yield audio_chunk

        final_chunk = vocoder_inference.flush(stream_state)
        if final_chunk.size(-1) > 0:
            yield final_chunk

    @torch.no_grad()
    def generate_audio(
        self,
        data: dict[str, Any],
        *,
        precision: str,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
        speaker_scale: float = 1.5,
    ) -> torch.Tensor:
        latent_patches = list(
            self._generate_latents_stream(
                data,
                precision=precision,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                speaker_scale=speaker_scale,
            )
        )
        logger.debug(
            logc("vocoder", "Vocoder decode started: latent_patch_count={}"),
            len(latent_patches),
        )
        audio = self._get_vocoder_inference().decode_latents(
            torch.cat(latent_patches, dim=1)
        )
        logger.debug(
            logc("vocoder", "Vocoder decode completed: waveform_samples={}"),
            audio.shape[-1],
        )
        return audio

    # endregion Public generation APIs
