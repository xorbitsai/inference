from __future__ import annotations

import logging
import math
import os

import torch
import torchaudio
from omegaconf import OmegaConf
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from auk.model import CFMEdit, Flux2Edit
from auk.model.vae import load_vae_model
from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAEConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger().addFilter(lambda record: "System prompt modified" not in record.getMessage())


_DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


class AukInfer:
    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        *,
        device: str | None = None,
        dtype: str = "bf16",
        qwen_path: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _DTYPE_MAP.get(dtype, torch.bfloat16)

        config = OmegaConf.load(config_path)
        if qwen_path:
            config.model.text_encoder.text_encoder_path = qwen_path
        # the VAE ships next to the checkpoint as vae.safetensors; use it if present, else keep config's path
        ckpt_dir_vae = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "vae.safetensors")
        if os.path.isfile(ckpt_dir_vae):
            config.model.vae.vae_model_path = ckpt_dir_vae

        self.config = config
        # AuK-Flash is a distilled release that only works under a fixed (t_grid, cfg); detect it
        self.is_flash = config.model.get("name", "") == "AuK-Flash"
        if self.is_flash:
            logger.info("Detected AuK-Flash release — locking sampling to the 4-step / CFG-off recipe.")

        vae_config = config.model.vae
        self.target_sample_rate = vae_config.target_sample_rate
        self.downsample_rate = vae_config.downsample_rate
        self.latent_dim = vae_config.latent_dim

        # --- text encoder (Qwen2.5-Omni) ---
        text_encoder_config = config.model.text_encoder

        logger.info(f"Loading Qwen text encoder from {text_encoder_config.text_encoder_path} ...")
        thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            text_encoder_config.text_encoder_path,
            torch_dtype=torch.bfloat16,
        )
        # keep the full multimodal Thinker (text + ref_audio); drop the unused vision tower
        if thinker.visual is not None:
            del thinker.visual
            thinker.visual = None
        text_encoder = thinker
        text_processor = Qwen2_5OmniProcessor.from_pretrained(text_encoder_config.text_encoder_path)

        # --- VAE model ---
        logger.info(f"Loading VAE from {vae_config.vae_model_path} ...")
        model_init_kwargs = OmegaConf.to_container(vae_config.get("model_init_kwargs", OmegaConf.create({})), resolve=True)
        vae_model_config = BigVGANFlowVAEConfig.from_dict(model_init_kwargs)
        vae_model = load_vae_model(
            vae_name=vae_config.vae_name,
            vae_cfg=vae_model_config,
            vae_ckpt=vae_config.vae_model_path,
            map_location="cpu",
        )
        vae_model = vae_model.to(self.device).eval()
        vae_model.requires_grad_(False)
        self.vae_model = vae_model

        # build CFMEdit (VAE-latent); Flux2Edit is the only supported backbone
        model_arc = OmegaConf.to_container(config.model.arch, resolve=True)
        model_arc["attn_backend"] = "torch"  # inference does not depend on flash_attn
        schedule_config = OmegaConf.to_container(config.model.get("schedule", OmegaConf.create({})), resolve=True)

        logger.info("Building CFMEdit model ...")
        model = CFMEdit(
            transformer=Flux2Edit(
                **model_arc,
                latent_dim=self.latent_dim,
            ),
            text_encoder=text_encoder,
            text_processor=text_processor,
            num_channels=self.latent_dim,
            **schedule_config,
        )
        model = model.to(torch.float32)

        # --- load EMA weights (strip "ema_model." prefix; text_encoder.* comes from Qwen snapshot) ---
        self._load_ema_weights(model, ckpt_path)
        self.model = model.to(self.device)
        self.model.eval()

    def _load_ema_weights(self, model: CFMEdit, ckpt_path: str):
        logger.info(f"Loading model checkpoint from {ckpt_path} ...")
        if ckpt_path.endswith(".safetensors"):
            # clean weights-only export: already stripped of the "ema_model." prefix
            from safetensors.torch import load_file

            state_dict = load_file(ckpt_path, device="cpu")
        else:
            # training checkpoint: pull the EMA weights and strip the "ema_model." prefix
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ema = checkpoint["ema_model_state_dict"]
            state_dict = {k.replace("ema_model.", ""): v for k, v in ema.items() if k not in ("initted", "step")}
            del checkpoint

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        n_missing_te = sum(1 for k in missing if k.startswith("text_encoder."))
        n_missing_other = len(missing) - n_missing_te
        logger.info(
            f"Loaded EMA weights | missing={len(missing)} (text_encoder.*={n_missing_te}, other={n_missing_other}) "
            f"| unexpected={len(unexpected)}"
        )
        if n_missing_other:
            logger.warning(
                "Some non-text-encoder weights are missing; check config/arch matches the checkpoint. "
                f"Examples: {[k for k in missing if not k.startswith('text_encoder.')][:10]}"
            )
        if unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected[:10]}")
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ helpers

    def _load_audio(self, source: str | tuple[torch.Tensor, int]) -> tuple[torch.Tensor, float]:
        if isinstance(source, str):
            audio, sr = torchaudio.load(source)
        else:
            audio, sr = source
            audio = audio.detach().to(device="cpu", dtype=torch.float32)
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            if audio.ndim != 2:
                raise ValueError(f"Audio tensor must have shape [channels, samples], got {tuple(audio.shape)}.")
            if not isinstance(sr, int) or sr <= 0:
                raise ValueError(f"Audio sample rate must be a positive integer, got {sr!r}.")
            if audio.shape[-1] == 0:
                raise ValueError("Audio tensor is empty.")
            if not torch.isfinite(audio).all():
                raise ValueError("Audio tensor contains NaN or Inf.")
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        ref_rms = torch.sqrt(torch.mean(torch.square(audio)))
        if sr != self.target_sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.target_sample_rate)(audio)
        return audio, float(ref_rms)

    @torch.inference_mode()
    def _run(
        self,
        ref_audio: torch.Tensor,  # [1, T] on cpu
        ref_rms: float | None,  # None => no reference audio, skip output RMS restore
        messages: list,  # single-sample chat messages (list of turns)
        gen_latent_len: int,
        *,
        nfe: int,
        cfg_strength: float,
        sway_sampling_coef: float,
        t_grid: list[float] | None,
        seed: int | None,
    ) -> torch.Tensor:
        if ref_rms is None:
            ref_latent_lens_t = torch.zeros(1, dtype=torch.long, device=self.device)
            total_latent_lens_t = torch.tensor([gen_latent_len], dtype=torch.long, device=self.device)
            ref_latents = torch.zeros(1, 0, self.latent_dim, device=self.device, dtype=torch.float32)
        else:
            ref_audio = ref_audio.to(self.device).unsqueeze(0)  # [1, 1, T]
            ref_latent_len = ref_audio.shape[-1] // self.downsample_rate
            total_latent_len = ref_latent_len + gen_latent_len

            ref_latent_lens_t = torch.tensor([ref_latent_len], dtype=torch.long, device=self.device)
            total_latent_lens_t = torch.tensor([total_latent_len], dtype=torch.long, device=self.device)
            audio_lens_t = ref_latent_lens_t * self.downsample_rate

            # --- online VAE encode + normalize ---
            ref_latents, enc_latent_lens = self.vae_model.encoding_and_normalization(
                ref_audio,
                sample_lengths=audio_lens_t,
            )
            ref_latent_lens_t = torch.minimum(ref_latent_lens_t, enc_latent_lens.to(ref_latent_lens_t.device))

        # --- CFM sample in latent space ---
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.device.startswith("cuda")):
            cond_inputs = self.model.build_cond_inputs([messages], self.model.text_processor)
            generated, _ = self.model.sample(
                cond=ref_latents,
                text=cond_inputs,
                duration=total_latent_lens_t,
                lens=ref_latent_lens_t,
                steps=nfe,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                t_grid=t_grid,
                no_ref_audio=False,
                seed=seed,
            )  # [1, T_total, D]

        gen = generated[0]
        rl = ref_latent_lens_t[0].item()
        tl = total_latent_lens_t[0].item()
        gen_latent = gen[rl:tl, :].unsqueeze(0)  # [1, T_new, D]
        if gen_latent.shape[1] == 0:
            raise RuntimeError("Empty generated latent (target duration collapsed to 0).")
        if torch.isnan(gen_latent).any() or torch.isinf(gen_latent).any():
            raise RuntimeError("Generated latent contains NaN/Inf.")

        gen_latent = self.vae_model.denormalize(gen_latent)
        gen_latent = gen_latent.permute(0, 2, 1)  # [1, D, T_new]

        gen_audio = self.vae_model.inference_from_latents(gen_latent).cpu()
        if gen_audio.ndim == 3:
            gen_audio = gen_audio.squeeze(0)  # [1, T_wav]
        if torch.isnan(gen_audio).any() or torch.isinf(gen_audio).any():
            raise RuntimeError("Generated audio contains NaN/Inf.")

        return gen_audio.to(torch.float32)

    # ------------------------------------------------------------------ public API

    def generate(
        self,
        messages: list,  # caller-composed ChatML turns (must carry a user audio item)
        *,
        audio: str | tuple[torch.Tensor, int] | None = None,
        gen_seconds: float | None = None,
        nfe: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        t_grid: list[float] | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        wav_path = audio or extract_audio_path(messages, required=False)
        if wav_path is not None:
            ref_audio, ref_rms = self._load_audio(wav_path)
        else:
            # no reference audio (text-only instruct TTS): empty reference, ref_rms=None
            ref_audio = torch.zeros(1, 0)
            ref_rms = None
            for m in messages:
                if m.get("role") != "user":
                    continue
                for c in m.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "text" and not c["text"].endswith("|<no_prompt_audio>|"):
                        c["text"] = c["text"] + "|<no_prompt_audio>|"
        ref_latent_len = ref_audio.shape[-1] // self.downsample_rate  # 0 when no reference

        if gen_seconds is not None:
            gen_latent_len = max(1, int(math.ceil(gen_seconds * self.target_sample_rate / self.downsample_rate)))
        else:
            # default: regenerate a segment as long as the source clip
            gen_latent_len = max(1, ref_latent_len)

        # AuK-Flash: ignore any caller-supplied sampling knobs and pin the distilled recipe —
        # 4-step time_grid (from the checkpoint metadata), CFG off. The DMD student bakes in its
        # own guidance, so re-adding CFG blows up the amplitude (clips hard). Base AuK is unrestricted.
        if self.is_flash:
            nfe = 4
            cfg_strength = 0.0
            sway_sampling_coef = None
            t_grid = [0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0]

        audio_out = self._run(
            ref_audio,
            ref_rms,
            messages,
            gen_latent_len,
            nfe=nfe,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            t_grid=t_grid,
            seed=seed,
        )
        return audio_out, self.target_sample_rate


def extract_audio_path(messages: list, *, required: bool = True) -> str | None:
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, dict) and c.get("type") == "audio":
                path = c.get("audio") or c.get("audio_url")
                if path:
                    return path
    if required:
        raise ValueError("generate() needs a user audio item (type=audio) in messages to VAE-encode.")
    return None


def get_gen_duration(
    audio: str | None = None,
    ref_text: str | None = None,
    gen_text: str | None = None,
    gen_seconds: float | None = None,
    speed: float = 1.0,
) -> float | None:
    if gen_seconds:
        return float(gen_seconds)

    ref_seconds = None
    if audio:
        info = torchaudio.info(audio)
        ref_seconds = info.num_frames / info.sample_rate

    if ref_text and gen_text and ref_seconds is not None:
        return ref_seconds * len(gen_text.encode("utf-8")) / max(1, len(ref_text.encode("utf-8"))) / speed
    return ref_seconds


def save_audio(audio: torch.Tensor, sample_rate: int, output_path: str):
    """Save a [1, T] / [T] float tensor to ``output_path`` (creates parent dirs)."""
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    torchaudio.save(output_path, audio.to(torch.float32), sample_rate)
