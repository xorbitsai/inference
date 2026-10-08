import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar


@dataclass
class ModelConfig:
    latent_dim: int = 128
    latent_patch_size: int = 1
    model_dim: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    mlp_ratio: float = 2.875
    text_mlp_ratio: float | None = 2.6
    speaker_mlp_ratio: float | None = 2.6
    dropout: float = 0.0
    text_vocab_size: int = 102400
    text_tokenizer_repo: str = "sbintuitions/sarashina2.2-0.5b"
    text_encoder_revision: str | None = None
    text_add_bos: bool = True
    text_encoder_type: str = "scratch"
    pretrained_projector_type: str = "linear"
    pretrained_projector_hidden_ratio: float = 2.0
    pretrained_projector_dropout: float = 0.0
    text_dim: int = 1280
    text_layers: int = 14
    text_heads: int = 10
    use_caption_condition: bool = False
    use_speaker_condition: bool | None = None
    caption_vocab_size: int | None = None
    caption_tokenizer_repo: str | None = None
    caption_add_bos: bool | None = None
    caption_dim: int | None = None
    caption_layers: int | None = None
    caption_heads: int | None = None
    caption_mlp_ratio: float | None = None
    speaker_dim: int = 1280
    speaker_layers: int = 14
    speaker_heads: int = 10
    speaker_patch_size: int = 1
    timestep_embed_dim: int = 512
    adaln_rank: int = 256
    norm_eps: float = 1e-5
    use_duration_predictor: bool = False
    duration_aux_dim: int = 14
    duration_hidden_dim: int = 1024
    duration_layers: int = 3
    duration_dropout: float = 0.1
    duration_attention_heads: int = 8
    duration_architecture: str = "token_sum_adarn_zero_no_aux"
    duration_token_init_frames: float = 9.0
    duration_speaker_fusion: str = "adarn_zero"
    duration_caption_fusion: str = "adarn_zero"
    duration_caption_pooling: str = "masked_mean"

    @property
    def patched_latent_dim(self) -> int:
        return self.latent_dim * self.latent_patch_size

    @property
    def speaker_patched_latent_dim(self) -> int:
        return self.patched_latent_dim * self.speaker_patch_size

    @property
    def use_speaker_condition_resolved(self) -> bool:
        # Legacy compatibility: old caption configs implied no speaker branch.
        if self.use_speaker_condition is None:
            return not bool(self.use_caption_condition)
        return bool(self.use_speaker_condition)

    @property
    def text_mlp_ratio_resolved(self) -> float:
        if self.text_mlp_ratio is None:
            return self.mlp_ratio
        return float(self.text_mlp_ratio)

    @property
    def use_pretrained_text_encoder(self) -> bool:
        return str(self.text_encoder_type).strip().lower() == "pretrained"

    @property
    def caption_vocab_size_resolved(self) -> int:
        if self.caption_vocab_size is None:
            return int(self.text_vocab_size)
        return int(self.caption_vocab_size)

    @property
    def caption_tokenizer_repo_resolved(self) -> str:
        if self.caption_tokenizer_repo is None:
            return self.text_tokenizer_repo
        return str(self.caption_tokenizer_repo)

    @property
    def caption_add_bos_resolved(self) -> bool:
        if self.caption_add_bos is None:
            return bool(self.text_add_bos)
        return bool(self.caption_add_bos)

    @property
    def caption_dim_resolved(self) -> int:
        if self.caption_dim is None:
            return int(self.text_dim)
        return int(self.caption_dim)

    @property
    def caption_layers_resolved(self) -> int:
        if self.caption_layers is None:
            return int(self.text_layers)
        return int(self.caption_layers)

    @property
    def caption_heads_resolved(self) -> int:
        if self.caption_heads is None:
            return int(self.text_heads)
        return int(self.caption_heads)

    @property
    def caption_mlp_ratio_resolved(self) -> float:
        if self.caption_mlp_ratio is None:
            return self.text_mlp_ratio_resolved
        return float(self.caption_mlp_ratio)

    @property
    def speaker_mlp_ratio_resolved(self) -> float:
        if self.speaker_mlp_ratio is None:
            return self.mlp_ratio
        return float(self.speaker_mlp_ratio)


@dataclass
class TrainConfig:
    manifest_path: str = ""
    output_dir: str = "outputs"
    batch_size: int = 8
    num_workers: int = 2
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: int = 2
    dataloader_cuda_prefetch: bool = False
    length_bucket_enabled: bool = False
    length_bucket_window_batches: int = 64
    latent_length_bucket_size: int = 0
    allow_tf32: bool = False
    compile_model: bool = False
    gradient_checkpointing: bool = False
    train_mode: str = "rf"
    learning_rate: float = 1e-4
    pretrained_text_encoder_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    optimizer: str = "muon"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    muon_momentum: float = 0.95
    muon_adjust_lr_fn: str = "match_rms_adamw"
    lr_scheduler: str = "none"
    warmup_steps: int = 0
    caption_warmup: bool = False
    caption_warmup_steps: int = 0
    pretrained_projector_warmup_steps: int = 0
    stable_steps: int = 0
    min_lr_scale: float = 0.1
    max_steps: int = 200000
    log_every: int = 100
    save_every: int = 1000
    checkpoint_best_n: int = 0
    valid_ratio: float = 0.0
    valid_every: int = 0
    progress: bool = True
    progress_all_ranks: bool = False
    precision: str = "bf16"
    grad_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    max_text_len: int = 256
    max_caption_len: int | None = None
    text_condition_dropout: float = 0.1
    caption_condition_dropout: float = 0.1
    speaker_condition_dropout: float = 0.1
    speaker_inversion_enabled: bool = False
    speaker_inversion_tokens: int = 16
    speaker_inversion_init_std: float = 0.02
    speaker_inversion_init_embedding: str | None = None
    max_latent_steps: int = 750
    ref_min_seconds: float = 1.0
    ref_max_seconds: float = 120.0
    fixed_target_latent_steps: int | None = 750
    fixed_target_full_mask: bool = True
    rf_loss_mode: str = "echo"
    duration_loss_weight: float = 0.1
    duration_backprop_to_condition: bool = False
    duration_speaker_dropout: float = 0.1
    duration_caption_dropout: float = 0.1
    duration_huber_delta: float = 0.1
    timestep_logit_mean: float = 0.0
    timestep_logit_std: float = 1.0
    timestep_stratified: bool = True
    timestep_min: float = 0.001
    timestep_max: float = 0.999
    wandb_enabled: bool = False
    wandb_project: str = "Irodori-TTS"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"
    ddp_find_unused_parameters: bool = False
    lora_enabled: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    lora_target_modules: str = "diffusion_attn"
    lora_modules_to_save: str | None = "auto"
    seed: int = 0


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def dump_configs(path: str | Path, model_cfg: ModelConfig, train_cfg: TrainConfig) -> None:
    save_json(path, {"model": asdict(model_cfg), "train": asdict(train_cfg)})


T = TypeVar("T")


def load_config_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a training config YAML. Returns {} for an empty document.
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for --config support. Install with `pip install pyyaml`."
        ) from exc

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return payload


def merge_dataclass_overrides(base: T, overrides: dict[str, Any] | None, section: str) -> T:
    """
    Merge mapping overrides into a dataclass instance with key validation.
    """
    if overrides is None:
        return base
    if not isinstance(overrides, dict):
        raise ValueError(f"Config section '{section}' must be a mapping.")

    allowed = {f.name for f in fields(base)}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in '{section}' config: {unknown}")

    merged = asdict(base)
    merged.update(overrides)
    return type(base)(**merged)
