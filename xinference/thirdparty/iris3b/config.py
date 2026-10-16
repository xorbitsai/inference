"""Typed configuration tree.

Configs compose as: structured defaults (this file) <- experiment YAML <- CLI
dotlist overrides. The model, text-encoder, flow and optimizer defaults
describe Iris-3B; per-stage data, resolution and schedule settings live in
``configs/iris3b``.
"""

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

DEFAULT_VALIDATION_PROMPTS: list[str] = [
    "a golden retriever puppy sitting in a field of tall grass at sunset",
    "close-up portrait of an elderly fisherman with a weathered face, soft window light, shallow depth of field",
    "a cozy reading nook with a window seat and rain on the glass, warm lamp light, watercolor illustration",
    "an astronaut tending a vegetable garden inside a glass dome on the moon, cinematic lighting",
    "a snow-covered mountain village at dawn with smoke rising from the chimneys, highly detailed",
    "a neon sign above a small night-market stall that reads \u201cOPEN LATE\u201d",
    "a hand-painted wooden sign in a flower shop window that says \u201cFresh Tulips Today\u201d",
]


@dataclass
class PixelStageConfig:
    """Pixel-level output refinement pathway (PiT blocks)."""

    enabled: bool = True
    depth: int = 4
    hidden_size: int = 16  # PiT per-pixel channel width
    attn_hidden_size: int = 1280  # PiT per-patch compacted-attention width
    num_heads: int = 10
    mlp_ratio: float = 4.0
    # "pre": DiT-style input modulation + gates (6 params/pixel).
    # "post": affine on branch outputs, no gates (4 params/pixel).
    modulation: str = "post"
    abs_pos_embed: bool = True  # fixed full-resolution 2D sincos on PiT pixel tokens


@dataclass
class ModelConfig:
    # patch-level stage
    block: str = "single_stream"  # mmdit (dual-stream) | single_stream
    # Hybrid trunk: the first ``dual_depth`` blocks are dual-stream MM-DiT, the
    # rest are ``block``. A dual block costs exactly 2x the parameters of a
    # single-stream one at the same FLOPs, so this allocates parameters at fixed
    # compute. 0 keeps the homogeneous ``block`` stack.
    dual_depth: int = 8
    # The last block's text output tokens are discarded downstream. "drop" stops
    # building that block's text output path (projection, MLP, their norms and
    # gates), so no parameter is left gradient-free and DDP no longer needs
    # find_unused_parameters. Text still enters that block's attention as keys
    # and values, so the image tokens still read the caption.
    final_block_text: str = "keep"  # keep | drop
    hidden_size: int = 2560
    depth: int = 24
    num_heads: int = 20
    # grouped-query attention; ``None`` keeps full MHA with a fused QKV projection
    num_kv_heads: int | None = 5
    gated_attention: bool = True  # token-wise sigmoid gate before the attention output projection
    sandwich_norm: bool = True  # RMSNorm on each branch output before its residual add
    patch_size: int = 16
    in_channels: int = 3
    mlp_ratio: float = 4.0  # SwiGLU applies the 2/3 width rule on top
    qkv_bias: bool = False
    qk_norm: bool = True
    norm_eps: float = 1e-6
    attn_backend: str = "sdpa"  # sdpa | torch_flash | torch_cudnn | fa3 | fa4
    # patch-block adaLN parameterization: "per_block" = one Linear(D, 6D) per
    # block per stream; "shared_lowrank" = one shared core per stream plus a
    # per-block rank-r conditional residual U_i V_i; "shared_bias" = the same
    # shared core plus a per-block learned bias, so blocks share one timestep
    # response and differ only by a constant offset.
    modulation: str = "shared_bias"
    modulation_rank: int = 64  # shared_lowrank only
    # timestep conditioning
    timestep_max_period: float = 10.0  # flow time lives in [0, 1000]
    # True zero-initializes every adaLN projection (adaLN-zero), so the patch
    # blocks start as identity maps; False keeps PyTorch's default init
    adaln_zero_init: bool = True
    # positional encoding
    rope_theta: float = 10000.0
    rope_scale: float = 16.0  # image RoPE coords span [0, scale] at any resolution
    # "square": both axes normalized to [0, scale] independently, so the
    # encoding is aspect-blind and its angular step is anisotropic on
    # non-square grids. "isotropic": one shared step, aspect ratio preserved.
    rope_aspect: str = "isotropic"
    # reserve the N slowest (x, y) frequency pairs for a frame axis, so a second
    # image can be tagged in-context after pretraining (see nn/rope.py). Those
    # pairs are numerically inert under rope_theta=1e4 + rope_scale=16, and a
    # constant frame index cancels in attention, so 0 vs N is behaviorally
    # equivalent for single-image training.
    rope_frame_pairs: int = 0
    rope_frame_theta: float = 10.0
    text_rope: bool = True
    text_rope_theta: float = 10000.0
    text_abs_pos_embed: bool = True  # learned N(0,1) positional table on text tokens
    # text stream
    text_dim: int = 2560
    text_len: int = 300
    # text adapter capacity: "linear" = Linear + RMSNorm; "blocks2" adds two
    # token-axis transformer blocks; "lap_blocks2" first aggregates selected
    # frozen-encoder layers, then applies the same token-axis refiner.
    text_adapter: str = "lap_blocks2"
    text_lap_num_layers: int = 12  # one per text_encoder.hidden_layers entry
    text_lap_num_heads: int = 32
    text_lap_mlp_ratio: float = 1.3
    # REPA feature capture (1-based block index; 0 disables)
    repa_layer: int = 10
    pixel: PixelStageConfig = field(default_factory=PixelStageConfig)

    def validate(self) -> None:
        if not 0 <= self.dual_depth <= self.depth:
            raise ValueError(
                f"model.dual_depth must be in [0, model.depth={self.depth}], got {self.dual_depth}; "
                "it is the number of leading dual-stream blocks, not a depth increment"
            )
        if self.final_block_text not in ("keep", "drop"):
            raise ValueError(f"unknown model.final_block_text '{self.final_block_text}' (keep | drop)")


@dataclass
class FlowConfig:
    """Rectified-flow objective and its discrete training schedule."""

    num_train_timesteps: int = 1000
    shift: float = 4.0  # sigma' = shift*sigma / (1 + (shift-1)*sigma)
    timestep_sampler: str = "logit_normal"  # logit_normal | uniform
    logit_mean: float = 0.0
    logit_std: float = 1.0
    prediction: str = "v"  # v | x
    x_pred_sigma_min: float = 0.05  # clamp for x-prediction velocity conversion
    # resolve `shift` from the stage's image token count instead of using it
    # verbatim. "sd3": shift * sqrt(tokens / shift_base_tokens) (2x linear
    # resolution -> 2x shift). "flux": exp(affine in token count), which grows
    # much more slowly. The result is a stage scalar and is recorded in the
    # resolved config.
    shift_law: str = "none"  # none | sd3 | flux
    shift_base_tokens: int = 256  # 256px at patch 16, the anchor for "sd3"


@dataclass
class TextEncoderConfig:
    name: str = "qwen3_vl"
    pretrained: str = "Qwen/Qwen3-VL-4B-Instruct"
    dim: int = 2560
    max_length: int = 300
    dtype: str = "bfloat16"
    # Hugging Face attention backend for encoders that expose it.
    attn_implementation: str = "sdpa"
    # 1-based post-block states returned for layerwise aggregation: every
    # third layer of the 36-layer language stack, starting at 2
    hidden_layers: list[int] = field(default_factory=lambda: [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35])
    null_embed_dir: str = "output/pretrained_models"
    # torch.compile the frozen decoder (default inductor mode)
    compile: bool = False
    # caption budget overflow: the template suffix is appended after the
    # caption is truncated, so the chat turn can never be eaten. "warn" counts
    # and logs overflows once per interval, "error" refuses the batch.
    on_caption_overflow: str = "warn"  # warn | error | silent


@dataclass
class PerfConfig:
    """Wall-clock training optimizations.

    Tier 1 flags are mathematically exact: they change kernel scheduling,
    never the computed update (bitwise drift from kernel selection only).
    Tier 2 flags change numerics within training noise.
    All flags default off.
    """

    # -- tier 1 (exact) --
    foreach_ema: bool = False  # multi-tensor EMA update (2 kernels instead of 2/param)
    fused_adamw: bool = False  # torch.optim.AdamW(fused=True); adamw only
    # draw the CFG dropout mask first and skip text encoding for dropped rows
    # (encoder consumes no RNG, so the draw sequence is unchanged)
    skip_dropped_text: bool = False
    # single post-backward sync per step: the non-finite guard moves from the
    # loss (pre-backward) to the clipped grad norm, which is a superset check
    lazy_sync: bool = False
    # -- tier 2 (numerics drift) --
    compile: bool = False  # torch.compile the diffusion model
    compile_mode: str = "max-autotune"
    # "block" compiles each transformer block separately: one graph per block
    # shape instead of one per whole-model shape, so a bucketed run pays the
    # autotune cost once per token count rather than once per (H, W) grid.
    compile_scope: str = "block"  # block | model
    # "auto" = static at a single fixed resolution, dynamic once the run uses
    # more than one input shape (bucketed / multi-resolution stages).
    compile_dynamic: str = "auto"  # auto | true | false
    compile_fullgraph: bool = False  # raise on graph breaks instead of falling back
    # dynamo's default cache limit is 8; exceeding it marks the frame SKIP_CODE
    # and silently drops to eager for the rest of the run. 0 keeps the default.
    recompile_limit: int = 64

    def validate(self) -> None:
        if self.compile_scope not in ("block", "model"):
            raise ValueError(f"train.perf.compile_scope must be block or model, got {self.compile_scope!r}")
        if self.compile_dynamic not in ("auto", "true", "false"):
            raise ValueError(
                f"train.perf.compile_dynamic must be auto, true or false, got {self.compile_dynamic!r}"
            )


@dataclass
class RepaConfig:
    """Representation alignment against a frozen vision teacher (stage 1 only)."""

    weight: float = 0.0  # 0 disables the loss and the teacher
    # "repa": 3-layer MLP projector on raw teacher tokens.
    # "irepa" (arXiv 2512.10794): conv3x3 projector on the student token grid
    # + spatial normalization of teacher tokens (global component removed).
    variant: str = "repa"  # repa | irepa
    spatial_norm_gamma: float = 1.0  # irepa: fraction of the token mean removed
    # "hub": torch.hub repo/entrypoint (DINOv2, or DINOv3 with a license-gated
    # local checkpoint). "hf": `teacher` is a transformers repo id loaded via
    # AutoModel (DINOv3 safetensors; register tokens are sliced off).
    teacher_source: str = "hub"  # hub | hf
    teacher_hub: str = "facebookresearch/dinov2"  # torch.hub repo
    teacher: str = "dinov2_vitb14"  # hub entrypoint or HF repo id
    # optional checkpoint path/URL forwarded to torch.hub as `weights=`;
    # unused for teacher_source=hf
    teacher_weights: str = ""
    teacher_dim: int = 768
    proj_hidden_dim: int = 2048
    teacher_image_size: int = 224  # 224/14 -> 16x16 teacher tokens (= student grid at 256px)
    teacher_patch_size: int = 14  # DINOv2/v3 ViT patch stride
    # size the teacher input so its grid EQUALS the student's, instead of the
    # fixed square above. Required for a patch-level target away from 256px:
    # otherwise a 32x32 student is pooled to 16x16 and four student patches
    # share one teacher target. Costs teacher FLOPs quadratically in the grid.
    teacher_match_student: bool = False


@dataclass
class DataConfig:
    data_dirs: list[str] = field(default_factory=list)
    type: str = "pixel"  # pixel (fixed square) | pixel_multiscale (AR buckets)
    image_size: int = 512
    # how a sample's training (H, W) is chosen from its native size, and hence
    # how batches are grouped. "fixed": one square image_size. "bucket": snap to
    # aspect_ratio_bucket's table. "area": table-free, keep the native ratio at
    # image_size^2 worth of tokens, both sides rounded to shape_align.
    # Left unset, `pixel_multiscale` still implies "bucket".
    shape_policy: str = ""
    aspect_ratio_bucket: str = "shared21-512"  # key into iris3b.data.buckets.TRAIN_BUCKETS
    shape_align: int = 32  # "area": both sides round to a multiple of this
    shape_max_ratio: float = 4.0  # "area": clamp on the long/short side ratio
    caption_field: str = "caption"
    # when non-empty: sample a caption uniformly from these info-dict keys
    caption_fields: list[str] = field(default_factory=list)
    # whole-shard validation holdout; empty disables the frozen-grid val loss
    # even when train.val_every_steps is set
    val_data_dirs: list[str] = field(default_factory=list)
    num_workers: int = 10

    def resolved_shape_policy(self) -> str:
        """The policy name, defaulting from the dataset-type switch."""
        if self.shape_policy:
            return self.shape_policy
        return "bucket" if self.type.endswith("multiscale") else "fixed"

    def validate(self, patch_size: int = 16) -> None:
        policy = self.resolved_shape_policy()
        if policy not in ("fixed", "bucket", "area"):
            raise ValueError(f"unknown data.shape_policy '{policy}' (fixed | bucket | area)")
        if policy in ("fixed", "area") and self.image_size % patch_size:
            raise ValueError(
                f"data.image_size {self.image_size} is not divisible by model.patch_size {patch_size}"
            )
        if policy == "area" and self.shape_align % patch_size:
            raise ValueError(
                f"data.shape_align {self.shape_align} is not a multiple of model.patch_size {patch_size}"
            )


ADAMW_DEFAULT_BETAS = (0.9, 0.95)


@dataclass
class OptimizerConfig:
    # muon: Dion's Muon on hidden matrices, AdamW on embeddings, heads and vectors
    name: str = "muon"  # muon | adamw
    lr: float = 1.0e-4
    betas: list[float] = field(default_factory=lambda: list(ADAMW_DEFAULT_BETAS))
    weight_decay: float = 0.0
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_adjust_lr: str = "rms_norm"  # rms_norm | spectral_norm | none
    # rescale lr by effective_batch / base_batch before training
    auto_lr: str = "none"  # none | sqrt | linear
    base_batch_size: int = 256
    schedule: str = "constant"  # constant | cosine
    warmup_steps: int = 2000
    # multiply warmup by world size (a prepared scheduler steps
    # num_processes times per optimizer step)
    scale_warmup_by_world: bool = True

    def validate(self) -> None:
        if self.name not in {"adamw", "muon"}:
            raise ValueError(f"unknown optimizer {self.name!r} (adamw | muon)")
        if self.name == "muon":
            if not 0.0 <= self.muon_momentum < 1.0:
                raise ValueError("train.optimizer.muon_momentum must be in [0, 1)")
            if self.muon_adjust_lr not in {"rms_norm", "spectral_norm", "none"}:
                raise ValueError("train.optimizer.muon_adjust_lr must be rms_norm, spectral_norm, or none")


@dataclass
class EMAConfig:
    enabled: bool = True
    decay: float = 0.9999


@dataclass
class MeshConfig:
    """Device-mesh factorization for sharded strategies.

    ``dp_shard`` ranks hold one copy of the sharded state; ``dp_replicate``
    copies are kept in sync by an all-reduce of the *already sharded*
    gradients. Setting ``dp_shard`` to the GPU count of one node is hybrid
    sharding: every all-gather stays on NVLink and only the sharded gradient
    crosses the network, which is ~1/G of the traffic full sharding sends.
    """

    dp_shard: int = 0  # 0 = the whole world (pure sharding, no replication)
    dp_replicate: int = 0  # 0 = world_size // dp_shard


@dataclass
class DDPConfig:
    """Bucketing and reduction knobs for replicated data parallelism."""

    # "auto" resolves from the model: a stack whose last block still computes
    # text outputs has a constant unused tail and needs the graph walk.
    find_unused_parameters: str = "auto"  # auto | true | false
    gradient_as_bucket_view: bool = False
    # one _foreach_copy_ plus one flat div_ per bucket instead of a copy and a
    # div per parameter
    batched_grad_copy: bool = False
    # asserts the unused set is identical on every rank for the whole run; a
    # rank-divergent set desyncs or hangs
    skip_all_reduce_unused_params: bool = False
    bucket_cap_mb: int = 0  # 0 = torch default (25)
    forward_sync_buffers: bool = True


@dataclass
class DistConfig:
    """Data-parallel strategy. Defaults reproduce plain single-node DDP.

    ``fsdp2`` shards per parameter as DTensor, so optimizers that need a
    parameter's own shape or norm stay valid and compiled graphs survive
    sharding.
    """

    strategy: str = "ddp"  # ddp | fsdp2
    # full = params + grads + optimizer states; grad_op = grads + states only;
    # the hybrid variants shard inside a mesh group and replicate across groups
    sharding: str = "full"  # full | grad_op | hybrid | hybrid_grad_op
    mesh: MeshConfig = field(default_factory=MeshConfig)
    param_dtype: str = "bf16"  # sharded parameter/all-gather precision
    # gradient reduction precision. fp32 costs bandwidth and removes the
    # bf16 rounding that grows with the number of ranks reduced over.
    reduce_dtype: str = "fp32"  # fp32 | bf16
    reshard_after_forward: bool = True  # False trades memory for comms
    # "auto" = a single rank-0 file on one node, per-node replicas beyond that.
    # "rank0" is correct only on a filesystem every rank can read.
    checkpoint_io: str = "auto"  # auto | rank0 | per_node | dcp
    ddp: DDPConfig = field(default_factory=DDPConfig)

    @property
    def sharded(self) -> bool:
        return self.strategy == "fsdp2"

    @property
    def hybrid(self) -> bool:
        return self.sharding in ("hybrid", "hybrid_grad_op")

    def validate(self, train: "TrainConfig") -> None:
        if self.strategy not in ("ddp", "fsdp2"):
            raise ValueError(f"train.dist.strategy must be ddp or fsdp2, got {self.strategy!r}")
        if self.checkpoint_io not in ("auto", "rank0", "per_node", "dcp"):
            raise ValueError(
                f"train.dist.checkpoint_io must be auto, rank0, per_node or dcp, got {self.checkpoint_io!r}"
            )
        if self.ddp.find_unused_parameters not in ("auto", "true", "false"):
            raise ValueError(
                "train.dist.ddp.find_unused_parameters must be auto, true or false, "
                f"got {self.ddp.find_unused_parameters!r}"
            )
        if not self.sharded:
            if self.sharding != "full":
                raise ValueError(
                    "train.dist.sharding applies to fsdp2 only; "
                    f"strategy={self.strategy!r} shards nothing"
                )
            if self.mesh.dp_shard or self.mesh.dp_replicate:
                raise ValueError("train.dist.mesh applies to fsdp2 only")
            if self.checkpoint_io == "dcp":
                raise ValueError("train.dist.checkpoint_io=dcp requires a sharded strategy")
            return
        if self.sharding not in ("full", "grad_op", "hybrid", "hybrid_grad_op"):
            raise ValueError(
                f"train.dist.sharding must be full, grad_op, hybrid or hybrid_grad_op, got {self.sharding!r}"
            )
        if self.hybrid and self.mesh.dp_shard <= 0:
            raise ValueError(
                "hybrid sharding needs train.dist.mesh.dp_shard (the ranks one copy is "
                "sharded over, normally the GPU count of one node)"
            )
        if self.reduce_dtype not in ("fp32", "bf16"):
            raise ValueError(f"train.dist.reduce_dtype must be fp32 or bf16, got {self.reduce_dtype!r}")
        if train.log_block_grad_norms:
            # per-parameter grads are local shards: per-block norms would be
            # rank-local values logged as if they were global
            raise ValueError(
                f"train.dist.strategy={self.strategy!r} does not support train.log_block_grad_norms"
            )
        if train.resume_optimizer != "full":
            # the "core" prefix load slices flat-index-keyed Adam state; sharded
            # optimizer state is FQN-keyed and has no stable flat order
            raise ValueError(f"train.dist.strategy={self.strategy!r} requires train.resume_optimizer=full")


@dataclass
class TrainConfig:
    batch_size: int = 8  # per GPU
    grad_accum_steps: int = 1
    # when set, the trainer refuses to start unless
    # batch_size x world_size x grad_accum_steps equals this exactly, so a
    # wrong node count or accumulation cannot silently change the global batch
    expected_global_batch: int | None = None
    num_epochs: int = 100
    max_steps: int = 0  # 0 = run num_epochs; else stop after this many optimizer steps
    mixed_precision: str = "bf16"  # bf16 | fp16 | no
    # "none" keeps every activation. "full" recomputes every block. The
    # selective policies keep the compute-intensive results (attention, matmul,
    # convolution) and recompute only cheap pointwise work, which is most of
    # the memory for a fraction of the recompute.
    activation_checkpointing: str = "none"  # none | full | selective_op | selective_layer
    ac_selective_every: int = 2  # selective_layer: fully recompute every Nth block
    gradient_clip: float = 0.5
    text_dropout: float = 0.1  # CFG null substitution probability
    seed: int = 1
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    dist: DistConfig = field(default_factory=DistConfig)
    load_from: str | None = None  # weights only
    resume_from: str | None = None  # weights + optimizer + scheduler + step
    # Required only to migrate checkpoints written before data_position was
    # persisted. It names the checkpoint's original distributed world size.
    resume_data_world_size: int | None = None
    # exact restores the checkpoint's saved cursor. new_phase intentionally
    # retains training state while starting the current dataset from its beginning.
    resume_data_policy: str = "exact"  # exact | new_phase
    # "full" replays the whole optimizer payload. "core" keeps per-param state
    # only for the core model parameters and rebuilds the groups from the live
    # config, so a resume whose auxiliary heads (the REPA projector) differ from
    # the checkpoint carries the surviving moments over and starts new heads cold.
    resume_optimizer: str = "full"  # full | core
    override_lr_on_resume: bool = False
    save_every_steps: int = 500
    save_every_epochs: int = 5
    # checkpoint retention: 0 keeps every step-checkpoint; N keeps the newest
    # N plus any step listed in milestone_steps (never pruned)
    keep_last_checkpoints: int = 0
    milestone_steps: list[int] = field(default_factory=list)
    log_every: int = 20
    # log per-component losses plus a per-block gradient-norm breakdown
    # (embedders, each transformer/PiT block, head; pre-clip, post-allreduce)
    log_block_grad_norms: bool = False
    sample_every_steps: int = 500
    # frozen-grid validation loss: every N steps, forward a fixed holdout slice
    # under a deterministic (noise, timestep) grid and log val/loss
    # (+ per-sigma-quartile bands). 0 disables; requires data.val_data_dirs.
    val_every_steps: int = 0
    val_samples: int = 8192  # 0 = the whole holdout
    validation_prompts: list[str] = field(default_factory=lambda: list(DEFAULT_VALIDATION_PROMPTS))
    nan_loss_tolerance: int = 20
    perf: PerfConfig = field(default_factory=PerfConfig)

    def validate_resume_data(self) -> None:
        if self.resume_data_policy not in {"exact", "new_phase"}:
            raise ValueError(
                f"train.resume_data_policy must be exact or new_phase, got {self.resume_data_policy!r}"
            )
        if self.resume_data_policy == "new_phase" and self.resume_data_world_size is not None:
            raise ValueError(
                "train.resume_data_policy=new_phase is incompatible with train.resume_data_world_size"
            )


@dataclass
class SampleConfig:
    steps: int = 100
    order: int = 2
    cfg_scale: float = 3.0
    cfg_interval: list[float] = field(default_factory=lambda: [0.0, 1.0])
    negative_prompt: str = ""


@dataclass
class Config:
    name: str = "iris-3b"
    work_dir: str = "output/run"
    wandb_project: str = "iris"
    report_to: str = "wandb"  # wandb | tensorboard | none
    tags: list[str] = field(default_factory=list)  # W&B run tags
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    repa: RepaConfig = field(default_factory=RepaConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Merge structured defaults <- YAML file <- dotlist overrides into a typed Config."""
    conf = OmegaConf.structured(Config)
    if path is not None:
        conf = OmegaConf.merge(conf, OmegaConf.load(str(path)))
    if overrides:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_object(conf)


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.structured(cfg), str(path))


# the sections a checkpoint needs at inference; everything else is training state
INFERENCE_SECTIONS = ("model", "text_encoder", "flow")


def _known_keys(schema, raw: dict) -> dict:
    """``raw`` restricted to the keys ``schema`` still defines, recursively."""
    kept: dict = {}
    for key, value in raw.items():
        if key not in schema:
            continue
        node = schema[key]
        kept[key] = _known_keys(node, value) if isinstance(value, dict) and OmegaConf.is_config(node) else value
    return kept


def inference_config(raw: dict, overrides: list[str] | None = None) -> Config:
    """Sampling config: the ``INFERENCE_SECTIONS`` of ``raw`` over the defaults.

    ``raw`` is a training config dict (as embedded in a checkpoint) or an
    exported ``config.yaml``. Keys the current schema does not define are
    dropped; a missing model key surfaces as a state-dict mismatch instead.
    The null-embedding cache location is machine-local and always defaults.
    """
    schema = OmegaConf.structured(Config)
    snapshot = _known_keys(schema, {key: raw[key] for key in INFERENCE_SECTIONS if key in raw})
    snapshot.get("text_encoder", {}).pop("null_embed_dir", None)
    conf = OmegaConf.merge(schema, snapshot)
    if overrides:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_object(conf)
