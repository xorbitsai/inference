from __future__ import annotations

import math

import torch

from .config import TrainConfig


class MuonWithAuxAdamW:
    """
    Simple single-process wrapper:
    - torch.optim.Muon for Muon-compatible parameters
    - AdamW for auxiliary parameters (embeddings, biases, output heads, etc.)
    """

    def __init__(self, muon_opt: torch.optim.Optimizer, aux_opt: torch.optim.Optimizer | None):
        self.muon_opt = muon_opt
        self.aux_opt = aux_opt
        self.param_groups = list(muon_opt.param_groups)
        if aux_opt is not None:
            self.param_groups.extend(aux_opt.param_groups)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon_opt.zero_grad(set_to_none=set_to_none)
        if self.aux_opt is not None:
            self.aux_opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.muon_opt.step()
        if self.aux_opt is not None:
            self.aux_opt.step()

    def state_dict(self) -> dict:
        return {
            "muon": self.muon_opt.state_dict(),
            "aux": None if self.aux_opt is None else self.aux_opt.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if "muon" not in state_dict:
            raise ValueError(
                "MuonWithAuxAdamW state_dict must contain 'muon' key (and optional 'aux' key)."
            )

        self.muon_opt.load_state_dict(state_dict["muon"])
        if self.aux_opt is not None and state_dict.get("aux") is not None:
            self.aux_opt.load_state_dict(state_dict["aux"])


class ScalarLRScheduler:
    """
    Scheduler that applies a scalar multiplier to all optimizer param-group LRs.
    Works with both torch optimizers and MuonWithAuxAdamW wrapper.
    """

    def __init__(self, optimizer, lr_lambda):
        self.optimizer = optimizer
        self.lr_lambda = lr_lambda
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_step = -1

    def step(self) -> None:
        self.last_step += 1
        scale = float(self.lr_lambda(self.last_step))
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups, strict=False):
            group["lr"] = base_lr * scale

    def state_dict(self) -> dict:
        return {
            "base_lrs": list(self.base_lrs),
            "last_step": int(self.last_step),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if "base_lrs" in state_dict:
            loaded_base_lrs = [float(x) for x in state_dict["base_lrs"]]
            if len(loaded_base_lrs) == len(self.optimizer.param_groups):
                self.base_lrs = loaded_base_lrs
        if "last_step" in state_dict:
            self.last_step = int(state_dict["last_step"])


def _use_weight_decay(name: str, p: torch.nn.Parameter) -> bool:
    """
    Approximate Echo/JAX no-decay mask semantics for PyTorch modules.

    Flax side no-decay keys include:
      bias, scale, weight, gate, shift, freqs, phases, out_proj,
      adaln_rank_shift, adaln_rank_scale, adaln_rank_gate.

    In PyTorch:
    - keep all `.bias` no-decay,
    - map Flax `weight` mostly to normalization weights (`*norm*.weight`),
    - apply token-based exclusions for AdaLN / out-proj / freq-phase terms.
    """
    lname = name.lower()

    if lname.endswith(".bias"):
        return False

    # Flax `weight` token mostly corresponds to norm weights in this model family.
    if lname.endswith(".weight") and "norm" in lname:
        return False

    # Echo Flax wd_mask matches exact keys. In this PyTorch model the closest
    # equivalent is to restrict shift/scale/gate exclusions to AdaLN modules.
    if (".attention_adaln." in lname or ".mlp_adaln." in lname) and any(
        token in lname for token in ("shift", "scale", "gate", "adaln_rank_")
    ):
        return False

    if "out_proj" in lname:
        return False

    if "freqs" in lname or "phases" in lname:
        return False

    return True


def _is_pretrained_text_backbone_parameter(name: str) -> bool:
    return name.startswith("pretrained_text_backbone.") or (".pretrained_text_backbone." in name)


def _partition_adamw_params(
    model: torch.nn.Module,
) -> tuple[
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    pretrained_decay: list[torch.nn.Parameter] = []
    pretrained_no_decay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        has_decay = _use_weight_decay(name, p)
        if _is_pretrained_text_backbone_parameter(name):
            if has_decay:
                pretrained_decay.append(p)
            else:
                pretrained_no_decay.append(p)
        elif has_decay:
            decay.append(p)
        else:
            no_decay.append(p)
    return decay, no_decay, pretrained_decay, pretrained_no_decay


def _partition_muon_params(
    model: torch.nn.Module,
) -> tuple[
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
]:
    """
    Split parameters into Muon-compatible tensors and aux-Adam tensors,
    each with decay/no-decay partitions.
    """
    muon_decay: list[torch.nn.Parameter] = []
    muon_no_decay: list[torch.nn.Parameter] = []
    aux_decay: list[torch.nn.Parameter] = []
    aux_no_decay: list[torch.nn.Parameter] = []
    pretrained_decay: list[torch.nn.Parameter] = []
    pretrained_no_decay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        has_decay = _use_weight_decay(name, p)
        if _is_pretrained_text_backbone_parameter(name):
            # Fine-tune every pretrained-backbone parameter with AdamW. Muon's
            # matrix update is intentionally not mixed into the same backbone.
            if has_decay:
                pretrained_decay.append(p)
            else:
                pretrained_no_decay.append(p)
            continue
        # Muon is intended for hidden matrix-like weights.
        # Keep embeddings/output heads/bias-like params on Adam.
        is_muon_candidate = (
            p.ndim >= 2 and "embedding" not in name and not name.endswith("out_proj.weight")
        )
        if is_muon_candidate:
            if has_decay:
                muon_decay.append(p)
            else:
                muon_no_decay.append(p)
        else:
            if has_decay:
                aux_decay.append(p)
            else:
                aux_no_decay.append(p)
    return (
        muon_decay,
        muon_no_decay,
        aux_decay,
        aux_no_decay,
        pretrained_decay,
        pretrained_no_decay,
    )


def _append_param_group(
    groups: list[dict],
    params: list[torch.nn.Parameter],
    *,
    weight_decay: float,
    learning_rate: float,
    group_name: str,
) -> None:
    if params:
        groups.append(
            {
                "params": params,
                "weight_decay": weight_decay,
                "lr": learning_rate,
                "group_name": group_name,
            }
        )


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig):
    opt_name = cfg.optimizer.lower()
    if opt_name == "adamw":
        decay, no_decay, pretrained_decay, pretrained_no_decay = _partition_adamw_params(model)
        param_groups: list[dict] = []
        _append_param_group(
            param_groups,
            decay,
            weight_decay=cfg.weight_decay,
            learning_rate=cfg.learning_rate,
            group_name="main_decay",
        )
        _append_param_group(
            param_groups,
            no_decay,
            weight_decay=0.0,
            learning_rate=cfg.learning_rate,
            group_name="main_no_decay",
        )
        _append_param_group(
            param_groups,
            pretrained_decay,
            weight_decay=cfg.weight_decay,
            learning_rate=cfg.pretrained_text_encoder_learning_rate,
            group_name="pretrained_text_encoder_decay",
        )
        _append_param_group(
            param_groups,
            pretrained_no_decay,
            weight_decay=0.0,
            learning_rate=cfg.pretrained_text_encoder_learning_rate,
            group_name="pretrained_text_encoder_no_decay",
        )
        return torch.optim.AdamW(
            param_groups if param_groups else model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=0.0,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps,
        )
    if opt_name == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "optimizer=muon requires torch.optim.Muon (available in newer PyTorch releases)."
            )
        adjust_lr_fn = cfg.muon_adjust_lr_fn
        if adjust_lr_fn not in {"original", "match_rms_adamw"}:
            raise ValueError(
                "muon_adjust_lr_fn must be one of ['original', 'match_rms_adamw'], "
                f"got {adjust_lr_fn!r}"
            )

        (
            muon_decay,
            muon_no_decay,
            aux_decay,
            aux_no_decay,
            pretrained_decay,
            pretrained_no_decay,
        ) = _partition_muon_params(model)
        muon_param_groups: list[dict] = []
        _append_param_group(
            muon_param_groups,
            muon_decay,
            weight_decay=cfg.weight_decay,
            learning_rate=cfg.learning_rate,
            group_name="main_muon_decay",
        )
        _append_param_group(
            muon_param_groups,
            muon_no_decay,
            weight_decay=0.0,
            learning_rate=cfg.learning_rate,
            group_name="main_muon_no_decay",
        )
        if not muon_param_groups:
            raise ValueError("No Muon-compatible parameters found for optimizer=muon.")

        muon_opt = torch.optim.Muon(
            muon_param_groups,
            lr=cfg.learning_rate,
            weight_decay=0.0,
            momentum=cfg.muon_momentum,
            adjust_lr_fn=adjust_lr_fn,
        )
        aux_opt = None
        aux_param_groups: list[dict] = []
        _append_param_group(
            aux_param_groups,
            aux_decay,
            weight_decay=cfg.weight_decay,
            learning_rate=cfg.learning_rate,
            group_name="main_aux_decay",
        )
        _append_param_group(
            aux_param_groups,
            aux_no_decay,
            weight_decay=0.0,
            learning_rate=cfg.learning_rate,
            group_name="main_aux_no_decay",
        )
        _append_param_group(
            aux_param_groups,
            pretrained_decay,
            weight_decay=cfg.weight_decay,
            learning_rate=cfg.pretrained_text_encoder_learning_rate,
            group_name="pretrained_text_encoder_decay",
        )
        _append_param_group(
            aux_param_groups,
            pretrained_no_decay,
            weight_decay=0.0,
            learning_rate=cfg.pretrained_text_encoder_learning_rate,
            group_name="pretrained_text_encoder_no_decay",
        )
        if aux_param_groups:
            aux_opt = torch.optim.AdamW(
                aux_param_groups,
                lr=cfg.learning_rate,
                weight_decay=0.0,
                betas=(cfg.adam_beta1, cfg.adam_beta2),
                eps=cfg.adam_eps,
            )
        return MuonWithAuxAdamW(muon_opt=muon_opt, aux_opt=aux_opt)

    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def build_scheduler(
    optimizer,
    cfg: TrainConfig,
):
    sched_name = cfg.lr_scheduler.lower()
    if sched_name == "none":
        return None
    if sched_name not in {"cosine", "wsd"}:
        raise ValueError(f"Unsupported lr_scheduler: {cfg.lr_scheduler}")

    max_steps = max(1, int(cfg.max_steps))
    warmup_steps = max(0, int(cfg.warmup_steps))
    stable_steps = max(0, int(cfg.stable_steps))
    min_lr_scale = float(max(0.0, min(1.0, cfg.min_lr_scale)))

    def lr_lambda(step: int) -> float:
        s = int(step)
        if warmup_steps > 0 and s < warmup_steps:
            return float(s + 1) / float(warmup_steps)

        if sched_name == "cosine":
            denom = max(1, max_steps - warmup_steps)
            progress = min(1.0, max(0.0, float(s - warmup_steps) / float(denom)))
        else:
            if s < warmup_steps + stable_steps:
                return 1.0
            denom = max(1, max_steps - warmup_steps - stable_steps)
            progress = min(1.0, max(0.0, float(s - warmup_steps - stable_steps) / float(denom)))

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return ScalarLRScheduler(optimizer, lr_lambda=lr_lambda)


def current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def current_pretrained_text_encoder_lr(optimizer) -> float | None:
    for group in optimizer.param_groups:
        if str(group.get("group_name", "")).startswith("pretrained_text_encoder_"):
            return float(group["lr"])
    return None
