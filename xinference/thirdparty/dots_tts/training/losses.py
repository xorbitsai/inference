"""Loss aggregation helpers shared by training and validation.

The model returns masked per-token / per-patch loss tensors. This module turns
them into numerators/denominators for logging, combines configured loss weights,
and provides distributed reduction helpers.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
import torch.distributed as dist

from ..utils.util import scalar_as_float


@dataclass(frozen=True)
class LossTerm:
    """A loss tensor paired with a same-shape mask.

    ``loss`` stores unreduced per-element values.
    ``mask`` stores the weighting/validity for the same positions.
    """

    loss: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.loss.shape != self.mask.shape:
            raise ValueError(
                "LossTerm expects loss and mask to have the same shape, "
                f"but got {tuple(self.loss.shape)} and {tuple(self.mask.shape)}."
            )


LossTerms: TypeAlias = dict[str, LossTerm]
LossMasks: TypeAlias = dict[str, torch.Tensor]


def _safe_average(numerator: Any, denominator: Any) -> Any:
    """Average safely when a mask may produce a zero denominator."""
    if isinstance(numerator, torch.Tensor):
        denom = denominator
        if not isinstance(denom, torch.Tensor):
            denom = numerator.new_tensor(float(denominator))
        if float(denom.detach().item()) <= 0.0:
            return numerator * 0.0
        return numerator / denom.clamp_min(1.0).to(numerator.dtype)

    denom = float(denominator)
    if denom <= 0.0:
        return 0.0
    return float(numerator) / denom


def _as_weight_map(loss_config) -> dict[str, float]:
    """Extract ``*_weight`` fields from config into ``*_loss`` weights."""
    weights = {}
    for name, value in loss_config.model_dump().items():
        if name.endswith("_weight"):
            weights[f"{name[:-7]}_loss"] = float(value)
    return weights


def accumulate_named_scalars_(
    target: dict[str, float],
    source: dict[str, float],
) -> dict[str, float]:
    """In-place add ``source`` scalar values into ``target`` by key."""
    for name, value in source.items():
        target[name] += float(value)
    return target


def to_host_named_scalars(values: dict[str, Any]) -> dict[str, float]:
    """Convert scalar tensors into plain Python floats for logging/serialization."""
    return {name: scalar_as_float(value) for name, value in values.items()}


def collapse_loss_masks(
    loss_masks: LossMasks,
) -> dict[str, Any]:
    """Reduce each loss mask to its total effective weight."""
    return {name: mask.sum() for name, mask in loss_masks.items()}


def collapse_loss_terms(
    loss_terms: LossTerms,
    *,
    indices: list[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert masked per-sample loss terms into ``sum(loss*mask)`` statistics.

    Returns ``(numerators, normalizers)`` so the caller can aggregate them across
    batches or ranks before taking the final average.
    """
    index = None
    if indices is not None:
        first = next(iter(loss_terms.values()))
        index = torch.tensor(indices, device=first.loss.device, dtype=torch.long)

    numerators = {}
    normalizers = {}
    for name, term in loss_terms.items():
        loss = term.loss
        mask = term.mask
        if index is not None:
            loss = loss.index_select(0, index)
            mask = mask.index_select(0, index)
        mask = mask.to(loss.dtype)
        numerators[name] = (loss * mask).sum()
        normalizers[name] = mask.sum()
    return numerators, normalizers


def collapse_loss_terms_by_source(
    loss_terms: LossTerms,
    *,
    source_names: list[str | None],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Group collapsed loss statistics by dataset/source name within a batch."""
    first = next(iter(loss_terms.values()))
    batch_size = int(first.loss.size(0))
    if len(source_names) != batch_size:
        raise RuntimeError(
            "source_names must align with the batch size for source loss statistics. "
            f"Expected {batch_size}, got {len(source_names)}."
        )

    source_indices: dict[str, list[int]] = defaultdict(list)
    for index, source_name in enumerate(source_names):
        if source_name is None:
            raise RuntimeError("source_names must not contain None.")
        source_indices[str(source_name)].append(index)

    numerators_by_source = {}
    normalizers_by_source = {}
    for source_name, indices in source_indices.items():
        numerators, normalizers = collapse_loss_terms(loss_terms, indices=indices)
        numerators_by_source[source_name] = to_host_named_scalars(numerators)
        normalizers_by_source[source_name] = to_host_named_scalars(normalizers)
    return numerators_by_source, normalizers_by_source


def reduce_loss_statistics(
    numerators: dict[str, Any],
    normalizers: dict[str, Any],
    *,
    loss_config,
) -> dict[str, Any]:
    """Turn aggregated numerators/normalizers into averaged metrics.

    The returned mapping includes each individual loss plus a weighted ``loss``
    field assembled from ``loss_config``.
    """
    weights = _as_weight_map(loss_config)
    reduced = {}
    total_loss: Any = 0.0
    for name, numerator in sorted(numerators.items()):
        value = _safe_average(numerator, normalizers[name])
        reduced[name] = value
        total_loss = total_loss + value * weights.get(name, 1.0)
    reduced["loss"] = total_loss
    return reduced


def reduce_loss_statistics_by_source(
    numerators_by_source: dict[str, dict[str, float]],
    normalizers_by_source: dict[str, dict[str, float]],
    *,
    loss_config,
) -> dict[str, dict[str, Any]]:
    """Apply :func:`reduce_loss_statistics` independently for each source."""
    return {
        source_name: reduce_loss_statistics(
            numerators,
            normalizers_by_source[source_name],
            loss_config=loss_config,
        )
        for source_name, numerators in sorted(numerators_by_source.items())
    }


def compute_gradient_loss(
    loss_terms: LossTerms,
    *,
    global_normalizers: dict[str, float],
    loss_config,
    ddp_world_size: int,
    gradient_accumulation_steps: int,
) -> torch.Tensor:
    """Build the scalar loss used for ``backward()``.

    ``global_normalizers`` is expected to already include cross-rank totals. The
    final scaling by world size and accumulation steps compensates for the mean
    reduction that DDP/Accelerate applies during gradient synchronization.
    """
    numerators, _ = collapse_loss_terms(loss_terms)
    weights = _as_weight_map(loss_config)

    total_loss: Any = 0.0
    for name, numerator in sorted(numerators.items()):
        total_loss = total_loss + _safe_average(
            numerator,
            global_normalizers[name],
        ) * weights.get(name, 1.0)
    return total_loss * float(ddp_world_size) * float(gradient_accumulation_steps)


def sum_named_scalars_across_ranks(
    values: dict[str, float],
    *,
    device: torch.device,
) -> dict[str, float]:
    """All-reduce a dict of scalar values and return summed host floats."""
    names = _gather_string_union_across_ranks(values, device=device)
    if not names:
        return {}

    packed = torch.tensor(
        [float(values.get(name, 0.0)) for name in names],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return {
        name: float(value)
        for name, value in zip(names, packed.tolist(), strict=True)
    }


def sum_grouped_named_scalars_across_ranks(
    values: dict[str, dict[str, float]],
    *,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """All-reduce nested ``group -> metric -> value`` scalar mappings."""
    group_names = _gather_string_union_across_ranks(values, device=device)
    if not group_names:
        return {}

    metric_names = _gather_string_union_across_ranks(
        (
            metric_name
            for group_values in values.values()
            for metric_name in group_values
        ),
        device=device,
    )
    if not metric_names:
        return {group_name: {} for group_name in group_names}

    packed = torch.tensor(
        [
            float(values.get(group_name, {}).get(metric_name, 0.0))
            for group_name in group_names
            for metric_name in metric_names
        ],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    packed = packed.view(len(group_names), len(metric_names))
    return {
        group_name: {
            metric_name: float(packed[group_index, metric_index].item())
            for metric_index, metric_name in enumerate(metric_names)
        }
        for group_index, group_name in enumerate(group_names)
    }


def accumulate_grouped_named_scalars_(
    target: dict[str, dict[str, float]],
    source: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """In-place add nested ``group -> metric -> value`` scalar mappings."""
    for group_name, values in source.items():
        group_target = target.get(group_name)
        if group_target is None:
            group_target = {name: 0.0 for name in values}
            target[group_name] = group_target
        for name, value in values.items():
            group_target[name] += float(value)
    return target


def _gather_string_union_across_ranks(
    values: Iterable[str],
    *,
    device: torch.device,
) -> list[str]:
    strings = sorted({str(value) for value in values})
    if not (dist.is_available() and dist.is_initialized()):
        return strings

    payload = _encode_string_list(strings)
    world_size = dist.get_world_size()
    local_size = torch.tensor([len(payload)], device=device, dtype=torch.int64)
    size_tensors = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_tensors, local_size)

    max_size = max(int(size.item()) for size in size_tensors)
    if max_size <= 0:
        return []

    local_bytes = torch.zeros(max_size, device=device, dtype=torch.uint8)
    if payload:
        local_bytes[: len(payload)] = torch.tensor(
            list(payload),
            device=device,
            dtype=torch.uint8,
        )

    gathered_bytes = [
        torch.empty(max_size, device=device, dtype=torch.uint8)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered_bytes, local_bytes)

    union = set(strings)
    for size_tensor, byte_tensor in zip(size_tensors, gathered_bytes, strict=True):
        size = int(size_tensor.item())
        if size <= 0:
            continue
        union.update(
            _decode_string_list(bytes(byte_tensor[:size].cpu().tolist()))
        )
    return sorted(union)


def _encode_string_list(values: list[str]) -> bytes:
    if any("\0" in value for value in values):
        raise ValueError("Distributed scalar keys must not contain NUL characters.")
    return "\0".join(values).encode("utf-8")


def _decode_string_list(payload: bytes) -> list[str]:
    if not payload:
        return []
    return payload.decode("utf-8").split("\0")


__all__ = [
    "accumulate_grouped_named_scalars_",
    "LossMasks",
    "LossTerm",
    "LossTerms",
    "accumulate_named_scalars_",
    "collapse_loss_masks",
    "collapse_loss_terms",
    "collapse_loss_terms_by_source",
    "compute_gradient_loss",
    "reduce_loss_statistics",
    "reduce_loss_statistics_by_source",
    "sum_grouped_named_scalars_across_ranks",
    "sum_named_scalars_across_ranks",
    "to_host_named_scalars",
]
