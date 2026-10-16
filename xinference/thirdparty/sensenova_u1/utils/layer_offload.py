"""Layer offload wrapper for memory-efficient inference.

Keeps each layer of an ``nn.ModuleList`` in CPU pinned memory and moves it
onto an accelerator device (CUDA or XPU) on demand. Two modes share a single
:class:`LayerOffloadWrapper`:

- ``prefetch_count == 0`` — synchronous: load before forward, evict after.
- ``prefetch_count >= 1`` — asynchronous: a dedicated CUDA stream prefetches
  the next ``prefetch_count`` layers so the H2D copy overlaps compute.

General-purpose: works with any ``nn.Module`` whose forward iterates over a
``nn.ModuleList`` attribute (``transformer_blocks``, ``layers``, …). Each
layer is evicted back to CPU immediately after its forward completes; in
async mode prefetch wraps around modulo the layer count so the last layer's
prefetch warms up early layers for the next forward pass.

Inference-only — the eviction-after-forward design destroys gradient flow,
so :meth:`__init__` rejects models in training mode.

Origin: adapted from `Lightricks/LTX-2 <https://github.com/Lightricks/LTX-2>`_.

Example
-------
>>> model = build_my_model(device=torch.device("cpu")).eval()
>>> model = LayerOffloadWrapper(
...     model,
...     layers_attr="transformer_blocks",
...     target_device=torch.device("cuda:0"),
...     prefetch_count=2,
... )
>>> out = model(inputs)
>>> model.teardown()
"""

from __future__ import annotations

import functools
import itertools
import logging
import math
from typing import Any

import torch
from torch import nn

from .accel import accel_module as _accel
from .accel import require_accelerator as _require_accelerator

logger = logging.getLogger(__name__)

_GROUP_SHARED = "shared"
_GROUP_UNDERSTANDING = "understanding"
_GROUP_GENERATION = "generation"
_ALL_TENSOR_GROUPS = frozenset({_GROUP_SHARED, _GROUP_UNDERSTANDING, _GROUP_GENERATION})
_GIB = 1024**3
DEFAULT_FAST_VRAM_FRACTION = 0.90
DEFAULT_FAST_VRAM_HEADROOM_GIB = 2.0
DEFAULT_FAST_ACTIVATION_RESERVE_GIB = 4.0
_PHASE_CALLBACK_ATTR = "_layer_offload_phase_callback"
_MISSING = object()
# The prefix understanding path uses eager attention with a float32 additive
# mask.  At its peak, BF16 scores, promoted/masked scores, and float32 softmax
# workspace overlap.  Twelve bytes per score is deliberately conservative and
# matched the measured 8,442-token H100 peak with roughly 10% safety margin.
_EAGER_ATTENTION_BYTES_PER_SCORE = 12


def _resident_memory_limit(
    total_memory_bytes: int,
    *,
    memory_fraction: float = DEFAULT_FAST_VRAM_FRACTION,
    headroom_bytes: int = int(DEFAULT_FAST_VRAM_HEADROOM_GIB * _GIB),
    budget_bytes: int | None = None,
) -> int:
    """Return the allocator watermark used by generation-weight residency.

    An explicit ``budget_bytes`` overrides the fraction-derived budget. Both
    paths still preserve ``headroom_bytes`` below physical device capacity.
    """
    if total_memory_bytes < 0:
        raise ValueError("total_memory_bytes must be >= 0")
    if not math.isfinite(memory_fraction) or not 0 < memory_fraction <= 1:
        raise ValueError("memory_fraction must satisfy 0 < value <= 1")
    if headroom_bytes < 0:
        raise ValueError("headroom_bytes must be >= 0")
    if budget_bytes is not None and budget_bytes <= 0:
        raise ValueError("budget_bytes must be > 0 when set")
    if total_memory_bytes <= headroom_bytes:
        return 0
    requested_limit = budget_bytes if budget_bytes is not None else int(total_memory_bytes * memory_fraction)
    return min(requested_limit, total_memory_bytes - headroom_bytes)


def _log_vram(label: str, target_device: torch.device, *, reset_peak: bool = False) -> None:
    """Cheap VRAM snapshot for diagnosing offload-mode leaks across repeated
    runs (notably under ComfyUI). Never raises; logs at INFO so it shows up
    without explicit DEBUG opt-in.
    """
    try:
        accel = _accel(target_device)
        if not accel.is_available():
            return
        alloc = accel.memory_allocated(target_device) / (1024**3)
        reserved = accel.memory_reserved(target_device) / (1024**3)
        peak = accel.max_memory_allocated(target_device) / (1024**3)
        logger.info(
            "[layer_offload vram] %-40s | alloc=%6.2f GiB  reserved=%6.2f GiB  peak=%6.2f GiB",
            label,
            alloc,
            reserved,
            peak,
        )
        if reset_peak:
            accel.reset_peak_memory_stats(target_device)
    except Exception as exc:  # pragma: no cover - diagnostic only
        logger.debug("vram log %r failed: %s", label, exc)


def _resolve_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    """Resolve a dotted attribute path like ``'model.language_model.layers'``."""
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}")
    return obj


def _resolve_modules(module: nn.Module, dotted_paths: tuple[str, ...]) -> tuple[nn.Module, ...]:
    """Resolve model-declared module paths used by inference phase offload."""
    resolved: list[nn.Module] = []
    for dotted_path in dotted_paths:
        obj: Any = module
        for part in dotted_path.split("."):
            obj = getattr(obj, part)
        if not isinstance(obj, nn.Module):
            raise TypeError(f"Expected nn.Module at '{dotted_path}', got {type(obj).__name__}")
        resolved.append(obj)
    return tuple(resolved)


def _unique_module_parameters(modules: tuple[nn.Module, ...]) -> tuple[nn.Parameter, ...]:
    """Return parameters from ``modules`` once, preserving declaration order."""
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            parameters.append(parameter)
    return tuple(parameters)


class _PrefixWeightStore:
    """Keep pinned CPU backing for weights needed only before denoising."""

    def __init__(self, modules: tuple[nn.Module, ...], target_device: torch.device) -> None:
        self._target_device = target_device
        self._parameters = _unique_module_parameters(modules)
        self._pinned: dict[int, torch.Tensor] = {}
        self._on_target = False
        # Allocate every backing tensor before rebinding any Parameter.  A
        # pinning failure must leave the model in its original state.
        for parameter in self._parameters:
            pinned = parameter.data.pin_memory(device=target_device.type)
            self._pinned[id(parameter)] = pinned
        for parameter in self._parameters:
            parameter.data = self._pinned[id(parameter)]

    def move_to_target(self) -> None:
        if self._on_target:
            return
        for parameter in self._parameters:
            parameter.data = self._pinned[id(parameter)].to(self._target_device, non_blocking=True)
        self._on_target = True

    def evict_to_cpu(self) -> None:
        if not self._on_target:
            return
        for parameter in self._parameters:
            parameter.data = self._pinned[id(parameter)]
        self._on_target = False

    def parameter_ids(self) -> set[int]:
        return {id(parameter) for parameter in self._parameters}

    def cleanup(self) -> None:
        # Parameter.data intentionally keeps the pinned CPU tensor.  A later
        # wrapper can reuse it without paying the pinning copy again.
        self._parameters = ()
        self._pinned.clear()


def _partition_layer_tensor_names(layer: nn.Module) -> dict[str, str]:
    """Classify layer tensors into shared, understanding, and generation groups.

    NEO-Unify stores the two task branches side by side. Generation modules
    carry a ``_mot_gen`` suffix while their understanding counterparts use the
    same path without the suffix. A tensor without such a paired counterpart
    (for example RoPE buffers) is shared by both branches.

    For ordinary transformer blocks with no ``_mot_gen`` names every tensor is
    classified as shared, preserving the wrapper's general-purpose behaviour.
    """
    names = {name for name, _tensor in itertools.chain(layer.named_parameters(), layer.named_buffers())}
    groups: dict[str, str] = {}
    for name in names:
        parts = name.split(".")
        if any(part.endswith("_mot_gen") for part in parts[:-1]):
            groups[name] = _GROUP_GENERATION
            continue

        has_generation_pair = False
        for idx in range(len(parts) - 1):
            candidate = parts.copy()
            candidate[idx] = f"{candidate[idx]}_mot_gen"
            if ".".join(candidate) in names:
                has_generation_pair = True
                break
        groups[name] = _GROUP_UNDERSTANDING if has_generation_pair else _GROUP_SHARED
    return groups


def _required_tensor_groups(kwargs: dict[str, Any]) -> frozenset[str]:
    """Return the weight groups required by one decoder-layer invocation.

    The model normalises the two branch flags to Python ``bool`` values before
    entering the layer loop. Unknown/custom callers fall back to all groups so
    branch-aware streaming can never omit a weight merely because a module has
    a different forward signature.
    """
    use_understanding = kwargs.get("exist_non_image_gen_tokens")
    use_generation = kwargs.get("exist_image_gen_tokens")
    if not isinstance(use_understanding, bool) or not isinstance(use_generation, bool):
        return _ALL_TENSOR_GROUPS
    if use_understanding and not use_generation:
        return frozenset({_GROUP_SHARED, _GROUP_UNDERSTANDING})
    if use_generation and not use_understanding:
        return frozenset({_GROUP_SHARED, _GROUP_GENERATION})
    return _ALL_TENSOR_GROUPS


def _is_cuda_malloc_async_backend() -> bool:
    """Detect whether the active CUDA caching allocator is ``cudaMallocAsync``.

    The native caching allocator and ``cudaMallocAsync`` differ on a point
    that matters for our cross-stream prefetch: ``cudaMallocAsync`` keeps a
    pool *per stream* and never reuses freed blocks across streams without
    explicit ordering, so allocating on the prefetch stream and freeing on
    the compute stream causes the reserved pool to grow without bound. The
    native allocator handles this case with ``record_stream`` and reuses
    blocks freely.

    ComfyUI launchers commonly set
    ``PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync``; standalone Python
    runs typically don't.
    """
    try:
        return torch.cuda.is_available() and torch.cuda.get_allocator_backend() == "cudaMallocAsync"
    except Exception:
        return False


def _audit_lazy_state(
    model: nn.Module,
    target_device: torch.device,
    managed_tensor_ids: set[int],
) -> int:
    """Move any params/buffers stranded off ``target_device`` after the first
    forward (lazy buffers materialised inside ``forward()``) onto it.

    Returns the number of tensors moved. Tensors already managed by the
    offload store are skipped — they are intentionally rotated between
    pinned CPU and GPU. Anything else that ends up on the wrong device is
    almost certainly a lazy buffer (e.g. an attention mask cache) that the
    constructor could not see, and it lives on GPU permanently from here on.
    """
    moved = 0
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        if id(tensor) in managed_tensor_ids:
            continue
        if tensor.device != target_device:
            tensor.data = tensor.data.to(target_device)
            moved += 1
    return moved


class _LayerStore:
    """Holds CPU-pinned copies of every parameter/buffer of every offloaded layer.

    Tracks which layers currently reside on GPU so the prefetcher and evictor
    can make correct decisions in async mode. In sync mode the bookkeeping is
    free overhead.
    """

    def __init__(self, layers: nn.ModuleList, target_device: torch.device) -> None:
        self.target_device = target_device
        self.num_layers = len(layers)

        # ``Tensor.pin_memory()`` defaults to CUDA; XPU needs an explicit
        # device kind so the host buffer is registered with the right driver.
        self._pin_device = target_device.type

        self._pinned: list[dict[str, torch.Tensor]] = []
        self._tensor_groups: list[dict[str, str]] = []
        self._resident_groups: list[set[str]] = []
        self._on_gpu: set[int] = set()
        self._managed_tensor_ids: set[int] = set()
        self._bytes_moved = 0
        self._bytes_moved_by_group = {group: 0 for group in _ALL_TENSOR_GROUPS}

        for layer in layers:
            pinned: dict[str, torch.Tensor] = {}
            for name, tensor in itertools.chain(layer.named_parameters(), layer.named_buffers()):
                self._managed_tensor_ids.add(id(tensor))
                pinned_tensor = tensor.data.pin_memory(device=self._pin_device)
                tensor.data = pinned_tensor
                pinned[name] = pinned_tensor
            self._pinned.append(pinned)
            self._tensor_groups.append(_partition_layer_tensor_names(layer))
            self._resident_groups.append(set())

    def _check_idx(self, idx: int) -> None:
        if idx < 0 or idx >= self.num_layers:
            raise IndexError(f"Layer index {idx} out of range [0, {self.num_layers})")

    def is_on_gpu(self, idx: int, groups: frozenset[str] | None = None) -> bool:
        self._check_idx(idx)
        if groups is None:
            return idx in self._on_gpu
        return groups.issubset(self._resident_groups[idx])

    def move_to_gpu(
        self,
        idx: int,
        layer: nn.Module,
        *,
        groups: frozenset[str] = _ALL_TENSOR_GROUPS,
        non_blocking: bool = False,
    ) -> None:
        """Move layer *idx* parameters from pinned CPU to ``target_device``."""
        self._check_idx(idx)
        missing_groups = groups - self._resident_groups[idx]
        if not missing_groups:
            return
        pinned = self._pinned[idx]
        tensor_groups = self._tensor_groups[idx]
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            group = tensor_groups.get(name, _GROUP_SHARED)
            if name in pinned and group in missing_groups:
                source = pinned[name]
                param.data = source.to(self.target_device, non_blocking=non_blocking)
                moved = source.numel() * source.element_size()
                self._bytes_moved += moved
                self._bytes_moved_by_group[group] += moved
        self._resident_groups[idx].update(missing_groups)
        self._on_gpu.add(idx)

    def evict_to_cpu(
        self,
        idx: int,
        layer: nn.Module,
        *,
        groups: frozenset[str] = _ALL_TENSOR_GROUPS,
    ) -> None:
        """Swap layer *idx* parameters back to their pinned CPU copies."""
        self._check_idx(idx)
        if idx not in self._on_gpu:
            return
        pinned = self._pinned[idx]
        tensor_groups = self._tensor_groups[idx]
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if name in pinned and tensor_groups.get(name, _GROUP_SHARED) in groups:
                param.data = pinned[name]
        self._resident_groups[idx].difference_update(groups)
        if not self._resident_groups[idx]:
            self._on_gpu.discard(idx)

    def tensors_for_groups(
        self,
        idx: int,
        layer: nn.Module,
        groups: frozenset[str],
    ) -> list[torch.Tensor]:
        """Return tensors belonging to ``groups`` for stream recording."""
        self._check_idx(idx)
        tensor_groups = self._tensor_groups[idx]
        return [
            tensor
            for name, tensor in itertools.chain(layer.named_parameters(), layer.named_buffers())
            if tensor_groups.get(name, _GROUP_SHARED) in groups
        ]

    def transfer_stats(self) -> tuple[int, dict[str, int]]:
        return self._bytes_moved, dict(self._bytes_moved_by_group)

    def group_nbytes(self, idx: int, group: str) -> int:
        """Return the pinned bytes belonging to one tensor group in a layer."""
        self._check_idx(idx)
        tensor_groups = self._tensor_groups[idx]
        return sum(
            tensor.numel() * tensor.element_size()
            for name, tensor in self._pinned[idx].items()
            if tensor_groups.get(name, _GROUP_SHARED) == group
        )

    def managed_tensor_ids(self) -> set[int]:
        return set(self._managed_tensor_ids)

    def cleanup(self) -> None:
        """Drop the pinned-tensor refs so they can be freed by the GC."""
        for pinned_dict in self._pinned:
            pinned_dict.clear()
        self._pinned.clear()
        self._tensor_groups.clear()
        self._resident_groups.clear()
        self._on_gpu.clear()
        self._managed_tensor_ids.clear()


class _AsyncPrefetcher:
    """Issues H2D transfers on a dedicated CUDA stream.

    Uses per-layer CUDA events so that the compute stream only waits for the
    specific layer it needs, not all pending transfers.
    """

    def __init__(self, store: _LayerStore, layers: nn.ModuleList) -> None:
        self._store = store
        self._layers = layers
        self._accel = _accel(store.target_device)
        self._stream = self._accel.Stream(device=store.target_device)
        self._events: dict[int, tuple[Any, frozenset[str]]] = {}

    def prefetch(self, idx: int, groups: frozenset[str]) -> None:
        """Begin async transfer of layer *idx* to GPU (no-op if already there)."""
        if self._store.is_on_gpu(idx, groups) or idx in self._events:
            return
        with self._accel.stream(self._stream):
            self._store.move_to_gpu(idx, self._layers[idx], groups=groups, non_blocking=True)
            event = self._accel.Event()
            event.record(self._stream)
            self._events[idx] = (event, groups)

    def wait(self, idx: int) -> None:
        """Block the compute stream until layer *idx*'s transfer completes."""
        pending = self._events.pop(idx, None)
        if pending is not None:
            event, _groups = pending
            self._accel.current_stream(self._store.target_device).wait_event(event)

    def clear_events(self) -> None:
        self._events.clear()

    def cleanup(self) -> None:
        """Drain pending work and release accelerator stream/event resources."""
        self._events.clear()
        self._stream = None
        self._layers = None
        self._store = None
        self._accel = None


class LayerOffloadWrapper(nn.Module):
    """Wraps a model to offload its sequential layers between CPU and GPU.

    Each layer is evicted immediately after its forward completes. With
    ``prefetch_count == 0`` the wrapper runs in synchronous mode (one layer
    on GPU at a time, no extra stream). With ``prefetch_count >= 1`` it
    pre-stages the next layers on a dedicated CUDA stream so H2D overlaps
    compute, with up to ``1 + prefetch_count`` layers resident on GPU.

    Parameters
    ----------
    model:
        The model to wrap, with all parameters on **CPU** and in eval mode.
    layers_attr:
        Dotted attribute path to the ``nn.ModuleList`` of sequential layers
        (e.g. ``"transformer_blocks"`` or ``"language_model.model.layers"``).
    target_device:
        The accelerator device to use for compute (CUDA or XPU). CPU / MPS
        are rejected.
    prefetch_count:
        ``0`` = synchronous (per-layer load/evict, lowest VRAM, slowest).
        ``>= 1`` = async prefetch this many layers ahead (faster, more VRAM).
    keep_generation_resident:
        Retain generation-branch weights across decoder forwards while they
        fit the configured fast-mode budget.
    fast_vram_fraction:
        Automatic budget as a fraction of physical device memory.
    fast_vram_headroom_gib:
        Physical device memory that must remain available.
    fast_activation_reserve_gib:
        Projected allocation added to each residency decision for work that
        happens after the decoder-layer hooks.
    fast_vram_budget_gib:
        Optional absolute budget overriding ``fast_vram_fraction``.
    """

    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
        prefetch_count: int = 0,
        keep_generation_resident: bool = False,
        fast_vram_fraction: float = DEFAULT_FAST_VRAM_FRACTION,
        fast_vram_headroom_gib: float = DEFAULT_FAST_VRAM_HEADROOM_GIB,
        fast_activation_reserve_gib: float = DEFAULT_FAST_ACTIVATION_RESERVE_GIB,
        fast_vram_budget_gib: float | None = None,
    ) -> None:
        super().__init__()
        _require_accelerator(target_device)
        if prefetch_count < 0:
            raise ValueError("prefetch_count must be >= 0")
        if model.training:
            raise RuntimeError(
                "LayerOffloadWrapper only supports inference; the per-forward "
                "evict-to-CPU step destroys gradient flow. Call model.eval() first."
            )

        self._model = model
        self._layers = _resolve_attr(model, layers_attr)
        self._target_device = target_device
        self._accel = _accel(target_device)
        # Clamp: no point prefetching more layers than (num_layers - 1).
        max_prefetch = max(len(self._layers) - 1, 0)
        self._prefetch_count = min(prefetch_count, max_prefetch)
        self._async_mode = self._prefetch_count >= 1
        self._keep_generation_resident = keep_generation_resident
        self._resident_generation_layers: set[int] = set()
        self._resident_generation_bytes = 0
        self._resident_understanding_layers: set[int] = set()
        self._resident_understanding_bytes = 0
        self._inference_phase: str | None = None
        self._prefix_attention_reserve_bytes = 0
        self._prefix_residency_active = False
        self._resident_memory_limit_bytes = 0
        if not math.isfinite(fast_vram_headroom_gib) or fast_vram_headroom_gib < 0:
            raise ValueError("fast_vram_headroom_gib must be finite and >= 0")
        if not math.isfinite(fast_activation_reserve_gib) or fast_activation_reserve_gib < 0:
            raise ValueError("fast_activation_reserve_gib must be finite and >= 0")
        if fast_vram_budget_gib is not None and (not math.isfinite(fast_vram_budget_gib) or fast_vram_budget_gib <= 0):
            raise ValueError("fast_vram_budget_gib must be finite and > 0 when set")
        self._resident_headroom_bytes = int(fast_vram_headroom_gib * _GIB)
        self._activation_reserve_bytes = int(fast_activation_reserve_gib * _GIB)
        # ``cudaMallocAsync`` keeps per-stream memory pools and never reuses
        # freed blocks across streams without explicit ordering. Detect the
        # backend at construction time so the hooks can pick the right
        # alloc/free pairing strategy: native allocator → record_stream
        # (fast, frees go to whatever stream is current); cudaMallocAsync →
        # wait_stream + free on prefetch stream (correct, slightly more
        # serialized). Only meaningful for CUDA; XPU always uses the native
        # caching allocator and takes the record_stream fast path.
        self._cuda_malloc_async = target_device.type == "cuda" and _is_cuda_malloc_async_backend()
        if self._async_mode:
            logger.info(
                "LayerOffloadWrapper: async prefetch enabled (prefetch_count=%d, allocator=%s, free_path=%s)",
                self._prefetch_count,
                "cudaMallocAsync" if self._cuda_malloc_async else "native",
                "prefetch-stream + wait_stream" if self._cuda_malloc_async else "compute-stream + record_stream",
            )
        if self._keep_generation_resident:
            try:
                total_memory = int(self._accel.get_device_properties(target_device).total_memory)
                budget_bytes = None if fast_vram_budget_gib is None else int(fast_vram_budget_gib * _GIB)
                self._resident_memory_limit_bytes = _resident_memory_limit(
                    total_memory,
                    memory_fraction=fast_vram_fraction,
                    headroom_bytes=self._resident_headroom_bytes,
                    budget_bytes=budget_bytes,
                )
                if self._activation_reserve_bytes < 0:
                    raise ValueError("fast_activation_reserve_gib must be >= 0")
                logger.info(
                    "LayerOffloadWrapper: generation residency budget=%.2f GiB "
                    "(fraction=%.3f, headroom=%.2f GiB, activation_reserve=%.2f GiB, explicit_budget=%s)",
                    self._resident_memory_limit_bytes / _GIB,
                    fast_vram_fraction,
                    self._resident_headroom_bytes / _GIB,
                    self._activation_reserve_bytes / _GIB,
                    "auto" if budget_bytes is None else f"{budget_bytes / _GIB:.2f} GiB",
                )
            except Exception as exc:  # pragma: no cover - backend-specific fallback
                raise ValueError(f"Invalid generation residency configuration: {exc}") from exc
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._audit_handle: torch.utils.hooks.RemovableHandle | None = None
        self._prefetcher: _AsyncPrefetcher | None = None
        self._active_groups: frozenset[str] | None = None
        self._prefix_weight_store: _PrefixWeightStore | None = None
        self._previous_phase_callback: Any = _MISSING
        self._phase_callback_installed = False

        _log_vram("wrapper.__init__: pre-setup", target_device, reset_peak=True)
        self._setup()
        _log_vram(
            f"wrapper.__init__: post-setup (async={self._async_mode}, "
            f"prefetch={self._prefetch_count}, layers={len(self._layers)})",
            target_device,
        )

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        # 1. Pin all layer tensors in CPU memory.
        self._store = _LayerStore(self._layers, self._target_device)

        denoise_module_paths = getattr(self._model, "_denoise_offload_module_paths", ())
        if denoise_module_paths:
            denoise_modules = _resolve_modules(self._model, tuple(denoise_module_paths))
            self._prefix_weight_store = _PrefixWeightStore(denoise_modules, self._target_device)

        # 2. Move all NON-layer params/buffers to GPU permanently.
        layer_tensor_ids: set[int] = set()
        for layer in self._layers:
            for t in itertools.chain(layer.parameters(), layer.buffers()):
                layer_tensor_ids.add(id(t))

        denoise_parameter_ids = (
            self._prefix_weight_store.parameter_ids() if self._prefix_weight_store is not None else set()
        )

        for p in self._model.parameters():
            if id(p) not in layer_tensor_ids and id(p) not in denoise_parameter_ids:
                p.data = p.data.to(self._target_device)
        for b in self._model.buffers():
            if id(b) not in layer_tensor_ids:
                b.data = b.data.to(self._target_device)
        if self._prefix_weight_store is not None:
            self._prefix_weight_store.move_to_target()

        # 3. In async mode spin up the prefetch stream. The first weights are
        #    loaded lazily because the layer kwargs tell us whether this pass
        #    needs the understanding branch or the generation branch.
        if self._async_mode:
            self._prefetcher = _AsyncPrefetcher(self._store, self._layers)

        # 4. Register layer load/evict hooks.
        self._register_hooks()

        # 5. One-shot audit: catch lazy params/buffers materialised inside the
        #    first forward (RoPE caches, attention masks, etc.) that escaped
        #    the construction-time scan.
        self._audit_handle = self._model.register_forward_hook(self._audit_first_forward)
        if self._prefix_weight_store is not None:
            # Install last so a setup failure cannot leave the model pointing
            # at a partially initialized wrapper.
            self._install_phase_callback()

    def _switch_async_groups(self, groups: frozenset[str], *, force: bool = False) -> None:
        """Reset stale prefetch state when inference changes task branch."""
        if self._active_groups == groups and not force:
            return
        if self._active_groups is not None:
            # Branch transitions are infrequent (typically prefix -> denoise).
            # A one-time drain is preferable to carrying the unused branch's
            # prefetched weights through every subsequent layer.
            logger.info(
                "LayerOffloadWrapper: branch transition %s -> %s "
                "(understanding_resident=%.2f GiB, generation_resident=%.2f GiB)",
                sorted(self._active_groups),
                sorted(groups),
                self._resident_understanding_bytes / _GIB,
                self._resident_generation_bytes / _GIB,
            )
            self._accel.synchronize(device=self._target_device)
            self._prefetcher.clear_events()  # type: ignore[union-attr]
            if self._cuda_malloc_async:
                with self._accel.stream(self._prefetcher._stream):  # type: ignore[union-attr]
                    for idx, layer in enumerate(self._layers):
                        self._store.evict_to_cpu(idx, layer)
            else:
                for idx, layer in enumerate(self._layers):
                    self._store.evict_to_cpu(idx, layer)
            self._resident_generation_layers.clear()
            self._resident_generation_bytes = 0
            self._resident_understanding_layers.clear()
            self._resident_understanding_bytes = 0
        self._active_groups = groups

    def _can_keep_resident(self, activation_reserve_bytes: int) -> bool:
        """Return whether current allocations plus workspace fit the fast watermark."""
        allocated = int(self._accel.memory_allocated(self._target_device))
        required_free_memory = self._resident_headroom_bytes + activation_reserve_bytes
        try:
            driver_free_memory = int(self._accel.mem_get_info(self._target_device)[0])
            # cudaMallocAsync owns separate pools per stream in this wrapper;
            # cached blocks from the prefetch stream cannot be assumed usable
            # by attention work on the compute stream.
            reclaimable_cache = 0
            if not self._cuda_malloc_async:
                reserved = int(self._accel.memory_reserved(self._target_device))
                reclaimable_cache = max(reserved - allocated, 0)
            effective_free_memory = driver_free_memory + reclaimable_cache
        except Exception:  # pragma: no cover - backend-specific fallback
            effective_free_memory = required_free_memory
        projected_peak = allocated + activation_reserve_bytes
        return projected_peak <= self._resident_memory_limit_bytes and effective_free_memory >= required_free_memory

    def _generation_groups_to_evict(self, idx: int, groups: frozenset[str]) -> frozenset[str]:
        """Keep generation weights resident while below the VRAM watermark."""
        if not self._keep_generation_resident or _GROUP_GENERATION not in groups:
            return groups

        keep_resident = self._can_keep_resident(self._activation_reserve_bytes)
        group_bytes = self._store.group_nbytes(idx, _GROUP_GENERATION)
        if keep_resident:
            if idx not in self._resident_generation_layers:
                self._resident_generation_layers.add(idx)
                self._resident_generation_bytes += group_bytes
            return groups - {_GROUP_GENERATION}
        if idx in self._resident_generation_layers:
            self._resident_generation_layers.remove(idx)
            self._resident_generation_bytes -= group_bytes
        return groups

    def _observe_prefix_forward(
        self,
        *,
        query_length: int,
        key_length: int,
        batch_size: int,
        num_heads: int,
    ) -> None:
        """Record the long-prefix workspace before retaining Think weights."""
        if not self._keep_generation_resident or self._inference_phase != "prefix":
            return
        if min(query_length, key_length, batch_size, num_heads) <= 0:
            return
        if query_length > 1:
            score_count = batch_size * num_heads * query_length * key_length
            reserve = score_count * _EAGER_ATTENTION_BYTES_PER_SCORE
            self._prefix_attention_reserve_bytes = max(self._prefix_attention_reserve_bytes, reserve)
            return
        if self._prefix_attention_reserve_bytes > 0:
            self._prefix_residency_active = True

    def _understanding_groups_to_evict(self, idx: int, groups: frozenset[str]) -> frozenset[str]:
        """Retain Think-path weights when the measured prefix budget permits."""
        if (
            not self._keep_generation_resident
            or self._inference_phase != "prefix"
            or not self._prefix_residency_active
            or _GROUP_UNDERSTANDING not in groups
        ):
            return groups

        activation_reserve = max(self._activation_reserve_bytes, self._prefix_attention_reserve_bytes)
        keep_resident = self._can_keep_resident(activation_reserve)
        group_bytes = self._store.group_nbytes(idx, _GROUP_UNDERSTANDING)
        if keep_resident:
            if idx not in self._resident_understanding_layers:
                self._resident_understanding_layers.add(idx)
                self._resident_understanding_bytes += group_bytes
            return groups - {_GROUP_UNDERSTANDING}
        if idx in self._resident_understanding_layers:
            self._resident_understanding_layers.remove(idx)
            self._resident_understanding_bytes -= group_bytes
        return groups

    def _register_hooks(self) -> None:
        idx_map: dict[int, int] = {id(layer): idx for idx, layer in enumerate(self._layers)}
        num_layers = len(self._layers)

        active_groups_by_layer: dict[int, frozenset[str]] = {}

        def _pre_hook(module: nn.Module, args: Any, kwargs: dict[str, Any], *, idx: int) -> None:
            groups = _required_tensor_groups(kwargs)
            active_groups_by_layer[idx] = groups
            if idx == 0 and _GROUP_UNDERSTANDING in groups and args and isinstance(args[0], torch.Tensor):
                hidden_states = args[0]
                attention_mask = kwargs.get("attention_mask")
                key_length = (
                    int(attention_mask.shape[-1])
                    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim >= 2
                    else int(hidden_states.shape[-2])
                )
                attention_config = getattr(getattr(module, "self_attn", None), "config", None)
                num_heads = int(getattr(attention_config, "num_attention_heads", 0))
                self._observe_prefix_forward(
                    query_length=int(hidden_states.shape[-2]),
                    key_length=key_length,
                    batch_size=int(hidden_states.shape[0]),
                    num_heads=num_heads,
                )
            if self._async_mode:
                self._switch_async_groups(groups)
                # Wait only for THIS layer's H2D transfer.
                self._prefetcher.wait(idx)  # type: ignore[union-attr]
                if not self._store.is_on_gpu(idx, groups):
                    self._store.move_to_gpu(idx, module, groups=groups)

                if not self._cuda_malloc_async:
                    # Native caching allocator fast path: tell the allocator
                    # the compute stream will read these weights so it does
                    # not reuse the blocks while the kernel is still running.
                    # Frees in _post_hook go to whatever stream is current
                    # (compute stream) and the allocator handles cross-stream
                    # reuse internally — no prefetch-stream barrier needed.
                    compute_stream = self._accel.current_stream(self._target_device)
                    for param in self._store.tensors_for_groups(idx, module, groups):
                        param.data.record_stream(compute_stream)

                # Kick off prefetch for upcoming layers (wraps around for next pass).
                for offset in range(1, self._prefetch_count + 1):
                    self._prefetcher.prefetch((idx + offset) % num_layers, groups)  # type: ignore[union-attr]
            else:
                # Sync mode: the H2D dispatches on the compute stream itself,
                # which serialises naturally with the kernel that follows.
                self._store.move_to_gpu(idx, module, groups=groups, non_blocking=True)

        def _post_hook(module: nn.Module, _args: Any, _output: Any, *, idx: int) -> None:
            groups = active_groups_by_layer.pop(idx, _ALL_TENSOR_GROUPS)
            groups_to_evict = self._understanding_groups_to_evict(idx, groups)
            groups_to_evict = self._generation_groups_to_evict(idx, groups_to_evict)
            if self._async_mode and self._cuda_malloc_async:
                # cudaMallocAsync slow-but-safe path: per-stream pools
                # require alloc and free on the same stream. Since
                # `_AsyncPrefetcher` allocates layer weights on the prefetch
                # stream, we must also free them there. Wait for the compute
                # stream to finish reading the weights first; wait_stream is
                # host-async so it does not stall Python. The cost is that
                # subsequent prefetches queued on the prefetch stream are
                # ordered after this wait, slightly reducing pipeline depth.
                prefetch_stream = self._prefetcher._stream  # type: ignore[union-attr]
                compute_stream = self._accel.current_stream(self._target_device)
                prefetch_stream.wait_stream(compute_stream)
                with self._accel.stream(prefetch_stream):
                    self._store.evict_to_cpu(idx, module, groups=groups_to_evict)
            else:
                # Native allocator path: just drop the GPU tensor refs on
                # the compute stream. record_stream in _pre_hook ensures the
                # blocks are not reused before the kernel finishes.
                self._store.evict_to_cpu(idx, module, groups=groups_to_evict)

        for layer in self._layers:
            idx = idx_map[id(layer)]
            h1 = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx), with_kwargs=True)
            h2 = layer.register_forward_hook(functools.partial(_post_hook, idx=idx))
            self._hooks.extend([h1, h2])

    def _audit_first_forward(self, _module: nn.Module, _inputs: Any, _outputs: Any) -> None:
        _log_vram("wrapper.audit: pre", self._target_device)
        managed_tensor_ids = self._store.managed_tensor_ids()
        if self._prefix_weight_store is not None:
            managed_tensor_ids.update(self._prefix_weight_store.parameter_ids())
        moved = _audit_lazy_state(self._model, self._target_device, managed_tensor_ids)
        if moved:
            logger.warning(
                "LayerOffloadWrapper: moved %d lazy param(s)/buffer(s) onto %s after "
                "the first forward. These will stay on GPU; they are not offloaded.",
                moved,
                self._target_device,
            )
        _log_vram(f"wrapper.audit: post (moved={moved})", self._target_device)
        if self._audit_handle is not None:
            self._audit_handle.remove()
            self._audit_handle = None

    def set_inference_phase(self, phase: str) -> None:
        """Swap branch and prefix-only weights for the next inference phase."""
        if phase not in {"prefix", "denoise"}:
            raise ValueError(f"Unsupported inference phase: {phase!r}")
        self._inference_phase = phase
        if phase == "prefix":
            self._prefix_attention_reserve_bytes = 0
            self._prefix_residency_active = False
            if self._async_mode:
                # A previous generation may have been cancelled during Think,
                # before its denoise phase had a chance to release resident
                # understanding weights. Always drain at a new run boundary.
                self._switch_async_groups(
                    frozenset({_GROUP_SHARED, _GROUP_UNDERSTANDING}),
                    force=True,
                )
            else:
                self._accel.synchronize(device=self._target_device)
                for idx, layer in enumerate(self._layers):
                    self._store.evict_to_cpu(idx, layer)
                self._resident_generation_layers.clear()
                self._resident_generation_bytes = 0
                self._resident_understanding_layers.clear()
                self._resident_understanding_bytes = 0
            if self._prefix_weight_store is not None:
                self._prefix_weight_store.move_to_target()
            return
        if self._async_mode:
            self._switch_async_groups(frozenset({_GROUP_SHARED, _GROUP_GENERATION}))
        else:
            self._accel.synchronize(device=self._target_device)
            for idx, layer in enumerate(self._layers):
                self._store.evict_to_cpu(
                    idx,
                    layer,
                    groups=frozenset({_GROUP_UNDERSTANDING}),
                )
        self._resident_understanding_layers.clear()
        self._resident_understanding_bytes = 0
        if self._prefix_weight_store is not None:
            self._prefix_weight_store.evict_to_cpu()

    def _install_phase_callback(self) -> None:
        self._previous_phase_callback = self._model.__dict__.get(_PHASE_CALLBACK_ATTR, _MISSING)
        setattr(self._model, _PHASE_CALLBACK_ATTR, self.set_inference_phase)
        self._phase_callback_installed = True

    def _restore_phase_callback(self) -> None:
        if not self._phase_callback_installed:
            return
        if self._previous_phase_callback is _MISSING:
            if _PHASE_CALLBACK_ATTR in self._model.__dict__:
                delattr(self._model, _PHASE_CALLBACK_ATTR)
        else:
            setattr(self._model, _PHASE_CALLBACK_ATTR, self._previous_phase_callback)
        self._previous_phase_callback = _MISSING
        self._phase_callback_installed = False

    def teardown(self) -> None:
        """Remove hooks, release pinned memory, and move parameters back to CPU.

        After this call the wrapper is inert: hooks are removed, the prefetch
        stream is drained and destroyed, all parameters reside on CPU, and
        the stores release their references. Parameters retain pinned CPU
        backing so a later wrapper can reuse it without another pinning copy.
        """
        _log_vram(
            f"wrapper.teardown: enter (on_gpu={len(self._store._on_gpu)}, "
            f"events={len(self._prefetcher._events) if self._prefetcher is not None else 0}, "
            f"resident_understanding_layers={len(self._resident_understanding_layers)}, "
            f"resident_understanding_gib={self._resident_understanding_bytes / _GIB:.2f}, "
            f"resident_generation_layers={len(self._resident_generation_layers)}, "
            f"resident_generation_gib={self._resident_generation_bytes / _GIB:.2f})",
            self._target_device,
        )
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        if self._audit_handle is not None:
            self._audit_handle.remove()
            self._audit_handle = None
        self._restore_phase_callback()

        # Drain in-flight H2D copies before tearing down stream resources, or
        # the accelerator driver can hit use-after-free during cleanup.
        self._accel.synchronize(device=self._target_device)
        if self._prefetcher is not None:
            self._prefetcher.cleanup()
            self._prefetcher = None

        if self._prefix_weight_store is not None:
            self._prefix_weight_store.evict_to_cpu()

        for idx, layer in enumerate(self._layers):
            self._store.evict_to_cpu(idx, layer)
        self._resident_generation_layers.clear()
        self._resident_generation_bytes = 0
        self._resident_understanding_layers.clear()
        self._resident_understanding_bytes = 0

        for p in self._model.parameters():
            p.data = p.data.to("cpu")
        for b in self._model.buffers():
            b.data = b.data.to("cpu")

        total_bytes, by_group = self._store.transfer_stats()
        logger.info(
            "LayerOffloadWrapper transfer totals: %.2f GiB (shared=%.2f, understanding=%.2f, generation=%.2f)",
            total_bytes / (1024**3),
            by_group[_GROUP_SHARED] / (1024**3),
            by_group[_GROUP_UNDERSTANDING] / (1024**3),
            by_group[_GROUP_GENERATION] / (1024**3),
        )
        self._store.cleanup()
        if self._prefix_weight_store is not None:
            self._prefix_weight_store.cleanup()
            self._prefix_weight_store = None
        _log_vram("wrapper.teardown: exit (pre-empty_cache)", self._target_device)

    # ------------------------------------------------------------------
    # Forward and attribute delegation
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the wrapped model.

        ``nn.Module.__getattr__`` is only called when normal lookup fails, so
        ``_model`` / ``_store`` etc. are still resolved via ``__dict__``.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)
