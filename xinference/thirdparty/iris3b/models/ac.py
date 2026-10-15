"""Activation-checkpointing policies for the transformer stacks.

Selected by ``train.activation_checkpointing``:

``none``
    Keep every activation; no checkpoint region is entered at all.
``full``
    Recompute every region - each trunk block and each pixel block.
``selective_op``
    Enter every region but keep the outputs of the compute-intensive ops
    (attention, matmul, convolution) and replay only the pointwise work, which
    is where most of the activation memory sits for a fraction of the FLOPs.
``selective_layer``
    Fully recompute every Nth region of a stack and keep the rest.

A region's cost profile is a property of the stage, not of the model, so the
choice belongs in the stage config: nothing here looks at resolution.
"""

import functools
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.checkpoint import (
    checkpoint,
    create_selective_checkpoint_contexts,
    noop_context_fn,
)

POLICIES = ("none", "full", "selective_op", "selective_layer")

# Kept under ``selective_op``: the attention and matmul-class kernels, where a
# recompute costs real FLOPs. Everything else - norms, SiLU, gating, adds,
# reshapes - is replayed. Recomputing the attention *interior* only pays off
# for a kernel that materialises the s x s score matrix; SDPA never does, so
# the policy here is the inverse: save the kernel result, replay the cheap ops
# around it. External attention backends (fa3 / fa4) dispatch outside aten and
# are therefore always recomputed.
SAVE_OP_NAMES: tuple[str, ...] = (
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_cudnn_attention",
    "_scaled_dot_product_attention_math",
    "_scaled_dot_product_fused_attention_overrideable",
    "_scaled_dot_product_flash_attention_for_cpu",
    "mm",
    "bmm",
    "addmm",
    "_scaled_mm",
    "convolution",
    "max",
)


def validate_policy(policy: str, every: int) -> None:
    if policy not in POLICIES:
        raise ValueError(
            f"unknown train.activation_checkpointing '{policy}' ({' | '.join(POLICIES)})"
        )
    if policy == "selective_layer" and every < 1:
        raise ValueError(
            "train.activation_checkpointing='selective_layer' recomputes every Nth block "
            f"and needs train.ac_selective_every >= 1, got {every}"
        )


@functools.cache
def resolve_save_ops(names: tuple[str, ...] = SAVE_OP_NAMES) -> tuple[Any, ...]:
    """The ``aten`` overloads for ``names`` that this torch build actually has.

    Op coverage moves between releases (the cuDNN and overrideable SDPA
    variants, ``_scaled_mm``) and ``create_selective_checkpoint_contexts``
    rejects a list holding anything that is not an ``OpOverload``, so each name
    is probed and the misses are dropped instead of raising.
    """
    ops = []
    for name in names:
        packet = getattr(torch.ops.aten, name, None)
        overload = getattr(packet, "default", None) if packet is not None else None
        if isinstance(overload, torch._ops.OpOverload):
            ops.append(overload)
    return tuple(ops)


@functools.cache
def checkpoint_context(policy: str) -> Callable[[], tuple[Any, Any]]:
    """The ``context_fn`` for ``torch.utils.checkpoint.checkpoint``.

    ``selective_op`` gets a partial, so every region builds its own pair of
    caching dispatch modes and no two regions share a save cache. Every other
    policy gets torch's own default, which keeps ``full`` bit-for-bit identical
    to a plain ``checkpoint`` call.
    """
    if policy != "selective_op":
        return noop_context_fn
    return functools.partial(create_selective_checkpoint_contexts, list(resolve_save_ops()))


def should_checkpoint(policy: str, block_index: int, every: int) -> bool:
    """Whether region ``block_index`` enters a checkpoint region at all.

    ``block_index`` is 0-based and counted within its own stack, so the trunk
    and the pixel refiner are strided independently.
    """
    if policy == "none":
        return False
    if policy == "selective_layer":
        return block_index % every == 0
    return True


def run(fn: Callable[..., Any], args: tuple[Any, ...], policy: str, block_index: int, every: int) -> Any:
    """Run ``fn(*args)`` under the checkpoint policy for this region.

    ``preserve_rng_state`` is left at its default: stashing and restoring the
    RNG state does cost time, but dropping it changes the dropout draws on the
    recompute, so adopting it would need a bitwise gradient check - and it is a
    no-op under torch.compile, which always preserves RNG.
    """
    if not should_checkpoint(policy, block_index, every):
        return fn(*args)
    return checkpoint(fn, *args, use_reentrant=False, context_fn=checkpoint_context(policy))
