"""Alignment-padding helpers for the standard diffusion transformer."""

from typing import Optional, Sequence

import torch


def mask_out_alignment_padding(
    attention_mask: torch.Tensor,
    pad_masks: Sequence[Optional[torch.Tensor]],
    offsets: Sequence[int],
) -> torch.Tensor:
    """Exclude per-item alignment padding from a 2D boolean attention mask."""

    if attention_mask.ndim != 2 or attention_mask.dtype != torch.bool:
        raise ValueError(
            "attention_mask must be a 2D boolean tensor, got "
            f"shape={attention_mask.shape}, dtype={attention_mask.dtype}"
        )
    batch_size = attention_mask.shape[0]
    if len(pad_masks) != batch_size or len(offsets) != batch_size:
        raise ValueError(
            "pad mask metadata must match the attention-mask batch size: "
            f"batch={batch_size}, pad_masks={len(pad_masks)}, offsets={len(offsets)}"
        )

    for item_index, (pad_mask, offset) in enumerate(zip(pad_masks, offsets)):
        if pad_mask is None or pad_mask.numel() == 0:
            continue
        if pad_mask.ndim != 1 or pad_mask.dtype != torch.bool:
            raise ValueError(
                "each alignment-pad mask must be a 1D boolean tensor: "
                f"item={item_index}, shape={pad_mask.shape}, dtype={pad_mask.dtype}"
            )
        offset = int(offset)
        end = offset + pad_mask.shape[0]
        if offset < 0 or end > attention_mask.shape[1]:
            raise ValueError(
                "alignment-pad mask falls outside the attention sequence: "
                f"item={item_index}, offset={offset}, length={pad_mask.shape[0]}, "
                f"sequence={attention_mask.shape[1]}"
            )
        attention_mask[item_index, offset:end].masked_fill_(pad_mask, False)

    return attention_mask
