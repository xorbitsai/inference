import random
from typing import Any

import numpy as np
import torch


def get_dtype(x):
    if x.lower() in ("bf16", "torch.bfloat16", "bfloat16"):
        return torch.bfloat16
    if x.lower() in ("fp16", "torch.float16", "float16"):
        return torch.float16
    if x.lower() in ("fp32", "torch.float32", "float32"):
        return torch.float32
    raise ValueError("Unsupported dtype value.")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mask_data(x, mask, masking_value=0.0):
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    if isinstance(masking_value, torch.Tensor):
        return torch.where(mask, masking_value.expand_as(x), x)
    return torch.where(
        mask, torch.full(x.shape, masking_value, dtype=x.dtype, device=x.device), x
    )


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len).to(lengths.device))
    return (ids < lengths.unsqueeze(1)).bool()


def scalar_as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().item())
    return float(value)
