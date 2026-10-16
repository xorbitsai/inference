"""Text encoder interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class TextEncoding:
    embeddings: torch.Tensor  # [B, L, dim], or [B, L, N, dim] for layerwise adapters
    # [B, L] (int, 1 = real token). Masked text adapters consume it.
    mask: torch.Tensor


class TextEncoder(ABC):
    """Frozen prompt encoder producing fixed-length token embeddings.

    Implementations own their prompt template, token-selection policy, and CFG
    null/negative-prompt policy. ``null`` returns the fixed-length encoding
    expected by training dropout and inference guidance.
    """

    dim: int
    max_length: int
    # caption-budget accounting, maintained by every implementation and read by
    # the trainer: encode calls, rows truncated, tokens dropped, longest caption
    calls: int
    overflow_rows: int
    overflow_tokens: int
    max_caption_tokens: int

    @abstractmethod
    def encode(self, prompts: list[str]) -> TextEncoding: ...

    @abstractmethod
    def null(self, negative_prompt: str = "") -> TextEncoding: ...

    @abstractmethod
    def to(self, device) -> "TextEncoder": ...
