# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Audio8TTSState:
    input_ids: Any = None
    vq_mask_tokens: Any | None = None
    vq_parts: Any | None = None
    num_codebooks: int = 10
    codebook_size: int = 4096
    max_new_tokens: int = 1024
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    do_sample: bool = True
    output_codes: Any | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    audio_samples: Any | None = None
    sample_rate: int = 44100
    response_format: str = "wav"

    @staticmethod
    def _to_list(value: Any) -> Any:
        return value.tolist() if isinstance(value, torch.Tensor) else value

    def to_dict(self) -> dict[str, Any]:
        data = {
            "input_ids": self._to_list(self.input_ids),
            "vq_mask_tokens": self._to_list(self.vq_mask_tokens),
            "vq_parts": (
                [self._to_list(part) for part in self.vq_parts]
                if self.vq_parts is not None
                else None
            ),
            "num_codebooks": self.num_codebooks,
            "codebook_size": self.codebook_size,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "output_codes": self._to_list(self.output_codes),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "audio_samples": self._to_list(self.audio_samples),
            "sample_rate": self.sample_rate,
            "response_format": self.response_format,
        }
        return {key: value for key, value in data.items() if value is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Audio8TTSState":
        vq_parts = data.get("vq_parts")
        if vq_parts is not None:
            vq_parts = [torch.as_tensor(part, dtype=torch.long) for part in vq_parts]
        output_codes = data.get("output_codes")
        return cls(
            input_ids=data.get("input_ids"),
            vq_mask_tokens=data.get("vq_mask_tokens"),
            vq_parts=vq_parts,
            num_codebooks=int(data.get("num_codebooks", 10)),
            codebook_size=int(data.get("codebook_size", 4096)),
            max_new_tokens=int(data.get("max_new_tokens", 1024)),
            temperature=float(data.get("temperature", 0.8)),
            top_p=float(data.get("top_p", 0.95)),
            top_k=int(data.get("top_k", 50)),
            do_sample=bool(data.get("do_sample", True)),
            output_codes=(
                torch.as_tensor(output_codes, dtype=torch.long)
                if output_codes is not None
                else None
            ),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            audio_samples=data.get("audio_samples"),
            sample_rate=int(data.get("sample_rate", 44100)),
            response_format=str(data.get("response_format", "wav")),
        )
