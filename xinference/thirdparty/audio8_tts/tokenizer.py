# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import inspect
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Reference:
    text: str
    vq_codes: torch.Tensor | None = None


class Audio8TokenizerAdapter:
    def __init__(
        self,
        tokenizer: Any,
        *,
        semantic_begin_id: int,
        semantic_end_id: int,
        eos_token_id: int,
    ) -> None:
        self._tok = tokenizer
        self.semantic_begin_id = int(semantic_begin_id)
        self.semantic_end_id = int(semantic_end_id)
        self.eos_token_ids = [int(eos_token_id)]

    def _encode(self, text: str) -> torch.Tensor:
        kwargs: dict[str, Any] = {"add_special_tokens": False}
        if "allowed_special" in inspect.signature(self._tok.encode).parameters:
            kwargs["allowed_special"] = "all"
        return torch.tensor(self._tok.encode(text, **kwargs), dtype=torch.long)

    @staticmethod
    def _clean(text: str) -> str:
        value = "".join(
            " " if char.isspace() else "" if unicodedata.category(char).startswith("C") else char
            for char in str(text)
        )
        return " ".join(value.split())

    @classmethod
    def _reference_text(cls, text: str) -> str:
        text = cls._clean(text)
        return text if re.search(r"<\|speaker:\d+\|>", text) else f"<|speaker:0|>{text}"

    def build_prompt(
        self,
        text: str,
        references: list[Reference] | None = None,
        *,
        num_codebooks: int = 10,
        **_: Any,
    ) -> dict[str, Any]:
        target = self._clean(text)
        if not target:
            raise ValueError("text must not be empty")

        def encode_parts(parts: list[str]) -> torch.Tensor:
            return torch.cat([self._encode(part) for part in parts])

        if not references:
            tokens = encode_parts(
                [
                    "<|im_start|>system\n",
                    "convert the provided text to speech",
                    "<|im_end|>\n",
                    "<|im_start|>user\n",
                    target,
                    "<|im_end|>\n",
                    "<|im_start|>assistant\n<|voice|>",
                ]
            )
            return {
                "input_ids": tokens,
                "vq_mask_tokens": torch.zeros(tokens.numel(), dtype=torch.bool),
                "vq_parts": [],
            }

        if len(references) != 1:
            raise ValueError("Audio8 TTS currently supports exactly one reference")
        reference = references[0]
        if not reference.text:
            raise ValueError("reference text is required")
        if reference.vq_codes is None:
            raise ValueError("reference VQ codes are required")
        codes = reference.vq_codes.to(dtype=torch.long, device="cpu")
        if codes.ndim != 2 or codes.shape[0] != num_codebooks or codes.shape[1] == 0:
            raise ValueError(
                f"reference codes must have shape [{num_codebooks}, T>0], got {tuple(codes.shape)}"
            )

        prefix = encode_parts(
            [
                "<|im_start|>system\n",
                "convert the provided text to speech reference to the following:\n\nText:\n",
                self._reference_text(reference.text),
                "\n\nSpeech:\n",
            ]
        )
        semantic = codes[0] + self.semantic_begin_id
        suffix = encode_parts(
            [
                "<|im_end|>\n",
                "<|im_start|>user\n",
                target,
                "<|im_end|>\n",
                "<|im_start|>assistant\n<|voice|>",
            ]
        )
        tokens = torch.cat((prefix, semantic, suffix))
        mask = torch.zeros(tokens.numel(), dtype=torch.bool)
        mask[prefix.numel() : prefix.numel() + semantic.numel()] = True
        return {"input_ids": tokens, "vq_mask_tokens": mask, "vq_parts": [codes]}
