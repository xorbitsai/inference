# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import TYPE_CHECKING, Optional

import PIL.Image
import torch

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

from ....device_utils import get_available_device
from ...utils import allow_trust_remote_code
from .ocr_family import OCRModel

logger = logging.getLogger(__name__)


class OvisOCR2Model(OCRModel):
    required_libs = ("transformers",)

    DEFAULT_PROMPT = (
        "Extract all readable content from the image in natural human reading "
        "order and output the result as a single Markdown document. For charts "
        "or images, represent them using an HTML image tag: "
        '<img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where '
        "left, top, right, bottom are bounding box coordinates scaled to "
        "[0, 1000). Format formulas as LaTeX. Format tables as HTML: "
        "<table>...</table>. Transcribe all other text as standard Markdown. "
        "Preserve the original text without translation or paraphrasing."
    )

    @staticmethod
    def _clean_truncated_repeats(
        text: str,
        min_text_len: int = 8000,
        max_period: int = 200,
        min_period: int = 1,
        min_repeat_chars: int = 100,
        min_repeat_times: int = 5,
    ) -> str:
        n = len(text)
        if n < min_text_len:
            return text

        max_period = min(max_period, n - 1)
        for unit_len in range(min_period, max_period + 1):
            if text[n - 1] != text[n - 1 - unit_len]:
                continue

            match_len = 1
            idx = n - 2
            while idx >= unit_len and text[idx] == text[idx - unit_len]:
                match_len += 1
                idx -= 1

            total_len = match_len + unit_len
            repeat_times = total_len // unit_len
            tail_len = total_len % unit_len
            if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
                return text[: n - total_len + unit_len] + text[n - tail_len :]

        return text

    @classmethod
    def _postprocess_output(cls, text: str, filter_imgtags: bool = True) -> str:
        text = text.strip()
        if filter_imgtags:
            text = "\n\n".join(
                block
                for block in text.split("\n\n")
                if not block.strip().startswith('<img src="images/bbox_')
            )
        return cls._clean_truncated_repeats(text)

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return model_family.model_name == "OvisOCR2"

    def __init__(
        self,
        model_uid: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        model_spec: Optional["ImageModelFamilyV2"] = None,
        **kwargs,
    ):
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._model = None
        self._processor = None
        self._abilities = model_spec.model_ability or []  # type: ignore
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._abilities

    def load(self):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        device = self._device or get_available_device()
        model_kwargs = self._kwargs.copy()
        model_kwargs.pop("cpu_offload", None)
        model_kwargs.setdefault(
            "torch_dtype", torch.float32 if device == "cpu" else torch.bfloat16
        )
        model_kwargs.setdefault(
            "trust_remote_code", allow_trust_remote_code(self.model_family)
        )

        self._processor = AutoProcessor.from_pretrained(
            self._model_path,
            trust_remote_code=allow_trust_remote_code(self.model_family),
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            self._model_path, **model_kwargs
        )
        if "device_map" not in model_kwargs:
            model = model.to(device)
        self._model = model.eval()

    def ocr(
        self,
        image: PIL.Image.Image,
        prompt: Optional[str] = None,
        filter_imgtags: bool = True,
        **kwargs,
    ) -> str:
        if self._model is None or self._processor is None:
            self.load()

        if not isinstance(image, PIL.Image.Image):
            raise ValueError("Input must be a PIL Image")
        image = image.convert("RGB")
        prompt = prompt or self.DEFAULT_PROMPT

        processor = self._processor
        model = self._model
        assert processor is not None
        assert model is not None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(model.device)

        generation_kwargs = kwargs.copy()
        generation_kwargs.setdefault("max_new_tokens", 16384)
        generation_kwargs.setdefault("do_sample", False)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)

        generated_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
        output_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not output_texts:
            logger.warning("OvisOCR2 returned empty decoded output.")
            return ""
        return self._postprocess_output(output_texts[0], filter_imgtags)
