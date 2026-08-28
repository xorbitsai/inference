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


class NaviDCOCRModel(OCRModel):
    required_libs = ("transformers",)

    DEFAULT_PROMPT = "Please output the text content from the image."
    DEFAULT_LAYOUT_PROMPT = "Analyze the image layout."
    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
    _IMAGE_PROMPT_PREFIX = "<image>\n"
    _LEGACY_PROMPT_MAP = {
        "OCR": DEFAULT_PROMPT,
        "<image>\nFree OCR.": DEFAULT_PROMPT,
        (
            "<image>\nFree OCR. Extract all text content from the image."
        ): DEFAULT_PROMPT,
        (
            "<image>\nConvert this document to clean markdown format. "
            "Extract the text content and format it properly using markdown syntax. "
            "Do not include any coordinate annotations or special formatting markers."
        ): DEFAULT_PROMPT,
        (
            "<image>\n<|grounding|>Convert the document to markdown with "
            "structure annotations. Include coordinate information for text regions "
            "and maintain the document structure."
        ): DEFAULT_LAYOUT_PROMPT,
    }
    NON_GENERATION_KWARGS = (
        "model_size",
        "test_compress",
        "save_results",
        "save_dir",
        "eval_mode",
        "request_id",
    )

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return model_family.model_name == "NaviDC-OCR"

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

    @classmethod
    def _normalize_prompt(cls, prompt: Optional[str]) -> str:
        if not prompt:
            return cls.DEFAULT_PROMPT
        mapped_prompt = cls._LEGACY_PROMPT_MAP.get(prompt)
        if mapped_prompt is not None:
            return mapped_prompt
        if prompt.startswith(cls._IMAGE_PROMPT_PREFIX):
            return prompt[len(cls._IMAGE_PROMPT_PREFIX) :]
        return prompt

    def load(self):
        from transformers import AutoModel, AutoProcessor

        device = self._device or get_available_device()
        model_kwargs = self._kwargs.copy()
        model_kwargs.pop("cpu_offload", None)
        use_fast = model_kwargs.pop("use_fast", True)
        model_kwargs.setdefault(
            "trust_remote_code", allow_trust_remote_code(self.model_family)
        )
        model_kwargs.setdefault(
            "torch_dtype", torch.float32 if device == "cpu" else torch.bfloat16
        )

        self._processor = AutoProcessor.from_pretrained(
            self._model_path,
            trust_remote_code=allow_trust_remote_code(self.model_family),
            use_fast=use_fast,
        )
        model = AutoModel.from_pretrained(self._model_path, **model_kwargs)
        if "device_map" not in model_kwargs:
            model = model.to(device)
        self._model = model.eval()

    def ocr(
        self,
        image: PIL.Image.Image,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        if self._model is None or self._processor is None:
            self.load()

        if not isinstance(image, PIL.Image.Image):
            raise ValueError("Input must be a PIL Image")
        image = image.convert("RGB")
        prompt = self._normalize_prompt(prompt)

        system_prompt = kwargs.pop("system_prompt", self.DEFAULT_SYSTEM_PROMPT)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        processor = self._processor
        model = self._model
        assert processor is not None
        assert model is not None
        chat_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[chat_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(device=model.device, dtype=model.dtype)

        generation_kwargs = kwargs.copy()
        for key in self.NON_GENERATION_KWARGS:
            generation_kwargs.pop(key, None)
        generation_kwargs.setdefault("use_cache", True)
        generation_kwargs.setdefault("max_new_tokens", 4096)
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
            logger.warning("NaviDC-OCR returned empty decoded output.")
            return ""
        return output_texts[0].strip()
