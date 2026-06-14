# Copyright 2022-2026 XProbe Inc.
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

from .ocr_family import OCRModel

logger = logging.getLogger(__name__)


class HunyuanOCRModel(OCRModel):
    required_libs = ("transformers",)

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return model_family.model_name == "HunyuanOCR"

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

    def _load(self):
        from transformers import AutoConfig, AutoProcessor

        try:
            from transformers import HunYuanVLForConditionalGeneration as ModelCls
        except ImportError:
            from transformers import AutoModelForSeq2SeqLM as ModelCls

        attn_impl = self._kwargs.get("attn_implementation", "eager")
        torch_dtype = self._kwargs.get("torch_dtype", torch.bfloat16)
        device_map = self._kwargs.get("device_map", "auto")

        self._processor = AutoProcessor.from_pretrained(
            self._model_path, use_fast=False, trust_remote_code=True
        )
        config = AutoConfig.from_pretrained(self._model_path, trust_remote_code=True)
        self._model = ModelCls.from_pretrained(
            self._model_path,
            config=config,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    def load(self):
        if self._model is None:
            self._load()

    def ocr(self, image: PIL.Image.Image, prompt: Optional[str] = None, **kwargs):
        if self._model is None or self._processor is None:
            self._load()

        if image.mode in ("RGBA", "CMYK"):
            image = image.convert("RGB")

        if prompt is None:
            prompt = "Detect and recognize text within images, then output the text coordinates in a formatted manner."

        messages = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        processor = self._processor
        assert processor is not None  # type check helper

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=image, padding=True, return_tensors="pt")

        max_new_tokens = kwargs.pop("max_new_tokens", 2048)
        do_sample = kwargs.pop("do_sample", False)
        temperature = kwargs.pop("temperature", None)

        with torch.no_grad():
            device = next(self._model.parameters()).device  # type: ignore
            inputs = inputs.to(device)
            generated_ids = self._model.generate(  # type: ignore
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )

        input_ids = inputs.get("input_ids", inputs.get("inputs"))
        if input_ids is None:
            logger.warning(
                "HunyuanOCR: input_ids missing in inputs, returning raw text"
            )
            return processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if isinstance(output_texts, list):
            if output_texts:
                return output_texts[0]
            logger.warning("HunyuanOCR returned empty decoded list.")
            return ""
        if output_texts is None:
            logger.warning("HunyuanOCR returned None output.")
            return ""
        return output_texts
