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

from typing import TYPE_CHECKING, Any, Optional

import PIL.Image
import torch

from ....device_utils import get_available_device
from ...utils import allow_trust_remote_code
from .ocr_family import OCRModel

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2


class XiaomiOCRModel(OCRModel):
    required_libs = ("transformers",)

    DEFAULT_PROMPT = (
        "Extract all information from the main body of the document image and represent it "
        "in markdown format, ignoring headers and footers. Tables should be expressed in OTSL "
        "format, formulas in the document should be represented using LATEX format, and the "
        "parsing should be organized according to the reading order."
    )
    NON_GENERATION_KWARGS = (
        "request_id",
        "model_size",
        "test_compress",
        "save_results",
        "save_dir",
        "eval_mode",
    )

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return (
            model_family.model_name == "xiaomi-ocr-0"
            and getattr(model_family, "model_format", None) == "pytorch"
        )

    def __init__(
        self,
        model_uid: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        model_spec: Optional["ImageModelFamilyV2"] = None,
        **kwargs: Any,
    ) -> None:
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._model = None
        self._processor = None
        self._abilities = model_spec.model_ability if model_spec is not None else []
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._abilities

    @classmethod
    def _normalize_prompt(cls, prompt: Optional[str]) -> str:
        if prompt and prompt.startswith("<image>"):
            prompt = prompt[len("<image>") :].lstrip()
        if not prompt or prompt in (
            "OCR",
            "Free OCR.",
            "Free OCR. Extract all text content from the image.",
        ):
            return cls.DEFAULT_PROMPT
        return prompt

    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = torch.device(self._device or get_available_device())
        model_kwargs = self._kwargs.copy()
        model_kwargs.pop("cpu_offload", None)
        trust_remote_code = allow_trust_remote_code(self.model_family)
        model_kwargs.setdefault("trust_remote_code", trust_remote_code)
        if "torch_dtype" not in model_kwargs:
            model_kwargs.setdefault(
                "dtype", torch.float32 if device.type == "cpu" else torch.float16
            )
        self._processor = AutoProcessor.from_pretrained(
            self._model_path, trust_remote_code=trust_remote_code
        )
        model = AutoModelForImageTextToText.from_pretrained(
            self._model_path, **model_kwargs
        )
        if "device_map" not in model_kwargs:
            model = model.to(device)
        self._model = model.eval()

    def ocr(
        self, image: PIL.Image.Image, prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        if not isinstance(image, PIL.Image.Image):
            raise ValueError("Input must be a PIL Image")
        if self._model is None or self._processor is None:
            self.load()

        assert self._model is not None
        assert self._processor is not None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": self._normalize_prompt(prompt)},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self._model.device)
        for key in self.NON_GENERATION_KWARGS:
            kwargs.pop(key, None)
        kwargs.setdefault("max_new_tokens", 4096)
        kwargs.setdefault("do_sample", False)
        with torch.inference_mode():
            output = self._model.generate(**inputs, **kwargs)
        generated_ids = output[:, inputs["input_ids"].shape[-1] :]
        texts = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return texts[0].strip() if texts else ""
