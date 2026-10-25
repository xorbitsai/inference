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


DEFAULT_PROMPT = (
    "Transcribe the provided document image into a clean Markdown format, "
    "preserving the natural reading order."
)
NON_GENERATION_KWARGS = (
    "request_id",
    "model_size",
    "test_compress",
    "save_results",
    "save_dir",
    "eval_mode",
)


def _normalize_prompt(prompt: Optional[str]) -> str:
    if prompt and prompt.startswith("<image>"):
        prompt = prompt[len("<image>") :].lstrip()
    return prompt or DEFAULT_PROMPT


class JinaOCRModel(OCRModel):
    required_libs = ("transformers",)

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return (
            model_family.model_name == "jina-ocr-v1"
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

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        device = torch.device(self._device or get_available_device())
        model_kwargs = self._kwargs.copy()
        model_kwargs.pop("cpu_offload", None)
        model_kwargs.setdefault(
            "dtype", torch.float32 if device.type == "cpu" else torch.bfloat16
        )
        trust_remote_code = allow_trust_remote_code(self.model_family)
        self._processor = AutoProcessor.from_pretrained(
            self._model_path, trust_remote_code=trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            self._model_path, trust_remote_code=trust_remote_code, **model_kwargs
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
        inputs = self._processor.prepare_ocr_inputs(
            image.convert("RGB"),
            prompt=_normalize_prompt(prompt),
            device=self._model.device,
        )
        for key in NON_GENERATION_KWARGS:
            kwargs.pop(key, None)
        kwargs.setdefault("max_new_tokens", 4096)
        kwargs.setdefault("do_sample", False)
        with torch.inference_mode():
            output = self._model.generate(**inputs, **kwargs)
        return self._processor.decode_ocr(output, inputs["input_ids"])
