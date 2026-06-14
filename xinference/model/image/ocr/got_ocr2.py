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

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

from .ocr_family import OCRModel

logger = logging.getLogger(__name__)


class GotOCR2Model(OCRModel):
    required_libs = ("transformers",)

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return model_family.model_name == "GOT-OCR2_0"

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
        # model info when loading
        self._model = None
        self._tokenizer = None
        # info
        self._model_spec = model_spec
        self._abilities = model_spec.model_ability or []  # type: ignore
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._abilities

    def load(self):
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self._model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="cuda",
            use_safetensors=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        self._model = model.eval().cuda()

    def ocr(
        self,
        image: PIL.Image,
        **kwargs,
    ):
        logger.info("Got OCR 2.0 kwargs: %s", kwargs)
        if "ocr_type" not in kwargs:
            kwargs["ocr_type"] = "ocr"
        if image.mode == "RGBA" or image.mode == "CMYK":
            # convert to RGB
            image = image.convert("RGB")
        assert self._model is not None
        # This chat API limits the max new tokens inside.
        result = self._model.chat(self._tokenizer, image, gradio_input=True, **kwargs)
        if result is None:
            logger.warning("Got OCR 2.0 returned empty result.")
            return ""
        return result
