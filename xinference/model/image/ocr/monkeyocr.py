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

# import fitz
from typing import TYPE_CHECKING, Optional, Union

from PIL import Image

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

from ...utils import allow_trust_remote_code
from .ocr_family import OCRModel

logger = logging.getLogger(__name__)


class MonkeyOCRModel(OCRModel):
    required_libs = ("transformers", "qwen_vl_utils")

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return model_family.model_name.lower() == "monkeyocr"

    def __init__(
        self,
        model_uid: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        model_spec: Optional["ImageModelFamilyV2"] = None,
        **kwargs,
    ):
        if model_path is None:
            raise ValueError("model_path is required for MonkeyOCR")

        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = f"{model_path}/Recognition"
        # model info when loading
        self._model = None
        self._tokenizer = None
        # info
        self._model_spec = model_spec
        self._device = device
        self._abilities = (model_spec.model_ability or []) if model_spec else []
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._abilities

    def load(self):
        from ....thirdparty.monkeyocr.magic_pdf.model.custom_model import (
            MonkeyChat_transformers,
        )

        self._model = MonkeyChat_transformers(
            self._model_path,
            device=self._device,
            trust_remote_code=allow_trust_remote_code(self.model_family),
        )

    def ocr(
        self,
        image: Union[str, Image.Image],
        **kwargs,
    ) -> str:
        if self._model is None:
            raise RuntimeError("Model must be loaded.")

        logger.info("MonkeyOCR kwargs: %s", kwargs)
        pre_question = "请按照markdown格式返回解析后的结果。"
        if "question" in kwargs:
            question = kwargs.pop("question")
            if isinstance(question, str):
                question = pre_question + question
            else:
                raise ValueError("The parameter question type must be str.")
        else:
            question = pre_question

        result = self._model.batch_inference([image], [question])
        return result[0] if result else ""
