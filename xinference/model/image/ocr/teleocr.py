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

import base64
import io
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import PIL.Image

from .navidc_ocr import NaviDCOCRModel

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2


class TeleOCRModel(NaviDCOCRModel):
    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return (
            model_family.model_name == "TeleOCR"
            and getattr(model_family, "model_format", None) != "ggufv2"
        )


class LlamaCppTeleOCRModel(TeleOCRModel):
    required_libs = ("xllamacpp",)

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return (
            model_family.model_name == "TeleOCR"
            and getattr(model_family, "model_format", None) == "ggufv2"
        )

    def load(self) -> None:
        from xllamacpp import CommonParams, Server

        if self._model_path is None:
            raise ValueError("A GGUF model path is required")

        model_path = self._model_path
        if os.path.isdir(model_path):
            if self.model_family is None:
                raise ValueError(
                    "A model specification is required for a GGUF directory"
                )
            template = getattr(self.model_family, "model_file_name_template", None)
            quantization = getattr(self.model_family, "quantization", None)
            if not template or not quantization:
                raise ValueError("A GGUF directory requires a quantization")
            model_path = os.path.join(
                model_path,
                template.format(quantization=quantization),
            )

        projector = (
            self._kwargs.get("multimodal_projector")
            or getattr(self.model_family, "multimodal_projector", None)
            or "NaviDC-OCR-mmproj-q8_0.gguf"
        )
        if not os.path.isabs(projector):
            projector = os.path.join(os.path.dirname(model_path), projector)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        if not os.path.isfile(projector):
            raise FileNotFoundError(projector)

        params = CommonParams()
        try:
            params.model = model_path
        except Exception:
            params.model.path = model_path
        params.mmproj.path = projector
        params.use_jinja = True
        params.n_parallel = 1
        for key in ("n_ctx", "n_batch", "n_gpu_layers"):
            if key in self._kwargs:
                setattr(params, key, self._kwargs[key])
        self._model = Server(params)

    def stop(self) -> None:
        self._model = None

    def ocr(
        self,
        image: PIL.Image.Image,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        if not isinstance(image, PIL.Image.Image):
            raise ValueError("Input must be a PIL Image")
        if self._model is None:
            self.load()

        image_buffer = io.BytesIO()
        image.convert("RGB").save(image_buffer, format="PNG")
        image_url = "data:image/png;base64," + base64.b64encode(
            image_buffer.getvalue()
        ).decode("ascii")
        messages = [
            {
                "role": "system",
                "content": kwargs.pop("system_prompt", self.DEFAULT_SYSTEM_PROMPT),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": self._normalize_prompt(prompt)},
                ],
            },
        ]
        max_tokens = kwargs.pop("max_tokens", kwargs.pop("max_new_tokens", 4096))
        request: Dict[str, Any] = {
            "model": self._model_uid,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": kwargs.pop("temperature", 0),
        }
        for key in ("top_p", "stop"):
            if key in kwargs:
                request[key] = kwargs[key]

        responses: List[Dict[str, Any]] = []

        def collect(response: Any) -> None:
            responses.extend(response if isinstance(response, list) else [response])

        assert self._model is not None
        self._model.handle_chat_completions(request, collect)
        if not responses:
            raise RuntimeError("llama.cpp returned no OCR response")
        for response in responses:
            if response.get("error"):
                raise RuntimeError(str(response["error"]))
            if response.get("choices"):
                return (response["choices"][0]["message"].get("content") or "").strip()
        raise RuntimeError("llama.cpp returned no OCR choices")
