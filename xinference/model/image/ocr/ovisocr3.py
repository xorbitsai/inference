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

import concurrent.futures
import threading
from typing import TYPE_CHECKING, Any, Optional

import PIL.Image
import torch

from ....device_utils import get_available_device
from ...utils import allow_trust_remote_code
from .ovisocr2 import OvisOCR2Model
from .xiaomi_ocr import XiaomiOCRModel

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

_engine_executor_lock = threading.Lock()


class OvisOCR3Model(XiaomiOCRModel):
    _model: Any
    DEFAULT_PROMPT = OvisOCR2Model.DEFAULT_PROMPT
    MIN_PIXELS = 448 * 448
    MAX_PIXELS = 2880 * 2880

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return (
            model_family.model_name == "OvisOCR3"
            and getattr(model_family, "model_format", None) == "pytorch"
        )

    @staticmethod
    def _messages(prompt: str, image: Optional[PIL.Image.Image] = None) -> list:
        image_content: dict = {"type": "image"}
        if image is not None:
            image_content["image"] = image
        return [
            {
                "role": "user",
                "content": [image_content, {"type": "text", "text": prompt}],
            }
        ]

    @classmethod
    def _generation_kwargs(cls, kwargs: dict) -> dict:
        kwargs = kwargs.copy()
        for key in cls.NON_GENERATION_KWARGS:
            kwargs.pop(key, None)
        max_tokens = kwargs.pop("max_tokens", 16384)
        kwargs.setdefault("max_new_tokens", max_tokens)
        return kwargs

    @classmethod
    def _process_output(cls, text: str, prompt: str, filter_imgtags: bool) -> str:
        # The upstream cleanup applies only to document_markdown. Other tasks
        # (including JSON and grounding) must retain their original output.
        if prompt == cls.DEFAULT_PROMPT:
            return OvisOCR2Model._postprocess_output(text, filter_imgtags)
        return text

    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = torch.device(self._device or get_available_device())
        model_kwargs = self._kwargs.copy()
        model_kwargs.pop("cpu_offload", None)
        min_pixels = model_kwargs.pop("min_pixels", self.MIN_PIXELS)
        max_pixels = model_kwargs.pop("max_pixels", self.MAX_PIXELS)
        trust_remote_code = allow_trust_remote_code(self.model_family)
        model_kwargs.setdefault("trust_remote_code", trust_remote_code)
        if "torch_dtype" not in model_kwargs:
            model_kwargs.setdefault(
                "dtype", torch.float32 if device.type == "cpu" else torch.bfloat16
            )
        self._processor = AutoProcessor.from_pretrained(
            self._model_path,
            trust_remote_code=trust_remote_code,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        model = AutoModelForImageTextToText.from_pretrained(
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
        **kwargs: Any,
    ) -> str:
        if not isinstance(image, PIL.Image.Image):
            raise ValueError("Input must be a PIL Image")
        if self._model is None or self._processor is None:
            self.load()
        assert self._model is not None
        assert self._processor is not None
        normalized_prompt = self._normalize_prompt(prompt)
        inputs = self._processor.apply_chat_template(
            self._messages(normalized_prompt, image.convert("RGB")),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self._model.device)
        generation_kwargs = self._generation_kwargs(kwargs)
        generation_kwargs.setdefault("do_sample", False)
        with torch.inference_mode():
            output = self._model.generate(**inputs, **generation_kwargs)
        texts = self._processor.batch_decode(
            output[:, inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return (
            self._process_output(texts[0], normalized_prompt, filter_imgtags)
            if texts
            else ""
        )


class ThreadedOvisOCR3Model(OvisOCR3Model):
    """Keep offline engine creation, generation and shutdown on one thread."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The worker serializes this object into the model actor before load().
        self._engine_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    def _run_on_engine_thread(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        with _engine_executor_lock:
            if self._engine_executor is None:
                self._engine_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1
                )
            executor = self._engine_executor
        return executor.submit(func, *args, **kwargs).result()

    def _shutdown_model(self) -> None:
        self._model.shutdown()

    def stop(self) -> None:
        try:
            if self._model is not None:
                self._run_on_engine_thread(self._shutdown_model)
        finally:
            self._model = None
            self._processor = None
            executor, self._engine_executor = self._engine_executor, None
            if executor is not None:
                executor.shutdown(wait=True)
