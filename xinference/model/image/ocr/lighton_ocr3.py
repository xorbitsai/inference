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
from typing import TYPE_CHECKING, Any, Optional, Tuple

import PIL.Image
import torch

from ....device_utils import get_available_device
from ...utils import allow_trust_remote_code
from .ocr_family import OCRModel

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

_engine_executor_lock = threading.Lock()


class LightOnOCR3Model(OCRModel):
    required_libs: Tuple[str, ...] = ("transformers",)
    MODEL_NAMES = ("LightOnOCR-3-0.8B", "LightOnOCR-3-1B", "LightOnOCR-3-4B")

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return (
            model_family.model_name in cls.MODEL_NAMES
            and model_family.model_format == "pytorch"
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
        self._kwargs = kwargs
        self._model: Any = None
        self._processor: Any = None

    @property
    def model_ability(self):
        return self.model_family.model_ability if self.model_family is not None else []

    @property
    def _is_1b(self) -> bool:
        return (
            self.model_family is not None
            and self.model_family.model_name == "LightOnOCR-3-1B"
        )

    @property
    def _chat_template_kwargs(self) -> dict:
        return {} if self._is_1b else {"enable_thinking": False}

    @property
    def _context_length(self) -> int:
        return 16384 if self._is_1b else 18000

    def _prepare_image(self, image: PIL.Image.Image) -> PIL.Image.Image:
        if not isinstance(image, PIL.Image.Image):
            raise ValueError("Input must be a PIL Image")
        longest_edge = 1540 if self._is_1b else 2048
        image = image.convert("RGB")
        image.thumbnail((longest_edge, longest_edge), PIL.Image.Resampling.LANCZOS)
        return image

    @staticmethod
    def _messages(
        prompt: Optional[str], image: Optional[PIL.Image.Image] = None
    ) -> list:
        prompt = (prompt or "").strip().removeprefix("<image>").lstrip()
        # Map the shared Web UI's OCR and layout prompts to the two model modes.
        if prompt.startswith("<|grounding|>"):
            prompt = "grounding"
        elif prompt in (
            "OCR",
            "Free OCR.",
            "Free OCR. Extract all text content from the image.",
        ) or prompt.startswith("Convert this document to clean markdown format."):
            prompt = ""

        image_content: dict = {"type": "image"}
        if image is not None:
            image_content["image"] = image
        content = [image_content]
        if prompt:
            content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    @staticmethod
    def _generation_kwargs(kwargs: dict) -> dict:
        kwargs = kwargs.copy()
        for key in (
            "request_id",
            "model_size",
            "test_compress",
            "save_results",
            "save_dir",
            "eval_mode",
            "filter_imgtags",
        ):
            kwargs.pop(key, None)
        max_tokens = kwargs.pop("max_tokens", None)
        kwargs.setdefault("max_new_tokens", 4096 if max_tokens is None else max_tokens)
        return kwargs

    def load(self) -> None:
        model_kwargs = self._kwargs.copy()
        if self._is_1b:
            # The 1B checkpoint declares model_type=mistral3. AutoModel would
            # select the Mistral3 class instead of the LightOnOCR implementation.
            from transformers import (
                LightOnOcrForConditionalGeneration,
                LightOnOcrProcessor,
            )

            model_class, processor_class = (
                LightOnOcrForConditionalGeneration,
                LightOnOcrProcessor,
            )
            # The original checkpoint nests the Qwen3 decoder under language_model.model.
            model_kwargs.setdefault(
                "key_mapping", {r"^language_model\.model\.": "model.language_model."}
            )
        else:
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

            model_class, processor_class = (
                Qwen3_5ForConditionalGeneration,
                AutoProcessor,
            )

        model_kwargs.pop("cpu_offload", None)
        trust_remote_code = allow_trust_remote_code(self.model_family)
        model_kwargs.setdefault("trust_remote_code", trust_remote_code)
        device = torch.device(self._device or get_available_device())
        if "torch_dtype" not in model_kwargs:
            model_kwargs.setdefault(
                "dtype",
                torch.float32 if device.type in ("cpu", "mps") else torch.bfloat16,
            )

        processor = processor_class.from_pretrained(
            self._model_path, trust_remote_code=trust_remote_code
        )
        model = model_class.from_pretrained(self._model_path, **model_kwargs)
        if model_kwargs.get("device_map") is None:
            model = model.to(device)
        self._model = model.eval()
        self._processor = processor

    def ocr(
        self, image: PIL.Image.Image, prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        image = self._prepare_image(image)
        if self._model is None or self._processor is None:
            self.load()

        inputs = self._processor.apply_chat_template(
            self._messages(prompt, image),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **self._chat_template_kwargs,
        ).to(device=self._model.device, dtype=self._model.dtype)
        generation_kwargs = self._generation_kwargs(kwargs)
        if generation_kwargs.setdefault(
            "do_sample", generation_kwargs.get("temperature") != 0
        ):
            generation_kwargs.setdefault("temperature", 0.1)
        with torch.inference_mode():
            outputs = self._model.generate(**inputs, **generation_kwargs)
        generated_ids = outputs[:, inputs["input_ids"].shape[-1] :]
        return self._processor.decode(
            generated_ids[0], skip_special_tokens=True
        ).strip()


class ThreadedLightOnOCR3Model(LightOnOCR3Model):
    """Keep offline engine creation, generation and shutdown on one thread."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The worker serializes the model before calling load().
        self._engine_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    def _run_on_engine_thread(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        with _engine_executor_lock:
            if self._engine_executor is None:
                self._engine_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1
                )
            executor = self._engine_executor
        return executor.submit(fn, *args, **kwargs).result()

    def _shutdown_model(self) -> None:
        self._model.shutdown()

    def stop(self) -> None:
        try:
            if self._model is not None:
                self._run_on_engine_thread(self._shutdown_model)
        finally:
            self._model = None
            self._processor = None
            with _engine_executor_lock:
                executor, self._engine_executor = self._engine_executor, None
            if executor is not None:
                executor.shutdown(wait=True)
