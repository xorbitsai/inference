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
import os
import subprocess
import threading
from typing import TYPE_CHECKING, Any, Optional, Tuple

import PIL.Image
import torch
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion

from ....core.exceptions import InvalidModelRequestError
from ....device_utils import get_available_device
from ...utils import allow_trust_remote_code
from .ocr_family import OCRModel

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

_engine_executor_lock = threading.Lock()


class LightOnOCR3Model(OCRModel):
    required_libs: Tuple[str, ...] = ("transformers",)
    MODEL_NAMES = ("LightOnOCR-3-0.8B", "LightOnOCR-3-1B", "LightOnOCR-3-4B")
    _transformers_versions = ">=5.5.4,<6"

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

    @staticmethod
    def _transformers_component_names(model_name: Optional[str]) -> Tuple[str, str]:
        if model_name == "LightOnOCR-3-1B":
            # The checkpoint declares mistral3, so the loader must be explicit.
            return "LightOnOcrForConditionalGeneration", "LightOnOcrProcessor"
        return "Qwen3_5ForConditionalGeneration", "AutoProcessor"

    @classmethod
    def _transformers_components(cls, model_name: Optional[str]) -> Tuple[Any, Any]:
        import transformers

        model_class, processor_class = cls._transformers_component_names(model_name)
        return getattr(transformers, model_class), getattr(
            transformers, processor_class
        )

    @classmethod
    def preflight(
        cls,
        model_spec: "ImageModelFamilyV2",
        model_engine: str,
        *,
        python_executable: Optional[str] = None,
    ) -> None:
        if model_engine.lower() != "transformers":
            return

        model_name = model_spec.model_name
        component_names = cls._transformers_component_names(model_name)
        requirement = (
            f"{model_name} requires transformers{cls._transformers_versions} with "
            f"{' and '.join(component_names)}. Upgrade Transformers or enable the model virtualenv."
        )
        try:
            if python_executable is None:
                from transformers import __version__ as installed_version

                model_class, processor_class = cls._transformers_components(model_name)
                # Access the loaders so Transformers dummy exports fail here too.
                _ = model_class.from_pretrained, processor_class.from_pretrained
            else:
                from ....core.utils import build_subpool_envs_for_python

                env = {
                    **os.environ,
                    **build_subpool_envs_for_python({}, python_executable),
                }
                result = subprocess.run(
                    [
                        python_executable,
                        "-c",
                        "import sys, transformers; "
                        "[getattr(transformers, name).from_pretrained for name in sys.argv[1:]]; "
                        "print(transformers.__version__)",
                        *component_names,
                    ],
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                installed_version = result.stdout.strip()
            compatible = installed_version in SpecifierSet(cls._transformers_versions)
        except (
            AttributeError,
            ImportError,
            InvalidVersion,
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
        ) as exc:
            raise ImportError(requirement) from exc
        if not compatible:
            raise ImportError(f"{requirement} Found transformers=={installed_version}.")

    def load(self) -> None:
        model_kwargs = self._kwargs.copy()
        model_name = (
            self.model_family.model_name if self.model_family is not None else None
        )
        model_class, processor_class = self._transformers_components(model_name)
        if self._is_1b:
            # The original checkpoint nests the Qwen3 decoder under language_model.model.
            model_kwargs.setdefault(
                "key_mapping", {r"^language_model\.model\.": "model.language_model."}
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
        generation_kwargs = self._generation_kwargs(kwargs)
        seed = generation_kwargs.pop("seed", None)
        if generation_kwargs.setdefault(
            "do_sample", generation_kwargs.get("temperature") != 0
        ):
            if seed is not None:
                raise InvalidModelRequestError(
                    "Seeded sampling is not supported by the Transformers engine for LightOnOCR-3."
                )
            generation_kwargs.setdefault("temperature", 0.1)

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
