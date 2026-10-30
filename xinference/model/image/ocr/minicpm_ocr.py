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
import concurrent.futures
import io
import os
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import PIL.Image
import torch

from ....device_utils import get_available_device
from ...utils import allow_trust_remote_code
from .ocr_family import OCRModel

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

_sglang_executor_lock = threading.Lock()


class MiniCPMOCRModel(OCRModel):
    required_libs: Tuple[str, ...] = ("transformers",)
    DEFAULT_PROMPT = "Text Recognition:"
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
            model_family.model_name == "MiniCPM-OCR-1.0"
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
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
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

    @classmethod
    def _messages(cls, image: PIL.Image.Image, prompt: Optional[str]) -> List[Dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": cls._normalize_prompt(prompt)},
                ],
            }
        ]

    @staticmethod
    def _prepare_image(image: PIL.Image.Image) -> PIL.Image.Image:
        if not isinstance(image, PIL.Image.Image):
            raise ValueError("Input must be a PIL Image")
        return image.convert("RGB")

    @staticmethod
    def _image_url(image: PIL.Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
            "ascii"
        )

    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = torch.device(self._device or get_available_device())
        model_kwargs = self._kwargs.copy()
        model_kwargs.pop("cpu_offload", None)
        trust_remote_code = allow_trust_remote_code(self.model_family)
        model_kwargs.setdefault("trust_remote_code", trust_remote_code)
        if "torch_dtype" not in model_kwargs:
            model_kwargs.setdefault(
                "dtype", torch.float32 if device.type == "cpu" else torch.bfloat16
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
        image = self._prepare_image(image)
        if self._model is None or self._processor is None:
            self.load()
        assert self._model is not None
        assert self._processor is not None
        inputs = self._processor.apply_chat_template(
            self._messages(image, prompt),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self._model.device)
        for key in self.NON_GENERATION_KWARGS:
            kwargs.pop(key, None)
        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is not None:
            kwargs.setdefault("max_new_tokens", max_tokens)
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


class SGLangMiniCPMOCRModel(MiniCPMOCRModel):
    required_libs = ("sglang", "transformers")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    def _run_on_sglang_thread(self, fn, *args, **kwargs):
        # Engine.generate uses the event loop captured during Engine construction.
        with _sglang_executor_lock:
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix=f"minicpm-ocr-{self._model_uid}"
                )
            executor = self._executor
        return executor.submit(fn, *args, **kwargs).result()

    def load(self) -> None:
        import sglang as sgl
        from transformers import AutoTokenizer

        from ...llm.sglang.runtime import prepare_jit_cache

        prepare_jit_cache()
        config = self._kwargs.copy()
        for key in ("cpu_offload", "device_map", "torch_dtype", "attn_implementation"):
            config.pop(key, None)
        config.setdefault(
            "trust_remote_code", allow_trust_remote_code(self.model_family)
        )
        config.setdefault("context_length", 40960)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            trust_remote_code=allow_trust_remote_code(self.model_family),
        )
        try:
            self._model = self._run_on_sglang_thread(
                sgl.Engine, model_path=self._model_path, **config
            )
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        try:
            if self._model is not None:
                self._run_on_sglang_thread(self._model.shutdown)
        finally:
            self._model = None
            self._tokenizer = None
            with _sglang_executor_lock:
                executor, self._executor = self._executor, None
            if executor is not None:
                executor.shutdown(wait=False)

    def ocr(
        self, image: PIL.Image.Image, prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        image = self._prepare_image(image)
        if self._model is None or self._tokenizer is None:
            self.load()
        assert self._model is not None
        assert self._tokenizer is not None
        text = self._tokenizer.apply_chat_template(
            self._messages(image, prompt),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for key in self.NON_GENERATION_KWARGS + ("use_cache",):
            kwargs.pop(key, None)
        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is not None:
            kwargs.setdefault("max_new_tokens", max_tokens)
        kwargs.setdefault("max_new_tokens", 4096)
        do_sample = kwargs.pop("do_sample", False)
        kwargs.setdefault("temperature", 1.0 if do_sample else 0.0)
        result = self._run_on_sglang_thread(
            self._model.generate,
            prompt=text,
            image_data=self._image_url(image),
            sampling_params=kwargs,
        )
        return (result.get("text") or "").strip()


class LlamaCppMiniCPMOCRModel(MiniCPMOCRModel):
    required_libs = ("xllamacpp",)

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return (
            model_family.model_name == "MiniCPM-OCR-1.0"
            and getattr(model_family, "model_format", None) == "ggufv2"
        )

    def load(self) -> None:
        from xllamacpp import CommonParams, Server

        if not self._model_path:
            raise ValueError("A GGUF model path is required")
        model_path = self._model_path
        if os.path.isdir(model_path):
            template = getattr(self.model_family, "model_file_name_template", None)
            quantization = getattr(self.model_family, "quantization", None)
            if not template or not quantization:
                raise ValueError("A GGUF directory requires a quantization")
            model_path = os.path.join(
                model_path, template.format(quantization=quantization)
            )
        projector = (
            self._kwargs.get("multimodal_projector")
            or getattr(self.model_family, "multimodal_projector", None)
            or "mmproj-model-f16-4x.gguf"
        )
        if not os.path.isabs(projector):
            projector = os.path.join(os.path.dirname(model_path), projector)
        for path in (model_path, projector):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
        params = CommonParams()
        try:
            params.model = model_path
        except (AttributeError, TypeError):
            params.model.path = model_path
        params.mmproj.path = projector
        params.use_jinja = True
        params.n_parallel = 1
        params.n_ctx = self._kwargs.get("n_ctx", 8192)
        for key in ("n_batch", "n_gpu_layers"):
            if key in self._kwargs:
                setattr(params, key, self._kwargs[key])
        self._model = Server(params)

    def stop(self) -> None:
        self._model = None

    def ocr(
        self, image: PIL.Image.Image, prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        image = self._prepare_image(image)
        if self._model is None:
            self.load()
        assert self._model is not None
        request: Dict[str, Any] = {
            "model": self._model_uid,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_url(image)},
                        },
                        {"type": "text", "text": self._normalize_prompt(prompt)},
                    ],
                }
            ],
            "stream": False,
            "max_tokens": kwargs.pop("max_tokens", kwargs.pop("max_new_tokens", 4096)),
            "temperature": kwargs.pop("temperature", 0),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for key in ("top_p", "top_k", "stop", "repeat_penalty", "seed"):
            if key in kwargs:
                request[key] = kwargs[key]
        responses: List[Dict[str, Any]] = []

        def collect(response: Any) -> None:
            responses.extend(response if isinstance(response, list) else [response])

        self._model.handle_chat_completions(request, collect)
        for response in responses:
            if response.get("error"):
                raise RuntimeError(str(response["error"]))
            if response.get("choices"):
                return (response["choices"][0]["message"].get("content") or "").strip()
        raise RuntimeError("llama.cpp returned no OCR choices")
