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

import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional, Union

import PIL.Image

from ....device_utils import is_vacc_available
from ....thirdparty.navidc_ocr import MODEL_ARCHITECTURE as NAVIDC_MODEL_ARCHITECTURE
from ...utils import allow_trust_remote_code
from .deepseek_ocr import DeepSeekOCRModel
from .got_ocr2 import GotOCR2Model
from .hunyuan_ocr import HunyuanOCRModel
from .navidc_ocr import NaviDCOCRModel
from .paddleocr_vl import PaddleOCRVLModel

logger = logging.getLogger(__name__)


def _load_vllm_model(model_path: str, model_kwargs: Dict[str, Any]):
    try:
        if is_vacc_available():
            import vllm_vacc  # noqa: F401
        from vllm import LLM
    except ImportError as exc:
        error_message = "Failed to import module 'vllm'"
        installation_guide = [
            "Please make sure 'vllm' is installed. ",
            "You can install it by `pip install vllm`\n",
        ]
        raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}") from exc

    filtered_kwargs = _filter_engine_args(model_kwargs)
    if filtered_kwargs.keys() != model_kwargs.keys():
        dropped = set(model_kwargs) - set(filtered_kwargs)
        logger.info("Dropping unsupported vLLM args: %s", sorted(dropped))
    import inspect

    llm_params = inspect.signature(LLM.__init__).parameters
    if "task" in llm_params:
        return LLM(model=model_path, task="generate", **filtered_kwargs)
    return LLM(model=model_path, **filtered_kwargs)


def _sanitize_vllm_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    vllm_kwargs = dict(kwargs)
    for key in (
        "device",
        "device_map",
        "torch_dtype",
        "attn_implementation",
        "use_fast",
    ):
        vllm_kwargs.pop(key, None)
    if "cpu_offload" in vllm_kwargs and "cpu_offload_gb" not in vllm_kwargs:
        vllm_kwargs["cpu_offload_gb"] = vllm_kwargs.pop("cpu_offload")
    return vllm_kwargs


def _filter_engine_args(model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from vllm.engine.arg_utils import EngineArgs
    except Exception:  # noqa: BLE001
        return model_kwargs

    import inspect

    valid_keys = set(inspect.signature(EngineArgs.__init__).parameters.keys())
    valid_keys.discard("self")
    return {key: value for key, value in model_kwargs.items() if key in valid_keys}


def _build_sampling_params(kwargs: Dict[str, Any]):
    from vllm import SamplingParams

    max_tokens = kwargs.pop("max_tokens", None)
    if max_tokens is None:
        max_tokens = kwargs.pop("max_new_tokens", 2048)

    do_sample = kwargs.pop("do_sample", False)
    temperature = kwargs.pop("temperature", None)
    if temperature is None and not do_sample:
        temperature = 0.0

    top_p = kwargs.pop("top_p", None)
    stop = kwargs.pop("stop", None)
    params: Dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        params["top_p"] = top_p
    if stop is not None:
        params["stop"] = stop
    return SamplingParams(**params)


def _extract_text(outputs: List[Any]) -> List[str]:
    texts: List[str] = []
    for output in outputs:
        if not output.outputs:
            texts.append("")
            continue
        texts.append((output.outputs[0].text or "").strip())
    return texts


def _shutdown_vllm_model(model: Any) -> None:
    if model is None:
        return
    try:
        shutdown = getattr(model, "shutdown", None)
        if callable(shutdown):
            shutdown()
            return
    except Exception:
        logger.debug("Failed to call vLLM model.shutdown()", exc_info=True)

    engine = getattr(model, "llm_engine", None) or getattr(model, "engine", None)
    if engine is None:
        return
    try:
        engine_shutdown = getattr(engine, "shutdown", None)
        if callable(engine_shutdown):
            engine_shutdown()
    except Exception:
        logger.debug("Failed to call vLLM engine.shutdown()", exc_info=True)
    try:
        model_executor = getattr(engine, "model_executor", None)
        executor_shutdown = getattr(model_executor, "shutdown", None)
        if callable(executor_shutdown):
            executor_shutdown()
    except Exception:
        logger.debug("Failed to call vLLM executor.shutdown()", exc_info=True)


class VLLMDeepSeekOCRModel(DeepSeekOCRModel):
    required_libs = ("vllm",)

    def load(self):
        vllm_kwargs = _sanitize_vllm_kwargs(self._kwargs)
        self._model = _load_vllm_model(self._model_path, vllm_kwargs)
        self._tokenizer = self._model.get_tokenizer()

    def stop(self):
        _shutdown_vllm_model(self._model)
        self._model = None
        self._tokenizer = None

    def _prepare_inputs(
        self, prompt: str, image: Union[PIL.Image.Image, List[PIL.Image.Image]]
    ) -> List[Dict[str, Any]]:
        images = image if isinstance(image, list) else [image]
        return [
            {"prompt": prompt, "multi_modal_data": {"image": [img]}} for img in images
        ]

    def ocr(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        if self._model is None:
            self.load()
        assert self._model is not None

        prompt = kwargs.pop("prompt", "<image>\nFree OCR.")
        kwargs.pop("model_size", None)
        kwargs.pop("test_compress", None)
        kwargs.pop("save_results", None)
        kwargs.pop("save_dir", None)
        kwargs.pop("eval_mode", None)

        sampling_params = _build_sampling_params(kwargs)
        inputs = self._prepare_inputs(prompt, image)
        outputs = self._model.generate(inputs, sampling_params)
        texts = _extract_text(outputs)

        def _as_response(text: str) -> Dict[str, Any]:
            return {
                "text": text,
                "model": "deepseek-ocr",
                "engine": "vllm",
                "success": True,
            }

        if isinstance(image, list):
            return [_as_response(text) for text in texts]
        return _as_response(texts[0] if texts else "")

    def visualize_ocr(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
        model_size: str = "gundam",
        save_results: bool = False,
        save_dir: Optional[str] = None,
        eval_mode: bool = False,
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        _ = (model_size, save_results, save_dir, eval_mode)
        response = self.ocr(image=image, prompt=prompt, **kwargs)
        if isinstance(response, list):
            return [
                {
                    **item,
                    "visualization": {
                        "has_annotations": False,
                        "num_bounding_boxes": 0,
                        "num_extracted_images": 0,
                    },
                }
                for item in response
            ]
        response["visualization"] = {
            "has_annotations": False,
            "num_bounding_boxes": 0,
            "num_extracted_images": 0,
        }
        return response


class VLLMGotOCR2Model(GotOCR2Model):
    required_libs = ("vllm",)


class VLLMHunyuanOCRModel(HunyuanOCRModel):
    required_libs = ("vllm",)

    def load(self):
        from transformers import AutoProcessor

        vllm_kwargs = _sanitize_vllm_kwargs(self._kwargs)
        self._model = _load_vllm_model(self._model_path, vllm_kwargs)
        self._tokenizer = self._model.get_tokenizer()
        self._processor = AutoProcessor.from_pretrained(
            self._model_path,
            use_fast=False,
            trust_remote_code=allow_trust_remote_code(self.model_family),
        )

    def stop(self):
        _shutdown_vllm_model(self._model)
        self._model = None
        self._tokenizer = None
        self._processor = None

    def _build_prompt(self, image: PIL.Image.Image, prompt: str) -> str:
        processor = self._processor
        assert processor is not None
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
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def ocr(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        prompt: Optional[str] = None,
        **kwargs,
    ) -> Union[str, List[str]]:
        if self._model is None or self._processor is None:
            self.load()
        assert self._model is not None

        if prompt is None:
            prompt = (
                "Detect and recognize text within images, then output the text "
                "coordinates in a formatted manner."
            )

        if isinstance(image, list):
            prompts = [self._build_prompt(img, prompt) for img in image]
            inputs = [
                {"prompt": text, "multi_modal_data": {"image": [img]}}
                for text, img in zip(prompts, image)
            ]
        else:
            text = self._build_prompt(image, prompt)
            inputs = [{"prompt": text, "multi_modal_data": {"image": [image]}}]

        sampling_params = _build_sampling_params(kwargs)
        outputs = self._model.generate(inputs, sampling_params)
        texts = _extract_text(outputs)

        if isinstance(image, list):
            return texts
        return texts[0] if texts else ""


class VLLMNaviDCOCRModel(NaviDCOCRModel):
    required_libs = ("vllm",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._actor_loop: Optional[asyncio.AbstractEventLoop] = None
        self._actor_loop_thread_id: Optional[int] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._actor_loop = loop
        self._actor_loop_thread_id = threading.get_ident()

    def _run_on_actor_loop(self, fn, *args, **kwargs):
        # vLLM's offline client must be created and called on the model actor's
        # event-loop thread; moving either operation to asyncio.to_thread stalls.
        loop = self._actor_loop
        if loop is None or threading.get_ident() == self._actor_loop_thread_id:
            return fn(*args, **kwargs)

        async def _invoke():
            return fn(*args, **kwargs)

        return asyncio.run_coroutine_threadsafe(_invoke(), loop).result()

    def load(self):
        from transformers import AutoProcessor

        vllm_kwargs = _sanitize_vllm_kwargs(self._kwargs)
        hf_overrides = vllm_kwargs.pop("hf_overrides", None) or {}
        if not isinstance(hf_overrides, dict):
            raise TypeError("NaviDC-OCR requires hf_overrides to be a dictionary")
        hf_overrides = dict(hf_overrides)
        hf_overrides["architectures"] = [NAVIDC_MODEL_ARCHITECTURE]
        vllm_kwargs["hf_overrides"] = hf_overrides
        vllm_kwargs.setdefault(
            "trust_remote_code", allow_trust_remote_code(self.model_family)
        )
        vllm_kwargs.setdefault("gpu_memory_utilization", 0.7)
        vllm_kwargs.setdefault("max_model_len", 16384)

        try:
            self._model = self._run_on_actor_loop(
                _load_vllm_model, self._model_path, vllm_kwargs
            )
            self._processor = AutoProcessor.from_pretrained(
                self._model_path,
                trust_remote_code=allow_trust_remote_code(self.model_family),
                use_fast=True,
            )
        except Exception:
            self.stop()
            raise

    def stop(self):
        try:
            self._run_on_actor_loop(_shutdown_vllm_model, self._model)
        except Exception:
            logger.exception("Failed to shut down NaviDC-OCR vLLM model")
        finally:
            self._model = None
            self._processor = None

    def _build_prompt(self, prompt: str, system_prompt: str) -> str:
        processor = self._processor
        assert processor is not None
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def ocr(
        self,
        image: PIL.Image.Image,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        if self._model is None or self._processor is None:
            self.load()
        assert self._model is not None

        if not isinstance(image, PIL.Image.Image):
            raise ValueError("Input must be a PIL Image")
        image = image.convert("RGB")
        prompt = self._normalize_prompt(prompt)

        system_prompt = kwargs.pop("system_prompt", self.DEFAULT_SYSTEM_PROMPT)
        chat_prompt = self._build_prompt(
            prompt,
            system_prompt,
        )
        inputs = [
            {
                "prompt": chat_prompt,
                "multi_modal_data": {"image": [image]},
            }
        ]

        kwargs.pop("use_cache", None)
        kwargs.setdefault("max_new_tokens", 4096)
        sampling_params = _build_sampling_params(kwargs)
        outputs = self._run_on_actor_loop(self._model.generate, inputs, sampling_params)
        texts = _extract_text(outputs)
        return texts[0] if texts else ""


class VLLMPaddleOCRVLModel(PaddleOCRVLModel):
    required_libs = ("vllm",)
