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
from typing import Any, Dict, List, Optional, Union

import PIL.Image

from ....device_utils import is_vacc_available
from .deepseek_ocr import DeepSeekOCRModel
from .got_ocr2 import GotOCR2Model
from .hunyuan_ocr import HunyuanOCRModel
from .mineru import MinerUModel
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
            self._model_path, use_fast=False, trust_remote_code=True
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


class VLLMPaddleOCRVLModel(PaddleOCRVLModel):
    required_libs = ("vllm",)


class VLLMMinerUModel(MinerUModel):
    """vLLM-based MinerU model for faster inference."""

    required_libs = ("vllm",)

    def load(self):
        try:
            from mineru_vl_utils import MinerUClient, MinerULogitsProcessor
        except ImportError:
            raise ImportError(
                "mineru-vl-utils is required for MinerU models. "
                "Please install it with: pip install 'mineru-vl-utils[vllm]'"
            )

        logger.info(f"Loading MinerU model with vLLM from {self._model_path}")

        vllm_kwargs = _sanitize_vllm_kwargs(self._kwargs)

        # Load vLLM model with MinerU logits processor
        from vllm import LLM

        self._model = LLM(
            model=self._model_path,
            logits_processors=[MinerULogitsProcessor],
            **vllm_kwargs,
        )

        # Create MinerU client with vLLM backend
        self._client = MinerUClient(
            backend="vllm-engine",
            vllm_llm=self._model,
        )

        logger.info("MinerU model loaded successfully with vLLM backend")

    def stop(self):
        _shutdown_vllm_model(self._model)
        self._model = None
        self._client = None
