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
import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

import torch

from ....constants import XINFERENCE_MAX_TOKENS
from ....device_utils import (
    get_device_preferred_dtype,
    gpu_count,
    is_hf_accelerate_supported,
)
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionChoice,
    CompletionChunk,
    CreateCompletionTorch,
    LoRA,
    PytorchGenerateConfig,
    PytorchModelConfig,
)
from ...scheduler.request import InferenceRequest
from ...utils import check_dependency_available, select_device
from ..core import LLM, chat_context_var
from ..llm_family import LLMFamilyV2, LLMSpecV1
from ..utils import (
    DEEPSEEK_TOOL_CALL_FAMILY,
    LLAMA3_TOOL_CALL_FAMILY,
    QWEN_TOOL_CALL_FAMILY,
    ChatModelMixin,
)
from .utils import (
    _get_pad_param,
    convert_to_cache_cls,
    get_context_length,
    get_max_src_len,
    pad_prefill_tokens,
)

logger = logging.getLogger(__name__)

# !!!!! Do not add model_name to this list; register architectures via `register_non_default_model` instead!
NON_DEFAULT_MODEL_LIST: List[str] = []


# Define the decorator to support multiple names registration
def register_non_default_model(*architectures: str):
    """
    Decorator for registering non-default model architectures.

    Args:
        *architectures (str): One or more architecture names to be treated as non-default.

    Returns:
        A decorator function that adds the provided model names to the NON_DEFAULT_MODEL_LIST.
    """

    def decorator(cls):
        """
        Inner decorator function that modifies the class by registering model names.

        Args:
            cls: The class to be decorated.

        Returns:
            The original class after registering the model names.
        """
        for name in architectures:
            if name not in NON_DEFAULT_MODEL_LIST:
                NON_DEFAULT_MODEL_LIST.append(name)
        return cls

    return decorator


class PytorchModel(LLM):
    allow_batch = True

    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV2",
        model_path: str,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
        peft_model: Optional[List[LoRA]] = None,
    ):
        super().__init__(model_uid, model_family, model_path)
        self._use_fast_tokenizer = True
        self._pytorch_model_config: PytorchModelConfig = self._sanitize_model_config(
            pytorch_model_config
        )
        self._context_length: Optional[int] = None
        self._peft_model = peft_model

    def _sanitize_model_config(
        self, pytorch_model_config: Optional[PytorchModelConfig]
    ) -> PytorchModelConfig:
        if pytorch_model_config is None:
            pytorch_model_config = PytorchModelConfig()
        pytorch_model_config.setdefault("revision", self.model_spec.model_revision)
        pytorch_model_config.setdefault("gptq_ckpt", None)
        pytorch_model_config.setdefault("gptq_wbits", 16)
        pytorch_model_config.setdefault("gptq_groupsize", -1)
        pytorch_model_config.setdefault("gptq_act_order", False)
        pytorch_model_config.setdefault("device", "auto")
        pytorch_model_config.setdefault("trust_remote_code", True)
        pytorch_model_config.setdefault("max_num_seqs", 16)
        pytorch_model_config.setdefault("enable_tensorizer", False)
        pytorch_model_config.setdefault("reasoning_content", False)
        pytorch_model_config.setdefault("quantization_config", {})
        return pytorch_model_config

    def _sanitize_generate_config(
        self,
        generate_config: Optional[PytorchGenerateConfig],
    ) -> PytorchGenerateConfig:
        if generate_config is None:
            generate_config = PytorchGenerateConfig(**CreateCompletionTorch().dict())
        else:
            # Validate generate_config and fill default values to the generate config.
            generate_config = PytorchGenerateConfig(
                **CreateCompletionTorch(**generate_config).dict()
            )
        if not generate_config.get("max_tokens") and XINFERENCE_MAX_TOKENS:
            generate_config["max_tokens"] = XINFERENCE_MAX_TOKENS  # type: ignore
        generate_config["model"] = self.model_uid
        return generate_config

    def _check_tensorizer_integrity(self):
        if not self._pytorch_model_config.get("enable_tensorizer"):
            return False

        from .tensorizer_utils import check_tensorizer_integrity

        integrity = check_tensorizer_integrity(
            self.model_path,
            [component[0] for component in self._get_components()],
        )
        logger.info(f"Tensorizer files integrity: {integrity} {self.model_uid}")
        return integrity

    def _load_tensorizer(self, **kwargs):
        enable_tensorizer = self._pytorch_model_config.get("enable_tensorizer", None)
        if enable_tensorizer:
            from .tensorizer_utils import load_from_tensorizer

            component_metadata = [
                (name, type, kwargs)
                for name, _, type, kwargs in self._get_components(**kwargs)
            ]
            model, tokenizer = load_from_tensorizer(
                self.model_path, component_metadata, self._get_model_class(), **kwargs
            )
            return model, tokenizer

    def _save_tensorizer(self, **kwargs):
        enable_tensorizer = self._pytorch_model_config.get("enable_tensorizer", None)
        if enable_tensorizer:
            from .tensorizer_utils import save_to_tensorizer

            components = [(name, obj) for name, obj, _, _ in self._get_components()]
            save_to_tensorizer(self.model_path, self._model, components, **kwargs)

    def _get_model_class(self):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM

    def _get_components(self, **kwargs):
        from transformers import AutoTokenizer

        return [
            (
                "tokenizer",
                getattr(self, "_tokenizer", None),
                AutoTokenizer,
                {
                    "use_fast": self._use_fast_tokenizer,
                    "trust_remote_code": kwargs.get("trust_remote_code", True),
                    "revision": kwargs.get("revision"),
                    "code_revision": kwargs.get("code_revision", None),
                },
            )
        ]

    def _load_model(self, **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            error_message = "Failed to import module 'transformers'"
            installation_guide = [
                "Please make sure 'transformers' is installed. ",
                "You can install it by `pip install transformers`\n",
            ]
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=self._use_fast_tokenizer,
            trust_remote_code=kwargs["trust_remote_code"],
            revision=kwargs["revision"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            **kwargs,
        )

        return model, tokenizer

    def _apply_lora(self):
        if self._peft_model is not None:
            try:
                from peft import PeftModel
            except ImportError:
                raise ImportError(
                    f"Failed to import 'PeftModel' from 'peft'. Please make sure 'peft' is installed.\n\n"
                )

            for i, peft_model in enumerate(self._peft_model):
                if i == 0:
                    self._model = PeftModel.from_pretrained(
                        self._model,
                        peft_model.local_path,
                        adapter_name=peft_model.lora_name,
                    )
                else:
                    self._model.load_adapter(
                        peft_model.local_path, adapter_name=peft_model.lora_name
                    )
                logger.info(
                    f"PEFT adaptor '{peft_model.lora_name}' successfully loaded for model '{self.model_uid}'."
                )

    def apply_bnb_quantization(
        self, kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        _kwargs = kwargs if kwargs is not None else {}
        quantization_config = (
            self._pytorch_model_config.get("quantization_config") or {}
        )
        if quantization_config:
            # If `load_in_4bit` is enabled, apply default quantization presets.
            if quantization_config.get("load_in_4bit", False):
                quantization_config.setdefault("bnb_4bit_compute_dtype", torch.float16)
                quantization_config.setdefault("bnb_4bit_use_double_quant", True)
                quantization_config.setdefault(
                    "llm_int8_skip_modules",
                    [
                        "lm_head",
                        "encoder",
                        "EncDecAttention",
                    ],
                )

            from transformers import BitsAndBytesConfig

            _kwargs["quantization_config"] = BitsAndBytesConfig(**quantization_config)
        return _kwargs

    def apply_fp_quantization(
        self, kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if self.model_spec.model_format != "fp4":
            return kwargs if kwargs is not None else {}

        _kwargs = kwargs if kwargs is not None else {}
        quantization_config = (
            self._pytorch_model_config.get("quantization_config") or {}
        )

        try:
            from transformers import FPQuantConfig
        except ImportError as exc:
            raise ImportError(
                "FP4 quantization requires `transformers` with FPQuantConfig support."
            ) from exc

        if isinstance(quantization_config, FPQuantConfig):
            fp_config = quantization_config
        elif isinstance(quantization_config, dict):
            fp_kwargs = dict(quantization_config)
            fp_kwargs.setdefault("pseudoquantization", True)
            if "forward_dtype" not in fp_kwargs:
                model_quant = (self.model_spec.quantization or "").lower()
                if model_quant in ("mxfp4", "nvfp4"):
                    fp_kwargs["forward_dtype"] = model_quant
            fp_config = FPQuantConfig(**fp_kwargs)
        else:
            raise ValueError(
                "fp4 quantization_config must be a dict or FPQuantConfig instance"
            )

        _kwargs["quantization_config"] = fp_config
        return _kwargs

    def apply_quantization_config(
        self, kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if self.model_spec.model_format == "fp4":
            return self.apply_fp_quantization(kwargs)
        if self.model_spec.model_format == "pytorch":
            return self.apply_bnb_quantization(kwargs)
        return kwargs if kwargs is not None else {}

    def load(self):
        num_gpus = gpu_count()
        device = self._pytorch_model_config.get("device", "auto")
        self._pytorch_model_config["device"] = select_device(device)
        self._device = self._pytorch_model_config["device"]

        kwargs = {}

        torch_dtype = self._pytorch_model_config.get("torch_dtype")
        if torch_dtype is not None:
            if isinstance(torch_dtype, str) and torch_dtype != "auto":
                torch_dtype = getattr(torch, torch_dtype)
            kwargs["torch_dtype"] = torch_dtype
        else:
            dtype = get_device_preferred_dtype(self._device)

            if dtype is not None:
                kwargs["torch_dtype"] = dtype
            else:
                raise ValueError(f"Device {self._device} is not supported in temporary")

        if self.model_spec.model_format == "fp4":
            kwargs["torch_dtype"] = torch.bfloat16

        kwargs["revision"] = self._pytorch_model_config.get(
            "revision", self.model_spec.model_revision
        )
        kwargs["trust_remote_code"] = self._pytorch_model_config.get(
            "trust_remote_code"
        )

        is_device_map_auto = False

        # This is required for Intel GPU to actually work with accelerate device_map until
        # https://github.com/intel/intel-extension-for-pytorch/issues/522
        # is resolved
        max_memory_env = os.getenv("ACCELERATE_MAX_MEMORY", None)

        if max_memory_env is not None:
            max_memory_raw = json.loads(max_memory_env)
            max_memory = {
                int(k) if k.isdigit() else k: max_memory_raw[k] for k in max_memory_raw
            }
            kwargs["max_memory"] = max_memory

        # handle quantization
        kwargs = self.apply_quantization_config(kwargs)

        if num_gpus > 0 and is_hf_accelerate_supported(self._device):
            kwargs.update({"device_map": "auto"})
            is_device_map_auto = True

        reasoning_content = self._pytorch_model_config.pop("reasoning_content")
        enable_thinking = self._pytorch_model_config.pop("enable_thinking", False)
        self.prepare_parse_reasoning_content(
            reasoning_content, enable_thinking=enable_thinking
        )
        self.prepare_parse_tool_calls()

        logger.debug("Loading Transformers model with kwargs: %s", kwargs)

        if self._check_tensorizer_integrity():
            self._model, self._tokenizer = self._load_tensorizer(**kwargs)
        else:
            self._model, self._tokenizer = self._load_model(**kwargs)

        self._apply_lora()

        if not is_device_map_auto:
            self._model.to(self._device)

        self._save_tensorizer(**kwargs)

        # set context length
        self._context_length = self._pytorch_model_config.get(
            "context_length", get_context_length(self._model.config)
        )

        logger.debug(f"Model Memory: {self._model.get_memory_footprint()}")

        # Initialize batch scheduler if batching is enabled
        self._batch_scheduler = None
        if self._should_use_batching():
            from ...scheduler.batch import BatchScheduler

            self._batch_scheduler = BatchScheduler(self)
            # Note: scheduler will be started when first request comes in

    def _should_use_batching(self) -> bool:
        """Check if this model should use batch scheduling"""
        # Apply the original allow_batching logic
        model_ability = getattr(self.model_family, "model_ability", [])

        # For multimodal models, check if they're in the allowed list
        if "vision" in model_ability or "audio" in model_ability:
            from ....core.model import XINFERENCE_BATCHING_ALLOWED_VISION_MODELS

            if (
                self.model_family.model_name
                in XINFERENCE_BATCHING_ALLOWED_VISION_MODELS
                or getattr(self.model_family, "model_family", None)
                in XINFERENCE_BATCHING_ALLOWED_VISION_MODELS
            ):
                max_num_seqs = self._pytorch_model_config.get("max_num_seqs", 16)
                return max_num_seqs > 1
            else:
                logger.warning(
                    f"Currently for multimodal models, "
                    f"xinference only supports {', '.join(XINFERENCE_BATCHING_ALLOWED_VISION_MODELS)} for batching. "
                    f"Your model {self.model_family.model_name} with model family {getattr(self.model_family, 'model_family', None)} is disqualified."
                )
                return False

        # For regular PytorchModel (non-multimodal), enable batching by default
        max_num_seqs = self._pytorch_model_config.get("max_num_seqs", 16)
        return max_num_seqs > 1

    async def _ensure_scheduler_started(self):
        """Ensure the batch scheduler is started"""
        if self._batch_scheduler and not self._batch_scheduler._running:
            await self._batch_scheduler.start()

    async def generate(self, prompt: str, generate_config: Optional[dict] = None):
        """Generate method that handles both batching and non-batching"""
        if self._batch_scheduler:
            await self._ensure_scheduler_started()
            # Use batching path
            from asyncio import Queue
            from concurrent.futures import Future as ConcurrentFuture

            # Check if streaming
            stream = generate_config and generate_config.get("stream", False)

            if stream:
                queue: Queue = Queue()
                await self._batch_scheduler.add_request(
                    prompt, queue, "generate", generate_config
                )
                # Return async generator for streaming
                return self._queue_to_async_generator(queue)
            else:
                future: ConcurrentFuture = ConcurrentFuture()
                await self._batch_scheduler.add_request(
                    prompt, future, "generate", generate_config
                )
                import asyncio

                fut = asyncio.wrap_future(future)
                return await fut
        else:
            # Use direct path - subclasses should implement this
            return await self._direct_generate(prompt, generate_config)

    async def _direct_generate(
        self, prompt: str, generate_config: Optional[dict] = None
    ):
        """Direct generate implementation - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _direct_generate")

    async def _queue_to_async_generator(self, queue):
        """Convert queue to async generator for streaming"""
        from ...scheduler.core import (
            XINFERENCE_STREAMING_ABORT_FLAG,
            XINFERENCE_STREAMING_DONE_FLAG,
            XINFERENCE_STREAMING_ERROR_FLAG,
        )

        while True:
            item = await queue.get()
            if item == XINFERENCE_STREAMING_DONE_FLAG:
                break
            elif isinstance(item, str) and item.startswith(
                XINFERENCE_STREAMING_ERROR_FLAG
            ):
                raise ValueError(item[len(XINFERENCE_STREAMING_ERROR_FLAG) :])
            elif item == XINFERENCE_STREAMING_ABORT_FLAG:
                raise RuntimeError("Request was aborted")
            else:
                yield item

    async def abort_request(self, request_id: str) -> Optional[str]:
        """Abort a request - delegate to batch scheduler if available"""
        if self._batch_scheduler:
            return await self._batch_scheduler.abort_request(request_id)
        else:
            # For non-batching models, indicate that model doesn't handle abort
            return None

    async def stop_scheduler(self):
        """Stop the batch scheduler"""
        if self._batch_scheduler:
            await self._batch_scheduler.stop()

    def stop(self):
        """Stop the model and clean up resources"""
        import asyncio

        if self._batch_scheduler:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If in async context, create a task
                    asyncio.create_task(self.stop_scheduler())
                else:
                    # If not in async context, run sync
                    asyncio.run(self.stop_scheduler())
            except Exception as e:
                logger.warning(f"Failed to stop scheduler: {e}")
        # Clean up model resources if needed
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_tokenizer"):
            del self._tokenizer

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        dep_check = check_dependency_available("transformers", "transformers")
        if dep_check != True:
            return dep_check
        return True

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if llm_spec.model_format not in ["pytorch", "gptq", "awq", "bnb", "fp4"]:
            return (
                False,
                "Transformers engine supports pytorch/gptq/awq/bnb/fp4 formats only",
            )
        if llm_family.matches_supported_architectures(NON_DEFAULT_MODEL_LIST):
            return (
                False,
                f"Model architectures {llm_family.architectures} require a custom transformer implementation",
            )
        if "generate" not in llm_family.model_ability:
            return False, "Transformers engine requires generate ability"
        return True

    def build_prefill_attention_mask(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        Build attention mask for prefill phase.
        Padding `0` on the left.
        Note that the parameter `seq_length` is from `input_ids`.
        """
        data = []
        for r in reqs:
            real_len = seq_length - r.padding_len
            r.extra_kwargs["attention_mask_seq_len"] = real_len

            if self._tokenizer.padding_side == "left":
                # [PAD][PAD]...[TOKEN]
                x = torch.cat(
                    [
                        torch.full((r.padding_len,), 0, dtype=torch.long),
                        torch.ones((real_len,), dtype=torch.long),
                    ]
                )
            else:  # right padding
                # [TOKEN]...[PAD][PAD]
                x = torch.cat(
                    [
                        torch.ones((real_len,), dtype=torch.long),
                        torch.full((r.padding_len,), 0, dtype=torch.long),
                    ]
                )
            data.append(x)

        return torch.stack(data).to(self._device)

    def build_decode_attention_mask(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        Build attention mask for decode phase.
        Note that the `seq_length` parameter is from merged kv_cache.
        So we need pad `0` on the left again.
        """
        data = []
        max_len = max(r.extra_kwargs["attention_mask_seq_len"] for r in reqs) + 1
        for r in reqs:
            r.extra_kwargs["attention_mask_seq_len"] += 1
            real_len = r.extra_kwargs["attention_mask_seq_len"]
            pad_len = max_len - real_len

            if self._tokenizer.padding_side == "left":
                x = torch.cat(
                    [
                        (
                            torch.full((pad_len,), 0, dtype=torch.long)
                            if pad_len > 0
                            else torch.tensor([], dtype=torch.long)
                        ),
                        torch.ones((real_len,), dtype=torch.long),
                    ]
                )
            else:
                x = torch.cat(
                    [
                        torch.ones((real_len,), dtype=torch.long),
                        torch.full((pad_len,), 0, dtype=torch.long),
                    ]
                )
            data.append(x)

        return torch.stack(data).to(self._device)

    def build_prefill_position_ids(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        Build position ids for prefill phase.
        Padding `0` on the left.
        Note that the parameter `seq_length` is from `input_ids`.
        Record the `max_position_id` on request for the decode phase.
        """
        res = []
        for r in reqs:
            real_seq_len = seq_length - r.padding_len
            res.append(
                torch.cat(
                    [
                        torch.full((r.padding_len,), 0, dtype=torch.long),
                        torch.arange(0, real_seq_len, dtype=torch.long),
                    ]
                )
            )
            r.extra_kwargs["max_position_id"] = real_seq_len - 1
        return torch.stack(res).to(self._device)

    def build_decode_position_ids(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        Build position ids for decode phase.
        For most models, just let the `max_position_id` in previous step += 1 and use the latest `max_position_id`
        """
        data = []
        for r in reqs:
            r.extra_kwargs["max_position_id"] += 1
            data.append([r.extra_kwargs["max_position_id"]])
        position_ids = torch.as_tensor(data, dtype=torch.long, device=self._device)
        return position_ids

    def build_prefill_token_type_ids(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        Build token_type_ids for prefill phase.
        For most models, this is not required.
        """
        return None

    def build_decode_token_type_ids(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        Build token_type_ids for decode phase.
        For most models, this is not required.
        """
        return None

    def build_prefill_inputs(self, prompts: List, req_list: List[InferenceRequest]):
        """
        Get inputs for inference. Models may have their own impl.
        """
        assert isinstance(prompts[0], str)
        inputs = self._tokenizer(prompts, padding=False).input_ids
        context_len = self.get_context_len()
        input_ids = torch.as_tensor(
            pad_prefill_tokens(inputs, context_len, req_list), device=self._device
        )
        return input_ids

    def build_prefill_kwargs(self, prompts: List, req_list: List[InferenceRequest]):
        """
        Get all inputs parameters for prefill phase. Models may have their own impl.
        """
        input_ids = self.build_prefill_inputs(prompts, req_list)
        res = {"input_ids": input_ids}
        batch_size, seq_len = input_ids.shape
        attention_mask = self.build_prefill_attention_mask(
            batch_size, seq_len, req_list
        )
        if attention_mask is not None:
            res["attention_mask"] = attention_mask
        position_ids = self.build_prefill_position_ids(batch_size, seq_len, req_list)
        if position_ids is not None:
            res["position_ids"] = position_ids
        token_type_ids = self.build_prefill_token_type_ids(
            batch_size, seq_len, req_list
        )
        if token_type_ids is not None:
            res["token_type_ids"] = token_type_ids
        return res

    def build_decode_kwargs(
        self,
        prompts: List,
        req_list: List[InferenceRequest],
        batch_size: int,
        seq_len: int,
    ):
        """
        Get all inputs parameters for decode phase. Models may have their own impl.
        """
        res = {"input_ids": torch.as_tensor(prompts, device=self._device)}
        attention_mask = self.build_decode_attention_mask(batch_size, seq_len, req_list)
        if attention_mask is not None:
            res["attention_mask"] = attention_mask
        position_ids = self.build_decode_position_ids(batch_size, seq_len, req_list)
        if position_ids is not None:
            res["position_ids"] = position_ids
        token_type_ids = self.build_decode_token_type_ids(batch_size, seq_len, req_list)
        if token_type_ids is not None:
            res["token_type_ids"] = token_type_ids
        return res

    @staticmethod
    def get_batch_size_and_seq_len_indexes_from_kv() -> Tuple[int, int]:
        """
        From huggingface transformers document, the `pask_key_values` has the shape of
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`.
        However, for some models, the shape may be changed.
        """
        return 0, 2

    def get_dtype(self):
        raise NotImplementedError("Not implemented.")

    @lru_cache
    def get_context_len(self):
        assert self._context_length is not None
        return self._context_length

    def get_max_num_seqs(self) -> int:
        return self._pytorch_model_config.get("max_num_seqs")  # type: ignore

    def prepare_sanitize_generate_config(self, req: InferenceRequest):
        return self._sanitize_generate_config(req.generate_config)

    def merge_kv_cache(self, past_cache, new_cache):
        from torch.nn.functional import pad
        from transformers import DynamicCache

        # Handle case where past_cache is None
        if past_cache is None:
            return new_cache

        # Convert both caches to DynamicCache if not already
        if not isinstance(past_cache, DynamicCache):
            past_cache = convert_to_cache_cls(past_cache)
        if not isinstance(new_cache, DynamicCache):
            new_cache = convert_to_cache_cls(new_cache)

        _, seq_len_idx = self.get_batch_size_and_seq_len_indexes_from_kv()

        # Handle empty caches
        if len(past_cache) == 0:
            return new_cache
        if len(new_cache) == 0:
            return past_cache

        # Get first layer seq_len safely
        past_first = past_cache[0] if len(past_cache) > 0 else (None, None)
        new_first = new_cache[0] if len(new_cache) > 0 else (None, None)

        if past_first[0] is None or past_first[1] is None:
            return new_cache
        if new_first[0] is None or new_first[1] is None:
            return past_cache

        past_seq_len = past_first[0].shape[seq_len_idx]
        new_seq_len = new_first[0].shape[seq_len_idx]

        # Pad the shorter cache
        if past_seq_len != new_seq_len:
            if past_seq_len > new_seq_len:
                padding_target = new_cache
                padding_len = past_seq_len - new_seq_len
            else:
                padding_target = past_cache
                padding_len = new_seq_len - past_seq_len

            pad_param = _get_pad_param(seq_len_idx, padding_len)
            for idx in range(len(padding_target)):
                k = padding_target.key_cache[idx]
                v = padding_target.value_cache[idx]
                if k is not None and v is not None:
                    padding_target.key_cache[idx] = pad(k, pad_param)
                    padding_target.value_cache[idx] = pad(v, pad_param)

        # Merge caches
        ret_kv = DynamicCache()
        max_layers = max(len(past_cache), len(new_cache))

        for idx in range(max_layers):
            past_k = past_cache.key_cache[idx] if idx < len(past_cache) else None
            past_v = past_cache.value_cache[idx] if idx < len(past_cache) else None
            new_k = new_cache.key_cache[idx] if idx < len(new_cache) else None
            new_v = new_cache.value_cache[idx] if idx < len(new_cache) else None

            if past_k is not None and new_k is not None:
                # Both layers exist - validate tensor dimensions before concatenation
                if past_k.dim() != new_k.dim():
                    logger.error(
                        f"KV cache tensor dimension mismatch at layer {idx}: "
                        f"past_k.dim()={past_k.dim()}, new_k.dim()={new_k.dim()}"
                    )
                    # Use the cache with higher batch size
                    if past_k.shape[0] >= new_k.shape[0]:
                        ret_kv.update(past_k, past_v, idx)
                    else:
                        ret_kv.update(new_k, new_v, idx)
                    continue

                if past_k.shape[1:] == new_k.shape[1:]:
                    # Shapes are compatible, concatenate along batch dimension
                    ret_kv.update(
                        torch.cat((new_k, past_k), 0).contiguous(),
                        torch.cat((new_v, past_v), 0).contiguous(),
                        idx,
                    )
                else:
                    # Detailed logging for shape mismatch
                    logger.warning(
                        f"KV cache shape mismatch at layer {idx}: "
                        f"past_k.shape={past_k.shape}, new_k.shape={new_k.shape}. "
                        f"This may be due to inconsistent batch sizes in continuous batching."
                    )

                    # Choose the cache with larger batch size to preserve more data
                    if past_k.shape[0] >= new_k.shape[0]:
                        ret_kv.update(past_k, past_v, idx)
                    else:
                        ret_kv.update(new_k, new_v, idx)
            elif past_k is not None:
                ret_kv.update(past_k, past_v, idx)
            elif new_k is not None:
                ret_kv.update(new_k, new_v, idx)
            else:
                # both None, fill with None
                ret_kv.update(None, None, idx)

        return ret_kv

    def prepare_batch_inference(self, req_list: List[InferenceRequest]):
        # check some parameters
        for r in req_list:
            try:
                if r.sanitized_generate_config is None:
                    r.sanitized_generate_config = self.prepare_sanitize_generate_config(
                        r
                    )
                if r.is_prefill:
                    # check some generate params
                    max_src_len = get_max_src_len(self.get_context_len(), r)  # type: ignore
                    if max_src_len < 0:
                        r.stopped = True
                        r.error_msg = "Max tokens exceeds model's max length"
                        continue
                    if r.stream_interval <= 0:
                        r.stopped = True
                        r.error_msg = "`stream_interval` must be greater than 0"
                        continue
                    stop_str = r.sanitized_generate_config.get("stop", None)
                    if stop_str and (
                        not (
                            isinstance(stop_str, str) or isinstance(stop_str, Iterable)
                        )
                    ):
                        r.stopped = True
                        r.error_msg = "Invalid `stop` field type"
                        continue
            # Catch exception here. If not catch exception, the request would hang.
            except Exception as e:
                logger.exception(f"prepare inference error with {e}")
                r.stopped = True
                r.error_msg = str(e)

    def get_builtin_stop_token_ids(self) -> Tuple:
        from ..utils import get_stop_token_ids_from_config_file

        try:
            stop_token_ids = get_stop_token_ids_from_config_file(self.model_path)
        except OSError:
            # some model lacks of generation_config.json
            stop_token_ids = None
        if stop_token_ids is not None:
            return tuple(stop_token_ids)
        else:
            return (
                tuple(self.model_family.stop_token_ids)
                if self.model_family.stop_token_ids
                else tuple()
            )

    def handle_batch_inference_results(self, req_list: List[InferenceRequest]):
        for req in req_list:
            if req.error_msg is None:
                # nothing need handle for non-stream case
                if req.stream:
                    results = []
                    for i, c in enumerate(req.completion):
                        if c == "<bos_stream>":
                            chunk = req.completion[i + 1]
                            results.append(
                                CompletionChunk(
                                    id=chunk["id"],
                                    object=chunk["object"],
                                    created=chunk["created"],
                                    model=chunk["model"],
                                    choices=[
                                        CompletionChoice(
                                            text="",
                                            index=0,
                                            logprobs=None,
                                            finish_reason=None,
                                        )
                                    ],
                                )
                            )
                            continue
                        elif c == "<eos_stream>":
                            break
                        else:
                            results.append(c)

                    if req.stopped and req.include_usage:
                        results.append(req.completion[-1])
                    req.completion = results

    def batch_inference(self, req_list: List[InferenceRequest]):
        from .utils import batch_inference_one_step

        self.prepare_batch_inference(req_list)
        batch_inference_one_step(
            self, req_list, self.model_uid, self._model, self._tokenizer
        )
        self.handle_batch_inference_results(req_list)

    def build_reduced_kv_cache(self, cache, skipped_indexes: Set[int]):
        batch_size = cache.key_cache[0].shape[0]
        batch_slices = [num for num in range(batch_size) if num not in skipped_indexes]
        for idx in range(len(cache)):
            cache.key_cache[idx] = cache.key_cache[idx][batch_slices, ::].contiguous()
            cache.value_cache[idx] = cache.value_cache[idx][
                batch_slices, ::
            ].contiguous()
        return cache


class PytorchChatModel(PytorchModel, ChatModelMixin):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV2",
        model_path: str,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
        peft_model: Optional[List[LoRA]] = None,
    ):
        super().__init__(
            model_uid,
            model_family,
            model_path,
            pytorch_model_config,
            peft_model,
        )

    def _sanitize_generate_config(
        self,
        generate_config: Optional[PytorchGenerateConfig],
    ) -> PytorchGenerateConfig:
        generate_config = super()._sanitize_generate_config(generate_config)
        if (not generate_config.get("stop")) and self.model_family.stop is not None:
            generate_config["stop"] = self.model_family.stop.copy()
        if (
            generate_config.get("stop_token_ids", None) is None
            and self.model_family.stop_token_ids is not None
        ):
            generate_config["stop_token_ids"] = self.model_family.stop_token_ids.copy()

        return generate_config

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if llm_spec.model_format not in ["pytorch", "gptq", "awq", "bnb", "fp4"]:
            return (
                False,
                "Transformers chat engine supports pytorch/gptq/awq/bnb/fp4 formats only",
            )
        if llm_family.matches_supported_architectures(NON_DEFAULT_MODEL_LIST):
            return (
                False,
                f"Model architectures {llm_family.architectures} require a custom transformer implementation",
            )
        if "chat" not in llm_family.model_ability:
            return False, "Transformers chat engine requires chat ability"
        return True

    async def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """Chat method that handles both batching and non-batching"""
        if self._batch_scheduler:
            await self._ensure_scheduler_started()
            # Use batching path
            from asyncio import Queue
            from concurrent.futures import Future as ConcurrentFuture

            # Check if streaming
            stream = generate_config and generate_config.get("stream", False)

            if stream:
                queue: Queue = Queue()
                await self._batch_scheduler.add_request(
                    messages, queue, "chat", generate_config
                )
                # Return async generator for streaming
                return self._queue_to_async_generator(queue)
            else:
                future: ConcurrentFuture = ConcurrentFuture()
                await self._batch_scheduler.add_request(
                    messages, future, "chat", generate_config
                )
                import asyncio

                fut = asyncio.wrap_future(future)
                return await fut
        else:
            # Use direct path - call the original implementation
            return await self._direct_chat(messages, generate_config)

    async def _direct_chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """Direct chat implementation - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _direct_chat")

    def load(self):
        super().load()

    def _get_full_prompt(self, messages: List[Dict], tools, generate_config: dict):
        model_family = self.model_family.model_family or self.model_family.model_name
        chat_template_kwargs = (
            self._get_chat_template_kwargs_from_generate_config(
                generate_config, self.reasoning_parser
            )
            or {}
        )
        chat_context_var.set(chat_template_kwargs)
        full_context_kwargs = chat_template_kwargs.copy()
        if (
            tools
            and model_family in QWEN_TOOL_CALL_FAMILY
            or model_family in LLAMA3_TOOL_CALL_FAMILY
            or model_family in DEEPSEEK_TOOL_CALL_FAMILY
        ):
            full_context_kwargs["tools"] = tools
        assert self.model_family.chat_template is not None
        full_prompt = self.get_full_context(
            messages,
            self.model_family.chat_template,
            tokenizer=self._tokenizer,
            **full_context_kwargs,
        )
        return full_prompt

    def prepare_batch_inference(self, req_list: List[InferenceRequest]):
        super().prepare_batch_inference(req_list)
        for r in req_list:
            try:
                if not r.stopped and r.is_prefill:
                    tools = r.generate_config.get("tools", None)
                    r.full_prompt = self._get_full_prompt(
                        r.prompt, tools, r.generate_config
                    )
                    if tools:
                        r.tools = tools
            except Exception as e:
                logger.exception(f"prepare inference error with {e}")
                r.stopped = True
                r.error_msg = str(e)

    def handle_chat_result_non_streaming(self, req: InferenceRequest):
        if req.tools:
            req.completion[0] = self._post_process_completion(
                self.model_family,
                self.model_uid,
                req.completion[0],
            )
        else:
            req.completion[0] = self._to_chat_completion(
                req.completion[0], self.reasoning_parser
            )

    def handle_chat_result_streaming(self, req: InferenceRequest):
        results = []
        for i, c in enumerate(req.completion):
            if c == "<bos_stream>":
                results.extend(
                    self._get_first_chat_completion_chunk(
                        req.completion[i + 1], self.reasoning_parser
                    )
                )
            elif c == "<eos_stream>":
                break
            else:
                results.append(
                    self._to_chat_completion_chunk(
                        c,
                        self.reasoning_parser,
                        req.previous_texts,
                        ensure_role=not results,
                    )
                )

        if req.stopped and req.include_usage:
            results.append(self._get_final_chat_completion_chunk(req.completion[-1]))
        req.completion = results

    def handle_batch_inference_results(self, req_list: List[InferenceRequest]):
        for req in req_list:
            if req.error_msg is None and req.completion:
                # The `generate` function can be called for some chat models.
                # So that we cannot convert completion chunk to chat completion chunk.
                if req.call_ability == "generate":
                    results = []
                    for c in req.completion:
                        if c == "<bos_stream>":
                            continue
                        elif c == "<eos_stream>":
                            break
                        else:
                            results.append(c)
                    req.completion = results
                    continue

                if req.stream:
                    self.handle_chat_result_streaming(req)
                else:
                    self.handle_chat_result_non_streaming(req)
