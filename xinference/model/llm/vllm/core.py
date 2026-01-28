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

import asyncio
import importlib
import importlib.util
import itertools
import json
import logging
import multiprocessing
import os
import sys
import threading
import time
import uuid
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypedDict,
    Union,
    cast,
)

import xoscar as xo
from packaging import version
from typing_extensions import NotRequired

from ....constants import XINFERENCE_MAX_TOKENS
from ....device_utils import is_vacc_available
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionUsage,
    LoRA,
)
from .. import BUILTIN_LLM_FAMILIES, LLM, LLMFamilyV2, LLMSpecV1
from ..core import chat_context_var
from ..llm_family import cache_model_tokenizer_and_config
from ..utils import (
    DEEPSEEK_TOOL_CALL_FAMILY,
    QWEN_TOOL_CALL_FAMILY,
    QWEN_TOOL_CALL_SYMBOLS,
    ChatModelMixin,
    generate_completion_chunk,
)
from .utils import vllm_check

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput

    # Handle ExecutorBase type import for different vLLM versions
    # vLLM >= 0.11.1: from vllm.v1.executor import Executor
    # vLLM < 0.11.1: from vllm.executor.executor_base import ExecutorBase
    try:
        from vllm.v1.executor import Executor as ExecutorBase
    except ImportError:
        try:
            from vllm.executor.executor_base import ExecutorBase
        except ImportError:
            # If vLLM is not installed, define a placeholder for type checking
            ExecutorBase = Any  # type: ignore


class VLLMModelConfig(TypedDict, total=False):
    tokenizer_mode: Optional[str]
    trust_remote_code: bool
    tensor_parallel_size: int
    pipeline_parallel_size: int
    nnodes: int
    node_rank: int
    distributed_executor_backend: str
    block_size: int
    swap_space: int  # GiB
    gpu_memory_utilization: float
    max_num_batched_tokens: int
    max_num_seqs: int
    quantization: Optional[str]
    max_model_len: Optional[int]
    limit_mm_per_prompt: Optional[Dict[str, int]]
    guided_decoding_backend: Optional[str]
    scheduling_policy: Optional[str]
    reasoning_content: bool
    model_quantization: Optional[str]
    mm_processor_kwargs: NotRequired[dict[str, Any]]
    min_pixels: NotRequired[int]
    max_pixels: NotRequired[int]
    enable_expert_parallel: bool
    speculative_config: Optional[Dict[str, Any]]
    rope_scaling: Optional[Dict[str, Any]]
    hf_overrides: Optional[Dict[str, Any]]


class VLLMGenerateConfig(TypedDict, total=False):
    lora_name: Optional[str]
    n: int
    best_of: Optional[int]
    presence_penalty: float
    frequency_penalty: float
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    stop_token_ids: Optional[List[int]]
    stop: Optional[Union[str, List[str]]]
    stream: bool  # non-sampling param, should not be passed to the engine.
    stream_options: Optional[Union[dict, None]]
    skip_special_tokens: Optional[bool]
    response_format: Optional[dict]
    guided_json: Optional[Union[str, dict]]
    guided_regex: Optional[str]
    guided_choice: Optional[List[str]]
    guided_grammar: Optional[str]
    guided_json_object: Optional[bool]
    guided_decoding_backend: Optional[str]
    guided_whitespace_pattern: Optional[str]
    ignore_eos: Optional[bool]


try:
    if is_vacc_available():
        import vllm_vacc  # noqa: F401

    import vllm  # noqa: F401

    if not getattr(vllm, "__version__", None):
        raise ImportError(
            "vllm not installed properly, or wrongly be found in sys.path"
        )

    VLLM_INSTALLED = True
    VLLM_VERSION = version.parse(vllm.__version__)
except ImportError:
    VLLM_INSTALLED = False
    VLLM_VERSION = None

DEFAULT_VLLM_VERSION = version.parse("0.13.0")


def _get_effective_vllm_version() -> version.Version:
    if VLLM_VERSION is not None:
        return VLLM_VERSION
    try:
        from ....constants import XINFERENCE_ENABLE_VIRTUAL_ENV
    except Exception:
        XINFERENCE_ENABLE_VIRTUAL_ENV = False
    if XINFERENCE_ENABLE_VIRTUAL_ENV:
        return DEFAULT_VLLM_VERSION
    return version.parse("0.0.0")


def _virtual_env_allows_missing_vllm() -> bool:
    try:
        from ....constants import XINFERENCE_ENABLE_VIRTUAL_ENV
    except Exception:
        return False
    return bool(XINFERENCE_ENABLE_VIRTUAL_ENV)


def _append_unique(target: List[str], *items: str) -> None:
    for item in items:
        if item not in target:
            target.append(item)


VLLM_SUPPORTED_MULTI_MODEL_LIST: List[str] = []
VLLM_SUPPORTED_MODELS = [
    "LlamaForCausalLM",
    "MistralForCausalLM",
]
VLLM_SUPPORTED_CHAT_MODELS = [
    "LlamaForCausalLM",
    "BaichuanForCausalLM",
    "InternLM2ForCausalLM",
    "QWenLMHeadModel",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "ChatGLMForConditionalGeneration",
    "GlmForCausalLM",
    "ChatGLMModel",
]


def _update_vllm_supported_lists() -> None:
    effective_version = _get_effective_vllm_version()
    if effective_version >= version.parse("0.3.0"):
        _append_unique(VLLM_SUPPORTED_CHAT_MODELS, "Qwen2ForCausalLM")
        _append_unique(VLLM_SUPPORTED_MODELS, "Qwen2ForCausalLM")

    if effective_version >= version.parse("0.3.2"):
        _append_unique(VLLM_SUPPORTED_CHAT_MODELS, "GemmaForCausalLM")

    if effective_version >= version.parse("0.3.3"):
        _append_unique(VLLM_SUPPORTED_CHAT_MODELS, "OrionForCausalLM")

    if effective_version >= version.parse("0.4.0"):
        _append_unique(
            VLLM_SUPPORTED_CHAT_MODELS, "Qwen2MoeForCausalLM", "CohereForCausalLM"
        )

    if effective_version >= version.parse("0.5.1"):
        _append_unique(
            VLLM_SUPPORTED_CHAT_MODELS,
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
            "Qwen3ForCausalLM",
        )

    if effective_version >= version.parse("0.6.1"):
        _append_unique(VLLM_SUPPORTED_MULTI_MODEL_LIST, "InternVLChatModel")

    if effective_version >= version.parse("0.6.2"):
        _append_unique(VLLM_SUPPORTED_CHAT_MODELS, "MiniCPM3ForCausalLM")

    if effective_version >= version.parse("0.6.3"):
        _append_unique(VLLM_SUPPORTED_MODELS, "MllamaForConditionalGeneration")
        _append_unique(
            VLLM_SUPPORTED_MULTI_MODEL_LIST,
            "MllamaForConditionalGeneration",
            "Qwen2VLForConditionalGeneration",
            "Qwen2AudioForConditionalGeneration",
        )

    if effective_version >= version.parse("0.7.0"):
        _append_unique(VLLM_SUPPORTED_CHAT_MODELS, "InternLM3ForCausalLM")

    if effective_version >= version.parse("0.7.2"):
        _append_unique(
            VLLM_SUPPORTED_MULTI_MODEL_LIST,
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen2AudioForConditionalGeneration",
        )

    if effective_version >= version.parse("0.7.3"):
        _append_unique(VLLM_SUPPORTED_MULTI_MODEL_LIST, "Qwen2_5OmniModel")

    if effective_version >= version.parse("0.8.0"):
        _append_unique(VLLM_SUPPORTED_CHAT_MODELS, "Gemma3ForCausalLM")
        _append_unique(
            VLLM_SUPPORTED_MULTI_MODEL_LIST, "Gemma3ForConditionalGeneration"
        )

    if effective_version >= version.parse("0.8.4"):
        _append_unique(VLLM_SUPPORTED_CHAT_MODELS, "Glm4ForCausalLM")

    if effective_version >= version.parse("0.9.1"):
        _append_unique(VLLM_SUPPORTED_CHAT_MODELS, "MiniCPMForCausalLM")

    if effective_version >= version.parse("0.9.2"):
        _append_unique(
            VLLM_SUPPORTED_CHAT_MODELS,
            "Ernie4_5ForCausalLM",
            "Qwen3MoeForCausalLM",
        )
        _append_unique(VLLM_SUPPORTED_MULTI_MODEL_LIST, "Glm4vForConditionalGeneration")

    if effective_version >= version.parse("0.10.0"):
        _append_unique(VLLM_SUPPORTED_CHAT_MODELS, "Glm4MoeForCausalLM")
        _append_unique(
            VLLM_SUPPORTED_MULTI_MODEL_LIST, "Glm4vMoeForConditionalGeneration"
        )

    if effective_version > version.parse("0.10.0"):
        _append_unique(VLLM_SUPPORTED_CHAT_MODELS, "GptOssForCausalLM")

    if effective_version >= version.parse("0.10.2"):
        _append_unique(
            VLLM_SUPPORTED_CHAT_MODELS, "SeedOssForCausalLM", "Qwen3NextForCausalLM"
        )
        _append_unique(VLLM_SUPPORTED_MULTI_MODEL_LIST, "MiniCPMV")

    if effective_version >= version.parse("0.11.0"):
        _append_unique(VLLM_SUPPORTED_CHAT_MODELS, "DeepseekV32ForCausalLM")
        _append_unique(
            VLLM_SUPPORTED_MULTI_MODEL_LIST,
            "Qwen3VLMoeForConditionalGeneration",
            "Qwen3OmniMoeForConditionalGeneration",
        )

    if effective_version >= version.parse("0.11.2"):
        _append_unique(VLLM_SUPPORTED_CHAT_MODELS, "MiniMaxM2ForCausalLM")


_update_vllm_supported_lists()


class VLLMModel(LLM):
    allow_batch = True

    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV2",
        model_path: str,
        model_config: Optional[VLLMModelConfig],
        peft_model: Optional[List[LoRA]] = None,
    ):
        super().__init__(model_uid, model_family, model_path)
        self._model_config = model_config
        self._engine = None
        self.lora_modules = peft_model
        self.lora_requests: List[Any] = []
        self._xavier_config = None
        self._context_length: Optional[int] = None
        # distributed inference
        self._device_count = None
        self._address = model_config.pop("address", None)  # type: ignore
        self._n_worker = model_config.pop("n_worker", 1)  # type: ignore
        self._shard = model_config.pop("shard", 0)  # type: ignore
        self._driver_info = model_config.pop("driver_info", None)  # type: ignore
        self._loading_thread: Optional[threading.Thread] = None
        self._loading_error = None
        # variables used for distributed inference and multiple GPUs
        self._pool_addresses = None
        self._worker_addresses: Optional[Dict[int, List[str]]] = None
        self._all_worker_ready: Optional[threading.Event] = None
        # used to call async
        self._loop = None

    def set_xavier_config(self, value: Optional[Dict]):
        self._xavier_config = value  # type: ignore

    def set_worker_addresses(self, shard: int, worker_addresses: List[str]):
        assert self._worker_addresses is not None
        self._worker_addresses[shard] = worker_addresses
        if (
            self._all_worker_ready is not None
            and len(self._worker_addresses) == self._n_worker
        ):
            self._all_worker_ready.set()

    @property
    def driver_info(self) -> Optional[dict]:
        return self._driver_info

    @property
    def need_create_pools(self):
        return True

    def set_pool_addresses(self, pool_addresses: List[str]):
        self._pool_addresses = pool_addresses  # type: ignore

    def get_pool_addresses(self) -> Optional[List[str]]:
        return self._pool_addresses

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        # loop will be passed into XinferenceDistributedExecutor,
        # to call aynsc method with asyncio.run_coroutine_threadsafe
        self._loop = loop  # type: ignore

    def _is_vllm_v1(self) -> bool:
        """
        Check if vLLM v1 is being used.

        For vLLM >= 0.11.1: v1 is the default, no VLLM_USE_V1 env var needed
        For vLLM < 0.11.1: check VLLM_USE_V1 environment variable
        """
        from vllm import envs

        # For vLLM >= 0.11.1, v1 is default
        if VLLM_VERSION > version.parse("0.11.0"):
            return True

        # For older versions, check the environment variable
        return envs.is_set("VLLM_USE_V1") and envs.VLLM_USE_V1

    def load(self):
        try:
            import vllm
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            from vllm.lora.request import LoRARequest

            # Handle ExecutorBase import for different vLLM versions
            # vLLM >= 0.11.1: from vllm.v1.executor import Executor
            # vLLM < 0.11.1: from vllm.executor.executor_base import ExecutorBase
            try:
                from vllm.v1.executor import Executor as ExecutorBase
            except ImportError:
                from vllm.executor.executor_base import ExecutorBase
        except ImportError:
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        if not getattr(vllm, "__version__", None):
            raise ImportError(
                "vllm not installed properly, or wrongly be found in sys.path"
            )
        global VLLM_INSTALLED, VLLM_VERSION
        VLLM_INSTALLED = True
        VLLM_VERSION = version.parse(vllm.__version__)
        _update_vllm_supported_lists()

        from ..llm_family import LlamaCppLLMSpecV2

        if version.parse("0.3.1") <= VLLM_VERSION <= version.parse("0.3.3"):
            # from vllm v0.3.1 to v0.3.3, it uses cupy as NCCL backend
            # in which cupy will fork a process
            # only for xoscar >= 0.3.0, new process is allowed in subpool
            # besides, xinference set start method as forkserver for unix
            # we need to set it to fork to make cupy NCCL work
            multiprocessing.set_start_method("fork", force=True)

        self._device_count = self._get_cuda_count()
        self._model_config = self._sanitize_model_config(self._model_config)
        reasoning_content = self._model_config.pop("reasoning_content")
        enable_thinking = self._model_config.pop("enable_thinking", False)
        self.prepare_parse_reasoning_content(
            reasoning_content, enable_thinking=enable_thinking
        )
        self.prepare_parse_tool_calls()

        if (
            isinstance(self.model_spec, LlamaCppLLMSpecV2)
            and self.model_spec.model_format == "ggufv2"
        ):
            # gguf
            self._preprocess_load_gguf()

        if self.lora_modules is None:
            self.lora_requests = []
        else:
            self.lora_requests = [
                LoRARequest(
                    lora_name=lora.lora_name,
                    lora_int_id=i,
                    lora_local_path=lora.local_path,
                )
                for i, lora in enumerate(self.lora_modules, start=1)
            ]

        enable_lora = len(self.lora_requests) > 0
        max_loras = len(self.lora_requests)

        logger.info(
            f"Loading {self.model_uid} with following model config: {self._model_config}"
            f"Enable lora: {enable_lora}. Lora count: {max_loras}."
        )

        if self._xavier_config is not None:
            from .xavier.engine import XavierEngine

            # Enabling Xavier means that `enable_prefix_caching` is enabled by default.
            self._model_config.setdefault("enable_prefix_caching", True)
            xavier_transfer_block_num = self._model_config.pop(
                "xavier_transfer_block_num", 512
            )
            self._xavier_config["transfer_block_num"] = xavier_transfer_block_num
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                enable_lora=enable_lora,
                max_loras=max_loras,
                **self._model_config,
            )

            logger.debug(f"Start xavier for vllm with config: {self._xavier_config}")
            self._engine = XavierEngine.from_engine_args(
                engine_args, xavier_config=self._xavier_config
            )
        elif self._n_worker > 1 or (
            self._device_count > 1 and VLLM_VERSION >= version.parse("0.7.0")
        ):
            from vllm.config import VllmConfig

            # model across multiple workers or GPUs
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                enable_lora=enable_lora,
                max_loras=max_loras,
                **self._model_config,
            )
            self._enable_v1_if_supported(engine_args)

            assert self._loop is not None
            self._worker_addresses = {}

            def _load():
                try:
                    assert self._pool_addresses

                    if self._shard > 0:
                        assert self._driver_info
                        address = self._driver_info["address"]

                        coro = xo.actor_ref(address, self.raw_model_uid)
                        model_ref = asyncio.run_coroutine_threadsafe(
                            coro, self._loop
                        ).result()
                        coro = model_ref.set_worker_addresses(
                            self._shard, self._pool_addresses
                        )
                        asyncio.run_coroutine_threadsafe(coro, self._loop).result()
                    else:
                        self.set_worker_addresses(0, self._pool_addresses)
                        self._driver_info = {"address": self._address}

                        if self._n_worker > 1:
                            self._all_worker_ready = threading.Event()
                            # if model across workers, wait for other workers ready
                            self._all_worker_ready.wait()

                        # gather all worker addresses
                        worker_addresses = list(
                            itertools.chain(
                                *[
                                    self._worker_addresses[shard]
                                    for shard in range(self._n_worker)
                                ]
                            )
                        )
                        assert worker_addresses
                        loop = self._loop

                        if not self._is_vllm_v1():
                            # vLLM v0
                            from .distributed_executor import (
                                XinferenceDistributedExecutor,
                            )

                            class XinferenceAsyncLLMEngine(AsyncLLMEngine):
                                @classmethod
                                def _get_executor_cls(
                                    cls, engine_config: VllmConfig
                                ) -> Type[ExecutorBase]:
                                    return partial(  # type: ignore
                                        XinferenceDistributedExecutor,
                                        pool_addresses=worker_addresses,
                                        n_worker=self._n_worker,
                                        loop=loop,
                                    )

                            self._engine = XinferenceAsyncLLMEngine.from_engine_args(
                                engine_args
                            )
                        else:
                            from vllm.v1.executor.abstract import Executor

                            # Import the appropriate executor based on vLLM version
                            if VLLM_VERSION > version.parse("0.11.0"):
                                from .distributed_executor_v1 import (
                                    XinferenceDistributedExecutorV1,
                                )
                            else:
                                from .distributed_executor import (
                                    XinferenceDistributedExecutorV1,
                                )

                            # vLLM V1
                            # NOTE: loop has to be None for vLLM v1
                            # in v1, a new process called EngineCore will be created via fork by default
                            # in which executor is initialized, we cannot pass loop, or it will be stuck,
                            # instead, a new loop will be created inside executor
                            executor_cls = partial(  # type: ignore
                                XinferenceDistributedExecutorV1,
                                pool_addresses=worker_addresses,
                                n_worker=self._n_worker,
                            )
                            # patch vllm Executor.get_class
                            Executor.get_class = lambda vllm_config: executor_cls
                            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
                except:  # noqa: E722
                    logger.exception("Creating vllm engine failed")
                    self._loading_error = sys.exc_info()

            self._loading_thread = threading.Thread(target=_load)
            self._loading_thread.start()
            # wait some time for init finish
            if self._shard == 0:
                self._loading_thread.join(1)
        else:
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                enable_lora=enable_lora,
                max_loras=max_loras,
                **self._model_config,
            )
            self._enable_v1_if_supported(engine_args)
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)

        self._check_health_task = None
        if hasattr(self._engine, "check_health"):
            # vLLM introduced `check_health` since v0.4.1
            self._check_health_task = self._loop.create_task(self._check_healthy())

    def wait_for_load(self):
        if self._loading_thread:
            self._loading_thread.join()
            if self._loading_error:
                _, err, tb = self._loading_error
                raise err.with_traceback(tb)

        # set context length after engine inited
        # if shard > 0, the engine will be inited in another process
        if self._engine:
            self._set_context_length()

    def _set_context_length(self):
        if not self._is_vllm_v1():
            # v0
            self._context_length = (
                self._engine.engine.vllm_config.model_config.max_model_len
            )
        else:
            # v1
            self._context_length = self._engine.model_config.max_model_len
        assert self._context_length is not None
        logger.debug("Model context length: %s", self._context_length)

    def _enable_v1_if_supported(self, engine_args: "vllm.AsyncEngineArgs"):
        # For vLLM >= 0.11.1, v1 is the default, no need to set environment variable
        if VLLM_VERSION >= version.parse("0.11.1"):
            logger.debug(
                "vLLM >= 0.11.1 detected, v1 is default, skip setting VLLM_USE_V1"
            )
            return

        if os.getenv("VLLM_USE_V1") is not None:
            logger.debug(
                "Setting vLLM v1 via environment variable already, skip checking"
            )
            return

        try:
            supported_func = engine_args._is_v1_supported_oracle
        except AttributeError:
            logger.debug(
                "Cannot get `EngineArgs._is_v1_supported_oracle` "
                "to decide enabling vLLM v1, perhaps vllm version is too old, "
                "version: %s",
                VLLM_VERSION,
            )
            return
        model_config = engine_args.create_model_config()
        old_main_thread = threading.main_thread()
        try:
            # HACK: patch main thread to let vllm pass check
            # vllm do some signal handling when on main thread
            # but they will skip registering signal if not on main thread,
            # however, the _is_v1_supported_oracle will return False
            # when not on main thread, we patched the main thread temporially,
            # It's OK because Xinference will take care of all processes
            threading.main_thread = lambda: threading.current_thread()

            if supported_func(model_config):
                logger.debug("Setting vLLM v1 by checking model config")
                os.environ["VLLM_USE_V1"] = "1"
            else:
                logger.debug("Use vLLM v0 due to not supported config")
        finally:
            # patch back
            threading.main_thread = lambda: old_main_thread

    def _preprocess_load_gguf(self):
        # check if it is multi gguf files
        if (
            not os.path.isfile(self.model_path)
            and self.model_spec.quantization_parts
            and self.quantization in self.model_spec.quantization_parts
        ):
            raise RuntimeError(
                "vllm does not support multiple gguf files, please merge them first and "
                "provide `model_path` with merged file"
            )

        if "tokenizer" not in self._model_config:
            # find pytorch format without quantization
            family = next(
                family
                for family in BUILTIN_LLM_FAMILIES
                if family.model_name == self.model_family.model_name
            ).copy()
            non_quant_spec = next(
                spec
                for spec in family.model_specs
                if spec.quantization == "none"
                and spec.model_size_in_billions
                == self.model_spec.model_size_in_billions
                and spec.model_hub == self.model_spec.model_hub
            )
            family.model_specs = [non_quant_spec]
            path = cache_model_tokenizer_and_config(family)
            # other than gguf file, vllm requires to provide tokenizer and hf_config_path
            self._model_config["tokenizer"] = self._model_config["hf_config_path"] = (
                path
            )

        if not os.path.isfile(self.model_path):
            self.model_path = os.path.realpath(
                os.path.join(
                    self.model_path,
                    self.model_spec.model_file_name_template.format(
                        quantization=self.quantization
                    ),
                )
            )

    def stop(self):
        # though the vLLM engine will shutdown when deleted,
        # but some issue e.g. GH#1682 reported
        # when deleting, the engine exists still
        logger.info("Stopping vLLM engine")
        if self._check_health_task:
            self._check_health_task.cancel()
        if self._engine:
            if not self._is_vllm_v1():
                # v0
                if model_executor := getattr(
                    self._engine.engine, "model_executor", None
                ):
                    model_executor.shutdown()
                self._engine = None
            else:
                # v1
                self._engine.shutdown()
                self._engine = None

    async def init_xavier(self):
        await self._engine.init_xavier()

    async def _check_healthy(self, interval: int = 30):
        from vllm.engine.async_llm_engine import AsyncEngineDeadError

        logger.debug("Begin to check health of vLLM")

        while self._engine is not None:
            try:
                await self._engine.check_health()
            except (AsyncEngineDeadError, RuntimeError):
                logger.info("Detecting vLLM is not health, prepare to quit the process")
                try:
                    self.stop()
                except:  # noqa: E722
                    # ignore error when stop
                    pass
                # Just kill the process and let xinference auto-recover the model
                os._exit(1)
            else:
                await asyncio.sleep(interval)

    def parse_str_field_to_dict(
        self, field_value, field_name: str = "config_field", default: dict = {}
    ) -> dict:
        """
        Generic function: Parse a string-type configuration field to a dictionary.
        Returns an empty default dict and logs a warning if parsing fails.

        Applicable scenarios: JSON-formatted strings passed via webui
        (e.g., speculative_config, mm_processor_kwargs fields)

        Args:
            field_value: Value of the field to parse (may be str/dict/other types)
            field_name: Name of the field (used for log messages, e.g., "speculative_config")
            default: Default value returned when parsing fails, empty dict by default

        Returns:
            Parsed dictionary (returns default if parsing fails or input is non-string type)
        """
        # Non-string type: Return original value if it's a dict, otherwise return default
        if not isinstance(field_value, str):
            return field_value if isinstance(field_value, dict) else default

        # String type: Attempt JSON parsing
        try:
            parsed_dict = json.loads(field_value)
            # Ensure parsing result is a dictionary (avoid list/number etc. from JSON string)
            if isinstance(parsed_dict, dict):
                return parsed_dict
            else:
                logger.warning(
                    f"Parsed result of {field_name} is not a dictionary (type: {type(parsed_dict)}), "
                    f"using default empty dict"
                )
                return default
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse {field_name} as JSON string, using default empty dict"
            )
            return default
        except Exception as e:
            logger.warning(
                f"Unexpected error parsing {field_name}: {str(e)}, using default empty dict"
            )
            return default

    def _sanitize_model_config(
        self, model_config: Optional[VLLMModelConfig]
    ) -> VLLMModelConfig:
        if model_config is None:
            model_config = VLLMModelConfig()

        model_config.setdefault("tokenizer_mode", "auto")
        model_config.setdefault("trust_remote_code", True)
        model_config.setdefault("tensor_parallel_size", self._device_count)  # type: ignore
        model_config.setdefault("pipeline_parallel_size", self._n_worker)  # type: ignore
        if (
            self._n_worker > 1
            and VLLM_VERSION
            and VLLM_VERSION >= version.parse("0.11.0")
        ):
            # vLLM v1 requires nnodes/node_rank for multi-node world sizes.
            model_config.setdefault("nnodes", self._n_worker)  # type: ignore
            model_config.setdefault("node_rank", self._shard)  # type: ignore
            # Use mp backend to satisfy vLLM validation; executor is patched later.
            model_config.setdefault("distributed_executor_backend", "mp")
        model_config.setdefault("block_size", 16)
        model_config.setdefault("swap_space", 4)
        model_config.setdefault("gpu_memory_utilization", 0.90)
        model_config.setdefault("max_num_seqs", 256)

        if "model_quantization" in model_config:
            model_config["quantization"] = model_config.pop("model_quantization")
        else:
            model_config.setdefault("quantization", None)
        model_config.setdefault("max_model_len", None)
        model_config.setdefault("reasoning_content", False)

        if "speculative_config" in model_config:
            model_config["speculative_config"] = self.parse_str_field_to_dict(
                model_config.get("speculative_config", {}), "speculative_config"
            )
        if "rope_scaling" in model_config:
            rope_scaling = self.parse_str_field_to_dict(
                model_config["rope_scaling"], "rope_scaling"
            )
            model_config["hf_overrides"] = {"rope_scaling": rope_scaling}
            model_config.pop("rope_scaling", {})

        # Add scheduling policy if vLLM version is 0.6.3 or higher
        if VLLM_VERSION >= version.parse("0.6.3"):
            model_config.setdefault("scheduling_policy", "fcfs")
            # init mm_processor_kwargs params
            mm_processor_kwargs = self.parse_str_field_to_dict(
                model_config.get("mm_processor_kwargs", {}), "mm_processor_kwargs"
            )
            pixel_params: Dict[str, int] = {}
            if "min_pixels" in model_config:
                pixel_params["min_pixels"] = model_config.pop("min_pixels")
            if "max_pixels" in model_config:
                pixel_params["max_pixels"] = model_config.pop("max_pixels")
            if pixel_params or mm_processor_kwargs:
                model_config["mm_processor_kwargs"] = {
                    **mm_processor_kwargs,
                    **pixel_params,
                }
        return model_config

    @staticmethod
    def _sanitize_generate_config(
        generate_config: Optional[Dict] = None,
    ) -> VLLMGenerateConfig:
        if not generate_config:
            generate_config = {}

        sanitized = VLLMGenerateConfig()

        response_format = generate_config.pop("response_format", None)
        guided_json_object = None
        guided_json = None

        if response_format is not None:
            if response_format.get("type") == "json_object":
                guided_json_object = True
            elif response_format.get("type") == "json_schema":
                json_schema = response_format.get("json_schema")
                assert json_schema is not None
                guided_json = json_schema.get("json_schema")

        sanitized.setdefault("lora_name", generate_config.get("lora_name", None))
        sanitized.setdefault("n", generate_config.get("n", 1))
        sanitized.setdefault("best_of", generate_config.get("best_of", None))
        sanitized.setdefault(
            "presence_penalty", generate_config.get("presence_penalty", 0.0)
        )
        sanitized.setdefault(
            "frequency_penalty", generate_config.get("frequency_penalty", 0.0)
        )
        sanitized.setdefault("temperature", generate_config.get("temperature", 1.0))
        sanitized.setdefault("top_p", generate_config.get("top_p", 1.0))
        sanitized.setdefault("top_k", generate_config.get("top_k", -1))
        sanitized.setdefault(  # type: ignore
            "max_tokens",
            generate_config.get("max_tokens", XINFERENCE_MAX_TOKENS)  # type: ignore
            or XINFERENCE_MAX_TOKENS,
        )
        sanitized.setdefault("stop", generate_config.get("stop", None))
        sanitized.setdefault(
            "stop_token_ids", generate_config.get("stop_token_ids", None)
        )
        sanitized.setdefault("stream", generate_config.get("stream", False))
        sanitized.setdefault(
            "stream_options", generate_config.get("stream_options", None)
        )
        sanitized.setdefault(
            "skip_special_tokens", generate_config.get("skip_special_tokens", True)
        )
        sanitized.setdefault(
            "guided_json", generate_config.get("guided_json", guided_json)
        )
        sanitized.setdefault("guided_regex", generate_config.get("guided_regex", None))
        sanitized.setdefault(
            "guided_choice", generate_config.get("guided_choice", None)
        )
        sanitized.setdefault(
            "guided_grammar", generate_config.get("guided_grammar", None)
        )
        sanitized.setdefault(
            "guided_whitespace_pattern",
            generate_config.get("guided_whitespace_pattern", None),
        )
        sanitized.setdefault(
            "guided_json_object",
            generate_config.get("guided_json_object", guided_json_object),
        )
        # 1. Try to get from generate config
        ignore_eos_val = generate_config.get("ignore_eos")

        # 2. else, get from extra_body
        # sometimes Xinference put unrecognized params into extra_body
        if ignore_eos_val is None:
            extra_body = generate_config.get("extra_body")
            if isinstance(extra_body, dict):
                ignore_eos_val = extra_body.get("ignore_eos")

        # 3. write into sanitized
        sanitized.setdefault(
            "ignore_eos", ignore_eos_val if ignore_eos_val is not None else False
        )

        return sanitized

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        try:
            importlib.import_module("vllm")
        except ImportError as exc:  # includes missing shared libs such as libcudart
            return False, f"Failed to import vLLM: {exc}"
        except OSError as exc:  # native extension load errors
            return False, f"Failed to load vLLM native extension: {exc}"
        return True

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if (
            not cls._has_cuda_device()
            and not cls._has_mlu_device()
            and not cls._has_vacc_device()
            and not cls._has_musa_device()
        ):
            return False, "vLLM requires CUDA or MLU GPUs or VACC GPUs or MUSA GPUs"
        if not cls._is_linux():
            return False, "vLLM backend is only supported on Linux"
        if llm_spec.model_format not in ["pytorch", "gptq", "awq", "fp4", "fp8", "bnb"]:
            return False, "vLLM supports pytorch/gptq/awq/fp4/fp8/bnb formats only"
        if llm_spec.model_format == "pytorch":
            if quantization not in (None, "none"):
                return (
                    False,
                    "pytorch format with quantization is not supported by vLLM",
                )
        if llm_spec.model_format == "awq":
            if "4" not in quantization:
                return False, "vLLM only supports 4-bit AWQ weights"
        if llm_spec.model_format == "gptq":
            if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.3.3"):
                if not any(q in quantization for q in ("3", "4", "8")):
                    return False, "gptq quantization must be 3/4/8 bit for vLLM >=0.3.3"
            else:
                if "4" not in quantization:
                    return False, "gptq quantization must be 4 bit for vLLM <0.3.3"
        if not llm_family.matches_supported_architectures(VLLM_SUPPORTED_MODELS):
            return (
                False,
                f"Model architectures {llm_family.architectures} are not supported by vLLM",
            )
        if "generate" not in llm_family.model_ability:
            return False, "vLLM base engine requires generate ability"
        if not VLLM_INSTALLED and not _virtual_env_allows_missing_vllm():
            return False, "vLLM library is not installed"
        return True

    @staticmethod
    def _convert_request_output_to_completion_chunk(
        request_id: str, model: str, request_output: "RequestOutput"
    ) -> Tuple[CompletionChunk, Optional[str]]:
        choices: List[CompletionChoice] = []
        finish_reason = None
        for output in request_output.outputs:
            choices.append(
                CompletionChoice(
                    text=output.text,
                    index=output.index,
                    logprobs=None,  # TODO: support logprobs.
                    finish_reason=None,
                )
            )
            finish_reason = output.finish_reason
        return (
            CompletionChunk(
                id=request_id,
                object="text_completion",
                created=int(time.time()),
                model=model,
                choices=choices,
            ),
            finish_reason,
        )

    @staticmethod
    def _convert_request_output_to_completion(
        request_id: str, model: str, request_output: "RequestOutput"
    ) -> Completion:
        choices = []
        for output in request_output.outputs:
            choices.append(
                CompletionChoice(
                    text=output.text,
                    index=output.index,
                    logprobs=None,  # TODO: support logprobs.
                    finish_reason=output.finish_reason,
                )
            )

        prompt_tokens = len(request_output.prompt_token_ids)
        completion_tokens = sum(
            len(output.token_ids) for output in request_output.outputs
        )
        usage = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return Completion(
            id=request_id,
            object="text_completion",
            created=int(time.time()),
            model=model,
            choices=choices,
            usage=usage,
        )

    async def _get_tokenizer(self, lora_request: Any) -> Any:
        try:
            # vLLM 0.11.0+ get_tokenizer doesn't accept lora_request parameter
            if (
                VLLM_VERSION >= version.parse("0.11.0")
                or VLLM_VERSION.base_version >= "0.11.0"
            ):
                return await self._engine.get_tokenizer()  # type: ignore
            else:
                return await self._engine.get_tokenizer(lora_request)  # type: ignore
        except AttributeError:
            # Fallback to get_tokenizer_async for older versions
            try:
                return await self._engine.get_tokenizer_async(lora_request)  # type: ignore
            except (AttributeError, TypeError):
                # If all else fails, try without parameters
                return await self._engine.get_tokenizer()  # type: ignore

    def _tokenize(self, tokenizer: Any, prompt: str, config: dict) -> List[int]:
        truncate_prompt_tokens = config.get("truncate_prompt_tokens")
        add_special_tokens = config.get("add_special_tokens", True)

        if truncate_prompt_tokens is None:
            encoded = tokenizer(prompt, add_special_tokens=add_special_tokens)
        elif truncate_prompt_tokens < 0:
            # Negative means we cap at the model's max length
            encoded = tokenizer(
                prompt,
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=self._context_length,
            )
        else:
            encoded = tokenizer(
                prompt,
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=truncate_prompt_tokens,
            )

        return encoded.input_ids

    async def _gen_tokens_prompt(
        self, tokenizer, prompt: Union[str, dict], config: dict
    ):
        from vllm import TokensPrompt

        token_ids = await asyncio.to_thread(
            self._tokenize,
            tokenizer,
            prompt,  # type: ignore
            config,
        )
        return TokensPrompt(prompt_token_ids=token_ids)

    @vllm_check
    async def async_generate(
        self,
        prompt: Union[str, Dict[str, Any]],
        generate_config: Optional[Dict] = None,
        tools: object = False,
        request_id: Optional[str] = None,
    ) -> Union[Completion, AsyncGenerator[CompletionChunk, None]]:
        try:
            from vllm.sampling_params import SamplingParams
        except ImportError:
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        sanitized_generate_config = self._sanitize_generate_config(generate_config)
        logger.debug(
            "Enter generate, prompt: %s, generate config: %s", prompt, generate_config
        )

        lora_model = sanitized_generate_config.pop("lora_name")

        lora_request = None
        if lora_model is not None:
            for lora in self.lora_requests:
                if lora_model == lora.lora_name:
                    lora_request = lora
                    break

        stream = sanitized_generate_config.pop("stream")
        stream_options = sanitized_generate_config.pop("stream_options", None)
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )

        if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.6.3"):
            # guided decoding only available for vllm >= 0.6.3
            GuidedDecodingParams = None
            StructuredOutputsParams = None
            supports_guided = VLLM_VERSION < version.parse("1.12.0")
            try:
                import vllm.sampling_params as _sampling_params
            except ImportError:
                if supports_guided:
                    logger.info(
                        "GuidedDecodingParams not found in vLLM %s, "
                        "trying StructuredOutputsParams fallback.",
                        VLLM_VERSION,
                    )
            else:
                if supports_guided and hasattr(
                    _sampling_params, "GuidedDecodingParams"
                ):
                    GuidedDecodingParams = _sampling_params.GuidedDecodingParams
                elif supports_guided:
                    logger.info(
                        "GuidedDecodingParams not found in vLLM %s, "
                        "trying StructuredOutputsParams fallback.",
                        VLLM_VERSION,
                    )

                if hasattr(_sampling_params, "StructuredOutputsParams"):
                    StructuredOutputsParams = _sampling_params.StructuredOutputsParams
                elif GuidedDecodingParams is None:
                    logger.warning(
                        "No guided decoding support found in vLLM %s "
                        "(GuidedDecodingParams / StructuredOutputsParams).",
                        VLLM_VERSION,
                    )

            # Extract guided decoding parameters
            guided_params: dict[str, Any] = {}
            guided_json = sanitized_generate_config.pop("guided_json", None)
            if guided_json:
                guided_params["json"] = guided_json

            guided_regex = sanitized_generate_config.pop("guided_regex", None)
            if guided_regex:
                guided_params["regex"] = guided_regex

            guided_choice = sanitized_generate_config.pop("guided_choice", None)
            if guided_choice:
                guided_params["choice"] = guided_choice

            guided_grammar = sanitized_generate_config.pop("guided_grammar", None)
            if guided_grammar:
                guided_params["grammar"] = guided_grammar

            guided_json_object = sanitized_generate_config.pop(
                "guided_json_object", None
            )
            if guided_json_object:
                guided_params["json_object"] = guided_json_object

            guided_backend = sanitized_generate_config.pop(
                "guided_decoding_backend", None
            )
            if guided_backend:
                guided_params["_backend"] = guided_backend

            guided_whitespace_pattern = sanitized_generate_config.pop(
                "guided_whitespace_pattern", None
            )
            if guided_whitespace_pattern:
                guided_params["whitespace_pattern"] = guided_whitespace_pattern

            # Create GuidedDecodingParams / StructuredOutputsParams if we have any guided parameters
            guided_options = None
            if guided_params and GuidedDecodingParams:
                try:
                    guided_options = GuidedDecodingParams(**guided_params)
                except Exception as e:
                    logger.warning(f"Failed to create GuidedDecodingParams: {e}")
                    guided_options = None
            elif guided_params and StructuredOutputsParams:
                try:
                    guided_options = StructuredOutputsParams(**guided_params)
                except Exception as e:
                    logger.warning(f"Failed to create StructuredOutputsParams: {e}")
                    guided_options = None

            try:
                import inspect

                sp_sig = inspect.signature(SamplingParams)
                unsupported_keys = [
                    key
                    for key in list(sanitized_generate_config.keys())
                    if key not in sp_sig.parameters
                ]
                config_dict = cast(Dict[str, Any], sanitized_generate_config)
                for key in unsupported_keys:
                    config_dict.pop(key, None)
                if unsupported_keys:
                    logger.warning(
                        "Dropping unsupported sampling params for vLLM %s: %s",
                        VLLM_VERSION,
                        unsupported_keys,
                    )
                # For v0.9.2 and similar versions, prioritize guided_decoding over structured_outputs
                # structured_outputs was introduced later (around v0.11.0) and may not accept
                # GuidedDecodingParams in earlier versions even if the parameter exists
                if "guided_decoding" in sp_sig.parameters:
                    sampling_params = SamplingParams(
                        guided_decoding=guided_options, **sanitized_generate_config
                    )
                elif "structured_outputs" in sp_sig.parameters:
                    try:
                        sampling_params = SamplingParams(
                            structured_outputs=guided_options,
                            **sanitized_generate_config,
                        )
                    except TypeError as e:
                        if "structured_outputs" in str(e):
                            # structured_outputs parameter exists but doesn't accept GuidedDecodingParams
                            # Fall back to no guided decoding
                            logger.warning(
                                f"structured_outputs parameter failed: {e}. "
                                "Falling back to no guided decoding for vLLM version compatibility."
                            )
                            sampling_params = SamplingParams(
                                **sanitized_generate_config
                            )
                        else:
                            raise
                else:
                    sampling_params = SamplingParams(**sanitized_generate_config)
            except Exception as e:
                logger.warning(
                    f"Failed to create SamplingParams with guided decoding: {e}"
                )
                sampling_params = SamplingParams(**sanitized_generate_config)
        else:
            # ignore generate configs for older versions
            sanitized_generate_config.pop("guided_json", None)
            sanitized_generate_config.pop("guided_regex", None)
            sanitized_generate_config.pop("guided_choice", None)
            sanitized_generate_config.pop("guided_grammar", None)
            sanitized_generate_config.pop("guided_json_object", None)
            sanitized_generate_config.pop("guided_decoding_backend", None)
            sanitized_generate_config.pop("guided_whitespace_pattern", None)
            import inspect

            sp_sig = inspect.signature(SamplingParams)
            unsupported_keys = [
                key
                for key in list(sanitized_generate_config.keys())
                if key not in sp_sig.parameters
            ]
            config_dict = cast(Dict[str, Any], sanitized_generate_config)
            for key in unsupported_keys:
                config_dict.pop(key, None)
            if unsupported_keys:
                logger.warning(
                    "Dropping unsupported sampling params for vLLM %s: %s",
                    VLLM_VERSION,
                    unsupported_keys,
                )
            sampling_params = SamplingParams(**sanitized_generate_config)

        prompt_or_token_ids: Union[str, Dict[str, Any], List[int]] = prompt
        if sampling_params.max_tokens is None:
            # no max_tokens set, try to get the max tokens
            # this requires tokenizing
            tokenizer = await self._get_tokenizer(lora_request)
            prompt_or_token_ids = await self._gen_tokens_prompt(
                tokenizer,
                prompt,
                sanitized_generate_config,  # type: ignore
            )
            sampling_params.max_tokens = max_tokens = self._context_length - len(  # type: ignore
                prompt_or_token_ids["prompt_token_ids"]  # type: ignore
            )
            logger.debug("No max_tokens set, setting to: %s", max_tokens)

        if not request_id:
            request_id = str(uuid.uuid1())

        assert self._engine is not None
        start_wall_time = time.time()
        start_perf = time.perf_counter()
        logger.debug(
            "Generate start, request_id: %s, stream: %s, start_time: %s",
            request_id,
            stream,
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_wall_time)),
        )
        results_generator = self._engine.generate(
            prompt_or_token_ids,
            sampling_params,
            request_id,
            lora_request=lora_request,
        )

        async def stream_results() -> AsyncGenerator[CompletionChunk, None]:
            previous_texts = [""] * sanitized_generate_config["n"]
            prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
            complete_response = ""
            match_tool_call_tmp_results = []
            is_match_tool_call = False
            chunk = None
            finish_reason = None
            async for _request_output in results_generator:
                chunk, finish_reason = self._convert_request_output_to_completion_chunk(
                    request_id=request_id,
                    model=self.model_uid,
                    request_output=_request_output,
                )

                for i, choice in enumerate(chunk["choices"]):
                    delta = choice["text"][len(previous_texts[i]) :]
                    previous_texts[i] = choice["text"]
                    choice["text"] = delta
                    complete_response += delta

                prompt_tokens = len(_request_output.prompt_token_ids)
                completion_tokens = sum(
                    len(output.token_ids) for output in _request_output.outputs
                )
                total_tokens = prompt_tokens + completion_tokens
                chunk["usage"] = CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

                if tools:
                    """
                    The qwen2 tool call returns format like this:
                    <tool_call>
                    {...}
                    </tool_call>
                    Here is to match this.
                    """
                    if (len(QWEN_TOOL_CALL_SYMBOLS[0]) > len(complete_response)) and (
                        not QWEN_TOOL_CALL_SYMBOLS[0].startswith(complete_response)
                    ):
                        for c in match_tool_call_tmp_results:
                            yield c
                        match_tool_call_tmp_results.clear()
                        yield chunk
                    elif (len(QWEN_TOOL_CALL_SYMBOLS[0]) > len(complete_response)) and (
                        QWEN_TOOL_CALL_SYMBOLS[0].startswith(complete_response)
                    ):
                        match_tool_call_tmp_results.append(chunk)
                    else:
                        assert len(QWEN_TOOL_CALL_SYMBOLS[0]) <= len(complete_response)
                        if not is_match_tool_call and complete_response.startswith(
                            QWEN_TOOL_CALL_SYMBOLS[0]
                        ):
                            is_match_tool_call = True
                            match_tool_call_tmp_results.clear()

                        if not is_match_tool_call:
                            for c in match_tool_call_tmp_results:
                                yield c
                            match_tool_call_tmp_results.clear()
                            yield chunk
                        else:
                            chunk["choices"][0]["text"] = complete_response
                else:
                    yield chunk

            if is_match_tool_call:
                assert chunk is not None
                yield chunk

            elapsed = time.perf_counter() - start_perf
            completion_tps = (
                completion_tokens / elapsed if elapsed > 0 else completion_tokens
            )
            total_tps = total_tokens / elapsed if elapsed > 0 else total_tokens
            logger.debug(
                "Generate finished, request_id: %s, stop reason: %s, prompt tokens: %s, "
                "completion tokens: %s, all tokens: %s, elapsed: %.3fs, "
                "throughput: completion %.2f tok/s, total %.2f tok/s",
                request_id,
                finish_reason,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                elapsed,
                completion_tps,
                total_tps,
            )

            # match OpenAI API stream
            yield generate_completion_chunk(
                chunk_text="",
                finish_reason=finish_reason,
                chunk_id=request_id,
                model_uid=self.model_uid,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

            if include_usage:
                chunk = CompletionChunk(
                    id=request_id,
                    object="text_completion",
                    created=int(time.time()),
                    model=self.model_uid,
                    choices=[],
                )
                chunk["usage"] = CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
                yield chunk

        if stream:
            return stream_results()
        else:
            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            assert final_output is not None
            return self._convert_request_output_to_completion(
                request_id, model=self.model_uid, request_output=final_output
            )


class VLLMChatModel(VLLMModel, ChatModelMixin):
    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if llm_spec.model_format not in [
            "pytorch",
            "gptq",
            "awq",
            "fp4",
            "fp8",
            "bnb",
            "ggufv2",
        ]:
            return (
                False,
                "vLLM chat mode supports pytorch/gptq/awq/fp4/fp8/bnb/ggufv2 formats only",
            )
        if llm_spec.model_format == "pytorch":
            if quantization not in (None, "none"):
                return (
                    False,
                    "pytorch format with quantization is not supported in vLLM chat",
                )
        if llm_spec.model_format == "awq":
            if not any(q in quantization for q in ("4", "8")):
                return False, "awq quantization must be 4 or 8 bit for vLLM chat"
        if llm_spec.model_format == "gptq":
            if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.3.3"):
                if not any(q in quantization for q in ("3", "4", "8")):
                    return False, "gptq quantization must be 3/4/8 bit for vLLM >=0.3.3"
            else:
                if "4" not in quantization:
                    return False, "gptq quantization must be 4 bit for vLLM <0.3.3"
        if llm_spec.model_format == "ggufv2":
            if not (VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.8.2")):
                return False, "ggufv2 support requires vLLM >= 0.8.2"
        if not llm_family.matches_supported_architectures(VLLM_SUPPORTED_CHAT_MODELS):
            return (
                False,
                f"Model architectures {llm_family.architectures} are not supported by vLLM chat",
            )
        if "chat" not in llm_family.model_ability:
            return False, "vLLM chat engine requires chat ability"
        if not VLLM_INSTALLED and not _virtual_env_allows_missing_vllm():
            return False, "vLLM library is not installed"
        return True

    def _sanitize_chat_config(
        self,
        generate_config: Optional[Dict] = None,
    ) -> Dict:
        if not generate_config:
            generate_config = {}

        if "reasoning" in getattr(self.model_family, "model_ability", []):
            generate_config.pop("stop", None)
            generate_config.pop("stop_token_ids", None)
        else:
            if not generate_config.get("stop") and self.model_family.stop:
                generate_config["stop"] = self.model_family.stop.copy()
            if (
                not generate_config.get("stop_token_ids")
                and self.model_family.stop_token_ids
            ):
                generate_config["stop_token_ids"] = (
                    self.model_family.stop_token_ids.copy()
                )

        # if response_format exists，generate guided_json
        if "response_format" in generate_config:
            resp_format = generate_config["response_format"]
            if (
                isinstance(resp_format, dict)
                and resp_format.get("type") == "json_schema"
                and "json_schema" in resp_format
            ):
                schema = resp_format["json_schema"].get("schema_")
                if schema:
                    generate_config["guided_json"] = schema

        return generate_config

    @staticmethod
    def is_tool_call_chunk_start(chunk):
        return chunk["choices"][0]["text"].startswith(QWEN_TOOL_CALL_SYMBOLS[0])

    @staticmethod
    def is_tool_call_chunk_end(chunk):
        return chunk["choices"][0]["text"].endswith(QWEN_TOOL_CALL_SYMBOLS[1])

    @staticmethod
    def prefill_messages(messages: List[Dict]) -> List[Dict]:
        """
        Preprocess messages to ensure content is not None

        Args:
            messages: Original message list

        Returns:
            Processed message list, where content is not None
        """
        processed_messages = []

        for msg in messages:
            if isinstance(msg, dict):
                if msg.get("content") is None:
                    msg_copy = msg.copy()
                    msg_copy["content"] = ""  # Replace None with empty string
                    processed_messages.append(msg_copy)
                else:
                    processed_messages.append(msg)
            else:
                processed_messages.append(msg)

        return processed_messages

    @vllm_check
    async def async_chat(
        self,
        messages: List[Dict],
        generate_config: Optional[Dict] = None,
        request_id: Optional[str] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        # Preprocess messages to ensure content is not None
        messages = self.prefill_messages(messages)

        tools = generate_config.pop("tools", []) if generate_config else None
        model_family = self.model_family.model_family or self.model_family.model_name
        chat_template_kwargs = (
            self._get_chat_template_kwargs_from_generate_config(
                generate_config, self.reasoning_parser
            )
            or {}
        )
        chat_context_var.set(chat_template_kwargs)
        full_context_kwargs = chat_template_kwargs.copy()
        if tools:
            if (
                model_family in QWEN_TOOL_CALL_FAMILY
                or model_family in DEEPSEEK_TOOL_CALL_FAMILY
            ):
                full_context_kwargs["tools"] = tools
        assert self.model_family.chat_template is not None

        generate_config = self._sanitize_chat_config(generate_config)
        stream = generate_config.get("stream", None)

        lora_request = None
        lora_model = generate_config.get("lora_name")
        if lora_model is not None:
            for lora in self.lora_requests:
                if lora_model == lora.lora_name:
                    lora_request = lora
                    break
        tokenizer = await self._get_tokenizer(lora_request)

        full_prompt = self.get_full_context(
            messages,
            self.model_family.chat_template,
            tokenizer=tokenizer,
            **full_context_kwargs,
        )

        if stream:
            agen = await self.async_generate(
                full_prompt, generate_config, tools, request_id=request_id
            )
            assert isinstance(agen, AsyncGenerator)
            if tools:
                return self._async_to_tool_completion_chunks(agen, chat_template_kwargs)
            return self._async_to_chat_completion_chunks(
                agen, self.reasoning_parser, chat_template_kwargs
            )
        else:
            c = await self.async_generate(
                full_prompt, generate_config, request_id=request_id
            )
            assert not isinstance(c, AsyncGenerator)
            if tools:
                return self._post_process_completion(
                    self.model_family, self.model_uid, c
                )
            return self._to_chat_completion(c, self.reasoning_parser)


class VLLMMultiModel(VLLMModel, ChatModelMixin):
    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if (
            not cls._has_cuda_device()
            and not cls._has_mlu_device()
            and not cls._has_vacc_device()
            and not cls._has_musa_device()
        ):
            return (
                False,
                "vLLM multimodal engine requires CUDA or MLU GPUs or VACC GPUs or MUSA GPUs",
            )
        if not cls._is_linux():
            return False, "vLLM multimodal engine is only supported on Linux"
        if llm_spec.model_format not in ["pytorch", "gptq", "awq", "fp4", "fp8", "bnb"]:
            return (
                False,
                "vLLM multimodal engine supports pytorch/gptq/awq/fp4/fp8/bnb formats only",
            )
        if llm_spec.model_format == "pytorch":
            if quantization not in (None, "none"):
                return (
                    False,
                    "pytorch format with quantization is not supported for vLLM multimodal",
                )
        if llm_spec.model_format == "awq":
            if not any(q in quantization for q in ("4", "8")):
                return False, "awq quantization must be 4 or 8 bit for vLLM multimodal"
        if llm_spec.model_format == "gptq":
            if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.3.3"):
                if not any(q in quantization for q in ("3", "4", "8")):
                    return False, "gptq quantization must be 3/4/8 bit for vLLM >=0.3.3"
            else:
                if "4" not in quantization:
                    return False, "gptq quantization must be 4 bit for vLLM <0.3.3"
        if not llm_family.matches_supported_architectures(
            VLLM_SUPPORTED_MULTI_MODEL_LIST
        ):
            return (
                False,
                f"Model architectures {llm_family.architectures} are not supported by vLLM multimodal engine",
            )
        if (
            "vision" not in llm_family.model_ability
            and "audio" not in llm_family.model_ability
            and "omni" not in llm_family.model_ability
        ):
            return (
                False,
                "vLLM multimodal engine requires vision, audio, or omni ability",
            )
        if not VLLM_INSTALLED:
            return False, "vLLM library is not installed"
        return True

    @staticmethod
    def _attach_video_metadata(
        videos: List[Any], fps_list: Optional[List[Any]]
    ) -> List[Any]:
        if not fps_list:
            return videos

        attached: List[Any] = []
        for idx, video in enumerate(videos):
            fps = fps_list[idx] if idx < len(fps_list) else None
            data = video
            metadata: Dict[str, Any] = {}
            if (
                isinstance(video, tuple)
                and len(video) == 2
                and isinstance(video[1], dict)
            ):
                data = video[0]
                metadata = dict(video[1])
            if fps is not None:
                metadata.setdefault("fps", fps)
                metadata.setdefault("video_fps", fps)
            attached.append((data, metadata) if metadata else data)
        return attached

    def _sanitize_model_config(
        self, model_config: Optional[VLLMModelConfig]
    ) -> VLLMModelConfig:
        model_config = super()._sanitize_model_config(model_config)
        if VLLM_VERSION >= version.parse("0.5.5"):
            raw_limit = model_config.get("limit_mm_per_prompt")
            if raw_limit:
                parsed_limit: Dict[str, int]
                if isinstance(raw_limit, dict):
                    parsed_limit = raw_limit
                else:
                    try:
                        if isinstance(raw_limit, list):
                            # Web UI may split the JSON string into multiple list items.
                            raw_value = ",".join(
                                str(item).strip() for item in raw_limit
                            )
                        else:
                            raw_value = str(raw_limit)
                        parsed_limit = json.loads(raw_value)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "Failed to parse limit_mm_per_prompt %r, fallback to default: %s",
                            raw_limit,
                            e,
                        )
                        parsed_limit = {}
                model_config["limit_mm_per_prompt"] = parsed_limit
            if not model_config.get("limit_mm_per_prompt"):
                if "omni" in self.model_family.model_ability:
                    model_config["limit_mm_per_prompt"] = {
                        "image": 2,
                        "video": 2,
                        "audio": 2,
                    }
                elif "vision" in self.model_family.model_ability:
                    model_config["limit_mm_per_prompt"] = {"image": 2, "video": 2}
                elif "audio" in self.model_family.model_ability:
                    model_config["limit_mm_per_prompt"] = {"audio": 2}
        return model_config

    def _sanitize_chat_config(
        self,
        generate_config: Optional[Dict] = None,
    ) -> Dict:
        from ..utils import get_stop_token_ids_from_config_file

        if not generate_config:
            generate_config = {}
        if generate_config.get("stop_token_ids", None) is None:
            stop_token_ids = get_stop_token_ids_from_config_file(self.model_path)
            if stop_token_ids is not None:
                generate_config.setdefault("stop_token_ids", stop_token_ids)
            else:
                if self.model_family.stop_token_ids:
                    generate_config.setdefault(
                        "stop_token_ids", self.model_family.stop_token_ids.copy()
                    )
        return generate_config

    async def _gen_tokens_prompt(
        self, tokenizer, prompt: Union[str, dict], config: dict
    ):
        from vllm import TokensPrompt

        if isinstance(prompt, str):
            return super()._gen_tokens_prompt(tokenizer, prompt, config)

        prompt_str = prompt["prompt"]
        multi_modal_data = prompt.get("multi_modal_data")

        token_ids = await asyncio.to_thread(
            self._tokenize,
            tokenizer,
            prompt_str,
            config,  # type: ignore
        )
        return TokensPrompt(
            prompt_token_ids=token_ids, multi_modal_data=multi_modal_data
        )

    @vllm_check
    async def async_chat(
        self,
        messages: List[ChatCompletionMessage],  # type: ignore
        generate_config: Optional[Dict] = None,
        request_id: Optional[str] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        tools = generate_config.pop("tools", []) if generate_config else None

        model_family = self.model_family.model_family or self.model_family.model_name
        audios, images, videos, video_kwargs = None, None, None, None
        if "internvl" not in model_family.lower():
            from qwen_omni_utils import (
                process_audio_info,
                process_mm_info,
                process_vision_info,
            )

            messages = self._transform_messages(messages)

            chat_template_kwargs = (
                self._get_chat_template_kwargs_from_generate_config(
                    generate_config, self.reasoning_parser
                )
                or {}
            )
            chat_context_var.set(chat_template_kwargs)
            full_context_kwargs = chat_template_kwargs.copy()
            if tools and model_family in QWEN_TOOL_CALL_FAMILY:
                full_context_kwargs["tools"] = tools
            assert self.model_family.chat_template is not None
            if "omni" in self.model_family.model_ability:
                audios, images, videos, video_kwargs = process_mm_info(
                    messages, use_audio_in_video=True, return_video_kwargs=True
                )
            elif "audio" in self.model_family.model_ability:
                audios = process_audio_info(messages, use_audio_in_video=False)
            elif "vision" in self.model_family.model_ability:
                images, videos, video_kwargs = process_vision_info(  # type: ignore
                    messages, return_video_kwargs=True
                )

            prompt = self.get_full_context(
                messages, self.model_family.chat_template, **full_context_kwargs
            )

        else:
            prompt, images = self.get_specific_prompt(model_family, messages)
        inputs = {"prompt": prompt, "multi_modal_data": {}, "mm_processor_kwargs": {}}
        if images:
            inputs["multi_modal_data"]["image"] = images
        if videos:
            fps_list = None
            if isinstance(video_kwargs, dict):
                fps_list = video_kwargs.get("fps")
            videos = self._attach_video_metadata(videos, fps_list)
            if fps_list:
                inputs["mm_processor_kwargs"]["video_fps"] = fps_list
            inputs["multi_modal_data"]["video"] = videos
        if audios:
            inputs["multi_modal_data"]["audio"] = audios
        if "omni" in self.model_family.model_ability:
            inputs["mm_processor_kwargs"]["use_audio_in_video"] = True
        if inputs["multi_modal_data"] == {}:
            inputs.pop("multi_modal_data")
        if inputs["mm_processor_kwargs"] == {}:
            inputs.pop("mm_processor_kwargs")
        generate_config = self._sanitize_chat_config(generate_config)

        stream = generate_config.get("stream", None)

        if stream:
            agen = await self.async_generate(
                inputs, generate_config, request_id=request_id
            )
            assert isinstance(agen, AsyncGenerator)
            return self._async_to_chat_completion_chunks(agen)
        else:
            c = await self.async_generate(
                inputs, generate_config, request_id=request_id
            )
            assert not isinstance(c, AsyncGenerator)
            return self._to_chat_completion(c)
