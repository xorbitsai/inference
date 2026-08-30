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
import importlib
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
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from typing_extensions import NotRequired
from xoscar.utils import get_next_port

from ....constants import XINFERENCE_MAX_TOKENS, XINFERENCE_TRUST_REMOTE_CODE
from ....device_utils import is_npu_available, is_vacc_available
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionLogprobs,
    CompletionUsage,
    LoRA,
)
from .. import BUILTIN_LLM_FAMILIES, LLM, LLMFamilyV2, LLMSpecV1
from ..core import chat_context_var, get_model_speculative_tokens_default
from ..llm_family import cache_model_tokenizer_and_config
from ..utils import (
    DEEPSEEK_TOOL_CALL_FAMILY,
    GEMMA_TOOL_CALL_FAMILY,
    GLM5_TOOL_CALL_FAMILY,
    KIMI_K3_TOOL_CALL_FAMILY,
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
    xinference_vllm_executor_backend: str
    tokenizer_mode: Optional[str]
    trust_remote_code: bool
    tensor_parallel_size: int
    pipeline_parallel_size: int
    nnodes: int
    node_rank: int
    distributed_executor_backend: str
    block_size: int
    swap_space: NotRequired[int]  # GiB
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
    # engine-neutral speculative decoding options, translated into
    # speculative_config and never forwarded to the engine as-is
    draft_model_path: NotRequired[Optional[str]]
    num_speculative_tokens: NotRequired[Optional[int]]


class VLLMGenerateConfig(TypedDict, total=False):
    lora_name: Optional[str]
    n: int
    best_of: Optional[int]
    seed: Optional[int]
    presence_penalty: float
    frequency_penalty: float
    repetition_penalty: float
    logprobs: Optional[int]
    prompt_logprobs: Optional[int]
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


def _get_transformers_version() -> Optional[version.Version]:
    try:
        import transformers
    except ImportError:
        return None
    return version.parse(transformers.__version__)


DEFAULT_VLLM_VERSION = version.parse("0.21.0")


def _get_effective_vllm_version() -> version.Version:
    try:
        from ....constants import XINFERENCE_ENABLE_VIRTUAL_ENV
    except Exception:
        XINFERENCE_ENABLE_VIRTUAL_ENV = False
    if XINFERENCE_ENABLE_VIRTUAL_ENV:
        return DEFAULT_VLLM_VERSION
    elif VLLM_VERSION is not None:
        return VLLM_VERSION
    return version.parse("0.0.0")


def _virtual_env_allows_missing_vllm() -> bool:
    # Delegate to the shared helper so engine discovery honors the effective
    # request-level virtualenv flag (via virtualenv_discovery_var) instead of
    # only the process-global default; falls back to the global constant when
    # called outside a discovery scope (e.g. the launch path).
    try:
        from ...utils import virtual_env_allows_missing_engine
    except Exception:
        try:
            from ....constants import XINFERENCE_ENABLE_VIRTUAL_ENV
        except Exception:
            return False
        return bool(XINFERENCE_ENABLE_VIRTUAL_ENV)
    return virtual_env_allows_missing_engine()


def _get_virtualenv_vllm_version(
    llm_family: "LLMFamilyV2",
) -> Optional[version.Version]:
    """Return the vLLM lower bound declared by a model virtualenv."""
    virtualenv = getattr(llm_family, "virtualenv", None)
    packages = getattr(virtualenv, "packages", None) or []
    for package in packages:
        requirement_text = package.split(";", 1)[0].strip()
        try:
            requirement = Requirement(requirement_text)
        except InvalidRequirement:
            continue
        if canonicalize_name(requirement.name) != "vllm":
            continue

        lower_bounds: List[version.Version] = []
        for specifier in requirement.specifier:
            if specifier.operator not in {">=", ">", "~=", "=="}:
                continue
            if "*" in specifier.version:
                continue
            try:
                lower_bounds.append(version.parse(specifier.version))
            except version.InvalidVersion:
                continue
        if lower_bounds:
            return max(lower_bounds)
    return None


def _get_effective_vllm_version_for_family(
    llm_family: "LLMFamilyV2",
) -> version.Version:
    if _virtual_env_allows_missing_vllm():
        declared_version = _get_virtualenv_vllm_version(llm_family)
        if declared_version is not None:
            return declared_version
        return DEFAULT_VLLM_VERSION
    if VLLM_VERSION is not None:
        return VLLM_VERSION
    return version.parse("0.0.0")


GuidedDecodingParams: Optional[Type[Any]] = None
StructuredOutputsParams: Optional[Type[Any]] = None


def _init_guided_decoding_classes() -> None:
    # Also re-invoked from VLLMModel after VLLM_VERSION is reassigned at runtime.
    # Detect into locals and publish at the end so a concurrent async_generate
    # coroutine (paused at an await) never observes a transient None.
    global GuidedDecodingParams, StructuredOutputsParams
    if not (
        VLLM_INSTALLED
        and VLLM_VERSION is not None
        and VLLM_VERSION >= version.parse("0.6.3")
    ):
        return
    supports_guided = VLLM_VERSION < version.parse("1.12.0")
    try:
        import vllm.sampling_params as _sampling_params
    except ImportError:
        if supports_guided:
            logger.debug(
                "GuidedDecodingParams not found in vLLM %s, "
                "trying StructuredOutputsParams fallback.",
                VLLM_VERSION,
            )
        return

    local_guided: Optional[Type[Any]] = None
    local_structured: Optional[Type[Any]] = None

    if supports_guided and hasattr(_sampling_params, "GuidedDecodingParams"):
        local_guided = _sampling_params.GuidedDecodingParams
    elif supports_guided:
        logger.debug(
            "GuidedDecodingParams not found in vLLM %s, "
            "trying StructuredOutputsParams fallback.",
            VLLM_VERSION,
        )

    if hasattr(_sampling_params, "StructuredOutputsParams"):
        local_structured = _sampling_params.StructuredOutputsParams
    elif local_guided is None:
        logger.warning(
            "No guided decoding support found in vLLM %s "
            "(GuidedDecodingParams / StructuredOutputsParams).",
            VLLM_VERSION,
        )

    GuidedDecodingParams = local_guided
    StructuredOutputsParams = local_structured


_init_guided_decoding_classes()


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
    "Qwen3_5MoeForCausalLM",
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

    if effective_version >= version.parse("0.15.0"):
        _append_unique(
            VLLM_SUPPORTED_MULTI_MODEL_LIST, "KimiK25ForConditionalGeneration"
        )
        _append_unique(VLLM_SUPPORTED_CHAT_MODELS, "Glm4MoeLiteForCausalLM")

    if effective_version >= version.parse("0.16.0"):
        _append_unique(VLLM_SUPPORTED_CHAT_MODELS, "GlmMoeDsaForCausalLM")

    if effective_version > version.parse("0.16.0"):
        _append_unique(
            VLLM_SUPPORTED_MULTI_MODEL_LIST,
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        )

    if effective_version >= version.parse("0.19.0"):
        _append_unique(
            VLLM_SUPPORTED_MULTI_MODEL_LIST, "Gemma4ForConditionalGeneration"
        )

    if effective_version >= version.parse("0.20.1"):
        _append_unique(VLLM_SUPPORTED_CHAT_MODELS, "DeepseekV4ForCausalLM")

    if effective_version >= version.parse("0.22.0"):
        _append_unique(
            VLLM_SUPPORTED_MULTI_MODEL_LIST, "MiniCPMV4_6ForConditionalGeneration"
        )
        _append_unique(
            VLLM_SUPPORTED_CHAT_MODELS,
            "HunYuanDenseV1ForCausalLM",
            "HYV3ForCausalLM",
        )

    if is_npu_available() and effective_version >= version.parse("0.18.0"):
        _append_unique(VLLM_SUPPORTED_CHAT_MODELS, "DeepseekV4ForCausalLM")

    if effective_version >= version.parse("0.24.0"):
        _append_unique(
            VLLM_SUPPORTED_MULTI_MODEL_LIST, "MiniMaxM3SparseForConditionalGeneration"
        )

    if effective_version >= version.parse("0.27.0"):
        _append_unique(
            VLLM_SUPPORTED_MULTI_MODEL_LIST, "KimiK3ForConditionalGeneration"
        )


_update_vllm_supported_lists()


class VLLMModel(LLM):
    allow_batch = True
    support_draft_model = True

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
        self._xinference_vllm_executor_backend = model_config.pop(  # type: ignore
            "xinference_vllm_executor_backend", None
        )
        self._address = model_config.pop("address", None)  # type: ignore
        self._n_worker = model_config.pop("n_worker", 1)  # type: ignore
        self._shard = model_config.pop("shard", 0)  # type: ignore
        self._driver_info = model_config.pop("driver_info", None)  # type: ignore
        self._loading_thread: Optional[threading.Thread] = None
        self._loading_error = None
        self._check_health_task = None
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

    def _get_xinference_executor_backend(self) -> str:
        backend = self._xinference_vllm_executor_backend
        if backend is None:
            backend = os.getenv("XINFERENCE_VLLM_EXECUTOR_BACKEND", "auto")
        backend = str(backend).strip().lower()
        if backend not in {"auto", "native_mp", "xoscar"}:
            raise ValueError(
                "Xinference vLLM executor backend must be one of "
                "'auto', 'native_mp', or 'xoscar'; "
                f"got {backend!r}"
            )
        return backend

    def _get_allocated_device_count(self) -> int:
        accelerators = getattr(self.model_family, "accelerators", None)
        if accelerators:
            return len(accelerators)

        if self._device_count is not None:
            return self._device_count

        configured_tp = (self._model_config or {}).get("tensor_parallel_size")
        configured_pp = (self._model_config or {}).get("pipeline_parallel_size")
        return int(configured_tp or 1) * int(configured_pp or 1)

    def _is_qwen4_exp(self) -> bool:
        return self.model_family.has_architecture("Qwen4ExpForConditionalGeneration")

    def _native_mp_route(self) -> Tuple[bool, str]:
        backend = self._get_xinference_executor_backend()
        if backend == "xoscar":
            return False, "explicit xoscar backend"

        if self._n_worker != 1:
            if backend == "native_mp":
                raise ValueError(
                    "native_mp only supports n_worker=1; "
                    "use xoscar for multi-worker deployment"
                )
            return False, "n_worker is not 1"

        if self._xavier_config is not None:
            if backend == "native_mp":
                raise ValueError("native_mp is not supported with Xavier")
            return False, "Xavier is enabled"

        device_count = self._get_allocated_device_count()
        if device_count <= 1:
            if backend == "native_mp":
                raise ValueError("native_mp requires more than one allocated GPU")
            return False, "allocated GPU count is not greater than 1"

        if VLLM_VERSION is None:
            if backend == "native_mp":
                raise ValueError(
                    "Cannot select native_mp because vLLM version is unavailable "
                    "in the model environment"
                )
            return False, "vLLM version is unavailable"

        if not self._is_vllm_v1():
            if backend == "native_mp":
                raise ValueError("native_mp requires vLLM V1")
            return False, "vLLM V1 is not enabled"

        min_version = version.parse("0.28.0")
        if VLLM_VERSION < min_version:
            if backend == "native_mp":
                raise ValueError("native_mp requires vLLM >= " f"{min_version}")
            return False, f"vLLM version is lower than {min_version}"

        if backend == "native_mp":
            self._get_native_mp_parallelism()
            return True, "explicit native_mp backend"
        if self._is_qwen4_exp():
            self._get_native_mp_parallelism()
            return True, "Qwen4Exp single-worker multi-GPU auto route"
        return False, "auto route is limited to Qwen4Exp architecture"

    def _use_native_mp_executor(self) -> bool:
        use_native_mp, _ = self._native_mp_route()
        return use_native_mp

    def _get_native_mp_parallelism(self) -> Tuple[int, int]:
        device_count = self._get_allocated_device_count()
        model_config = self._model_config or {}
        tensor_parallel_size = int(
            model_config.get("tensor_parallel_size", device_count)
        )
        pipeline_parallel_size = int(model_config.get("pipeline_parallel_size", 1))
        if tensor_parallel_size <= 0 or pipeline_parallel_size <= 0:
            raise ValueError(
                "native_mp tensor_parallel_size and pipeline_parallel_size "
                "must both be positive integers"
            )
        if tensor_parallel_size * pipeline_parallel_size != device_count:
            raise ValueError(
                "native_mp requires tensor_parallel_size * "
                "pipeline_parallel_size to equal the allocated GPU count; "
                f"got TP={tensor_parallel_size}, PP={pipeline_parallel_size}, "
                f"allocated_gpu_count={device_count}"
            )
        return tensor_parallel_size, pipeline_parallel_size

    @property
    def need_create_pools(self) -> bool:
        return not self._use_native_mp_executor()

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
        if VLLM_VERSION is not None and VLLM_VERSION > version.parse("0.11.0"):
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
        _init_guided_decoding_classes()
        # XINFERENCE_MODEL_UID is injected via the env= dict in
        # xinference.core.worker.WorkerActor._create_subpool so the sub-pool
        # and its vLLM descendants (EngineCore / GPU workers) inherit it. Do
        # NOT set os.environ here: load() runs in a worker thread under
        # concurrent launches, and os.environ.__setitem__ -> putenv is not
        # thread-safe; moreover the sub-pool was already forked by the time
        # this runs, so a main-process env mutation would never reach it.
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
        elif VLLM_VERSION and VLLM_VERSION >= version.parse("0.14.0"):
            self.lora_requests = [
                LoRARequest(
                    lora_name=lora.lora_name,
                    lora_int_id=i,
                    lora_path=lora.local_path,
                )
                for i, lora in enumerate(self.lora_modules, start=1)
            ]
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

        use_native_mp, native_mp_reason = self._native_mp_route()
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
        elif use_native_mp:
            tensor_parallel_size, pipeline_parallel_size = (
                self._get_native_mp_parallelism()
            )
            configured_backend = self._model_config.get("distributed_executor_backend")
            if configured_backend not in (None, "mp"):
                logger.warning(
                    "Overriding distributed_executor_backend=%r with 'mp' for "
                    "Xinference native_mp route of model %s",
                    configured_backend,
                    self.model_uid,
                )
            self._model_config["tensor_parallel_size"] = tensor_parallel_size
            self._model_config["pipeline_parallel_size"] = pipeline_parallel_size
            self._model_config["distributed_executor_backend"] = "mp"
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                enable_lora=enable_lora,
                max_loras=max_loras,
                **self._model_config,
            )
            self._enable_v1_if_supported(engine_args)

            def _load_native_mp():
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
                architectures = getattr(
                    self.model_family, "_resolve_architectures", lambda: None
                )()
                logger.info(
                    "vLLM executor route: model_uid=%s, model_name=%s, "
                    "requested_backend=%s, selected_backend=native_mp_local, "
                    "reason=%s, architecture=%s, n_worker=%s, "
                    "allocated_device_count=%s, vllm_version=%s, TP=%s, PP=%s, "
                    "distributed_executor_backend=mp, multiproc_method=spawn, "
                    "CUDA_VISIBLE_DEVICES=%s",
                    self.model_uid,
                    getattr(self.model_family, "model_name", None),
                    self._get_xinference_executor_backend(),
                    native_mp_reason,
                    architectures,
                    self._n_worker,
                    self._get_allocated_device_count(),
                    VLLM_VERSION,
                    tensor_parallel_size,
                    pipeline_parallel_size,
                    os.environ.get("CUDA_VISIBLE_DEVICES"),
                )
                try:
                    self._engine = AsyncLLMEngine.from_engine_args(engine_args)
                except Exception:
                    logger.exception("Creating vllm native mp engine failed")
                    self._loading_error = sys.exc_info()

            self._loading_thread = threading.Thread(target=_load_native_mp)
            self._loading_thread.start()
            self._loading_thread.join(1)
        elif self._n_worker > 1 or (
            self._device_count > 1
            and VLLM_VERSION is not None
            and VLLM_VERSION >= version.parse("0.7.0")
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
                            if VLLM_VERSION >= version.parse("0.19.0"):
                                executor_cls.supports_async_scheduling = lambda: True  # type: ignore
                            # patch vllm Executor.get_class
                            Executor.get_class = lambda vllm_config: executor_cls
                            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
                except Exception:
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

            def _load():
                # Force spawn to avoid fork deadlock: vLLM v1 creates
                # EngineCore via fork, which inherits parent's multi-thread
                # lock state causing deadlock. spawn creates a clean process.
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
                try:
                    self._engine = AsyncLLMEngine.from_engine_args(engine_args)
                except Exception:
                    logger.exception("Creating vllm engine failed")
                    self._loading_error = sys.exc_info()

            self._loading_thread = threading.Thread(target=_load)
            self._loading_thread.start()
            self._loading_thread.join(1)

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

        # Create health check here after engine is fully ready.
        # Previously in load(), but self._engine was None after
        # _loading_thread.join(1) for threaded paths (multi-GPU
        # and single-GPU), causing health check to be silently
        # skipped. This fix applies to ALL vLLM models that use
        # _loading_thread (both multi-GPU and single-GPU).
        # Use call_soon_threadsafe + create_task instead of
        # run_coroutine_threadsafe: the latter wraps the coroutine
        # in a concurrent.futures.Future whose exceptions are
        # silently swallowed if nobody checks the Future. create_task
        # produces an asyncio.Task whose unhandled exceptions are
        # logged by the asyncio default exception handler.
        self._check_health_task = None
        if self._engine and hasattr(self._engine, "check_health") and self._loop:
            logger.info(
                "Creating vLLM health check task for model %s",
                self.model_uid,
            )

            def _start_health_check():
                if self._engine is not None:
                    self._check_health_task = self._loop.create_task(
                        self._check_healthy()
                    )

            self._loop.call_soon_threadsafe(_start_health_check)

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
        # Wait for loading thread to finish so EngineCore subprocess
        # can be properly shut down below.
        if self._loading_thread and self._loading_thread.is_alive():
            self._loading_thread.join(timeout=30)
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
        logger.info("Begin to check health of vLLM")

        while self._engine is not None:
            try:
                await self._engine.check_health()
            except RuntimeError:
                logger.info("Detecting vLLM is not health, prepare to quit the process")
                try:
                    self.stop()
                except Exception:
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

    # vLLM grew a dedicated Gemma 4 MTP path in 0.22.0; before that it treats an
    # assistant checkpoint as a generic draft model and fails to initialize
    # against a multimodal target.
    MTP_MIN_VLLM_VERSION = version.parse("0.22.0")
    # Gemma4AssistantConfig first shipped in Transformers 5.8.0.
    MTP_MIN_TRANSFORMERS_VERSION = version.parse("5.8.0")
    # Generic fallback for MTP families without a model-specific recipe.
    DEFAULT_NUM_SPECULATIVE_TOKENS = 1

    def _default_num_speculative_tokens(self) -> int:
        return get_model_speculative_tokens_default(
            getattr(self.model_family, "model_name", None),
            getattr(self.model_spec, "model_size_in_billions", None),
            self.DEFAULT_NUM_SPECULATIVE_TOKENS,
        )

    def _apply_draft_model(self, model_config: VLLMModelConfig) -> None:
        """Turn a downloaded drafter into vLLM's ``speculative_config``.

        ``draft_model_path`` / ``num_speculative_tokens`` are the engine-neutral
        launch options; they must never reach ``AsyncEngineArgs``, so they are
        consumed here whether or not they end up being used.
        """
        draft_model_path = model_config.pop("draft_model_path", None)  # type: ignore[typeddict-item]
        num_speculative_tokens = model_config.pop("num_speculative_tokens", None)  # type: ignore[typeddict-item]
        if not draft_model_path:
            return

        if model_config.get("speculative_config"):
            # An explicit speculative_config wins: the user is driving vLLM
            # directly and may be running a different method entirely.
            logger.info(
                "Ignoring the drafter of %s, speculative_config was set explicitly",
                self.model_uid,
            )
            return

        if VLLM_VERSION is not None and VLLM_VERSION < self.MTP_MIN_VLLM_VERSION:
            raise ValueError(
                f"Speculative decoding with a Gemma 4 style drafter needs "
                f"vllm>={self.MTP_MIN_VLLM_VERSION}, but {VLLM_VERSION} is installed. "
                f"Upgrade vLLM, or launch without `enable_mtp`."
            )
        transformers_version = _get_transformers_version()
        if transformers_version is None or (
            transformers_version < self.MTP_MIN_TRANSFORMERS_VERSION
        ):
            installed = (
                str(transformers_version)
                if transformers_version is not None
                else "not installed"
            )
            raise ValueError(
                "Speculative decoding with a Gemma 4 style drafter needs "
                f"transformers>={self.MTP_MIN_TRANSFORMERS_VERSION}, but "
                f"{installed} is installed. Upgrade Transformers, or launch "
                "without `enable_mtp`."
            )

        from ..core import parse_num_speculative_tokens

        requested = parse_num_speculative_tokens(num_speculative_tokens)
        model_config["speculative_config"] = {
            "method": "mtp",
            "model": draft_model_path,
            "num_speculative_tokens": (
                requested
                if requested is not None
                else self._default_num_speculative_tokens()
            ),
        }
        logger.info(
            "Speculative decoding enabled for %s: %s",
            self.model_uid,
            model_config["speculative_config"],
        )

    def _sanitize_model_config(
        self, model_config: Optional[VLLMModelConfig]
    ) -> VLLMModelConfig:
        if model_config is None:
            model_config = VLLMModelConfig()

        architectures = getattr(self.model_family, "architectures", []) or []
        if "DeepseekV32ForCausalLM" in architectures:
            model_config.setdefault("tokenizer_mode", "deepseek_v32")
        else:
            model_config.setdefault("tokenizer_mode", "auto")
        # Respect the XINFERENCE_TRUST_REMOTE_CODE setting.
        model_config["trust_remote_code"] = (
            bool(model_config.get("trust_remote_code", XINFERENCE_TRUST_REMOTE_CODE))
            and XINFERENCE_TRUST_REMOTE_CODE
        )
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
            # vLLM's init_distributed_environment overrides distributed_init_method
            # with parallel_config.master_addr/master_port when nnodes > 1.
            # We must set them to avoid falling back to the defaults
            # ("127.0.0.1" and 29501).
            if self._address and ":" in self._address:
                master_addr = self._address.split(":", 1)[0]
            else:
                master_addr = self._address
            model_config.setdefault("master_addr", master_addr)  # type: ignore
            model_config.setdefault("master_port", get_next_port())  # type: ignore
        is_deepseek_v4 = "DeepseekV4ForCausalLM" in architectures
        if is_deepseek_v4:
            default_block_size = 128 if is_npu_available() else 256
        else:
            default_block_size = 16
        model_config.setdefault("block_size", default_block_size)
        if VLLM_VERSION < version.parse("0.18.0"):
            model_config.setdefault("swap_space", 4)
        model_config.setdefault("gpu_memory_utilization", 0.90)
        model_config.setdefault("max_num_seqs", 256)

        if "model_quantization" in model_config:
            model_config["quantization"] = model_config.pop("model_quantization")

        if self.model_spec.model_format == "fp8":
            if model_config.get("quantization") in (None, "none"):
                model_config["quantization"] = "fp8"
        else:
            model_config.setdefault("quantization", None)
        model_config.setdefault("max_model_len", None)
        model_config.setdefault("reasoning_content", False)

        config_dict_list = [
            "additional_config",
            "compilation_config",
            "model_loader_extra_config",
        ]
        for field in config_dict_list:
            if field in model_config:
                model_config[field] = self.parse_str_field_to_dict(  # type: ignore
                    model_config.get(field, {}), field
                )
        if "speculative_config" in model_config:
            model_config["speculative_config"] = self.parse_str_field_to_dict(
                model_config.get("speculative_config", {}), "speculative_config"
            )
        self._apply_draft_model(model_config)
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
                # Real serialized key is the field name `schema_` (the model
                # aliases the reserved `schema`); fall back to `schema` for raw
                # passthrough. Check `is None` rather than truthiness so a valid
                # empty schema ({}) is not dropped.
                guided_json = json_schema.get("schema_")
                if guided_json is None:
                    guided_json = json_schema.get("schema")

        sanitized.setdefault("lora_name", generate_config.get("lora_name", None))
        sanitized.setdefault("n", generate_config.get("n", 1))
        if VLLM_VERSION < version.parse("0.21.0"):
            sanitized.setdefault("best_of", generate_config.get("best_of", None))
        sanitized.setdefault("seed", generate_config.get("seed", None))
        sanitized.setdefault(
            "presence_penalty", generate_config.get("presence_penalty", 0.0)
        )
        sanitized.setdefault(
            "frequency_penalty", generate_config.get("frequency_penalty", 0.0)
        )
        sanitized.setdefault(
            "repetition_penalty", generate_config.get("repetition_penalty", 1.0)
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
        # Legacy completions use an integer ``logprobs`` value directly. Chat
        # completions use ``logprobs`` as a boolean and put the requested count
        # in ``top_logprobs``. vLLM uses None, not 0, to disable logprobs.
        logprobs_req = generate_config.get("logprobs")
        if isinstance(logprobs_req, bool):
            top_logprobs_req = generate_config.get("top_logprobs")
            vllm_logprobs = max(int(top_logprobs_req or 0), 0) if logprobs_req else None
        else:
            vllm_logprobs = logprobs_req
        sanitized.setdefault("logprobs", vllm_logprobs)
        sanitized.setdefault(
            "prompt_logprobs", generate_config.get("prompt_logprobs", None)
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
    def _build_logprobs(
        output: "RequestOutput", prompt_offset: int = 0
    ) -> Optional[CompletionLogprobs]:
        """Build a legacy-completions ``CompletionLogprobs`` from vLLM output.

        Returns ``None`` when the engine did not produce logprobs (i.e. the
        caller did not request them via ``generate_config``), preserving the
        previous behaviour. When logprobs are present, they are mapped to the
        ``text_offset`` / ``tokens`` / ``token_logprobs`` / ``top_logprobs``
        shape defined by ``CompletionLogprobs``.
        """
        output_logprobs = getattr(output, "logprobs", None)
        if not output_logprobs:
            return None
        token_ids = getattr(output, "token_ids", []) or []
        tokens: List[str] = []
        token_logprobs: List[Optional[float]] = []
        top_logprobs: List[Optional[Dict[str, float]]] = []
        text_offset: List[int] = []
        offset = prompt_offset
        for i, token_id in enumerate(token_ids):
            lp_dict = output_logprobs[i] if i < len(output_logprobs) else None
            if not lp_dict:
                tokens.append("")
                token_logprobs.append(None)
                top_logprobs.append(None)
                text_offset.append(offset)
                continue
            sampled = lp_dict.get(token_id)
            sampled_decoded = (
                getattr(sampled, "decoded_token", None) if sampled is not None else None
            )
            token_text = sampled_decoded if sampled_decoded else ""
            raw_token_lp = (
                getattr(sampled, "logprob", None) if sampled is not None else None
            )
            token_lp = (
                max(float(raw_token_lp), -9999.0) if raw_token_lp is not None else None
            )
            tokens.append(token_text)
            token_logprobs.append(token_lp)
            decoded_logprobs: Dict[str, float] = {}
            for lp in lp_dict.values():
                decoded_token = getattr(lp, "decoded_token", None)
                logprob = getattr(lp, "logprob", None)
                if decoded_token is not None and logprob is not None:
                    decoded_logprobs[decoded_token] = max(float(logprob), -9999.0)
            top_logprobs.append(decoded_logprobs)
            text_offset.append(offset)
            if token_text:
                offset += len(token_text)
        return CompletionLogprobs(
            text_offset=text_offset,
            token_logprobs=token_logprobs,
            tokens=tokens,
            top_logprobs=top_logprobs,
        )

    @staticmethod
    def _slice_logprobs(logprobs: CompletionLogprobs, start: int) -> CompletionLogprobs:
        """Return the newly generated portion of cumulative vLLM logprobs."""
        return CompletionLogprobs(
            text_offset=logprobs["text_offset"][start:],
            token_logprobs=logprobs["token_logprobs"][start:],
            tokens=logprobs["tokens"][start:],
            top_logprobs=logprobs["top_logprobs"][start:],
        )

    @staticmethod
    def _convert_request_output_to_completion_chunk(
        request_id: str, model: str, request_output: "RequestOutput"
    ) -> Tuple[CompletionChunk, Optional[str]]:
        choices: List[CompletionChoice] = []
        finish_reason = None
        prompt = getattr(request_output, "prompt", None)
        prompt_offset = len(prompt) if isinstance(prompt, str) else 0
        for output in request_output.outputs:
            choices.append(
                CompletionChoice(
                    text=output.text,
                    index=output.index,
                    logprobs=VLLMModel._build_logprobs(output, prompt_offset),
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
        prompt = getattr(request_output, "prompt", None)
        prompt_offset = len(prompt) if isinstance(prompt, str) else 0
        for output in request_output.outputs:
            choices.append(
                CompletionChoice(
                    text=output.text,
                    index=output.index,
                    logprobs=VLLMModel._build_logprobs(output, prompt_offset),
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
        import inspect

        try:
            # vLLM 0.11.0+ get_tokenizer doesn't accept lora_request parameter
            if (
                VLLM_VERSION >= version.parse("0.11.0")
                or VLLM_VERSION.base_version >= "0.11.0"
            ):
                result = self._engine.get_tokenizer()  # type: ignore
                # In vLLM v1 (>= 0.15.0), get_tokenizer may return tokenizer directly
                # instead of a coroutine. Check if we need to await.
                if inspect.iscoroutine(result):
                    return await result
                return result
            else:
                result = self._engine.get_tokenizer(lora_request)  # type: ignore
                if inspect.iscoroutine(result):
                    return await result
                return result
        except AttributeError:
            # Fallback to get_tokenizer_async for older versions
            try:
                result = self._engine.get_tokenizer_async(lora_request)  # type: ignore
                if inspect.iscoroutine(result):
                    return await result
                return result
            except (AttributeError, TypeError):
                # If all else fails, try without parameters
                result = self._engine.get_tokenizer()  # type: ignore
                if inspect.iscoroutine(result):
                    return await result
                return result

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

        # When enable_thinking is True, don't skip special tokens
        # Check chat_template_kwargs or reasoning_parser for enable_thinking
        enable_thinking = False
        if generate_config:
            chat_template_kwargs = generate_config.get("chat_template_kwargs")
            if chat_template_kwargs:
                if isinstance(chat_template_kwargs, dict):
                    enable_thinking = chat_template_kwargs.get("enable_thinking", False)
                elif isinstance(chat_template_kwargs, str):
                    try:
                        kwargs_dict = json.loads(chat_template_kwargs)
                        enable_thinking = kwargs_dict.get("enable_thinking", False)
                    except json.JSONDecodeError:
                        pass
            elif not enable_thinking and self.reasoning_parser:
                enable_thinking = self.reasoning_parser.enable_thinking

        if (enable_thinking or tools) and generate_config:
            generate_config["skip_special_tokens"] = False

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
            # guided decoding only available for vllm >= 0.6.3;
            # GuidedDecodingParams / StructuredOutputsParams are resolved at
            # module load by _init_guided_decoding_classes().
            # Extract guided decoding parameters
            guided_params: dict[str, Any] = {}
            guided_json = sanitized_generate_config.pop("guided_json", None)
            # Check `is not None` rather than truthiness so a valid empty
            # schema ({}) is forwarded to vLLM instead of being dropped.
            if guided_json is not None:
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
            previous_logprobs_counts = [0] * sanitized_generate_config["n"]
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
                    logprobs = choice["logprobs"]
                    if logprobs is not None:
                        current_count = len(logprobs["tokens"])
                        choice["logprobs"] = self._slice_logprobs(
                            logprobs, previous_logprobs_counts[i]
                        )
                        previous_logprobs_counts[i] = current_count
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
        if (
            llm_family.has_architecture("Qwen3_5MoeForCausalLM")
            and VLLM_INSTALLED
            and VLLM_VERSION < version.parse("0.27.0")
            and not _virtual_env_allows_missing_vllm()
        ):
            return False, "Qwen3_5MoeForCausalLM requires vLLM >= 0.27.0"
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

        tools = list(generate_config.pop("tools", [])) if generate_config else None
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
                or model_family in GEMMA_TOOL_CALL_FAMILY
                or model_family in DEEPSEEK_TOOL_CALL_FAMILY
                or model_family in GLM5_TOOL_CALL_FAMILY
                or model_family in KIMI_K3_TOOL_CALL_FAMILY
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
        logger.debug("tokenizer class: %s", type(tokenizer).__name__)
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
                full_prompt, generate_config, tools, request_id=request_id
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
        supported_architectures = list(VLLM_SUPPORTED_MULTI_MODEL_LIST)
        effective_version = _get_effective_vllm_version_for_family(llm_family)
        if effective_version >= version.parse("0.27.0"):
            _append_unique(supported_architectures, "KimiK3ForConditionalGeneration")
        if not llm_family.matches_supported_architectures(supported_architectures):
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
        # Align with VLLMChatModel: in virtualenv mode vLLM is installed on
        # demand, so a missing local install must not hide the engine at listing
        # time. Hardware/OS checks above stay unconditional because virtualenv
        # cannot add a GPU or change the OS.
        if not VLLM_INSTALLED and not _virtual_env_allows_missing_vllm():
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
            return await super()._gen_tokens_prompt(tokenizer, prompt, config)

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

    def _handle_base64_images(self, messages, temp_files):
        import base64
        import re
        import tempfile

        # Regex to match data URI scheme
        data_uri_pattern = re.compile(
            r"data:([a-zA-Z0-9]+/[a-zA-Z0-9-.+]+);base64,(.*)"
        )

        for msg in messages:
            if isinstance(msg, dict) and isinstance(msg.get("content"), list):
                for content in msg["content"]:
                    if isinstance(content, dict):
                        # check image_url
                        if "image_url" in content and isinstance(
                            content["image_url"], dict
                        ):
                            url = content["image_url"].get("url", "")
                            if isinstance(url, str) and url.startswith("data:"):
                                match = data_uri_pattern.match(url)
                                if match:
                                    mime_type, b64_data = match.groups()
                                    try:
                                        # Create temp file
                                        suffix = ".bin"
                                        if "pdf" in mime_type:
                                            suffix = ".pdf"
                                        elif "png" in mime_type:
                                            suffix = ".png"
                                        elif "jpeg" in mime_type or "jpg" in mime_type:
                                            suffix = ".jpg"

                                        with tempfile.NamedTemporaryFile(
                                            delete=False, suffix=suffix
                                        ) as tmp:
                                            tmp.write(base64.b64decode(b64_data))
                                            content["image_url"]["url"] = tmp.name
                                            temp_files.append(tmp.name)
                                            logger.debug(
                                                f"Decoded base64 content to temp file: {tmp.name}"
                                            )
                                    except Exception as e:
                                        logger.error(
                                            f"Failed to decode base64 file: {e}"
                                        )

    async def _get_chat_template_and_tokenizer(
        self, model_family: str
    ) -> Tuple[Optional[str], Any]:
        chat_template: Optional[str] = self.model_family.chat_template
        tokenizer = None
        if not chat_template:
            tokenizer = await self._get_tokenizer(None)
            if tokenizer is not None:
                chat_template = getattr(tokenizer, "chat_template", None)
        if not chat_template:
            supports_native_renderer = (
                model_family in KIMI_K3_TOOL_CALL_FAMILY
                and tokenizer is not None
                and callable(getattr(tokenizer, "apply_chat_template", None))
            )
            if not supports_native_renderer:
                raise ValueError(
                    f"chat_template is required for model {self.model_uid}, but none was provided."
                )
        return chat_template, tokenizer

    @vllm_check
    async def async_chat(
        self,
        messages: List[ChatCompletionMessage],  # type: ignore
        generate_config: Optional[Dict] = None,
        request_id: Optional[str] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        tools = list(generate_config.pop("tools", [])) if generate_config else None

        model_family = self.model_family.model_family or self.model_family.model_name
        audios, images, videos, video_kwargs = None, None, None, None
        if "internvl" not in model_family.lower():
            from qwen_omni_utils import (
                process_audio_info,
                process_mm_info,
                process_vision_info,
            )

            # Pre-process messages to handle base64 data URIs BEFORE transform
            temp_files: List[str] = []
            if (
                "vision" in self.model_family.model_ability
                or "omni" in self.model_family.model_ability
            ):
                self._handle_base64_images(messages, temp_files)

            messages = self._transform_messages(messages)

            chat_template_kwargs = (
                self._get_chat_template_kwargs_from_generate_config(
                    generate_config, self.reasoning_parser
                )
                or {}
            )
            chat_context_var.set(chat_template_kwargs)
            full_context_kwargs = chat_template_kwargs.copy()
            if tools and (
                model_family in QWEN_TOOL_CALL_FAMILY
                or model_family in GEMMA_TOOL_CALL_FAMILY
                or model_family in GLM5_TOOL_CALL_FAMILY
                or model_family in KIMI_K3_TOOL_CALL_FAMILY
            ):
                full_context_kwargs["tools"] = tools
            assert self.model_family.chat_template is not None

            # Kimi-K3 has no Jinja template and renders through its tokenizer's
            # custom Python apply_chat_template implementation.
            chat_template, tokenizer = await self._get_chat_template_and_tokenizer(
                model_family
            )

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
                messages, chat_template, tokenizer=tokenizer, **full_context_kwargs
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
                inputs,
                generate_config,
                tools=True if tools else False,
                request_id=request_id,
            )
            assert isinstance(agen, AsyncGenerator)
            if tools:
                return self._async_to_tool_completion_chunks(agen, chat_template_kwargs)
            return self._async_to_chat_completion_chunks(
                agen, self.reasoning_parser, chat_template_kwargs
            )
        else:
            c = await self.async_generate(
                inputs,
                generate_config,
                tools=True if tools else False,
                request_id=request_id,
            )
            assert not isinstance(c, AsyncGenerator)
            if tools:
                return self._post_process_completion(
                    self.model_family, self.model_uid, c
                )
            return self._to_chat_completion(c, self.reasoning_parser)
