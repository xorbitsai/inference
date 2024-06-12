# Copyright 2022-2023 XProbe Inc.
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
import multiprocessing
import time
import uuid
from typing import (
    TYPE_CHECKING,
    AsyncGenerator,
    Dict,
    Iterable,
    List,
    Optional,
    TypedDict,
    Union,
)

from ....constants import XINFERENCE_DISABLE_VLLM
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionUsage,
    LoRA,
    ToolCallFunction,
    ToolCalls,
)
from .. import LLM, LLMFamilyV1, LLMSpecV1
from ..llm_family import CustomLLMFamilyV1
from ..utils import ChatModelMixin

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput


class VLLMModelConfig(TypedDict, total=False):
    tokenizer_mode: Optional[str]
    trust_remote_code: bool
    tensor_parallel_size: int
    block_size: int
    swap_space: int  # GiB
    gpu_memory_utilization: float
    max_num_batched_tokens: int
    max_num_seqs: int
    quantization: Optional[str]
    max_model_len: Optional[int]


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


try:
    import vllm  # noqa: F401

    VLLM_INSTALLED = True
except ImportError:
    VLLM_INSTALLED = False

VLLM_SUPPORTED_MODELS = [
    "llama-2",
    "llama-3",
    "baichuan",
    "internlm-16k",
    "mistral-v0.1",
    "codestral-v0.1",
    "Yi",
    "Yi-1.5",
    "code-llama",
    "code-llama-python",
    "deepseek",
    "deepseek-coder",
]
VLLM_SUPPORTED_CHAT_MODELS = [
    "llama-2-chat",
    "llama-3-instruct",
    "vicuna-v1.3",
    "vicuna-v1.5",
    "baichuan-chat",
    "baichuan-2-chat",
    "internlm-chat-7b",
    "internlm-chat-8k",
    "internlm-chat-20b",
    "internlm2-chat",
    "qwen-chat",
    "Yi-chat",
    "Yi-1.5-chat",
    "Yi-1.5-chat-16k",
    "code-llama-instruct",
    "mistral-instruct-v0.1",
    "mistral-instruct-v0.2",
    "mistral-instruct-v0.3",
    "mixtral-instruct-v0.1",
    "mixtral-8x22B-instruct-v0.1",
    "chatglm3",
    "chatglm3-32k",
    "chatglm3-128k",
    "glm4-chat",
    "glm4-chat-1m",
    "deepseek-chat",
    "deepseek-coder-instruct",
]
if VLLM_INSTALLED and vllm.__version__ >= "0.3.0":
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen1.5-chat")
    VLLM_SUPPORTED_MODELS.append("codeqwen1.5")
    VLLM_SUPPORTED_CHAT_MODELS.append("codeqwen1.5-chat")
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen2-instruct")

if VLLM_INSTALLED and vllm.__version__ >= "0.3.2":
    VLLM_SUPPORTED_CHAT_MODELS.append("gemma-it")

if VLLM_INSTALLED and vllm.__version__ >= "0.3.3":
    VLLM_SUPPORTED_CHAT_MODELS.append("orion-chat")
    VLLM_SUPPORTED_CHAT_MODELS.append("orion-chat-rag")

if VLLM_INSTALLED and vllm.__version__ >= "0.4.0":
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen1.5-moe-chat")
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen2-moe-instruct")
    VLLM_SUPPORTED_CHAT_MODELS.append("c4ai-command-r-v01")


class VLLMModel(LLM):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        model_config: Optional[VLLMModelConfig],
        peft_model: Optional[List[LoRA]] = None,
    ):
        try:
            from vllm.lora.request import LoRARequest
        except ImportError:
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)
        self._model_config = model_config
        self._engine = None
        self.lora_modules = peft_model
        self.lora_requests: List[LoRARequest] = []

    def load(self):
        try:
            import vllm
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            from vllm.lora.request import LoRARequest
        except ImportError:
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        if vllm.__version__ >= "0.3.1":
            # from vllm v0.3.1, it uses cupy as NCCL backend
            # in which cupy will fork a process
            # only for xoscar >= 0.3.0, new process is allowed in subpool
            # besides, xinference set start method as forkserver for unix
            # we need to set it to fork to make cupy NCCL work
            multiprocessing.set_start_method("fork", force=True)

        self._model_config = self._sanitize_model_config(self._model_config)

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

        engine_args = AsyncEngineArgs(
            model=self.model_path,
            enable_lora=enable_lora,
            max_loras=max_loras,
            **self._model_config,
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

    def _sanitize_model_config(
        self, model_config: Optional[VLLMModelConfig]
    ) -> VLLMModelConfig:
        if model_config is None:
            model_config = VLLMModelConfig()

        cuda_count = self._get_cuda_count()

        model_config.setdefault("tokenizer_mode", "auto")
        model_config.setdefault("trust_remote_code", True)
        model_config.setdefault("tensor_parallel_size", cuda_count)
        model_config.setdefault("block_size", 16)
        model_config.setdefault("swap_space", 4)
        model_config.setdefault("gpu_memory_utilization", 0.90)
        model_config.setdefault("max_num_seqs", 256)
        model_config.setdefault("quantization", None)
        model_config.setdefault("max_model_len", 4096)

        return model_config

    @staticmethod
    def _sanitize_generate_config(
        generate_config: Optional[Dict] = None,
    ) -> VLLMGenerateConfig:
        if not generate_config:
            generate_config = {}

        sanitized = VLLMGenerateConfig()
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
        sanitized.setdefault("max_tokens", generate_config.get("max_tokens", 1024))
        sanitized.setdefault("stop", generate_config.get("stop", None))
        sanitized.setdefault(
            "stop_token_ids", generate_config.get("stop_token_ids", None)
        )
        sanitized.setdefault("stream", generate_config.get("stream", False))
        sanitized.setdefault(
            "stream_options", generate_config.get("stream_options", None)
        )

        return sanitized

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if XINFERENCE_DISABLE_VLLM:
            return False
        if not cls._has_cuda_device():
            return False
        if not cls._is_linux():
            return False
        if llm_spec.model_format not in ["pytorch", "gptq", "awq"]:
            return False
        if llm_spec.model_format == "pytorch":
            if quantization != "none" and not (quantization is None):
                return False
        if llm_spec.model_format == "awq":
            # Currently, only 4-bit weight quantization is supported for AWQ, but got 8 bits.
            if "4" not in quantization:
                return False
        if llm_spec.model_format == "gptq":
            if VLLM_INSTALLED and vllm.__version__ >= "0.3.3":
                if not any(q in quantization for q in ("3", "4", "8")):
                    return False
            else:
                if "4" not in quantization:
                    return False
        if isinstance(llm_family, CustomLLMFamilyV1):
            if llm_family.model_family not in VLLM_SUPPORTED_MODELS:
                return False
        else:
            if llm_family.model_name not in VLLM_SUPPORTED_MODELS:
                return False
        if "generate" not in llm_family.model_ability:
            return False
        return VLLM_INSTALLED

    @staticmethod
    def _convert_request_output_to_completion_chunk(
        request_id: str, model: str, request_output: "RequestOutput"
    ) -> CompletionChunk:
        choices: List[CompletionChoice] = []
        for output in request_output.outputs:
            choices.append(
                CompletionChoice(
                    text=output.text,
                    index=output.index,
                    logprobs=None,  # TODO: support logprobs.
                    finish_reason=output.finish_reason,
                )
            )
        return CompletionChunk(
            id=request_id,
            object="text_completion",
            created=int(time.time()),
            model=model,
            choices=choices,
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

    async def async_generate(
        self,
        prompt: str,
        generate_config: Optional[Dict] = None,
        tools: object = False,
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
        sampling_params = SamplingParams(**sanitized_generate_config)
        request_id = str(uuid.uuid1())

        assert self._engine is not None
        results_generator = self._engine.generate(
            prompt, sampling_params, request_id, lora_request=lora_request
        )

        async def stream_results() -> AsyncGenerator[CompletionChunk, None]:
            previous_texts = [""] * sanitized_generate_config["n"]
            tools_token_filter = ChatModelMixin._tools_token_filter(self.model_family)
            prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
            async for _request_output in results_generator:
                chunk = self._convert_request_output_to_completion_chunk(
                    request_id=request_id,
                    model=self.model_uid,
                    request_output=_request_output,
                )

                for i, choice in enumerate(chunk["choices"]):
                    delta = choice["text"][len(previous_texts[i]) :]
                    previous_texts[i] = choice["text"]
                    choice["text"] = delta

                if tools:
                    # only handle the first choice
                    choice = chunk["choices"][0]
                    if choice["finish_reason"] is not None:
                        # use previous text for evaluation temporarily
                        choice_delta = choice["text"]
                        choice["text"] = previous_texts[0]
                        _content, func, args = ChatModelMixin._eval_tool_arguments(
                            self.model_family, chunk, tools
                        )
                        choice["text"] = tools_token_filter(
                            tokens=previous_texts[0], delta=choice_delta
                        )
                        if func is not None:
                            choice["text"] = None
                            choice["finish_reason"] = "tool_calls"
                            choice["tool_calls"] = [
                                ToolCalls(
                                    id=str(uuid.uuid4()),
                                    type="function",
                                    function=ToolCallFunction(
                                        name=func,
                                        arguments=json.dumps(args, ensure_ascii=False),
                                    ),
                                )
                            ]
                    else:
                        # use a filter function to skip Qwen's react thought process
                        choice["text"] = tools_token_filter(
                            tokens=previous_texts[0], delta=choice["text"]
                        )
                        if not choice["text"]:
                            continue
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
                yield chunk
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
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if XINFERENCE_DISABLE_VLLM:
            return False
        if llm_spec.model_format not in ["pytorch", "gptq", "awq"]:
            return False
        if llm_spec.model_format == "pytorch":
            if quantization != "none" and not (quantization is None):
                return False
        if llm_spec.model_format == "awq":
            # Currently, only 4-bit weight quantization is supported for AWQ, but got 8 bits.
            if "4" not in quantization:
                return False
        if llm_spec.model_format == "gptq":
            if VLLM_INSTALLED and vllm.__version__ >= "0.3.3":
                if not any(q in quantization for q in ("3", "4", "8")):
                    return False
            else:
                if "4" not in quantization:
                    return False
        if isinstance(llm_family, CustomLLMFamilyV1):
            if llm_family.model_family not in VLLM_SUPPORTED_CHAT_MODELS:
                return False
        else:
            if llm_family.model_name not in VLLM_SUPPORTED_CHAT_MODELS:
                return False
        if "chat" not in llm_family.model_ability:
            return False
        return VLLM_INSTALLED

    def _sanitize_chat_config(
        self,
        generate_config: Optional[Dict] = None,
    ) -> Dict:
        if not generate_config:
            generate_config = {}
        if self.model_family.prompt_style:
            if (
                not generate_config.get("stop")
            ) and self.model_family.prompt_style.stop:
                generate_config["stop"] = self.model_family.prompt_style.stop.copy()
            if self.model_family.prompt_style.stop_token_ids:
                generate_config.setdefault(
                    "stop_token_ids",
                    self.model_family.prompt_style.stop_token_ids.copy(),
                )
        return generate_config

    async def async_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[Dict] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        assert self.model_family.prompt_style is not None
        prompt_style = self.model_family.prompt_style.copy()
        if system_prompt:
            prompt_style.system_prompt = system_prompt
        chat_history = chat_history or []
        tools = generate_config.pop("tools", []) if generate_config else None
        full_prompt = self.get_prompt(prompt, chat_history, prompt_style, tools=tools)

        generate_config = self._sanitize_chat_config(generate_config)
        # TODO(codingl2k1): qwen hacky to set stop for function call.
        model_family = self.model_family.model_family or self.model_family.model_name
        if tools and model_family in ["qwen-chat", "qwen1.5-chat"]:
            stop = generate_config.get("stop")
            if isinstance(stop, str):
                generate_config["stop"] = [stop, "Observation:"]
            elif isinstance(stop, Iterable):
                assert not isinstance(stop, str)
                generate_config["stop"] = list(stop) + ["Observation:"]
            else:
                generate_config["stop"] = "Observation:"

        stream = generate_config.get("stream", None)

        if stream:
            agen = await self.async_generate(full_prompt, generate_config, tools)
            assert isinstance(agen, AsyncGenerator)
            return self._async_to_chat_completion_chunks(agen)
        else:
            c = await self.async_generate(full_prompt, generate_config)
            assert not isinstance(c, AsyncGenerator)
            if tools:
                return self._tool_calls_completion(
                    self.model_family, self.model_uid, c, tools
                )
            return self._to_chat_completion(c)
