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

import asyncio
import json
import logging
import multiprocessing
import os
import time
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

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
from .. import LLM, LLMFamilyV1, LLMSpecV1
from ..llm_family import CustomLLMFamilyV1
from ..utils import (
    QWEN_TOOL_CALL_FAMILY,
    QWEN_TOOL_CALL_SYMBOLS,
    ChatModelMixin,
    generate_completion_chunk,
)
from .utils import vllm_check

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
    limit_mm_per_prompt: Optional[Dict[str, int]]
    guided_decoding_backend: Optional[str]


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
    response_format: Optional[dict]
    guided_json: Optional[Union[str, dict]]
    guided_regex: Optional[str]
    guided_choice: Optional[List[str]]
    guided_grammar: Optional[str]
    guided_json_object: Optional[bool]
    guided_decoding_backend: Optional[str]
    guided_whitespace_pattern: Optional[str]


try:
    import vllm  # noqa: F401

    VLLM_INSTALLED = True
except ImportError:
    VLLM_INSTALLED = False

VLLM_SUPPORTED_VISION_MODEL_LIST: List[str] = []
VLLM_SUPPORTED_MODELS = [
    "llama-2",
    "llama-3",
    "mistral-v0.1",
    "codestral-v0.1",
    "Yi",
    "Yi-1.5",
    "code-llama",
    "code-llama-python",
    "deepseek",
    "deepseek-coder",
    "yi-coder",
]
VLLM_SUPPORTED_CHAT_MODELS = [
    "llama-2-chat",
    "llama-3-instruct",
    "baichuan-2-chat",
    "internlm2-chat",
    "internlm2.5-chat",
    "internlm2.5-chat-1m",
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
    "codegeex4",
    "deepseek-chat",
    "deepseek-coder-instruct",
    "yi-coder-chat",
]
if VLLM_INSTALLED and vllm.__version__ >= "0.3.0":
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen1.5-chat")
    VLLM_SUPPORTED_MODELS.append("codeqwen1.5")
    VLLM_SUPPORTED_CHAT_MODELS.append("codeqwen1.5-chat")
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen2-instruct")
    VLLM_SUPPORTED_MODELS.append("qwen2.5")
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen2.5-instruct")
    VLLM_SUPPORTED_MODELS.append("qwen2.5-coder")
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen2.5-coder-instruct")
    VLLM_SUPPORTED_CHAT_MODELS.append("QwQ-32B-Preview")


if VLLM_INSTALLED and vllm.__version__ >= "0.3.2":
    VLLM_SUPPORTED_CHAT_MODELS.append("gemma-it")

if VLLM_INSTALLED and vllm.__version__ >= "0.3.3":
    VLLM_SUPPORTED_CHAT_MODELS.append("orion-chat")
    VLLM_SUPPORTED_CHAT_MODELS.append("orion-chat-rag")

if VLLM_INSTALLED and vllm.__version__ >= "0.4.0":
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen1.5-moe-chat")
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen2-moe-instruct")
    VLLM_SUPPORTED_CHAT_MODELS.append("c4ai-command-r-v01")

if VLLM_INSTALLED and vllm.__version__ >= "0.5.1":
    VLLM_SUPPORTED_CHAT_MODELS.append("deepseek-v2-chat")
    VLLM_SUPPORTED_CHAT_MODELS.append("deepseek-v2-chat-0628")
    VLLM_SUPPORTED_CHAT_MODELS.append("deepseek-v2.5")

if VLLM_INSTALLED and vllm.__version__ >= "0.5.3":
    VLLM_SUPPORTED_CHAT_MODELS.append("gemma-2-it")
    VLLM_SUPPORTED_CHAT_MODELS.append("mistral-nemo-instruct")
    VLLM_SUPPORTED_CHAT_MODELS.append("mistral-large-instruct")

if VLLM_INSTALLED and vllm.__version__ > "0.5.3":
    VLLM_SUPPORTED_MODELS.append("llama-3.1")
    VLLM_SUPPORTED_CHAT_MODELS.append("llama-3.1-instruct")

if VLLM_INSTALLED and vllm.__version__ >= "0.6.1":
    VLLM_SUPPORTED_VISION_MODEL_LIST.append("internvl2")

if VLLM_INSTALLED and vllm.__version__ >= "0.6.3":
    VLLM_SUPPORTED_MODELS.append("llama-3.2-vision")
    VLLM_SUPPORTED_VISION_MODEL_LIST.append("llama-3.2-vision-instruct")
    VLLM_SUPPORTED_VISION_MODEL_LIST.append("qwen2-vl-instruct")


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

        self._check_health_task = None
        if hasattr(self._engine, "check_health"):
            # vLLM introduced `check_health` since v0.4.1
            self._check_health_task = asyncio.create_task(self._check_healthy())

    def stop(self):
        # though the vLLM engine will shutdown when deleted,
        # but some issue e.g. GH#1682 reported
        # when deleting, the engine exists still
        logger.info("Stopping vLLM engine")
        if self._check_health_task:
            self._check_health_task.cancel()
        if model_executor := getattr(self._engine.engine, "model_executor", None):
            model_executor.shutdown()
        self._engine = None

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
                except:
                    # ignore error when stop
                    pass
                # Just kill the process and let xinference auto-recover the model
                os._exit(1)
            else:
                await asyncio.sleep(interval)

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
        model_config.setdefault("max_model_len", None)
        model_config.setdefault("guided_decoding_backend", "outlines")

        return model_config

    @staticmethod
    def _sanitize_generate_config(
        generate_config: Optional[Dict] = None,
    ) -> VLLMGenerateConfig:
        if not generate_config:
            generate_config = {}

        sanitized = VLLMGenerateConfig()

        response_format = generate_config.pop("response_format", None)
        guided_decoding_backend = generate_config.get("guided_decoding_backend", None)
        guided_json_object = None
        guided_json = None

        if response_format is not None:
            if response_format.get("type") == "json_object":
                guided_json_object = True
            elif response_format.get("type") == "json_schema":
                json_schema = response_format.get("json_schema")
                assert json_schema is not None
                guided_json = json_schema.get("json_schema")
                if guided_decoding_backend is None:
                    guided_decoding_backend = "outlines"

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
        sanitized.setdefault(
            "guided_decoding_backend",
            generate_config.get("guided_decoding_backend", guided_decoding_backend),
        )

        return sanitized

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if not cls._has_cuda_device():
            return False
        if not cls._is_linux():
            return False
        if llm_spec.model_format not in ["pytorch", "gptq", "awq", "fp8"]:
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

        if VLLM_INSTALLED and vllm.__version__ >= "0.6.3":
            # guided decoding only available for vllm >= 0.6.3
            from vllm.sampling_params import GuidedDecodingParams

            guided_options = GuidedDecodingParams.from_optional(
                json=sanitized_generate_config.pop("guided_json", None),
                regex=sanitized_generate_config.pop("guided_regex", None),
                choice=sanitized_generate_config.pop("guided_choice", None),
                grammar=sanitized_generate_config.pop("guided_grammar", None),
                json_object=sanitized_generate_config.pop("guided_json_object", None),
                backend=sanitized_generate_config.pop("guided_decoding_backend", None),
                whitespace_pattern=sanitized_generate_config.pop(
                    "guided_whitespace_pattern", None
                ),
            )

            sampling_params = SamplingParams(
                guided_decoding=guided_options, **sanitized_generate_config
            )
        else:
            # ignore generate configs
            sanitized_generate_config.pop("guided_json", None)
            sanitized_generate_config.pop("guided_regex", None)
            sanitized_generate_config.pop("guided_choice", None)
            sanitized_generate_config.pop("guided_grammar", None)
            sanitized_generate_config.pop("guided_json_object", None)
            sanitized_generate_config.pop("guided_decoding_backend", None)
            sanitized_generate_config.pop("guided_whitespace_pattern", None)
            sampling_params = SamplingParams(**sanitized_generate_config)

        if not request_id:
            request_id = str(uuid.uuid1())

        assert self._engine is not None
        results_generator = self._engine.generate(
            prompt,
            sampling_params,
            request_id,
            lora_request,
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
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if llm_spec.model_format not in ["pytorch", "gptq", "awq", "fp8"]:
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
        if not generate_config.get("stop") and self.model_family.stop:
            generate_config["stop"] = self.model_family.stop.copy()
        if (
            not generate_config.get("stop_token_ids")
            and self.model_family.stop_token_ids
        ):
            generate_config["stop_token_ids"] = self.model_family.stop_token_ids.copy()
        return generate_config

    @staticmethod
    def is_tool_call_chunk(chunk):
        return chunk["choices"][0]["text"].startswith(QWEN_TOOL_CALL_SYMBOLS[0])

    async def _async_to_tool_completion_chunks(
        self,
        chunks: AsyncGenerator[CompletionChunk, None],
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        i = 0
        async for chunk in chunks:
            if i == 0:
                yield self._get_first_chat_completion_chunk(chunk)
            # usage
            choices = chunk.get("choices")
            if not choices:
                yield self._get_final_chat_completion_chunk(chunk)
            else:
                if self.is_tool_call_chunk(chunk):
                    yield self._tool_calls_completion_chunk(
                        self.model_family, self.model_uid, chunk
                    )
                else:
                    yield self._to_chat_completion_chunk(chunk)
            i += 1

    @vllm_check
    async def async_chat(
        self,
        messages: List[Dict],
        generate_config: Optional[Dict] = None,
        request_id: Optional[str] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        tools = generate_config.pop("tools", []) if generate_config else None
        model_family = self.model_family.model_family or self.model_family.model_name
        full_context_kwargs = {}
        if tools and model_family in QWEN_TOOL_CALL_FAMILY:
            full_context_kwargs["tools"] = tools
        assert self.model_family.chat_template is not None
        full_prompt = self.get_full_context(
            messages, self.model_family.chat_template, **full_context_kwargs
        )

        generate_config = self._sanitize_chat_config(generate_config)
        stream = generate_config.get("stream", None)

        if stream:
            agen = await self.async_generate(
                full_prompt, generate_config, tools, request_id=request_id
            )
            assert isinstance(agen, AsyncGenerator)
            if tools:
                return self._async_to_tool_completion_chunks(agen)
            return self._async_to_chat_completion_chunks(agen)
        else:
            c = await self.async_generate(
                full_prompt, generate_config, request_id=request_id
            )
            assert not isinstance(c, AsyncGenerator)
            if tools:
                return self._tool_calls_completion(self.model_family, self.model_uid, c)
            return self._to_chat_completion(c)


class VLLMVisionModel(VLLMModel, ChatModelMixin):
    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if not cls._has_cuda_device():
            return False
        if not cls._is_linux():
            return False
        if llm_spec.model_format not in ["pytorch", "gptq", "awq", "fp8"]:
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
            if llm_family.model_family not in VLLM_SUPPORTED_VISION_MODEL_LIST:
                return False
        else:
            if llm_family.model_name not in VLLM_SUPPORTED_VISION_MODEL_LIST:
                return False
        if "vision" not in llm_family.model_ability:
            return False
        return VLLM_INSTALLED

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
        model_config.setdefault("max_model_len", None)
        model_config["limit_mm_per_prompt"] = (
            json.loads(model_config.get("limit_mm_per_prompt"))  # type: ignore
            if model_config.get("limit_mm_per_prompt")
            else {
                "image": 2,  # default 2 images all chat
            }
        )

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

    @vllm_check
    async def async_chat(
        self,
        messages: List[ChatCompletionMessage],  # type: ignore
        generate_config: Optional[Dict] = None,
        request_id: Optional[str] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        messages = self._transform_messages(messages)
        tools = generate_config.pop("tools", []) if generate_config else None

        model_family = self.model_family.model_family or self.model_family.model_name

        if "internvl2" not in model_family.lower():
            from qwen_vl_utils import process_vision_info

            full_context_kwargs = {}
            if tools and model_family in QWEN_TOOL_CALL_FAMILY:
                full_context_kwargs["tools"] = tools
            assert self.model_family.chat_template is not None
            prompt = self.get_full_context(
                messages, self.model_family.chat_template, **full_context_kwargs
            )
            images, video_inputs = process_vision_info(messages)
            if video_inputs:
                raise ValueError("Not support video input now.")
        else:
            prompt, images = self.get_specific_prompt(model_family, messages)

        if not images:
            inputs = {
                "prompt": prompt,
            }
        elif len(images) == 1:
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": images[-1]},  # type: ignore
            }
        else:
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": images},  # type: ignore
            }
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
