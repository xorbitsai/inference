# Copyright 2022-2024 XProbe Inc.
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
import time
import uuid
from typing import (  # TYPE_CHECKING,; Any,; Iterable,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    TypedDict,
    Union,
)

from ....constants import XINFERENCE_ENABLE_SGLANG
from ....types import (  # ChatCompletion,; ChatCompletionChunk,; ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionUsage,
)
from .. import LLM, LLMFamilyV1, LLMSpecV1
from ..llm_family import CustomLLMFamilyV1

# from ..utils import ChatModelMixin

logger = logging.getLogger(__name__)


class SGLANGModelConfig(TypedDict, total=False):
    tokenizer_mode: str
    trust_remote_code: bool
    tp_size: int
    mem_fraction_static: float
    log_level: str


class SGLANGGenerateConfig(TypedDict, total=False):
    presence_penalty: float
    frequency_penalty: float
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    stop: Optional[Union[str, List[str]]]
    ignore_eos: bool
    stream: bool


try:
    import sglang  # noqa: F401

    SGLANG_INSTALLED = True
except ImportError:
    SGLANG_INSTALLED = False

SGLANG_SUPPORTED_MODELS = ["llama-2", "qwen-chat", "qwen1.5-chat"]
# VLLM_SUPPORTED_CHAT_MODELS = [
#     "llama-2-chat",
#     "vicuna-v1.3",
#     "vicuna-v1.5",
#     "baichuan-chat",
#     "internlm-chat-7b",
#     "internlm-chat-8k",
#     "internlm-chat-20b",
#     "qwen-chat",
#     "Yi",
#     "Yi-chat",
#     "code-llama",
#     "code-llama-python",
#     "code-llama-instruct",
#     "mistral-instruct-v0.1",
#     "mistral-instruct-v0.2",
#     "mixtral-instruct-v0.1",
#     "chatglm3",
# ]
# if VLLM_INSTALLED and vllm.__version__ >= "0.3.0":
#     VLLM_SUPPORTED_CHAT_MODELS.append("qwen1.5-chat")


class SGLANGModel(LLM):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        model_config: Optional[SGLANGModelConfig],
    ):
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)
        self._model_config = model_config
        self._engine = None

    def load(self):
        try:
            import sglang as sgl
        except ImportError:
            error_message = "Failed to import module 'sglang'"
            installation_guide = [
                "Please make sure 'sglang' is installed. ",
                "You can install it by `pip install 'sglang[all]'`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        self._model_config = self._sanitize_model_config(self._model_config)
        logger.info(
            f"Loading {self.model_uid} with following model config: {self._model_config}"
        )

        self._engine = sgl.Runtime(
            model_path=self.model_path,
            tokenizer_path=self.model_path,
            trust_remote_code=True,
            **self._model_config,
        )

    def _sanitize_model_config(
        self, model_config: Optional[SGLANGModelConfig]
    ) -> SGLANGModelConfig:
        if model_config is None:
            model_config = SGLANGModelConfig()

        cuda_count = self._get_cuda_count()
        model_config.setdefault("tokenizer_mode", "auto")
        model_config.setdefault("trust_remote_code", True)
        model_config.setdefault("tp_size", cuda_count)
        model_config.setdefault("mem_fraction_static", 0.9)
        model_config.setdefault("log_level", "info")

        return model_config

    @staticmethod
    def _sanitize_generate_config(
        generate_config: Optional[SGLANGGenerateConfig] = None,
    ) -> SGLANGGenerateConfig:
        if generate_config is None:
            generate_config = SGLANGGenerateConfig()

        generate_config.setdefault("presence_penalty", 0.0)
        generate_config.setdefault("frequency_penalty", 0.0)
        generate_config.setdefault("temperature", 1.0)
        generate_config.setdefault("top_p", 1.0)
        generate_config.setdefault("top_k", -1)
        # See https://github.com/sgl-project/sglang/blob/main/python/sglang/lang/ir.py#L120
        # 16 is too less, so here set 256 by default
        generate_config.setdefault("max_new_tokens", 256)
        generate_config.setdefault("stop", [])
        generate_config.setdefault("stream", False)
        generate_config.setdefault("ignore_eos", False)

        return generate_config

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if not XINFERENCE_ENABLE_SGLANG:
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
        if llm_spec.model_format in ["gptq", "awq"]:
            # Currently, only 4-bit weight quantization is supported for GPTQ, but got 8 bits.
            if "4" not in quantization:
                return False
        if isinstance(llm_family, CustomLLMFamilyV1):
            if llm_family.model_family not in SGLANG_SUPPORTED_MODELS:
                return False
        else:
            if llm_family.model_name not in SGLANG_SUPPORTED_MODELS:
                return False
        # if "generate" not in llm_family.model_ability:
        #     return False
        return SGLANG_INSTALLED

    @staticmethod
    def _convert_state_to_completion_chunk(
        request_id: str, model: str, output_text: str, meta_info: Dict
    ) -> CompletionChunk:
        choices: List[CompletionChoice] = [
            CompletionChoice(
                text=output_text,
                index=0,
                logprobs=None,
                finish_reason=None,
            )
        ]
        chunk = CompletionChunk(
            id=request_id,
            object="text_completion",
            created=int(time.time()),
            model=model,
            choices=choices,
        )
        prompt_tokens = meta_info["prompt_tokens"]
        completion_tokens = meta_info["completion_tokens"]
        chunk["usage"] = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return chunk

    @staticmethod
    def _convert_state_to_completion(
        request_id: str, model: str, output_text: str, meta_info: Dict
    ) -> Completion:
        choices = [
            CompletionChoice(
                text=output_text,
                index=0,
                logprobs=None,
                finish_reason=None,
            )
        ]

        usage = CompletionUsage(
            prompt_tokens=meta_info["prompt_tokens"],
            completion_tokens=meta_info["completion_tokens"],
            total_tokens=meta_info["prompt_tokens"] + meta_info["completion_tokens"],
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
        generate_config: Optional[SGLANGGenerateConfig] = None,
    ) -> Union[Completion, AsyncGenerator[CompletionChunk, None]]:
        try:
            import sglang as sgl
            from sglang import assistant, gen, user
        except ImportError:
            error_message = "Failed to import module 'sglang'"
            installation_guide = [
                "Please make sure 'sglang' is installed. ",
                "You can install it by `pip install sglang[all]`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        @sgl.function
        def pipeline(s, question):
            s += user(question)
            s += assistant(gen("answer"))

        sanitized_generate_config = self._sanitize_generate_config(generate_config)
        logger.debug(
            "Enter generate, prompt: %s, generate config: %s", prompt, generate_config
        )
        stream = sanitized_generate_config.pop("stream")
        request_id = str(uuid.uuid1())
        state = pipeline.run(
            question=prompt,
            backend=self._engine,
            stream=stream,
            **sanitized_generate_config,
        )
        if not stream:
            return self._convert_state_to_completion(
                request_id,
                model=self.model_uid,
                output_text=state["answer"],
                meta_info=state.get_meta_info(),
            )
        else:

            async def stream_results() -> AsyncGenerator[CompletionChunk, None]:
                async for out, meta_info in state.text_async_iter(
                    var_name="answer", return_meta_data=True
                ):
                    chunk = self._convert_state_to_completion_chunk(
                        request_id, self.model_uid, output_text=out, meta_info=meta_info
                    )
                    yield chunk

            return stream_results()


# class VLLMChatModel(VLLMModel, ChatModelMixin):
#     @classmethod
#     def match(
#         cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
#     ) -> bool:
#         if XINFERENCE_DISABLE_VLLM:
#             return False
#         if llm_spec.model_format not in ["pytorch", "gptq", "awq"]:
#             return False
#         if llm_spec.model_format == "pytorch":
#             if quantization != "none" and not (quantization is None):
#                 return False
#         if llm_spec.model_format in ["gptq", "awq"]:
#             # Currently, only 4-bit weight quantization is supported for GPTQ, but got 8 bits.
#             if "4" not in quantization:
#                 return False
#         if isinstance(llm_family, CustomLLMFamilyV1):
#             if llm_family.model_family not in VLLM_SUPPORTED_CHAT_MODELS:
#                 return False
#         else:
#             if llm_family.model_name not in VLLM_SUPPORTED_CHAT_MODELS:
#                 return False
#         if "chat" not in llm_family.model_ability:
#             return False
#         return VLLM_INSTALLED
#
#     def _sanitize_chat_config(
#         self,
#         generate_config: Optional[Dict] = None,
#     ) -> Dict:
#         if not generate_config:
#             generate_config = {}
#         if self.model_family.prompt_style:
#             if (
#                 not generate_config.get("stop")
#             ) and self.model_family.prompt_style.stop:
#                 generate_config["stop"] = self.model_family.prompt_style.stop.copy()
#             if self.model_family.prompt_style.stop_token_ids:
#                 generate_config.setdefault(
#                     "stop_token_ids",
#                     self.model_family.prompt_style.stop_token_ids.copy(),
#                 )
#         return generate_config
#
#     async def async_chat(
#         self,
#         prompt: str,
#         system_prompt: Optional[str] = None,
#         chat_history: Optional[List[ChatCompletionMessage]] = None,
#         generate_config: Optional[Dict] = None,
#     ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
#         assert self.model_family.prompt_style is not None
#         prompt_style = self.model_family.prompt_style.copy()
#         if system_prompt:
#             prompt_style.system_prompt = system_prompt
#         chat_history = chat_history or []
#         tools = generate_config.pop("tools", []) if generate_config else None
#         full_prompt = self.get_prompt(prompt, chat_history, prompt_style, tools=tools)
#
#         generate_config = self._sanitize_chat_config(generate_config)
#         # TODO(codingl2k1): qwen hacky to set stop for function call.
#         model_family = self.model_family.model_family or self.model_family.model_name
#         if tools and "qwen-chat" == model_family:
#             stop = generate_config.get("stop")
#             if isinstance(stop, str):
#                 generate_config["stop"] = [stop, "Observation:"]
#             elif isinstance(stop, Iterable):
#                 assert not isinstance(stop, str)
#                 generate_config["stop"] = list(stop) + ["Observation:"]
#             else:
#                 generate_config["stop"] = "Observation:"
#
#         stream = generate_config.get("stream", None)
#
#         if stream:
#             agen = await self.async_generate(full_prompt, generate_config)
#             assert isinstance(agen, AsyncGenerator)
#             return self._async_to_chat_completion_chunks(agen)
#         else:
#             c = await self.async_generate(full_prompt, generate_config)
#             assert not isinstance(c, AsyncGenerator)
#             if tools:
#                 return self._tool_calls_completion(
#                     self.model_family, self.model_uid, c, tools
#                 )
#             return self._to_chat_completion(c)
