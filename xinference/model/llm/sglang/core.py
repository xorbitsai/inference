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
import multiprocessing
import os
import sys
import threading
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional, Tuple, TypedDict, Union

from xoscar.utils import get_next_port

from ....constants import XINFERENCE_MAX_TOKENS
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionUsage,
)
from ...utils import check_dependency_available
from .. import LLM, LLMFamilyV2, LLMSpecV1
from ..core import chat_context_var
from ..utils import (
    DEEPSEEK_TOOL_CALL_FAMILY,
    QWEN_TOOL_CALL_FAMILY,
    QWEN_TOOL_CALL_SYMBOLS,
    ChatModelMixin,
    generate_completion_chunk,
)

logger = logging.getLogger(__name__)


class SGLANGModelConfig(TypedDict, total=False):
    tokenizer_mode: str
    trust_remote_code: bool
    tp_size: int
    mem_fraction_static: float
    log_level: str
    attention_reduce_in_fp32: bool  # For gemma
    quantization: Optional[str]
    dtype: Optional[str]
    # distributed
    nnodes: Optional[int]
    node_rank: Optional[int]
    dist_init_addr: Optional[str]
    reasoning_content: bool


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
    stream_options: Optional[Union[dict, None]]
    json_schema: Optional[dict]
    response_format: dict


try:
    import sglang  # noqa: F401

    SGLANG_INSTALLED = True
except ImportError:
    SGLANG_INSTALLED = False

SGLANG_SUPPORTED_MODELS = [
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "Qwen2ForCausalLM",
    "OPTForCausalLM",
]
SGLANG_SUPPORTED_CHAT_MODELS = [
    "LlamaForCausalLM",
    "QWenLMHeadModel",
    "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "GemmaForCausalLM",
    "Gemma3ForCausalLM",
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "Qwen3ForCausalLM",
]
SGLANG_SUPPORTED_VISION_MODEL_LIST = [
    "Qwen2_5_VLForConditionalGeneration",
    "Gemma3ForConditionalGeneration",
    "MiniCPMV",
    "MllamaForConditionalGeneration",
]


class SGLANGModel(LLM):
    allow_batch = True

    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV2",
        model_path: str,
        model_config: Optional[SGLANGModelConfig],
    ):
        super().__init__(model_uid, model_family, model_path)
        self._model_config = model_config
        self._engine = None
        self._address = model_config.pop("address", None)  # type: ignore
        self._n_worker = model_config.pop("n_worker", 1)  # type: ignore
        self._shard = model_config.pop("shard", 0)  # type: ignore
        self._driver_info = model_config.pop("driver_info", None)  # type: ignore
        self._loading_thread = None
        self._loading_error = None

    @property
    def driver_info(self) -> Optional[dict]:
        return self._driver_info

    def load(self):
        try:
            venv_bin = os.path.dirname(sys.executable)
            if venv_bin:
                path_entries = os.environ.get("PATH", "").split(os.pathsep)
                if venv_bin not in path_entries:
                    os.environ["PATH"] = os.pathsep.join(
                        [venv_bin] + ([p for p in path_entries if p] or [])
                    )
            import sglang as sgl
        except ImportError:
            error_message = "Failed to import module 'sglang'"
            installation_guide = [
                "Please make sure 'sglang' is installed. ",
                "You can install it by `pip install 'sglang[all]'`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        self._model_config = self._sanitize_model_config(self._model_config)
        reasoning_content = self._model_config.pop("reasoning_content")
        enable_thinking = self._model_config.pop("enable_thinking", False)
        self.prepare_parse_reasoning_content(
            reasoning_content, enable_thinking=enable_thinking
        )
        self.prepare_parse_tool_calls()

        # Fix: GH#2169
        if sgl.__version__ >= "0.2.14":
            self._model_config.setdefault("triton_attention_reduce_in_fp32", False)
        else:
            self._model_config.setdefault("attention_reduce_in_fp32", False)

        # gen port for sgl Runtime,
        # this is useful for sglang service on a same machine.
        # sglang typically find a port between [port, 40000]
        # we need to ensure the generated port < 40000
        sgl_port = None
        for _ in range(10):
            sgl_port = get_next_port()
            if sgl_port >= 40000:
                sgl_port = None
            else:
                break
        if sgl_port is None:
            raise ValueError("Failed to find a port for sglang")

        # fork may cause sglang stuck, force set to spawn
        multiprocessing.set_start_method("spawn")

        if self._n_worker > 1:
            # distributed inference
            self._model_config["nnodes"] = self._n_worker
            self._model_config["node_rank"] = self._shard
            # model across multiple workers
            if self._shard == 0:
                # distributed, need to init driver_info
                assert self._driver_info is None
                # This must run inside Xoscar pool
                dist_init_addr = f"{self._address.split(':', 1)[0]}:{get_next_port()}"
                self._driver_info = {"dist_init_addr": dist_init_addr}
                self._model_config["dist_init_addr"] = dist_init_addr
            else:
                assert self._driver_info is not None
                self._model_config["dist_init_addr"] = self._driver_info[
                    "dist_init_addr"
                ]

            logger.info(
                f"Loading {self.model_uid}, shard({self._shard} of {self._n_worker}) with following model config: {self._model_config}"
            )

            def _load():
                try:
                    self._engine = sgl.Runtime(
                        model_path=self.model_path,
                        tokenizer_path=self.model_path,
                        port=sgl_port,
                        **self._model_config,
                    )
                except:
                    logger.exception("Creating sglang Runtime failed")
                    self._loading_error = sys.exc_info()

            self._loading_thread = threading.Thread(target=_load)
            self._loading_thread.start()
            if self._shard == 0:
                # wait for 3 seconds to ensure torch distributed inited first
                self._loading_thread.join(3)
        else:
            logger.info(
                f"Loading {self.model_uid} with following model config: {self._model_config}"
            )

            self._engine = sgl.Runtime(
                model_path=self.model_path,
                tokenizer_path=self.model_path,
                port=sgl_port,
                **self._model_config,
            )

    def wait_for_load(self):
        if self._loading_thread:
            if self._shard == 0:
                # for the shard 0, we wait it to complete
                # the sglang will serve forever for the other shards,
                # so we only check if any error happens.
                self._loading_thread.join()
            if self._loading_error:
                _, err, tb = self._loading_error
                raise err.with_traceback(tb)

    def stop(self):
        logger.info("Stopping SGLang engine, sglang pid: %s", self._engine.pid)
        self._engine.shutdown()

    def _sanitize_model_config(
        self, model_config: Optional[SGLANGModelConfig]
    ) -> SGLANGModelConfig:
        if model_config is None:
            model_config = SGLANGModelConfig()

        cuda_count = self._get_cuda_count()
        model_config.setdefault("tokenizer_mode", "auto")
        model_config.setdefault("trust_remote_code", True)
        model_config.setdefault("tp_size", cuda_count * self._n_worker)
        # See https://github.com/sgl-project/sglang/blob/00023d622a6d484e67ef4a0e444f708b8fc861c8/python/sglang/srt/server_args.py#L100-L109
        mem_fraction_static = model_config.get("mem_fraction_static")
        if mem_fraction_static is None:
            tp_size = model_config.get("tp_size", cuda_count)
            if tp_size >= 16:
                model_config["mem_fraction_static"] = 0.79
            elif tp_size >= 8:
                model_config["mem_fraction_static"] = 0.81
            elif tp_size >= 4:
                model_config["mem_fraction_static"] = 0.85
            elif tp_size >= 2:
                model_config["mem_fraction_static"] = 0.87
            else:
                model_config["mem_fraction_static"] = 0.88
        model_config.setdefault("log_level", "info")
        model_config.setdefault("reasoning_content", False)
        self._apply_fp4_config(model_config)

        return model_config

    def _apply_fp4_config(self, model_config: SGLANGModelConfig) -> None:
        if self.model_spec.model_format != "fp4":
            return

        if "quantization" in model_config:
            logger.warning(
                "SGLang fp4 expects offline-quantized weights; ignoring quantization=%s",
                model_config["quantization"],
            )
            model_config.pop("quantization", None)
        model_config.setdefault("dtype", "bfloat16")

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
        max_tokens = (
            generate_config.pop("max_tokens", XINFERENCE_MAX_TOKENS)  # type: ignore
            or XINFERENCE_MAX_TOKENS
        )
        generate_config.setdefault("max_new_tokens", max_tokens)  # type: ignore
        generate_config.setdefault("stop", [])
        generate_config.setdefault("stream", False)
        stream_options = generate_config.get("stream_options")
        generate_config.setdefault("stream_options", stream_options)
        generate_config.setdefault("ignore_eos", False)
        response_format = generate_config.pop("response_format", None)
        if response_format:
            json_schema_config = response_format.pop("json_schema", None)
            json_schema = None
            if "schema_" in json_schema_config:
                json_schema = json_schema_config.pop("schema_")
            elif "schema" in json_schema_config:
                json_schema = json_schema_config.pop("schema")
            if json_schema:
                generate_config.setdefault("json_schema", json.dumps(json_schema))  # type: ignore

        return generate_config

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        dep_check = check_dependency_available("sglang", "sglang")
        if dep_check != True:
            return dep_check
        return True

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if not cls._has_cuda_device():
            return False, "SGLang requires CUDA GPUs"
        if not cls._is_linux():
            return False, "SGLang backend is only supported on Linux"
        if llm_spec.model_format not in ["pytorch", "gptq", "awq", "fp8", "bnb"]:
            return (
                False,
                "SGLang supports pytorch/gptq/awq/fp8/bnb formats only",
            )
        if llm_spec.model_format == "fp4":
            return (
                False,
                "SGLang does not support fp4 online quantization; use offline fp4 weights with a compatible SGLang version",
            )
        if llm_spec.model_format == "pytorch":
            if quantization not in (None, "none"):
                return (
                    False,
                    "pytorch format with quantization is not supported by SGLang",
                )
        if not llm_family.matches_supported_architectures(SGLANG_SUPPORTED_MODELS):
            return (
                False,
                f"Model architectures {llm_family.architectures} are not supported by SGLang",
            )
        if "generate" not in llm_family.model_ability:
            return False, "SGLang base engine requires generate ability"
        if not SGLANG_INSTALLED:
            return False, "sglang library is not installed"
        return True

    @staticmethod
    def _convert_state_to_completion_chunk(
        request_id: str, model: str, output_text: str, meta_info: Dict
    ) -> CompletionChunk:
        finish_reason_raw = meta_info.get("finish_reason", None)
        finish_reason: Optional[str] = None
        if isinstance(finish_reason_raw, dict) and "type" in finish_reason_raw:
            finish_reason = (
                str(finish_reason_raw["type"])
                if finish_reason_raw["type"] is not None
                else None
            )
        elif isinstance(finish_reason_raw, str):
            finish_reason = finish_reason_raw
        choices: List[CompletionChoice] = [
            CompletionChoice(
                text=output_text,
                index=0,
                logprobs=None,
                finish_reason=finish_reason,
            )
        ]
        usage = CompletionUsage(
            prompt_tokens=meta_info["prompt_tokens"],
            completion_tokens=meta_info["completion_tokens"],
            total_tokens=meta_info["prompt_tokens"] + meta_info["completion_tokens"],
        )
        chunk = CompletionChunk(
            id=request_id,
            object="text_completion",
            created=int(time.time()),
            model=model,
            choices=choices,
            usage=usage,
        )
        return chunk

    @staticmethod
    def _convert_state_to_completion(
        request_id: str, model: str, output_text: str, meta_info: Dict
    ) -> Completion:
        finish_reason_raw = meta_info.get("finish_reason", None)
        finish_reason: Optional[str] = None
        if isinstance(finish_reason_raw, dict) and "type" in finish_reason_raw:
            finish_reason = (
                str(finish_reason_raw["type"])
                if finish_reason_raw["type"] is not None
                else None
            )
        elif isinstance(finish_reason_raw, str):
            finish_reason = finish_reason_raw
        choices = [
            CompletionChoice(
                text=output_text,
                index=0,
                logprobs=None,
                finish_reason=finish_reason,
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

    @classmethod
    def _filter_sampling_params(cls, sampling_params: dict):
        if not sampling_params.get("lora_name"):
            sampling_params.pop("lora_name", None)
        return sampling_params

    async def _stream_generate(
        self,
        prompt: str,
        image_data: Optional[Union[List[str], str]] = None,
        **sampling_params,
    ):
        import aiohttp

        sampling_params = self._filter_sampling_params(sampling_params)
        json_data = {
            "text": prompt,
            "image_data": image_data,
            "sampling_params": sampling_params,
            "stream": True,
        }
        pos = 0

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.post(
                self._engine.generate_url, json=json_data  # type: ignore
            ) as response:
                async for chunk, _ in response.content.iter_chunks():
                    chunk = chunk.decode("utf-8")
                    if chunk and chunk.startswith("data:"):
                        stop = "data: [DONE]\n\n"
                        need_stop = False
                        if chunk.endswith(stop):
                            chunk = chunk[: -len(stop)]
                            need_stop = True
                        if chunk:
                            data = json.loads(chunk[5:].strip("\n"))
                            cur = data["text"][pos:]
                            if cur:
                                yield data["meta_info"], cur
                            pos += len(cur)
                            if need_stop:
                                break

    async def _non_stream_generate(
        self,
        prompt: str,
        image_data: Optional[Union[List[str], str]] = None,
        **sampling_params,
    ) -> dict:
        import aiohttp

        sampling_params = self._filter_sampling_params(sampling_params)
        json_data = {
            "text": prompt,
            "image_data": image_data,
            "sampling_params": sampling_params,
        }
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.post(
                self._engine.generate_url, json=json_data  # type: ignore
            ) as response:
                return await response.json()

    async def async_generate(
        self,
        prompt: str,
        *,
        image_data: Optional[Union[List[str], str]] = None,
        generate_config: Optional[SGLANGGenerateConfig] = None,
        tools: Optional[List[Dict]] = None,
        request_id: Optional[str] = None,
    ) -> Union[Completion, AsyncGenerator[CompletionChunk, None]]:
        sanitized_generate_config = self._sanitize_generate_config(generate_config)
        logger.debug(
            "Enter generate, prompt: %s, generate config: %s", prompt, generate_config
        )
        stream = sanitized_generate_config.pop("stream")
        stream_options = sanitized_generate_config.pop("stream_options")

        include_usage = (
            stream_options.pop("include_usage")
            if isinstance(stream_options, dict)
            else False
        )
        if not request_id:
            request_id = str(uuid.uuid1())
        if not stream:
            state = await self._non_stream_generate(
                prompt, image_data, **sanitized_generate_config
            )
            return self._convert_state_to_completion(
                request_id,
                model=self.model_uid,
                output_text=state["text"],
                meta_info=state["meta_info"],
            )
        else:

            async def stream_results() -> AsyncGenerator[CompletionChunk, None]:
                prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
                complete_response = ""
                match_tool_call_tmp_results: List[CompletionChunk] = []
                is_match_tool_call = False
                chunk = None
                finish_reason = None
                async for meta_info, out in self._stream_generate(
                    prompt, image_data, **sanitized_generate_config
                ):
                    chunk = self._convert_state_to_completion_chunk(
                        request_id,
                        self.model_uid,
                        output_text=out,
                        meta_info=meta_info,
                    )
                    complete_response += out
                    finish_reason = meta_info["finish_reason"]
                    prompt_tokens = meta_info["prompt_tokens"]
                    completion_tokens = meta_info["completion_tokens"]
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
                        if (
                            len(QWEN_TOOL_CALL_SYMBOLS[0]) > len(complete_response)
                        ) and (
                            not QWEN_TOOL_CALL_SYMBOLS[0].startswith(complete_response)
                        ):
                            for c in match_tool_call_tmp_results:
                                yield c
                            match_tool_call_tmp_results.clear()
                            yield chunk
                        elif (
                            len(QWEN_TOOL_CALL_SYMBOLS[0]) > len(complete_response)
                        ) and (QWEN_TOOL_CALL_SYMBOLS[0].startswith(complete_response)):
                            match_tool_call_tmp_results.append(chunk)
                        else:
                            assert len(QWEN_TOOL_CALL_SYMBOLS[0]) <= len(
                                complete_response
                            )
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

                finish_reason = (
                    "stop"
                    if finish_reason is None
                    or (
                        isinstance(finish_reason, str)
                        and finish_reason.lower() == "none"
                    )
                    else finish_reason
                )
                yield generate_completion_chunk(
                    "",
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

            return stream_results()


class SGLANGChatModel(SGLANGModel, ChatModelMixin):
    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if llm_spec.model_format not in ["pytorch", "gptq", "awq", "fp8", "bnb"]:
            return (
                False,
                "SGLang chat engine supports pytorch/gptq/awq/fp8/bnb formats only",
            )
        if llm_spec.model_format == "fp4":
            return (
                False,
                "SGLang chat engine does not support fp4 online quantization; use offline fp4 weights with a compatible SGLang version",
            )
        if llm_spec.model_format == "pytorch":
            if quantization not in (None, "none"):
                return (
                    False,
                    "pytorch format with quantization is not supported by SGLang chat",
                )
        if not llm_family.matches_supported_architectures(SGLANG_SUPPORTED_CHAT_MODELS):
            return (
                False,
                f"Model architectures {llm_family.architectures} are not supported by SGLang chat",
            )
        if "chat" not in llm_family.model_ability:
            return False, "SGLang chat engine requires chat ability"
        if not SGLANG_INSTALLED:
            return False, "sglang library is not installed"
        return True

    def _sanitize_chat_config(
        self,
        generate_config: Optional[Dict] = None,
    ) -> Dict:
        if not generate_config:
            generate_config = {}
        if self.model_family.stop:
            if (not generate_config.get("stop")) and self.model_family.stop:
                generate_config["stop"] = self.model_family.stop.copy()
        generate_config.pop("chat_template_kwargs", None)
        return generate_config

    @staticmethod
    def is_tool_call_chunk_start(chunk):
        return chunk["choices"][0]["text"].startswith(QWEN_TOOL_CALL_SYMBOLS[0])

    @staticmethod
    def is_tool_call_chunk_end(chunk):
        return chunk["choices"][0]["text"].endswith(QWEN_TOOL_CALL_SYMBOLS[1])

    async def async_chat(
        self,
        messages: List[Dict],
        generate_config: Optional[Dict] = None,
        request_id: Optional[str] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        assert self.model_family.chat_template is not None
        # fix: Object of type list_iterator is not JSON serializable
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
                or model_family in DEEPSEEK_TOOL_CALL_FAMILY
            ):
                full_context_kwargs["tools"] = tools
        full_prompt = self.get_full_context(
            messages, self.model_family.chat_template, **full_context_kwargs
        )
        generate_config = self._sanitize_chat_config(generate_config)
        stream = generate_config.get("stream", None)
        if stream:
            agen = await self.async_generate(full_prompt, generate_config=generate_config, tools=tools)  # type: ignore
            assert isinstance(agen, AsyncGenerator)
            if tools:
                return self._async_to_tool_completion_chunks(agen)
            return self._async_to_chat_completion_chunks(
                agen, self.reasoning_parser, chat_template_kwargs
            )
        else:
            c = await self.async_generate(full_prompt, generate_config=generate_config, tools=tools)  # type: ignore
            assert not isinstance(c, AsyncGenerator)
            if tools:
                return self._post_process_completion(
                    self.model_family, self.model_uid, c
                )
            return self._to_chat_completion(c, self.reasoning_parser)


class SGLANGVisionModel(SGLANGModel, ChatModelMixin):
    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if not cls._has_cuda_device():
            return False, "SGLang vision engine requires CUDA GPUs"
        if not cls._is_linux():
            return False, "SGLang vision engine is only supported on Linux"
        if llm_spec.model_format not in ["pytorch", "gptq", "awq", "fp8", "bnb"]:
            return (
                False,
                "SGLang vision engine supports pytorch/gptq/awq/fp8/bnb formats only",
            )
        if llm_spec.model_format == "fp4":
            return (
                False,
                "SGLang vision engine does not support fp4 online quantization; use offline fp4 weights with a compatible SGLang version",
            )
        if llm_spec.model_format == "pytorch":
            if quantization not in (None, "none"):
                return (
                    False,
                    "pytorch format with quantization is not supported by SGLang vision",
                )
        if not llm_family.matches_supported_architectures(
            SGLANG_SUPPORTED_VISION_MODEL_LIST
        ):
            return (
                False,
                f"Model architectures {llm_family.architectures} are not supported by SGLang vision",
            )
        if "vision" not in llm_family.model_ability:
            return False, "SGLang vision engine requires vision ability"
        if not SGLANG_INSTALLED:
            return False, "sglang library is not installed"
        return True

    def _sanitize_chat_config(
        self,
        generate_config: Optional[Dict] = None,
    ) -> Dict:
        if not generate_config:
            generate_config = {}
        if self.model_family.stop:
            if (not generate_config.get("stop")) and self.model_family.stop:
                generate_config["stop"] = self.model_family.stop.copy()
        return generate_config

    async def async_chat(
        self,
        messages: List[ChatCompletionMessage],  # type: ignore
        generate_config: Optional[Dict] = None,
        request_id: Optional[str] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        import base64
        from io import BytesIO

        from PIL import Image
        from qwen_vl_utils import process_vision_info

        messages = self._transform_messages(messages)

        chat_template: str = (
            self.model_family.chat_template if self.model_family.chat_template else ""
        )
        chat_template_kwargs = (
            self._get_chat_template_kwargs_from_generate_config(
                generate_config, self.reasoning_parser
            )
            or {}
        )
        chat_context_var.set(chat_template_kwargs)
        full_context_kwargs = chat_template_kwargs.copy()
        prompt = self.get_full_context(messages, chat_template, **full_context_kwargs)

        images, video_inputs = process_vision_info(messages)
        if video_inputs:
            raise ValueError("Not support video input now.")

        base64_images: Optional[List[str]] = None
        if images:
            base64_images = []
            for image in images:
                if isinstance(image, Image.Image):
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG", quality=100)
                    base64_images.append(base64.b64encode(buffered.getvalue()).decode())
                elif isinstance(image, str):
                    base64_images.append(image)
                else:
                    raise ValueError(
                        f"Unsupported image type: {type(image)}, only support PIL.Image and base64 string"
                    )

        generate_config = self._sanitize_chat_config(generate_config)
        stream = generate_config.get("stream", None)
        if stream:
            agen = await self.async_generate(prompt, image_data=base64_images, generate_config=generate_config)  # type: ignore
            assert isinstance(agen, AsyncGenerator)
            return self._async_to_chat_completion_chunks(agen, self.reasoning_parser)
        else:
            c = await self.async_generate(prompt, image_data=base64_images, generate_config=generate_config)  # type: ignore
            assert not isinstance(c, AsyncGenerator)
            return self._to_chat_completion(c, self.reasoning_parser)
