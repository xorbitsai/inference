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
import datetime
import logging
import os
import time
from typing import Iterable, Iterator, List, Optional, Union

from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChunk,
    CompletionUsage,
    CreateCompletionLlamaCpp,
    LlamaCppGenerateConfig,
    LlamaCppModelConfig,
)
from ..core import LLM
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import QWEN_TOOL_CALL_FAMILY, ChatModelMixin

logger = logging.getLogger(__name__)


class LlamaCppModel(LLM):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        llamacpp_model_config: Optional[LlamaCppModelConfig] = None,
    ):
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)

        self._llamacpp_model_config: LlamaCppModelConfig = self._sanitize_model_config(
            llamacpp_model_config
        )
        self._llm = None

    def _can_apply_cublas(self):
        # TODO: figure out the quantizations supported.
        return True

    def _sanitize_model_config(
        self, llamacpp_model_config: Optional[LlamaCppModelConfig]
    ) -> LlamaCppModelConfig:
        if llamacpp_model_config is None:
            llamacpp_model_config = LlamaCppModelConfig()

        if self.model_family.context_length:
            llamacpp_model_config.setdefault("n_ctx", self.model_family.context_length)
        llamacpp_model_config.setdefault("use_mmap", False)
        llamacpp_model_config.setdefault("use_mlock", True)

        if (
            "llama-2" in self.model_family.model_name
            and self.model_spec.model_size_in_billions == 70
        ):
            llamacpp_model_config["use_mlock"] = False
            llamacpp_model_config["n_gqa"] = 8

        if self._is_darwin_and_apple_silicon():
            llamacpp_model_config.setdefault("n_gpu_layers", -1)
        elif self._is_linux() and self._can_apply_cublas():
            llamacpp_model_config.setdefault("n_gpu_layers", -1)

        return llamacpp_model_config

    def _sanitize_generate_config(
        self, generate_config: Optional[LlamaCppGenerateConfig]
    ) -> LlamaCppGenerateConfig:
        if generate_config is None:
            generate_config = LlamaCppGenerateConfig(
                **CreateCompletionLlamaCpp().dict()
            )
        else:
            from llama_cpp import LlamaGrammar

            grammar = generate_config.get("grammar")
            if grammar is not None and not isinstance(grammar, LlamaGrammar):
                generate_config["grammar"] = LlamaGrammar.from_string(
                    generate_config["grammar"]
                )
            # Validate generate_config and fill default values to the generate config.
            generate_config = LlamaCppGenerateConfig(
                **CreateCompletionLlamaCpp(**generate_config).dict()
            )
        # Currently, llama.cpp does not support lora
        generate_config.pop("lora_name", None)  # type: ignore
        return generate_config

    def _convert_ggml_to_gguf(self, model_path: str) -> str:
        from .tools import convert

        root_dir = os.path.dirname(os.path.dirname(model_path))
        gguf_dir = os.path.join(
            root_dir,
            "{}-ggufv2-{}b".format(
                self.model_family.model_name, self.model_spec.model_size_in_billions
            ),
        )
        os.makedirs(gguf_dir, exist_ok=True)
        gguf_path = os.path.join(
            gguf_dir,
            "{}.{}.ggufv2".format(self.model_family.model_name, self.quantization),
        )
        # trick for validation, use a mark file to make sure the gguf file is converted
        mark_file = os.path.join(gguf_dir, f"__valid_{self.quantization}")
        if os.path.exists(mark_file):
            return gguf_path
        else:
            logger.warning(
                "You are using a model with ggmlv3, "
                "and it will take some time to convert to ggufv2"
            )
            convert(model_path, gguf_path)
            with open(mark_file, "w") as f:
                f.write(str(datetime.datetime.now()))
            return gguf_path

    def load(self):
        try:
            import llama_cpp
            from llama_cpp import Llama

            if llama_cpp.__version__ < "0.2.0":
                raise ValueError(
                    "The llama_cpp version must be greater than 0.2.0. "
                    "Please upgrade your version via `pip install -U llama_cpp` or refer to "
                    "https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal."
                )
        except ImportError:
            error_message = "Failed to import module 'llama_cpp'"
            installation_guide = [
                "Please make sure 'llama_cpp' is installed. ",
                "You can install it by visiting the installation section of the git repo:\n",
                "https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        # handle legacy cache.
        model_path = os.path.join(
            self.model_path,
            self.model_spec.model_file_name_template.format(
                quantization=self.quantization
            ),
        )
        legacy_model_file_path = os.path.join(self.model_path, "model.bin")
        if os.path.exists(legacy_model_file_path):
            model_path = legacy_model_file_path

        if self.model_spec.model_format == "ggmlv3":
            model_path = self._convert_ggml_to_gguf(model_path)

        try:
            self._llm = Llama(
                model_path=model_path,
                verbose=True,
                **self._llamacpp_model_config,
            )
        except AssertionError:
            raise RuntimeError(f"Load model {self.model_family.model_name} failed")

    @classmethod
    def match(
        cls, llm_family: LLMFamilyV1, llm_spec: LLMSpecV1, quantization: str
    ) -> bool:
        if llm_spec.model_format not in ["ggmlv3", "ggufv2"]:
            return False
        if "qwen" in llm_family.model_name:
            return False
        if "generate" not in llm_family.model_ability:
            return False
        return True

    def generate(
        self, prompt: str, generate_config: Optional[LlamaCppGenerateConfig] = None
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        def generator_wrapper(
            _prompt: str,
            _generate_config: LlamaCppGenerateConfig,
        ) -> Iterator[CompletionChunk]:
            assert self._llm is not None
            prompt_token_ids: List[int] = (
                (
                    self._llm.tokenize(prompt.encode("utf-8"), special=True)
                    if prompt != ""
                    else [self._llm.token_bos()]
                )
                if isinstance(prompt, str)
                else prompt
            )
            prompt_tokens = len(prompt_token_ids)
            completion_tokens, total_tokens = 0, 0
            request_id = 0
            for index, _completion_chunk in enumerate(
                self._llm(prompt=_prompt, **_generate_config)
            ):
                request_id = _completion_chunk["id"]
                choice = _completion_chunk["choices"][0]
                if choice["finish_reason"] is not None:
                    completion_tokens = index
                total_tokens = prompt_tokens + completion_tokens
                _completion_chunk["usage"] = CompletionUsage(
                    prompt_tokens=total_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
                yield _completion_chunk
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

        logger.debug(
            "Enter generate, prompt: %s, generate config: %s", prompt, generate_config
        )

        generate_config = self._sanitize_generate_config(generate_config)
        stream = generate_config.get("stream", False)
        stream_options = generate_config.pop("stream_options", None)
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )

        if not stream:
            assert self._llm is not None
            completion = self._llm(prompt=prompt, **generate_config)

            return completion
        else:
            return generator_wrapper(prompt, generate_config)


class LlamaCppChatModel(LlamaCppModel, ChatModelMixin):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        llamacpp_model_config: Optional[LlamaCppModelConfig] = None,
    ):
        super().__init__(
            model_uid,
            model_family,
            model_spec,
            quantization,
            model_path,
            llamacpp_model_config,
        )

    @classmethod
    def match(
        cls, llm_family: LLMFamilyV1, llm_spec: LLMSpecV1, quantization: str
    ) -> bool:
        if llm_spec.model_format not in ["ggmlv3", "ggufv2"]:
            return False
        if "chat" not in llm_family.model_ability:
            return False
        return True

    def _sanitize_generate_config(
        self, generate_config: Optional[LlamaCppGenerateConfig]
    ) -> LlamaCppGenerateConfig:
        generate_config = super()._sanitize_generate_config(generate_config)
        if self.model_family.prompt_style and self.model_family.prompt_style.stop:
            generate_config["stop"] = self.model_family.prompt_style.stop
        return generate_config

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[LlamaCppGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        assert self.model_family.prompt_style is not None
        prompt_style = self.model_family.prompt_style.copy()
        if system_prompt:
            prompt_style.system_prompt = system_prompt

        chat_history = chat_history or []
        assert prompt_style is not None
        tools = generate_config.pop("tools", []) if generate_config else None
        full_prompt = self.get_prompt(prompt, chat_history, prompt_style, tools=tools)

        generate_config = self._sanitize_generate_config(generate_config)
        # TODO(codingl2k1): qwen hacky to set stop for function call.
        model_family = self.model_family.model_family or self.model_family.model_name
        if tools and model_family in QWEN_TOOL_CALL_FAMILY:
            stop = generate_config.get("stop")
            if isinstance(stop, str):
                generate_config["stop"] = [stop, "Observation:"]
            elif isinstance(stop, Iterable):
                assert not isinstance(stop, str)
                generate_config["stop"] = stop + ["Observation:"]  # type: ignore
            else:
                generate_config["stop"] = "Observation:"

        stream = generate_config.get("stream", False)
        if stream:
            it = self.generate(full_prompt, generate_config)
            assert isinstance(it, Iterator)
            return self._to_chat_completion_chunks(it)
        else:
            c = self.generate(full_prompt, generate_config)
            assert not isinstance(c, Iterator)
            if tools:
                return self._tool_calls_completion(
                    self.model_family, self.model_uid, c, tools
                )
            return self._to_chat_completion(c)
