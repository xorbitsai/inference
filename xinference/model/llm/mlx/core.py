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

import logging
import platform
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypedDict, Union

from ....fields import max_tokens_field
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    Completion,
    CompletionChunk,
    CompletionUsage,
    LoRA,
)
from ..core import LLM
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import QWEN_TOOL_CALL_FAMILY, ChatModelMixin, generate_completion_chunk

logger = logging.getLogger(__name__)


class MLXModelConfig(TypedDict, total=False):
    revision: Optional[str]
    max_gpu_memory: str
    trust_remote_code: bool


class MLXGenerateConfig(TypedDict, total=False):
    max_tokens: int
    temperature: float
    repetition_penalty: Optional[float]
    repetition_context_size: Optional[float]
    top_p: float
    logit_bias: Optional[Dict[int, float]]
    stop: Optional[Union[str, List[str]]]
    stop_token_ids: Optional[Union[int, List[int]]]
    stream: bool
    stream_options: Optional[Union[dict, None]]
    tools: Optional[List[Dict]]
    lora_name: Optional[str]


@dataclass
class PromptCache:
    cache: List[Any] = field(default_factory=list)
    model_key: Tuple[str, Optional[str]] = ("", None)
    tokens: List[int] = field(default_factory=list)


class MLXModel(LLM):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        model_config: Optional[MLXModelConfig] = None,
        peft_model: Optional[List[LoRA]] = None,
    ):
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)
        self._use_fast_tokenizer = True
        self._model_config: MLXModelConfig = self._sanitize_model_config(model_config)
        self._max_kv_size = None
        self._prompt_cache = None
        if peft_model is not None:
            raise ValueError("MLX engine has not supported lora yet")

    def _sanitize_model_config(
        self, model_config: Optional[MLXModelConfig]
    ) -> MLXModelConfig:
        if model_config is None:
            model_config = MLXModelConfig()
        model_config.setdefault("revision", self.model_spec.model_revision)
        model_config.setdefault("trust_remote_code", True)
        return model_config

    def _sanitize_generate_config(
        self,
        generate_config: Optional[MLXGenerateConfig],
    ) -> MLXGenerateConfig:
        if generate_config is None:
            generate_config = MLXGenerateConfig()

        generate_config.setdefault("max_tokens", max_tokens_field.default)
        # default config is adapted from
        # https://github.com/ml-explore/mlx-examples/blob/f212b770d8b5143e23102eda20400ae43340f844/llms/mlx_lm/utils.py#L129
        generate_config.setdefault("temperature", 0.0)
        generate_config.setdefault("repetition_penalty", None)
        generate_config.setdefault("repetition_context_size", 20)
        generate_config.setdefault("top_p", 1.0)
        generate_config.setdefault("logit_bias", None)
        return generate_config

    def _load_model(self, **kwargs):
        try:
            import mlx.core as mx
            from mlx_lm import load
        except ImportError:
            error_message = "Failed to import module 'mlx_lm'"
            installation_guide = [
                "Please make sure 'mlx_lm' is installed. ",
                "You can install it by `pip install mlx_lm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        tokenizer_config = dict(
            use_fast=self._use_fast_tokenizer,
            trust_remote_code=kwargs["trust_remote_code"],
            revision=kwargs["revision"],
        )
        logger.debug(
            "loading model with tokenizer config: %s, model config: %s",
            tokenizer_config,
            self._model_config,
        )

        cache_limit_gb = kwargs.get("cache_limit_gb", None)
        if cache_limit_gb:
            logger.debug(f"Setting cache limit to {cache_limit_gb} GB")
            mx.metal.set_cache_limit(cache_limit_gb * 1024 * 1024 * 1024)

        self._max_kv_size = kwargs.get("max_kv_size", None)
        self._prompt_cache = PromptCache()

        return load(
            self.model_path,
            tokenizer_config=tokenizer_config,
            model_config=self._model_config,
        )

    def load(self):
        kwargs = {}
        kwargs["revision"] = self._model_config.get(
            "revision", self.model_spec.model_revision
        )
        kwargs["trust_remote_code"] = self._model_config.get("trust_remote_code")
        kwargs["cache_limit_gb"] = self._model_config.pop("cache_limit_gb", None)

        self._model, self._tokenizer = self._load_model(**kwargs)

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if llm_spec.model_format not in ["mlx"]:
            return False
        if sys.platform != "darwin" or platform.processor() != "arm":
            # only work for Mac M chips
            return False
        if "generate" not in llm_family.model_ability:
            return False
        return True

    def _get_prompt_cache(self, prompt, lora_name: Optional[str] = None):
        from mlx_lm.models.cache import make_prompt_cache

        assert self._prompt_cache is not None
        cache_len = len(self._prompt_cache.tokens)
        model_key = (self.model_path, lora_name)
        if (
            self._prompt_cache.model_key != model_key
            or cache_len >= len(prompt)
            or self._prompt_cache.tokens != prompt[:cache_len]
        ):
            self._prompt_cache.model_key = model_key
            self._prompt_cache.cache = make_prompt_cache(self._model, self._max_kv_size)
            self._prompt_cache.tokens = []
            logger.debug("Making new prompt cache for %s", self.model_uid)
        else:
            prompt = prompt[cache_len:]
            logger.debug("Cache hit for %s", self.model_uid)
        self._prompt_cache.tokens.extend(prompt)
        return prompt

    def _generate_stream(self, prompt: str, kwargs: MLXGenerateConfig):
        from mlx_lm.utils import make_sampler, stream_generate

        model = self._model
        model_uid = self.model_uid
        tokenizer = self._tokenizer
        max_tokens = kwargs["max_tokens"]
        chunk_id = str(uuid.uuid4())
        stop_token_ids = kwargs.get("stop_token_ids", [])
        stream = kwargs.get("stream", False)
        lora_name = kwargs.get("lora_name")
        stream_options = kwargs.pop("stream_options", None)
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )

        prompt_token_ids = tokenizer.encode(prompt)
        prompt_token_ids = self._get_prompt_cache(prompt_token_ids, lora_name)
        input_echo_len = len(prompt_token_ids)

        i = 0
        start = time.time()
        output = ""
        tokens = []
        sampler = make_sampler(temp=kwargs["temperature"], top_p=kwargs["top_p"])
        for chunk_resp, i in zip(
            stream_generate(
                model,
                tokenizer,
                prompt_token_ids,
                max_tokens=max_tokens,
                sampler=sampler,
                repetition_penalty=kwargs["repetition_penalty"],
                repetition_context_size=kwargs["repetition_context_size"],
                prompt_cache=self._prompt_cache.cache,  # type: ignore
            ),
            range(max_tokens),
        ):
            token = chunk_resp.token
            tokens.append(token)

            out = chunk_resp.text
            if stream:
                # this special character is mainly for qwen
                out = out.strip("�")
                output = out
            else:
                output += out

            completion_usage = CompletionUsage(
                prompt_tokens=input_echo_len,
                completion_tokens=i,
                total_tokens=(input_echo_len + i),
            )

            yield generate_completion_chunk(
                chunk_text=output,
                finish_reason=None,
                chunk_id=chunk_id,
                model_uid=model_uid,
                prompt_tokens=input_echo_len,
                completion_tokens=i,
                total_tokens=(input_echo_len + i),
            ), completion_usage

            if token == tokenizer.eos_token_id or token in stop_token_ids:  # type: ignore
                break

        logger.info(
            f"Average generation speed: {i / (time.time() - start):.2f} tokens/s."
        )

        self._prompt_cache.tokens.extend(tokens)  # type: ignore

        if i == max_tokens - 1:
            finish_reason = "length"
        else:
            finish_reason = "stop"

        completion_usage = CompletionUsage(
            prompt_tokens=input_echo_len,
            completion_tokens=i,
            total_tokens=(input_echo_len + i),
        )
        if stream:
            yield generate_completion_chunk(
                "",
                finish_reason=finish_reason,
                chunk_id=chunk_id,
                model_uid=model_uid,
                prompt_tokens=input_echo_len,
                completion_tokens=i,
                total_tokens=(input_echo_len + i),
            ), completion_usage
        else:
            yield generate_completion_chunk(
                output,
                finish_reason=finish_reason,
                chunk_id=chunk_id,
                model_uid=model_uid,
                prompt_tokens=input_echo_len,
                completion_tokens=i,
                total_tokens=(input_echo_len + i),
            ), completion_usage

        if include_usage:
            completion_chunk = CompletionChunk(
                id=chunk_id,
                object="text_completion",
                created=int(time.time()),
                model=model_uid,
                choices=[],
            )
            yield completion_chunk, completion_usage

    def generate(
        self, prompt: str, generate_config: Optional[MLXGenerateConfig] = None
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        def generator_wrapper(
            prompt: str, generate_config: MLXGenerateConfig
        ) -> Iterator[CompletionChunk]:
            for completion_chunk, completion_usage in self._generate_stream(
                prompt,
                generate_config,
            ):
                completion_chunk["usage"] = completion_usage
                yield completion_chunk

        logger.debug(
            "Enter generate, prompt: %s, generate config: %s", prompt, generate_config
        )

        generate_config = self._sanitize_generate_config(generate_config)

        assert self._model is not None
        assert self._tokenizer is not None

        stream = generate_config.get("stream", False)
        if not stream:
            for completion_chunk, completion_usage in self._generate_stream(
                prompt,
                generate_config,
            ):
                pass
            completion = Completion(
                id=completion_chunk["id"],
                object=completion_chunk["object"],
                created=completion_chunk["created"],
                model=completion_chunk["model"],
                choices=completion_chunk["choices"],
                usage=completion_usage,
            )
            return completion
        else:
            return generator_wrapper(prompt, generate_config)


class MLXChatModel(MLXModel, ChatModelMixin):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        model_config: Optional[MLXModelConfig] = None,
        peft_model: Optional[List[LoRA]] = None,
    ):
        super().__init__(
            model_uid,
            model_family,
            model_spec,
            quantization,
            model_path,
            model_config,
            peft_model,
        )

    def _sanitize_generate_config(
        self,
        generate_config: Optional[MLXGenerateConfig],
    ) -> MLXGenerateConfig:
        generate_config = super()._sanitize_generate_config(generate_config)
        if (not generate_config.get("stop")) and self.model_family.stop:
            generate_config["stop"] = self.model_family.stop.copy()
        if (
            generate_config.get("stop_token_ids", None) is None
            and self.model_family.stop_token_ids
        ):
            generate_config["stop_token_ids"] = self.model_family.stop_token_ids.copy()

        return generate_config

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if llm_spec.model_format not in ["mlx"]:
            return False
        if sys.platform != "darwin" or platform.processor() != "arm":
            # only work for Mac M chips
            return False
        if "chat" not in llm_family.model_ability:
            return False
        return True

    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[MLXGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        model_family = self.model_family.model_family or self.model_family.model_name
        tools = generate_config.pop("tools", []) if generate_config else None
        full_context_kwargs = {}
        if tools and model_family in QWEN_TOOL_CALL_FAMILY:
            full_context_kwargs["tools"] = tools
        assert self.model_family.chat_template is not None
        full_prompt = self.get_full_context(
            messages, self.model_family.chat_template, **full_context_kwargs
        )

        generate_config = self._sanitize_generate_config(generate_config)

        stream = generate_config.get("stream", False)
        if stream:
            it = self.generate(full_prompt, generate_config)
            assert isinstance(it, Iterator)
            return self._to_chat_completion_chunks(it)
        else:
            c = self.generate(full_prompt, generate_config)
            assert not isinstance(c, Iterator)
            if tools:
                return self._tool_calls_completion(self.model_family, self.model_uid, c)
            return self._to_chat_completion(c)
