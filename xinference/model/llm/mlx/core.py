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
from ..utils import (
    DEEPSEEK_TOOL_CALL_FAMILY,
    QWEN_TOOL_CALL_FAMILY,
    ChatModelMixin,
    generate_completion_chunk,
)

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
        generate_config.setdefault("logit_bias", None)
        generate_config.setdefault("repetition_penalty", None)
        generate_config.setdefault("repetition_context_size", 20)
        generate_config.setdefault("top_p", 1.0)
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
        if "chat" in llm_family.model_ability or "vision" in llm_family.model_ability:
            # do not process chat or vision
            return False
        return True

    def _get_prompt_cache(
        self, prompt, lora_name: Optional[str] = None, model: Any = None
    ):
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
            self._prompt_cache.cache = make_prompt_cache(
                model or self._model, self._max_kv_size
            )
            self._prompt_cache.tokens = []
            logger.debug("Making new prompt cache for %s", self.model_uid)
        else:
            prompt = prompt[cache_len:]
            logger.debug("Cache hit for %s", self.model_uid)
        self._prompt_cache.tokens.extend(prompt)
        return prompt

    def _generate_stream_inner(self, **kwargs):
        from mlx_lm.utils import make_logits_processors, make_sampler, stream_generate

        sampler = make_sampler(
            temp=kwargs.pop("temperature"), top_p=kwargs.pop("top_p")
        )
        prompt_token_ids = kwargs.pop("prompt_token_ids")
        logits_processors = make_logits_processors(
            logit_bias=kwargs.pop("logits_bias", None),
            repetition_penalty=kwargs.pop("repetition_penalty"),
            repetition_context_size=kwargs.pop("repetition_context_size"),
        )
        yield from stream_generate(
            self._model,
            self._tokenizer,
            prompt_token_ids,
            sampler=sampler,
            logits_processors=logits_processors,
            **kwargs,
        )

    def _prepare_inputs(
        self, prompt: Union[str, Dict[str, Any]], kwargs
    ) -> Tuple[Any, int]:
        prompt_token_ids = self._tokenizer.encode(prompt)
        prompt_token_ids = self._get_prompt_cache(
            prompt_token_ids, kwargs.get("lora_name")
        )
        return prompt_token_ids, len(prompt_token_ids)

    def _generate_stream(
        self, prompt: Union[str, Dict[str, Any]], kwargs: MLXGenerateConfig
    ):
        model_uid = self.model_uid
        tokenizer = self._tokenizer
        max_tokens = kwargs["max_tokens"]
        chunk_id = str(uuid.uuid4())
        stop_token_ids = kwargs.get("stop_token_ids", [])
        stream = kwargs.get("stream", False)
        stream_options = kwargs.pop("stream_options", None)
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )

        prompt_token_ids, input_echo_len = self._prepare_inputs(prompt, kwargs)

        i = 0
        start = time.time()
        output = ""
        tokens = []
        for chunk_resp, i in zip(
            self._generate_stream_inner(
                prompt_token_ids=prompt_token_ids,
                max_tokens=max_tokens,
                temperature=kwargs["temperature"],
                top_p=kwargs["top_p"],
                repetition_penalty=kwargs["repetition_penalty"],
                repetition_context_size=kwargs["repetition_context_size"],
                prompt_cache=self._prompt_cache.cache if self._prompt_cache else None,  # type: ignore
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

        if self._prompt_cache:
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
        self,
        prompt: Union[str, Dict[str, Any]],
        generate_config: Optional[MLXGenerateConfig] = None,
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        def generator_wrapper(
            prompt: Union[str, Dict[str, Any]], generate_config: MLXGenerateConfig
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
        if "vision" in llm_family.model_ability:
            # do not process vision
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
        if tools:
            if model_family in QWEN_TOOL_CALL_FAMILY:
                full_context_kwargs["tools"] = tools
            elif model_family in DEEPSEEK_TOOL_CALL_FAMILY:
                self._tools_to_messages_for_deepseek(messages, tools)
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


class MLXVisionModel(MLXModel, ChatModelMixin):
    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if llm_spec.model_format not in ["mlx"]:
            return False
        if sys.platform != "darwin" or platform.processor() != "arm":
            # only work for Mac M chips
            return False
        if "vision" not in llm_family.model_ability:
            return False
        return True

    def _load_model(self, **kwargs):
        try:
            from mlx_vlm import load
        except ImportError:
            error_message = "Failed to import module 'mlx_vlm'"
            installation_guide = [
                "Please make sure 'mlx_vlm' is installed. ",
                "You can install it by `pip install mlx_vlm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        self._prompt_cache = PromptCache()

        return load(self.model_path)

    def load(self):
        kwargs = {}
        kwargs["revision"] = self._model_config.get(
            "revision", self.model_spec.model_revision
        )
        kwargs["trust_remote_code"] = self._model_config.get("trust_remote_code")
        kwargs["cache_limit_gb"] = self._model_config.pop("cache_limit_gb", None)

        self._model, self._processor = self._load_model(**kwargs)
        self._tokenizer = self._processor.tokenizer

    def _generate_stream_inner(self, **kwargs):
        import mlx.core as mx
        from mlx_lm.utils import GenerationResponse
        from mlx_vlm.utils import generate_step

        inputs = kwargs["prompt_token_ids"]

        max_tokens = kwargs.pop("max_tokens")
        input_ids, pixel_values, mask, kwargs = inputs

        tokenizer = self._processor.tokenizer
        detokenizer = self._processor.detokenizer

        detokenizer.reset()
        tic = time.perf_counter()
        for (token, logprobs), n in zip(
            generate_step(input_ids, self._model, pixel_values, mask, **kwargs),
            range(max_tokens),
        ):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = len(input_ids) / prompt_time
                tic = time.perf_counter()
            if token == tokenizer.eos_token_id:
                break
            detokenizer.add_token(token)

            # Yield the last segment if streaming
            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                prompt_tokens=len(input_ids),
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.metal.get_peak_memory() / 1e9,
            )

        detokenizer.finalize()
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            prompt_tokens=len(input_ids),
            prompt_tps=prompt_tps,
            generation_tokens=n + 1,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.metal.get_peak_memory() / 1e9,
        )

    def _prepare_inputs(
        self, prompt: Union[str, Dict[str, Any]], kwargs
    ) -> Tuple[Any, int]:
        import mlx.core as mx
        from mlx_vlm import prepare_inputs

        prompt_str = prompt.get("prompt")  # type: ignore
        images = prompt.get("multi_modal_data", {}).get("image")  # type: ignore
        if images and not isinstance(images, list):
            images = [images]
        resize_shape = kwargs.pop("resize_shape", None)
        image_token_index = getattr(self._model.config, "image_token_index", None)

        processor = self._processor
        tokenizer = processor if hasattr(processor, "encode") else processor.tokenizer
        prompt_tokens = mx.array(tokenizer.encode(prompt_str))

        if not images:
            input_ids = prompt_tokens[None, :]
            pixel_values = mask = None
            kwargs = {}
            input_token_len = input_ids.size
        else:
            inputs = prepare_inputs(
                processor, images, prompt_str, image_token_index, resize_shape
            )
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            mask = inputs["attention_mask"]
            kwargs = {
                k: v
                for k, v in inputs.items()
                if k not in ["input_ids", "pixel_values", "attention_mask"]
            }
            input_token_len = int(mask.sum())
        return (input_ids, pixel_values, mask, kwargs), input_token_len

    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[MLXGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        messages = self._transform_messages(messages)  # type: ignore
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
            prompt, images = self.get_specific_prompt(model_family, messages)  # type: ignore

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
        generate_config = self._sanitize_generate_config(generate_config)

        stream = generate_config.get("stream", False)
        if stream:
            it = self.generate(inputs, generate_config)
            assert isinstance(it, Iterator)
            return self._to_chat_completion_chunks(it)
        else:
            c = self.generate(inputs, generate_config)
            assert not isinstance(c, Iterator)
            if tools:
                return self._tool_calls_completion(self.model_family, self.model_uid, c)
            return self._to_chat_completion(c)
