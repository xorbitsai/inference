# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""TensorRT-LLM's Python LLM API, owned by a Xinference model process."""

import importlib.metadata
import logging
import sys
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Optional, Union

from ....constants import XINFERENCE_MAX_TOKENS
from ....types import Completion, CompletionChunk, CompletionUsage
from ...utils import check_dependency_available
from ..context_length import ContextLengthExceeded, validate_context_length
from ..core import LLM
from ..utils import ChatModelMixin

logger = logging.getLogger(__name__)


class TRTLLMModel(LLM, ChatModelMixin):
    allow_batch = True

    def __init__(self, model_uid, model_family, model_path, model_config=None):
        super().__init__(model_uid, model_family, model_path)
        self._model_config = dict(model_config or {})
        self._engine = None
        self._tokenizer = None
        self._requests: Dict[str, Any] = {}
        self._cache_enabled = False

    @classmethod
    def check_lib(cls):
        return check_dependency_available("tensorrt_llm", "tensorrt-llm")

    @classmethod
    def match_json(cls, llm_family, llm_spec, quantization):
        if sys.platform != "linux":
            return False, "TRT-LLM requires Linux and NVIDIA CUDA GPUs"
        if llm_spec.model_format != "pytorch" or quantization not in (None, "none"):
            return False, "TRT-LLM currently requires unquantized PyTorch weights"
        family = llm_family.model_family or llm_family.model_name
        if family != "qwen3" or "vision" in llm_family.model_ability:
            return False, "TRT-LLM currently supports dense Qwen3 text models"
        return True

    def load(self):
        if importlib.metadata.version("tensorrt-llm") != "1.2.1":
            raise ValueError("The TRT-LLM adapter requires tensorrt-llm==1.2.1")
        from tensorrt_llm import LLM as Engine

        config = dict(self._model_config)
        for name in ("address", "shard", "driver_info"):
            config.pop(name, None)
        if config.pop("n_worker", 1) != 1:
            raise ValueError("TRT-LLM currently requires one worker per replica")
        cache_config = config.pop("_xavier_cache_config", None)
        self._enable_thinking = config.pop("enable_thinking", True)
        reasoning = config.pop("reasoning_content", False)
        self.prepare_parse_reasoning_content(reasoning, self._enable_thinking)
        if config.get("max_seq_len") is None:
            config["max_seq_len"] = self.model_family.context_length
        self._context_length = validate_context_length(config["max_seq_len"])
        if config.get("backend", "pytorch") != "pytorch":
            raise ValueError("The TRT-LLM adapter requires the PyTorch backend")
        config["backend"] = "pytorch"
        if cache_config is not None:
            from .xavier import configure_xavier

            configure_xavier(self.model_path, config, cache_config)
            self._cache_enabled = True
        try:
            self._engine = Engine(model=self.model_path, **config)
            self._tokenizer = self._engine.tokenizer
        except BaseException:
            self.stop()
            raise

    def stop(self):
        try:
            for request in list(self._requests.values()):
                try:
                    request.abort()
                except Exception:
                    logger.warning(
                        "Could not abort TRT-LLM request during shutdown", exc_info=True
                    )
            if self._engine is not None:
                self._engine.shutdown()
        finally:
            self._requests.clear()
            self._engine = None
            if self._cache_enabled:
                connector = sys.modules.get("xinference.model.llm.trt_llm.connector")
                if connector is not None:
                    connector.close_clients()

    async def abort_request(self, request_id: str) -> str:
        request = self._requests.get(request_id)
        if request is None:
            return "NO_OP"
        request.abort()
        return "DONE"

    def _sampling_params(self, prompt: str, config: dict):
        from tensorrt_llm import SamplingParams

        if config.get("cache_salt") is not None:
            raise ValueError("TRT-LLM cache_salt is not supported by this adapter yet")
        for name in ("tools", "response_format", "logprobs", "prompt_logprobs"):
            if config.get(name):
                raise ValueError(
                    f"TRT-LLM does not support {name} through this adapter"
                )
        if config.get("n", 1) != 1:
            raise ValueError("TRT-LLM currently requires n=1")
        tokens = self._tokenizer.encode(prompt, add_special_tokens=False)
        remaining = self._context_length - len(tokens)
        if remaining <= 0:
            raise ContextLengthExceeded("The prompt leaves no room for output tokens")
        maximum = config.get("max_tokens", XINFERENCE_MAX_TOKENS)
        if maximum is not None and (type(maximum) is not int or maximum <= 0):
            raise ValueError("max_tokens must be a positive integer")
        options = {
            name: config[name]
            for name in (
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "seed",
                "repetition_penalty",
                "presence_penalty",
                "frequency_penalty",
                "ignore_eos",
                "skip_special_tokens",
            )
            if name in config
        }
        options.update(
            max_tokens=min(maximum, remaining) if maximum is not None else remaining,
            stop=config.get("stop", self.model_family.stop),
            stop_token_ids=config.get(
                "stop_token_ids", self.model_family.stop_token_ids
            ),
        )
        if self.reasoning_parser is not None:
            options.setdefault("skip_special_tokens", False)
        return tokens, SamplingParams(**options)

    @staticmethod
    def _usage(output) -> CompletionUsage:
        prompt = len(output.prompt_token_ids)
        generated = sum(len(choice.token_ids) for choice in output.outputs)
        return CompletionUsage(
            prompt_tokens=prompt,
            completion_tokens=generated,
            total_tokens=prompt + generated,
        )

    async def async_generate(
        self,
        prompt: str,
        generate_config: Optional[dict] = None,
        request_id: Optional[str] = None,
    ) -> Union[Completion, AsyncGenerator[CompletionChunk, None]]:
        config = dict(generate_config or {})
        tokens, params = self._sampling_params(prompt, config)
        request_id = request_id or str(uuid.uuid4())
        created = int(time.time())

        def start():
            if request_id in self._requests:
                raise ValueError("A TRT-LLM request with this ID is already active")
            request = self._engine.generate_async(tokens, params, streaming=True)
            self._requests[request_id] = request
            return request

        def finish(request):
            try:
                if self._engine is not None and not request.finished:
                    request.abort()
            except Exception:
                logger.warning(
                    "Could not abort TRT-LLM request during cleanup", exc_info=True
                )
            finally:
                self._requests.pop(request_id, None)

        async def stream():
            request = start()
            try:
                async for output in request:
                    yield CompletionChunk(
                        id=request_id,
                        object="text_completion",
                        created=created,
                        model=self.model_uid,
                        choices=[
                            dict(
                                text=choice.text_diff,
                                index=choice.index,
                                logprobs=None,
                                finish_reason=choice.finish_reason,
                            )
                            for choice in output.outputs
                        ],
                    )
                if (config.get("stream_options") or {}).get("include_usage"):
                    yield CompletionChunk(
                        id=request_id,
                        object="text_completion",
                        created=created,
                        model=self.model_uid,
                        choices=[],
                        usage=self._usage(request),
                    )
            finally:
                finish(request)

        if config.get("stream", False):
            return stream()
        request = start()
        try:
            output = await request
            return Completion(
                id=request_id,
                object="text_completion",
                created=created,
                model=self.model_uid,
                usage=self._usage(output),
                choices=[
                    dict(
                        text=choice.text,
                        index=choice.index,
                        logprobs=None,
                        finish_reason=choice.finish_reason,
                    )
                    for choice in output.outputs
                ],
            )
        finally:
            finish(request)

    async def async_chat(self, messages, generate_config=None, request_id=None):
        config = dict(generate_config or {})
        if config.get("tools"):
            raise ValueError(
                "TRT-LLM tool calling is not supported by this adapter yet"
            )
        for message in messages:
            if not isinstance(message.get("content"), (str, type(None))):
                raise ValueError("TRT-LLM currently requires plain text messages")
        context = dict(config.get("chat_template_kwargs") or {})
        context.setdefault("enable_thinking", self._enable_thinking)
        prompt = self.get_full_context(
            messages,
            self.model_family.chat_template,
            tokenizer=self._tokenizer,
            **context,
        )
        result = await self.async_generate(prompt, config, request_id)
        if config.get("stream"):
            return self._async_to_chat_completion_chunks(result, self.reasoning_parser)
        return self._to_chat_completion(result, self.reasoning_parser)
