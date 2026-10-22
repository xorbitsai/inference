# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""TensorRT-LLM's Python LLM API, owned by a Xinference model process."""

import importlib.metadata
import logging
import sys
import time
import uuid
from copy import deepcopy
from typing import Any, AsyncGenerator, Dict, Optional, Union, cast

from ....constants import XINFERENCE_MAX_TOKENS
from ....types import Completion, CompletionChunk, CompletionUsage
from ...utils import check_dependency_available
from ..context_length import ContextLengthExceeded, validate_context_length
from ..core import LLM, chat_context_var, parse_num_speculative_tokens
from ..utils import ChatModelMixin
from .support import match_model

logger = logging.getLogger(__name__)


class TRTLLMModel(LLM, ChatModelMixin):
    allow_batch = True
    support_draft_model = True

    def __init__(self, model_uid, model_family, model_path, model_config=None):
        super().__init__(model_uid, model_family, model_path)
        self._model_config = dict(model_config or {})
        self._engine = None
        self._tokenizer = None
        self._requests: Dict[str, Any] = {}
        self._cache_enabled = False
        self._pd = None

    @classmethod
    def check_lib(cls):
        return check_dependency_available("tensorrt_llm", "tensorrt-llm")

    @classmethod
    def match_json(cls, llm_family, llm_spec, quantization):
        if sys.platform != "linux":
            return False, "TRT-LLM requires Linux and NVIDIA CUDA GPUs"
        return match_model(llm_family, llm_spec, quantization)

    @classmethod
    def prepare_virtualenv(cls, llm_family):
        """Make architecture-based support available to per-model environments."""
        from ...core import VirtualEnvSettings
        from ...utils import _collect_virtualenv_engine_markers

        if not any(
            cls.match_json(llm_family, spec, spec.quantization) is True
            for spec in llm_family.model_specs
        ):
            return
        settings = llm_family.virtualenv
        packages = list(settings.packages or []) if settings else []
        # A guarded entry extends an explicit allowlist. For unrestricted
        # families, provision dependencies without creating a new allowlist.
        marker = "#trt_llm_dependencies#"
        if _collect_virtualenv_engine_markers(llm_family):
            marker += ' ; #engine# == "TRT-LLM"'
        if marker not in packages:
            packages.append(marker)
            llm_family.virtualenv = (
                settings.copy(update={"packages": packages})
                if settings
                else VirtualEnvSettings(packages=packages)
            )

    def load(self):
        if importlib.metadata.version("tensorrt-llm") != "1.2.1":
            raise ValueError("The TRT-LLM adapter requires tensorrt-llm==1.2.1")
        from tensorrt_llm import LLM as Engine

        config = dict(self._model_config)
        self._apply_draft_model(config)
        for name in ("address", "shard", "driver_info"):
            config.pop(name, None)
        if config.pop("n_worker", 1) != 1:
            raise ValueError("TRT-LLM currently requires one worker per replica")
        cache_config = config.pop("_xavier_cache_config", None)
        native_pd = config.pop("_nixl_config", None)
        if cache_config is not None and native_pd is not None:
            raise ValueError("Choose either TRT-LLM native or Xavier transport")
        if config.get("speculative_config") is not None and (
            cache_config is not None or native_pd is not None
        ):
            raise ValueError(
                "TRT-LLM speculative decoding requires ordinary replicas "
                "without Xavier or P/D"
            )
        self._enable_thinking = config.pop("enable_thinking", True)
        reasoning = config.pop("reasoning_content", False)
        self.prepare_parse_reasoning_content(reasoning, self._enable_thinking)
        if config.get("max_seq_len") is None:
            config["max_seq_len"] = self.model_family.context_length
        self._context_length = validate_context_length(config["max_seq_len"])
        if config.get("backend", "pytorch") != "pytorch":
            raise ValueError("The TRT-LLM adapter requires the PyTorch backend")
        if "_mpi_session" in config or "mpi_session" in config:
            raise ValueError("The TRT-LLM MPI session is managed by Xinference")
        config["backend"] = "pytorch"
        pd_contract = None
        if native_pd is not None:
            from .pd import configure_native

            pd_contract = configure_native(self.model_path, config, native_pd)
        if cache_config is not None:
            from .xavier import configure_xavier

            configure_xavier(self.model_path, config, cache_config)
            self._cache_enabled = True
        try:
            from .compat import prepare_runtime

            prepare_runtime(config)
            if isinstance(config.get("speculative_config"), dict):
                from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig

                config["speculative_config"] = DecodingBaseConfig.from_dict(
                    config["speculative_config"]
                )
            self._engine = Engine(model=self.model_path, **config)
            self._tokenizer = self._engine.tokenizer
            role = (native_pd or cache_config or {}).get("role")
            if role is not None:
                from .pd import TRTLLMPD

                self._pd = TRTLLMPD(
                    role, "nixl" if native_pd is not None else "xavier", pd_contract
                )
                self._pd.bind_engine(self._engine)
        except BaseException:
            self.stop()
            raise

    def _apply_draft_model(self, config: dict) -> None:
        """Consume common launch options before calling TRT's Python API."""
        draft = config.pop("draft_model_path", None)
        depth = config.pop("num_speculative_tokens", None)
        if config.get("speculative_config") is not None:
            # TRT's config parser pops decoding_type; never mutate launch state.
            if isinstance(config["speculative_config"], dict):
                config["speculative_config"] = deepcopy(config["speculative_config"])
        elif draft:
            config["speculative_config"] = {
                "decoding_type": "DraftTarget",
                "speculative_model": draft,
                "max_draft_len": parse_num_speculative_tokens(depth) or 3,
            }
        elif depth is not None:
            raise ValueError(
                "num_speculative_tokens requires draft_model_path or speculative_config"
            )
        if config.get("speculative_config") is not None:
            config.setdefault("disable_overlap_scheduler", True)

    def stop(self):
        if self._pd is not None:
            try:
                self._pd.close()
            except Exception:
                logger.warning("Could not close TRT-LLM P/D handoffs", exc_info=True)
        for request in list(self._requests.values()):
            try:
                request.abort()
            except Exception:
                logger.warning(
                    "Could not abort TRT-LLM request during shutdown", exc_info=True
                )
        if self._cache_enabled:
            from .gpu import close_service

            # Stop the importer before freeing C++ memory. If stopping fails,
            # retain the owning engine so live DMA cannot access freed slots.
            close_service()
        try:
            if self._engine is not None:
                self._engine.shutdown()
                if self._pd is not None:
                    self._pd.engine_stopped()
        finally:
            self._requests.clear()
            self._engine = None
            if self._cache_enabled:
                connector = sys.modules.get("xinference.model.llm.trt_llm.connector")
                if connector is not None:
                    connector.close_clients()

    async def release_pd_request(self, request_id: str) -> None:
        if self._pd is not None:
            await self._pd.release(request_id)

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
        for name in ("tools", "response_format", "prompt_logprobs", "echo"):
            if config.get(name):
                raise ValueError(
                    f"TRT-LLM does not support {name} through this adapter"
                )
        if config.get("logprobs") is not None and config["logprobs"] is not False:
            raise ValueError("TRT-LLM does not support logprobs through this adapter")
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
        if type(options["stop_token_ids"]) is int:
            options["stop_token_ids"] = [options["stop_token_ids"]]
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
        pd_state = None
        prefill = self._pd is not None and self._pd.role == "prefill"
        if prefill and config.get("stream"):
            raise ValueError("TRT-LLM prefill requires a non-streaming P/D request")
        if self._pd is not None:
            pd_state = await self._pd.prepare(
                request_id, tokens, config.get("_pd_kv_transfer_params")
            )
            if prefill:
                params.max_tokens = 1
        elif config.get("_pd_kv_transfer_params") is not None:
            raise ValueError("TRT-LLM remote prefill requires explicit P/D roles")

        def start():
            if request_id in self._requests:
                raise ValueError("A TRT-LLM request with this ID is already active")
            request = (
                self._pd.submit(self._engine, tokens, params, pd_state)
                if self._pd is not None
                else self._engine.generate_async(tokens, params, streaming=True)
            )
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
                if self._pd is not None and self._pd.backend == "xavier":
                    from .pd import forget_request

                    forget_request(request)

        owns_prepared_state = (
            pd_state is not None
            and self._pd is not None
            and self._pd.backend == "xavier"
        )

        async def stream():
            request = None
            try:
                if owns_prepared_state:
                    yield None  # Enter cleanup before transferring this stream.
                request = start()
                async for output in request:
                    if self._pd is not None:
                        self._pd.check_import(pd_state)
                    chunk = CompletionChunk(
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
                    # The shared chat parser consumes text only before finish.
                    # A native terminal result may still contain generated text.
                    if any(
                        c["text"] and c["finish_reason"] is not None
                        for c in chunk["choices"]
                    ):
                        yield {
                            **chunk,
                            "choices": [
                                {**c, "finish_reason": None} for c in chunk["choices"]
                            ],
                        }
                        chunk["choices"] = [{**c, "text": ""} for c in chunk["choices"]]
                    yield chunk
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
                if request is not None:
                    finish(request)
                if self._pd is not None:
                    await self._pd.release(request_id)

        if config.get("stream", False):
            result_stream = stream()
            if owns_prepared_state:
                await anext(result_stream)
            return result_stream
        request = None
        published = False
        try:
            request = start()
            output = await request
            if self._pd is not None:
                self._pd.check_import(pd_state)
            result = Completion(
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
            if prefill:
                cast(Dict[str, Any], result)["_pd_kv_transfer_params"] = (
                    await self._pd.publish(request_id, tokens, output)
                )
                published = True
            return result
        finally:
            if request is not None:
                finish(request)
            if self._pd is not None and (not prefill or not published):
                await self._pd.release(request_id)

    async def async_chat(self, messages, generate_config=None, request_id=None):
        config = dict(generate_config or {})
        if config.get("tools"):
            raise ValueError(
                "TRT-LLM tool calling is not supported by this adapter yet"
            )
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                if any(
                    not isinstance(part, dict)
                    or part.get("type") != "text"
                    or not isinstance(part.get("text"), str)
                    for part in content
                ):
                    raise ValueError("TRT-LLM currently requires plain text messages")
            elif not isinstance(content, (str, type(None))):
                raise ValueError("TRT-LLM currently requires plain text messages")
        if config.get("chat_template_kwargs") is None:
            config.pop("chat_template_kwargs", None)
        context = (
            self._get_chat_template_kwargs_from_generate_config(
                config, self.reasoning_parser
            )
            or {}
        )
        context.setdefault("enable_thinking", self._enable_thinking)
        prompt = self.get_full_context(
            [dict(message) for message in messages],
            self.model_family.chat_template,
            tokenizer=self._tokenizer,
            **context,
        )
        result = await self.async_generate(prompt, config, request_id)
        if config.get("stream"):
            chunks = self._async_to_chat_completion_chunks(
                result, self.reasoning_parser, context
            )
            if self._pd is not None and self._pd.backend == "xavier":

                async def owned_chat():
                    try:
                        yield None
                        async for chunk in chunks:
                            yield chunk
                    finally:
                        try:
                            await chunks.aclose()
                        finally:
                            await result.aclose()

                owned = owned_chat()
                await anext(owned)
                return owned
            return chunks
        token = chat_context_var.set(context)
        try:
            response = self._to_chat_completion(result, self.reasoning_parser)
            if "_pd_kv_transfer_params" in result:
                response["_pd_kv_transfer_params"] = result["_pd_kv_transfer_params"]
            return response
        finally:
            chat_context_var.reset(token)
