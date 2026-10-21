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
import uuid
from abc import abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionChunk,
    PytorchGenerateConfig,
)
from ..core import chat_context_var
from ..media import validate_messages_media
from ..utils import ChatModelMixin, generate_completion, generate_completion_chunk


class PytorchDirectChatMixin(ChatModelMixin):
    """Shared direct-chat generation for processor-based Transformers models."""

    model_uid: str
    model_family: Any
    _tokenizer: Any
    tool_parser: Any
    reasoning_parser: Any

    @abstractmethod
    def build_inputs_from_messages(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ):
        """Build the model inputs for one OpenAI-formatted chat request."""
        raise NotImplementedError

    @abstractmethod
    def build_generate_kwargs(
        self,
        generate_config: Dict,
    ) -> Dict[str, Any]:
        """Build the keyword arguments passed to ``model.generate``."""
        raise NotImplementedError

    @abstractmethod
    def build_streaming_iter(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ) -> Tuple[Iterator, int]:
        """Return a text iterator and the number of prompt tokens."""
        raise NotImplementedError

    def get_stop_strs(self) -> List[str]:
        return []

    def check_conditions(self, new_text: str) -> Tuple[str, bool]:
        for stop_str in self.get_stop_strs():
            if new_text.endswith(stop_str):
                new_text = new_text[: -len(stop_str)]
                break
        return new_text, False

    def count_completion_tokens(self, text: str, fallback: int) -> int:
        encode = getattr(self._tokenizer, "encode", None)
        if encode is None:
            return fallback
        try:
            return len(encode(text, add_special_tokens=False))
        except (TypeError, ValueError):
            return fallback

    def _iter_text(
        self, streamer: Iterator, stop: Optional[Union[str, List[str]]]
    ) -> Iterator[str]:
        stops = [stop] if isinstance(stop, str) else stop or []
        pending = ""
        for new_text in streamer:
            new_text, should_stop = self.check_conditions(new_text)
            if should_stop:
                break
            pending += new_text
            positions = [pending.find(s) for s in stops if s in pending]
            if positions:
                yield pending[: min(positions)]
                # Let background generation finish before releasing the request.
                for _ in streamer:
                    pass
                return
            # Keep any suffix that could become a stop string in the next chunk.
            held = max(
                (
                    size
                    for s in stops
                    for size in range(1, min(len(s), len(pending) + 1))
                    if pending.endswith(s[:size])
                ),
                default=0,
            )
            ready = len(pending) - held
            yield pending[:ready]
            pending = pending[ready:]
        if pending:
            yield pending

    def generate_non_streaming(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> ChatCompletion:
        config: Dict[str, Any] = dict(generate_config or {})
        tools = config.get("tools")
        streamer, prompt_tokens = self.build_streaming_iter(messages, config)
        completion_tokens, total_tokens = 0, prompt_tokens
        response = ""
        for chunk_count, new_text in enumerate(
            self._iter_text(streamer, config.get("stop")), start=1
        ):
            response += new_text
            completion_tokens = self.count_completion_tokens(response, chunk_count)
            total_tokens = prompt_tokens + completion_tokens
        completion = generate_completion(
            self.model_uid,
            response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens if prompt_tokens != -1 else -1,
            total_tokens=total_tokens if prompt_tokens != -1 else -1,
        )
        if tools and self.tool_parser:
            return self._post_process_completion(
                self.model_family,
                self.model_uid,
                completion,
                tools,
            )
        return self._to_chat_completion(completion, self.reasoning_parser)

    def generate_streaming(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Iterator[CompletionChunk]:
        config: Dict[str, Any] = dict(generate_config or {})
        tools = config.get("tools")
        use_tool_calls = bool(tools and self.tool_parser)
        streamer, prompt_tokens = self.build_streaming_iter(messages, config)
        stream_options = config.get("stream_options")
        include_usage = (
            stream_options.get("include_usage", False)
            if isinstance(stream_options, dict)
            else False
        )

        completion_id = str(uuid.uuid4())
        completion_tokens, total_tokens = 0, prompt_tokens
        previous_texts = [""]
        previous_tools_texts = [""]
        tool_call_state = {"seen": False}
        is_first_chunk = True
        response = ""
        for chunk_count, new_text in enumerate(
            self._iter_text(streamer, config.get("stop")), start=1
        ):
            response += new_text
            completion_tokens = self.count_completion_tokens(response, chunk_count)
            total_tokens = prompt_tokens + completion_tokens
            completion_chunk = generate_completion_chunk(
                chunk_text=new_text,
                finish_reason=None,
                chunk_id=completion_id,
                model_uid=self.model_uid,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens if prompt_tokens != -1 else -1,
                total_tokens=total_tokens if prompt_tokens != -1 else -1,
                has_choice=True,
                has_content=True,
            )
            if use_tool_calls:
                chat_chunk = self._to_chat_completion_chunk(
                    completion_chunk,
                    self.reasoning_parser,
                    previous_texts,
                    ensure_role=is_first_chunk,
                )
                reasoning_chunk, tool_chunk = self._split_reasoning_tool_chunk(
                    chat_chunk
                )
                if reasoning_chunk is not None:
                    yield cast(CompletionChunk, reasoning_chunk)
                if tool_chunk is not None:
                    processed_chunk = self._post_process_completion_chunk(
                        self.model_family,
                        self.model_uid,
                        tool_chunk,
                        previous_texts=previous_tools_texts,
                        tool_call_state=tool_call_state,
                        tools=tools,
                    )
                    if processed_chunk:
                        yield cast(CompletionChunk, processed_chunk)
            else:
                yield completion_chunk
            is_first_chunk = False

        completion_chunk = generate_completion_chunk(
            chunk_text=None,
            finish_reason="stop",
            chunk_id=completion_id,
            model_uid=self.model_uid,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens if prompt_tokens != -1 else -1,
            total_tokens=total_tokens if prompt_tokens != -1 else -1,
            has_choice=True,
            has_content=False,
        )
        if use_tool_calls:
            chat_chunk = self._to_chat_completion_chunk(
                completion_chunk,
                self.reasoning_parser,
                previous_texts,
                ensure_role=is_first_chunk,
            )
            processed_chunk = self._post_process_completion_chunk(
                self.model_family,
                self.model_uid,
                chat_chunk,
                previous_texts=previous_tools_texts,
                tool_call_state=tool_call_state,
                tools=tools,
            )
            if processed_chunk:
                yield cast(CompletionChunk, processed_chunk)
        else:
            yield completion_chunk

        if include_usage:
            usage_chunk = generate_completion_chunk(
                chunk_text=None,
                finish_reason=None,
                chunk_id=completion_id,
                model_uid=self.model_uid,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens if prompt_tokens != -1 else -1,
                total_tokens=total_tokens if prompt_tokens != -1 else -1,
                has_choice=False,
                has_content=False,
            )
            if use_tool_calls:
                chat_chunk = self._to_chat_completion_chunk(
                    usage_chunk,
                    self.reasoning_parser,
                    previous_texts,
                    ensure_role=is_first_chunk,
                )
                processed_chunk = self._post_process_completion_chunk(
                    self.model_family,
                    self.model_uid,
                    chat_chunk,
                    previous_texts=previous_tools_texts,
                    tool_call_state=tool_call_state,
                    tools=tools,
                )
                if processed_chunk:
                    yield cast(CompletionChunk, processed_chunk)
            else:
                yield usage_chunk

    def build_direct_chat_result(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        # Single funnel for every direct-chat model, including families that
        # never call _transform_messages (deepseek_vl2) or override it (qwen2_audio).
        validate_messages_media(messages)
        ctx = (
            self._get_chat_template_kwargs_from_generate_config(
                generate_config, self.reasoning_parser
            )
            or {}
        )
        token = chat_context_var.set(ctx)
        try:
            stream = bool(generate_config and generate_config.get("stream"))
            if stream:
                use_tool_calls = bool(
                    generate_config
                    and generate_config.get("tools")
                    and self.tool_parser
                )
                parser = self.reasoning_parser
                if use_tool_calls or (parser and not parser.check_content_parser()):
                    parser = None
                return self._to_chat_completion_chunks(
                    self.generate_streaming(messages, generate_config), parser, ctx=ctx
                )
            return self.generate_non_streaming(messages, generate_config)
        finally:
            chat_context_var.reset(token)
