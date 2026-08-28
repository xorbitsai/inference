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

    def generate_non_streaming(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> ChatCompletion:
        generate_config = dict(generate_config or {})
        tools = generate_config.get("tools")
        streamer, prompt_tokens = self.build_streaming_iter(messages, generate_config)
        completion_tokens, total_tokens = 0, prompt_tokens
        response = ""
        for chunk_count, new_text in enumerate(streamer, start=1):
            new_text, should_stop = self.check_conditions(new_text)
            if should_stop:
                break
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
            )
        return self._to_chat_completion(completion, self.reasoning_parser)

    def generate_streaming(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Iterator[CompletionChunk]:
        generate_config = dict(generate_config or {})
        tools = generate_config.get("tools")
        use_tool_calls = bool(tools and self.tool_parser)
        streamer, prompt_tokens = self.build_streaming_iter(messages, generate_config)
        stream_options = generate_config.get("stream_options")
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
        for chunk_count, new_text in enumerate(streamer, start=1):
            new_text, should_stop = self.check_conditions(new_text)
            if should_stop:
                break
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
                delta = chat_chunk["choices"][0]["delta"]
                if delta.get("reasoning_content") is not None:
                    yield cast(CompletionChunk, chat_chunk)
                else:
                    processed_chunk = self._post_process_completion_chunk(
                        self.model_family,
                        self.model_uid,
                        chat_chunk,
                        previous_texts=previous_tools_texts,
                        tool_call_state=tool_call_state,
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
        stream = bool(generate_config and generate_config.get("stream"))
        if stream:
            return self._to_chat_completion_chunks(
                self.generate_streaming(messages, generate_config)
            )
        return self.generate_non_streaming(messages, generate_config)
