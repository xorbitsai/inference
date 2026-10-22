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

from copy import deepcopy
from typing import TYPE_CHECKING, AsyncGenerator, Dict, List, Optional, Union, cast

if TYPE_CHECKING:
    from ...types import ChatCompletion, ChatCompletionChunk


class HarmonyStreamParser:
    _RAW_ANALYSIS_MARKER = "\x00analysis"
    _RAW_FINAL_MARKER = "\x00assistantfinal"
    _MARKERS = (
        (
            "<|start|>assistant<|channel|>analysis<|message|>",
            _RAW_ANALYSIS_MARKER,
        ),
        ("<|start|>assistant<|channel|>final<|message|>", _RAW_FINAL_MARKER),
        ("<|channel|>analysis<|message|>", _RAW_ANALYSIS_MARKER),
        ("<|channel|>final<|message|>", _RAW_FINAL_MARKER),
        ("<|end|>", ""),
        ("<|return|>", ""),
        ("<|endoftext|>", ""),
    )

    def __init__(self, structured_output: bool = False):
        # Current channel: either 'analysis', 'final', or None if not started yet
        self.current_channel: Optional[str] = None
        # Legacy compact streams reuse a parser across messages. Raw Harmony
        # framing is unambiguous, so only the legacy final marker enables the
        # bare ``analysis`` reset used by those streams.
        self._final_channel_is_raw = False
        # Buffer for accumulating text when looking for 'assistantfinal' marker
        self.buffer = ""
        # Raw vLLM control tokens can be split across streaming chunks.
        self._marker_buffer = ""
        # Constrained decoding can suppress the final-channel marker and leave
        # a bare ``analysis`` prefix before the JSON value.
        self.structured_output = structured_output

    def _normalize_markers(self, text: str) -> str:
        """Convert raw Harmony control tokens to the compact stream markers."""
        self._marker_buffer += text
        normalized: List[str] = []
        marker_texts = [marker for marker, _ in self._MARKERS]
        while self._marker_buffer:
            matches = [
                (self._marker_buffer.find(marker), marker, replacement)
                for marker, replacement in self._MARKERS
                if self._marker_buffer.find(marker) >= 0
            ]
            if matches:
                position, marker, replacement = min(
                    matches, key=lambda item: (item[0], -len(item[1]))
                )
                normalized.append(self._marker_buffer[:position])
                normalized.append(replacement)
                self._marker_buffer = self._marker_buffer[position + len(marker) :]
                continue

            safe_length = len(self._marker_buffer)
            for marker in marker_texts:
                for width in range(1, min(len(marker), len(self._marker_buffer)) + 1):
                    if self._marker_buffer.endswith(marker[:width]):
                        safe_length = min(safe_length, len(self._marker_buffer) - width)
            if safe_length == 0:
                break
            normalized.append(self._marker_buffer[:safe_length])
            self._marker_buffer = self._marker_buffer[safe_length:]
        return "".join(normalized)

    def _consume_structured_analysis(self) -> List[Dict[str, str]]:
        """Resolve a possible bare analysis prefix in constrained output."""
        stripped = self.buffer.lstrip()
        if not stripped:
            return []
        if stripped[0] in "[{":
            self.current_channel = "final"
            self.buffer = ""
            return [{"channel": "final", "content": stripped}]
        content = self.buffer
        self.buffer = ""
        return [{"channel": "analysis", "content": content}]

    def _feed_normalized(self, text: str) -> List[Dict[str, str]]:
        """Parse text after raw Harmony markers have been normalized."""
        if not text:
            return []
        segments: List[Dict[str, str]] = []

        if self.current_channel == "analysis":
            self.buffer += text
            final_markers = (self._RAW_FINAL_MARKER, "assistantfinal")
            matches = [
                (self.buffer.find(marker), marker)
                for marker in final_markers
                if self.buffer.find(marker) >= 0
            ]
            if matches:
                position, marker = min(matches, key=lambda item: item[0])
                before = self.buffer[:position]
                after = self.buffer[position + len(marker) :]
                if before:
                    segments.append({"channel": "analysis", "content": before})
                self.current_channel = "final"
                self._final_channel_is_raw = marker == self._RAW_FINAL_MARKER
                self.buffer = ""
                if after:
                    segments.extend(self._feed_normalized(after))
                return segments
            if self.structured_output:
                return self._consume_structured_analysis()
            if self.buffer.endswith(tuple("assistantfinal"[:i] for i in range(1, 14))):
                return segments
            if self.buffer:
                segments.append({"channel": "analysis", "content": self.buffer})
                self.buffer = ""
            return segments

        if self.current_channel == "final":
            # Raw channel markers are unambiguous. A literal word such as
            # ``analysis`` can be a valid final answer and must stay content.
            if self._RAW_ANALYSIS_MARKER in text:
                before, after = text.split(self._RAW_ANALYSIS_MARKER, 1)
                if before:
                    segments.append({"channel": "final", "content": before})
                self.current_channel = None
                self._final_channel_is_raw = False
                self.buffer = ""
                segments.extend(self._feed_normalized(after))
                return segments
            if not self._final_channel_is_raw and text.startswith("analysis"):
                self.current_channel = None
                self.buffer = ""
                return self._feed_normalized(text)
            text = text.replace(self._RAW_FINAL_MARKER, "")
            if text:
                segments.append({"channel": "final", "content": text})
            return segments

        if text.startswith(self._RAW_ANALYSIS_MARKER):
            self.current_channel = "analysis"
            return self._feed_normalized(text[len(self._RAW_ANALYSIS_MARKER) :])
        if text.startswith(self._RAW_FINAL_MARKER):
            self.current_channel = "final"
            self._final_channel_is_raw = True
            return self._feed_normalized(text[len(self._RAW_FINAL_MARKER) :])
        if text.startswith("analysis"):
            self.current_channel = "analysis"
            rest = text[len("analysis") :]
            if rest:
                return self._feed_normalized(rest)
            return segments
        if text.startswith("assistantfinal"):
            self.current_channel = "final"
            self._final_channel_is_raw = False
            return self._feed_normalized(text[len("assistantfinal") :])
        if self.structured_output:
            self.current_channel = "final"
            segments.append({"channel": "final", "content": text})
        return segments

    def feed(self, text: str) -> List[Dict[str, str]]:
        """Feed one stream chunk and return parsed channel segments."""
        return self._feed_normalized(self._normalize_markers(text))

    def flush(self) -> List[Dict[str, str]]:
        """Flush text held while checking for a marker split across chunks."""
        segments: List[Dict[str, str]] = []
        pending_marker = self._marker_buffer
        self._marker_buffer = ""
        if pending_marker:
            if self.current_channel is None:
                self.current_channel = "final"
                segments.append({"channel": "final", "content": pending_marker})
            else:
                segments.extend(self._feed_normalized(pending_marker))
        if self.buffer:
            channel = self.current_channel or "final"
            segments.append({"channel": channel, "content": self.buffer})
            self.buffer = ""
        return segments


async def async_stream_harmony_chat_completion(
    chunks: Union[
        "ChatCompletion",
        AsyncGenerator["ChatCompletionChunk", None],
    ],
    expose_reasoning: bool = True,
    structured_output: bool = False,
) -> AsyncGenerator[Union["ChatCompletion", "ChatCompletionChunk"], None]:
    """
    Parse Harmony-formatted content from either a full ChatCompletion (non-streaming)
    or an async stream of ChatCompletionChunk (streaming), using the HarmonyStreamParser defined in this file.

    Yields parsed objects incrementally.
    """

    # --- Non-streaming: ChatCompletion ---
    if isinstance(chunks, dict) and chunks.get("object") == "chat.completion":
        out_data = deepcopy(chunks)

        for message_choice in out_data["choices"]:
            parser = HarmonyStreamParser(structured_output=structured_output)
            msg = message_choice["message"]

            # Backup original content & reasoning
            original_content = msg.get("content") or ""
            original_reasoning = msg.get("reasoning_content") or ""

            # Reset fields before parsing
            msg["content"] = ""
            msg["reasoning_content"] = ""
            msg.setdefault("tool_calls", [])

            # Feed original content and flush a literal partial marker at EOF.
            segments = parser.feed(original_content)
            segments.extend(parser.feed(original_reasoning))
            segments.extend(parser.flush())
            for seg in segments:
                ch, c = seg["channel"], seg["content"]
                if ch == "final":
                    msg["content"] = (msg.get("content") or "") + c
                elif ch == "analysis":
                    msg["reasoning_content"] += c
                elif ch == "tool":
                    msg["tool_calls"].append(c)

            # Clean up reasoning_content: set to None if no reasoning content was parsed
            if not msg["reasoning_content"] and not original_reasoning:
                msg["reasoning_content"] = None  # type: ignore
            if not expose_reasoning:
                msg.pop("reasoning_content", None)

        yield out_data

    else:
        parsers_per_choice: Dict[int, HarmonyStreamParser] = {}
        last_chunk = None
        stream_chunks = cast(AsyncGenerator["ChatCompletionChunk", None], chunks)

        async for chunk in stream_chunks:
            last_chunk = chunk
            if not chunk.get("choices"):
                if chunk.get("usage") is not None:
                    flushed_choices: List[Dict[str, object]] = []
                    for i, parser in parsers_per_choice.items():
                        segments = parser.flush()
                        if not segments:
                            continue
                        content = "".join(
                            seg["content"]
                            for seg in segments
                            if seg["channel"] == "final"
                        )
                        reasoning = "".join(
                            seg["content"]
                            for seg in segments
                            if seg["channel"] == "analysis"
                        )
                        metadata_delta: Dict[str, object] = {"content": content}
                        if expose_reasoning and reasoning:
                            metadata_delta["reasoning_content"] = reasoning
                        flushed_choices.append(
                            {
                                "index": i,
                                "delta": metadata_delta,
                                "finish_reason": None,
                            }
                        )
                    if flushed_choices:
                        yield cast(
                            "ChatCompletionChunk",
                            {
                                "id": chunk["id"],
                                "model": chunk["model"],
                                "object": chunk["object"],
                                "created": chunk["created"],
                                "choices": flushed_choices,
                            },
                        )
                # Preserve usage-only and terminal metadata chunks.
                yield chunk
                continue

            output_choices: List[Dict[str, object]] = []
            out_chunk: Dict[str, object] = {
                "id": chunk["id"],
                "model": chunk["model"],
                "object": chunk["object"],
                "created": chunk["created"],
                "choices": output_choices,
            }
            if chunk.get("usage") is not None:
                out_chunk["usage"] = chunk["usage"]

            for i, stream_choice in enumerate(chunk["choices"]):
                stream_delta = stream_choice.get("delta", {})
                text = stream_delta.get("content") or ""
                parser = parsers_per_choice.setdefault(
                    i, HarmonyStreamParser(structured_output=structured_output)
                )
                segments = parser.feed(text)
                if stream_choice.get("finish_reason") is not None:
                    segments.extend(parser.flush())

                content_parts: List[str] = []
                reasoning_parts: List[str] = []
                tool_parts: List[str] = []
                for seg in segments:
                    if seg["channel"] == "final":
                        content_parts.append(seg["content"])
                    elif seg["channel"] == "analysis":
                        reasoning_parts.append(seg["content"])
                    elif seg["channel"] == "tool":
                        tool_parts.append(seg["content"])

                content = "".join(content_parts)
                reasoning_content = "".join(reasoning_parts)
                curr_delta: Dict[str, object] = {"content": content}
                if tool_parts:
                    curr_delta["tool_calls"] = tool_parts
                if expose_reasoning:
                    curr_delta["reasoning_content"] = reasoning_content
                role = stream_delta.get("role")
                if role is not None:
                    curr_delta["role"] = role
                    if (
                        stream_delta.get("content") is None
                        and not content
                        and not reasoning_content
                    ):
                        curr_delta["content"] = None

                if reasoning_content:
                    if not content:
                        curr_delta["content"] = None
                elif content and expose_reasoning:
                    curr_delta["reasoning_content"] = None
                elif (
                    stream_choice.get("finish_reason") is not None and expose_reasoning
                ):
                    curr_delta["reasoning_content"] = None

                output_choice: Dict[str, object] = {
                    "index": i,
                    "delta": curr_delta,
                    "finish_reason": stream_choice.get("finish_reason"),
                }
                choice_logprobs = stream_choice.get("logprobs")
                if content and choice_logprobs is not None:
                    # Reasoning tokens are not visible completion tokens. Keep
                    # logprobs on the chunk that emits final content.
                    output_choice["logprobs"] = choice_logprobs
                if "stop_sequence" in stream_choice:
                    output_choice["stop_sequence"] = stream_choice["stop_sequence"]
                output_choices.append(output_choice)

            def has_visible_output(choice: Dict[str, object]) -> bool:
                choice_delta = choice.get("delta")
                if not isinstance(choice_delta, dict):
                    return choice.get("finish_reason") is not None
                return bool(
                    choice_delta.get("content")
                    or choice_delta.get("reasoning_content")
                    or choice_delta.get("role")
                    or choice.get("finish_reason") is not None
                )

            has_content = any(has_visible_output(choice) for choice in output_choices)
            if has_content:
                yield cast("ChatCompletionChunk", out_chunk)

        # A generator can end without a terminal choice. Preserve a literal
        # suffix held while checking for a split raw control marker.
        if last_chunk is not None:
            for i, parser in parsers_per_choice.items():
                segments = parser.flush()
                if not segments:
                    continue
                content = "".join(
                    seg["content"] for seg in segments if seg["channel"] == "final"
                )
                reasoning = "".join(
                    seg["content"] for seg in segments if seg["channel"] == "analysis"
                )
                flush_delta: Dict[str, object] = {"content": content}
                if expose_reasoning and reasoning:
                    flush_delta["reasoning_content"] = reasoning
                yield cast(
                    "ChatCompletionChunk",
                    {
                        "id": last_chunk["id"],
                        "model": last_chunk["model"],
                        "object": last_chunk["object"],
                        "created": last_chunk["created"],
                        "choices": [
                            {"index": i, "delta": flush_delta, "finish_reason": None}
                        ],
                    },
                )


async def async_stream_harmony_chat_completion_chunks(
    chunks: AsyncGenerator["ChatCompletionChunk", None],
    expose_reasoning: bool = True,
    structured_output: bool = False,
) -> AsyncGenerator["ChatCompletionChunk", None]:
    """Typed streaming adapter for ``async_stream_harmony_chat_completion``."""
    async for chunk in async_stream_harmony_chat_completion(
        chunks,
        expose_reasoning=expose_reasoning,
        structured_output=structured_output,
    ):
        yield cast("ChatCompletionChunk", chunk)


async def parse_harmony_chat_completion(
    completion: "ChatCompletion",
    expose_reasoning: bool = True,
    structured_output: bool = False,
) -> "ChatCompletion":
    """Typed non-streaming adapter for ``async_stream_harmony_chat_completion``."""
    async for parsed in async_stream_harmony_chat_completion(
        completion,
        expose_reasoning=expose_reasoning,
        structured_output=structured_output,
    ):
        return cast("ChatCompletion", parsed)
    raise RuntimeError("Harmony completion parser returned no completion")
