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
from typing import TYPE_CHECKING, AsyncGenerator, Dict, Union

if TYPE_CHECKING:
    from ...types import ChatCompletion, ChatCompletionChunk


class HarmonyStreamParser:
    _MARKERS = (
        ("<|start|>assistant<|channel|>analysis<|message|>", "analysis"),
        ("<|start|>assistant<|channel|>final<|message|>", "assistantfinal"),
        ("<|channel|>analysis<|message|>", "analysis"),
        ("<|channel|>final<|message|>", "assistantfinal"),
        ("<|end|>", ""),
        ("<|return|>", ""),
        ("<|endoftext|>", ""),
    )

    def __init__(self, structured_output=False):
        # Current channel: either 'analysis', 'final', or None if not started yet
        self.current_channel = None
        # Buffer for accumulating text when looking for 'assistantfinal' marker
        self.buffer = ""
        # Raw vLLM control tokens can be split across streaming chunks.
        self._marker_buffer = ""
        # Constrained decoding can suppress the final-channel marker and leave
        # a bare ``analysis`` prefix before the JSON value.
        self.structured_output = structured_output

    def _normalize_markers(self, text):
        """Convert raw Harmony control tokens to the compact stream markers."""
        self._marker_buffer += text
        normalized = []
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

    def _consume_structured_analysis(self):
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

    def feed(self, text):
        """
        Feed a chunk of text into the parser and return parsed segments.

        Each segment is a dict:
        {
            "channel": "analysis" | "final",
            "content": <string>
        }

        The parser detects 'assistantfinal' markers inside reasoning text,
        splits the reasoning and final content correctly, and switches the channel.
        """
        text = self._normalize_markers(text)
        if not text:
            return []
        segments = []

        # If we are currently in 'analysis' mode
        if self.current_channel == "analysis":
            # Add text to buffer and check for 'assistantfinal' marker
            self.buffer += text
            if "assistantfinal" in self.buffer:
                # Split reasoning and final content
                before, after = self.buffer.split("assistantfinal", 1)
                if before:
                    segments.append({"channel": "analysis", "content": before})
                # Switch to final channel
                self.current_channel = "final"
                self.buffer = ""
                if after:
                    segments.append({"channel": "final", "content": after})
                return segments
            if self.structured_output:
                return self._consume_structured_analysis()
            else:
                # Check if buffer ends with partial 'assistantfinal'
                if any(
                    self.buffer.endswith("assistantfinal"[:i])
                    for i in range(1, len("assistantfinal") + 1)
                ):
                    # Don't emit anything yet, wait for more text
                    return segments
                else:
                    # Emit what we have so far and keep buffer for next time
                    if self.buffer:
                        segments.append({"channel": "analysis", "content": self.buffer})
                        self.buffer = ""
                    return segments

        # If we are currently in 'final' mode
        if self.current_channel == "final":
            if self.structured_output and text.startswith("assistantfinal"):
                text = text[len("assistantfinal") :]
                if not text:
                    return segments
            # Check if this is actually a new message starting with 'analysis'
            if not self.structured_output and text.startswith("analysis"):
                # Reset parser state for new message
                self.current_channel = None
                self.buffer = ""
                # Re-process this text with the new state
                return self.feed(text)
            else:
                segments.append({"channel": "final", "content": text})
                return segments

        # If no channel has been started yet
        if text.startswith("analysis"):
            self.current_channel = "analysis"
            rest = text[len("analysis") :]
            if "assistantfinal" in rest:
                # Split immediately if marker is found in the first chunk
                before, after = rest.split("assistantfinal", 1)
                if before:
                    segments.append({"channel": "analysis", "content": before})
                self.current_channel = "final"
                if after:
                    segments.append({"channel": "final", "content": after})
            elif self.structured_output:
                # Keep the prefix buffered until the first generated value
                # identifies this as constrained JSON or ordinary analysis.
                self.buffer = rest
                segments.extend(self._consume_structured_analysis())
            else:
                # Start buffering for potential 'assistantfinal' marker
                self.buffer = rest
                # Check if buffer ends with partial 'assistantfinal'
                if any(
                    self.buffer.endswith("assistantfinal"[:i])
                    for i in range(1, len("assistantfinal") + 1)
                ):
                    # Don't emit anything yet, wait for more text
                    pass
                else:
                    # Emit what we have so far
                    if self.buffer:
                        segments.append({"channel": "analysis", "content": self.buffer})
                        self.buffer = ""
        elif text.startswith("assistantfinal"):
            self.current_channel = "final"
            rest = text[len("assistantfinal") :]
            if rest:
                segments.append({"channel": "final", "content": rest})
        elif self.structured_output:
            self.current_channel = "final"
            segments.append({"channel": "final", "content": text})

        return segments


async def async_stream_harmony_chat_completion(
    chunks: Union[
        "ChatCompletion",
        AsyncGenerator["ChatCompletionChunk", None],
    ],
    expose_reasoning: bool = True,
    structured_output: bool = False,
) -> AsyncGenerator["ChatCompletion", None]:
    """
    Parse Harmony-formatted content from either a full ChatCompletion (non-streaming)
    or an async stream of ChatCompletionChunk (streaming), using the HarmonyStreamParser defined in this file.

    Yields parsed objects incrementally.
    """

    # --- Non-streaming: ChatCompletion ---
    if isinstance(chunks, dict) and chunks.get("object") == "chat.completion":
        out_data = deepcopy(chunks)

        for choice in out_data["choices"]:
            parser = HarmonyStreamParser(structured_output=structured_output)
            msg = choice["message"]

            # Backup original content & reasoning
            original_content = msg.get("content") or ""
            original_reasoning = msg.get("reasoning_content") or ""

            # Reset fields before parsing
            msg["content"] = ""
            msg["reasoning_content"] = ""
            msg.setdefault("tool_calls", [])

            # Feed original content
            for seg in parser.feed(original_content):
                ch, c = seg["channel"], seg["content"]
                if ch == "final":
                    msg["content"] += c
                elif ch == "analysis":
                    msg["reasoning_content"] += c
                elif ch == "tool":
                    msg["tool_calls"].append(c)

            # Feed original reasoning_content
            for seg in parser.feed(original_reasoning):
                if seg["channel"] == "analysis":
                    msg["reasoning_content"] += seg["content"]
                elif seg["channel"] == "tool":
                    msg["tool_calls"].append(seg["content"])

            # Clean up reasoning_content: set to None if no reasoning content was parsed
            if not msg["reasoning_content"] and not original_reasoning:
                msg["reasoning_content"] = None  # type: ignore
            if not expose_reasoning:
                msg.pop("reasoning_content", None)

        yield out_data

    else:
        # Streaming: handle async generator
        parsers_per_choice = {}

        async for chunk in chunks:  # type: ignore
            if not chunk.get("choices"):
                # Preserve usage-only and terminal metadata chunks.
                yield chunk
                continue
            out_chunk = {  # type: ignore
                "id": chunk["id"],
                "model": chunk["model"],
                "object": chunk["object"],
                "created": chunk["created"],
                "choices": [],
            }
            if chunk.get("usage") is not None:
                out_chunk["usage"] = chunk["usage"]

            for i, choice in enumerate(chunk["choices"]):
                delta = choice.get("delta", {})
                text = delta.get("content") or ""  # type: ignore

                if i not in parsers_per_choice:
                    parsers_per_choice[i] = HarmonyStreamParser(
                        structured_output=structured_output
                    )

                # Feed text to parser and collect current delta only
                curr_delta: Dict[str, object] = {
                    "content": "",
                    "reasoning_content": "",
                    "tool_calls": [],
                }

                for seg in parsers_per_choice[i].feed(text):
                    ch = seg["channel"]
                    c = seg["content"]
                    if ch == "final":
                        curr_delta["content"] += c  # type: ignore
                    elif ch == "analysis":
                        curr_delta["reasoning_content"] += c  # type: ignore
                    elif ch == "tool":
                        curr_delta["tool_calls"].append(c)  # type: ignore

                reasoning_content = curr_delta.get("reasoning_content") or ""
                if not expose_reasoning:
                    curr_delta.pop("reasoning_content", None)

                if reasoning_content:
                    if not curr_delta["content"]:
                        curr_delta["content"] = None

                elif curr_delta["content"]:
                    if expose_reasoning:
                        curr_delta["reasoning_content"] = None

                elif choice.get("finish_reason") is not None and expose_reasoning:
                    # For the final chunk, if there's no new reasoning content,
                    # don't include empty reasoning_content to avoid clearing existing state
                    curr_delta["reasoning_content"] = None

                out_chunk["choices"].append(  # type: ignore
                    {
                        "index": i,
                        "delta": curr_delta,
                        "finish_reason": choice.get("finish_reason"),
                    }
                )

            # Only yield if we have either content or reasoning_content
            has_content = any(
                choice["delta"].get("content")  # type: ignore
                or choice["delta"].get("reasoning_content")  # type: ignore
                or choice.get("finish_reason") is not None  # type: ignore
                for choice in out_chunk["choices"]  # type: ignore
            )
            if has_content:
                yield out_chunk  # type: ignore
