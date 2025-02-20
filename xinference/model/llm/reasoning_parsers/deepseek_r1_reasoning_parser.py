import re
from typing import Optional, Union, cast

from ....types import (
    ChatCompletionChunkDelta,
    ChatCompletionMessage,
    Completion,
    CompletionChunk,
)
from .abs_reasoning_parsers import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module("deepseek-r1")
@ReasoningParserManager.register_module("deepseek-r1-distill-qwen")
@ReasoningParserManager.register_module("deepseek-r1-distill-llama")
class DeepSeekR1ReasoningParser(ReasoningParser):
    """Reasoning parser for DeepSeek-R1 model."""

    def __init__(
        self, reasoning_start_tag: str = "<think>", reasoning_end_tag: str = "</think>"
    ):
        super().__init__(reasoning_start_tag, reasoning_end_tag)
        self.reasoning_regex = re.compile(
            rf"{self.reasoning_start_tag}(.*?){self.reasoning_end_tag}", re.DOTALL
        )

    async def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: Union[str, CompletionChunk],
    ) -> ChatCompletionChunkDelta:
        """Extract reasoning content from DeepSeek-R1 model output in a streaming fashion.

        Args:
            previous_text (str): The previous accumulated text content.
            current_text (Union[str, ChatCompletionChunk]): The current text chunk or completion chunk.

        Yields:
            str: Extracted reasoning content chunks.
        """

        if not isinstance(delta_text, str):
            delta_text = delta_text["choices"][0]["text"]

        # Check if <think> is present in previous or delta.
        # Keep compatibility with models that don't generate <think> tokens.
        if self.reasoning_start_tag in previous_text:
            if self.reasoning_end_tag in delta_text:
                # <think> in previous, </think> in delta,
                # extract reasoning content
                end_idx = delta_text.find(self.reasoning_end_tag)
                reasoning_content = delta_text[:end_idx]
                content = delta_text[end_idx + len(self.reasoning_end_tag) :]
                delta = {
                    "reasoning_content": reasoning_content,
                }
                if content is not None:
                    delta["content"] = content
                return cast(ChatCompletionChunkDelta, delta)
            elif self.reasoning_end_tag in previous_text:
                # <think> in previous, </think> in previous,
                # <think> in previous, </think> in previous,
                # reasoning content ends
                return cast(ChatCompletionChunkDelta, {"content": delta_text})
            else:
                # <think> in previous, no </think> in previous or delta,
                # reasoning content continues
                return cast(ChatCompletionChunkDelta, {"reasoning_content": delta_text})
        elif self.reasoning_start_tag in delta_text:
            if self.reasoning_end_tag in delta_text:
                # <think> in delta, </think> in delta, extract reasoning content
                start_idx = delta_text.find(self.reasoning_start_tag)
                end_idx = delta_text.find(self.reasoning_end_tag)
                reasoning_content = delta_text[
                    start_idx + len(self.reasoning_start_tag) : end_idx
                ]
                content = delta_text[end_idx + len(self.reasoning_end_tag) :]
                delta = {
                    "reasoning_content": reasoning_content,
                }
                if content is not None:
                    delta["content"] = content
                return cast(ChatCompletionChunkDelta, delta)
            else:
                # <think> in delta, no </think> in delta,
                # reasoning content continues
                return cast(ChatCompletionChunkDelta, {"reasoning_content": delta_text})
        else:
            # No <think> in previous or delta, also need to check for </think>.
            # Because the model may have generated </think> without <think>
            # Ref https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f
            if self.reasoning_end_tag in delta_text:
                # </think> in delta with more tokens,
                # extract reasoning content and content
                end_idx = delta_text.find(self.reasoning_end_tag)
                reasoning_content = delta_text[:end_idx]
                content = delta_text[end_idx + len(self.reasoning_end_tag) :]
                delta = {
                    "reasoning_content": reasoning_content,
                }
                if content is not None:
                    delta["content"] = content
                return cast(ChatCompletionChunkDelta, delta)
            elif self.reasoning_end_tag in previous_text:
                # </think> in previous, thinking content ends
                return cast(ChatCompletionChunkDelta, {"content": delta_text})
            else:
                # no </think> in previous or delta, reasoning content continues
                return cast(ChatCompletionChunkDelta, {"reasoning_content": delta_text})

    def extract_reasoning_content(
        self, model_output: Union[str, Completion]
    ) -> Optional[ChatCompletionMessage]:
        """Extract reasoning content from DeepSeek-R1 model output.

        Args:
            content (str): The model output content to parse.

        Returns:
            Optional[str]: Extracted reasoning content, or None if no reasoning content found.
        """
        if not isinstance(model_output, str):
            model_output = model_output["choices"][0]["text"]
        # DeepSeek R1 doesn't generate <think> now.
        # Thus we assume the reasoning content is always at the start.
        # Ref https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f
        if self.reasoning_end_tag not in model_output:
            return ChatCompletionMessage(role="user", content=model_output)
        else:
            # Add a start token if it's missing to keep compatibility.
            if self.reasoning_start_tag not in model_output:
                model_output = f"{self.reasoning_start_tag}{model_output}"
            # Use a regex to find the reasoning content
            reasoning_content = self.reasoning_regex.findall(model_output)[0]

            end_index = len(
                f"{self.reasoning_start_tag}{reasoning_content}{self.reasoning_end_tag}"
            )
            final_output = model_output[end_index:]

            if len(final_output) == 0:
                return ChatCompletionMessage(
                    role="user",
                    content=final_output,
                    reasoning_content=reasoning_content,
                )
            return ChatCompletionMessage(
                role="user", content=final_output, reasoning_content=reasoning_content
            )
