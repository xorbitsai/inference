import re
from typing import Optional, Tuple, Union

from ...types import ChatCompletionChunkDelta, CompletionChoice


class ReasoningParser:
    """Reasoning parser for reasoning model."""

    def __init__(
        self,
        reasoning_content: bool = False,
        reasoning_start_tag: str = "<think>",
        reasoning_end_tag: str = "</think>",
    ):
        self.reasoning_content = reasoning_content
        self.reasoning_start_tag = reasoning_start_tag
        self.reasoning_end_tag = reasoning_end_tag
        self.reasoning_regex = re.compile(
            rf"{self.reasoning_start_tag}(.*?){self.reasoning_end_tag}", re.DOTALL
        )

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> ChatCompletionChunkDelta:
        """Extract reasoning content from DeepSeek-R1 model output in a streaming fashion.

        Args:
            previous_text (str): The previous accumulated text content.
            current_text (Union[str, ChatCompletionChunk]): The current text chunk or completion chunk.

        Yields:
            str: Extracted reasoning content chunks.
        """
        delta = ChatCompletionChunkDelta()

        # Check if <think> is present in previous or delta.
        # Keep compatibility with models that don't generate <think> tokens.
        if self.reasoning_start_tag in previous_text:
            if self.reasoning_end_tag in delta_text:
                # <think> in previous, </think> in delta,
                # extract reasoning content
                end_idx = delta_text.find(self.reasoning_end_tag)
                reasoning_content = delta_text[:end_idx]
                content = delta_text[end_idx + len(self.reasoning_end_tag) :]
                delta["reasoning_content"] = reasoning_content
                if content:
                    delta["content"] = content
                else:
                    delta["content"] = None
                return delta
            elif self.reasoning_end_tag in previous_text:
                # <think> in previous, </think> in previous,
                # <think> in previous, </think> in previous,
                # reasoning content ends
                delta["reasoning_content"] = None
                delta["content"] = delta_text
                return delta
            else:
                # <think> in previous, no </think> in previous or delta,
                # reasoning content continues
                delta["reasoning_content"] = delta_text
                delta["content"] = None
                return delta
        elif self.reasoning_start_tag in delta_text:
            start_idx = delta_text.find(self.reasoning_start_tag)
            if self.reasoning_end_tag in delta_text:
                # <think> in delta, </think> in delta, extract reasoning content
                end_idx = delta_text.find(self.reasoning_end_tag)
                reasoning_content = delta_text[
                    start_idx + len(self.reasoning_start_tag) : end_idx
                ]
                content = delta_text[end_idx + len(self.reasoning_end_tag) :]
                delta["reasoning_content"] = reasoning_content
                if content:
                    delta["content"] = content
                else:
                    delta["content"] = None
                return delta
            else:
                # <think> in delta, no </think> in delta,
                # reasoning content continues
                reasoning_content = delta_text[
                    start_idx + len(self.reasoning_start_tag) :
                ]
                delta["reasoning_content"] = reasoning_content
                delta["content"] = None
                return delta
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
                delta["reasoning_content"] = reasoning_content
                if content:
                    delta["content"] = content
                else:
                    delta["content"] = None
                return delta
            elif self.reasoning_end_tag in previous_text:
                # </think> in previous, thinking content ends
                delta["reasoning_content"] = None
                delta["content"] = delta_text
                return delta
            else:
                # no </think> in previous or delta, reasoning content continues
                delta["reasoning_content"] = delta_text
                delta["content"] = None
                return delta

    def extract_reasoning_content(
        self, model_output: Union[str, CompletionChoice]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract reasoning content from DeepSeek-R1 model output.

        Args:
            content (str): The model output content to parse.

        Returns:
            Optional[str]: Extracted reasoning content, or None if no reasoning content found.
        """
        if not isinstance(model_output, str):
            model_output = model_output["text"]
        # DeepSeek R1 doesn't generate <think> now.
        # Thus we assume the reasoning content is always at the start.
        # Ref https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f
        if self.reasoning_end_tag not in model_output:
            return model_output, ""
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
                return reasoning_content, ""
            return reasoning_content, final_output

    def extract_content(self, model_output: Union[str, CompletionChoice]) -> str:
        """Ensures that the model output string starts with the reasoning_start_tag.

        If the model_output is not a string (e.g., CompletionChoice), it extracts
        the text content. If the reasoning_start_tag is not found in the text,
        it prepends the tag to the text.

        Args:
            model_output (Union[str, CompletionChoice]): The model output, can be a raw string
                or a CompletionChoice object from which text needs to be extracted.

        Returns:
            str: The model output string, guaranteed to start with reasoning_start_tag
                 if it was initially missing.
        """
        text_to_process = model_output
        # If model_output is a CompletionChoice object, extract the actual text.
        if not isinstance(text_to_process, str):
            text_to_process = text_to_process["text"]

        # If the reasoning_start_tag (e.g., "<think>") is not in the text,
        # prepend it to the beginning of the text.
        if self.reasoning_start_tag not in text_to_process:
            text_to_process = f"{self.reasoning_start_tag}{text_to_process}"
            return text_to_process

        # If the tag is already present, return the text as is.
        return text_to_process

    def check_content_parser(self):
        return self.reasoning_content
