import logging
from typing import Any, Dict, List, Optional, Tuple

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

logger = logging.getLogger(__name__)


@register_tool_parser("llama3")
class Llama3ToolParser(ToolParser):
    """
    Tool parser implementation for Llama3 model.

    This parser handles the specific format used by Llama3 for tool calls,
    which uses Python dictionary format that needs to be evaluated safely.

    """

    def __init__(self):
        """
        Initialize the Llama3 tool parser.
        """
        super().__init__()

    def extract_tool_calls(
        self, model_output: str
    ) -> List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        """
        Extract tool calls from complete model output.

        Parses the model output using eval() to extract tool call information.
        This method expects the output to be a valid Python dictionary format.

        Args:
            model_output (str): The complete output string from the model.

        Returns:
            List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
            A list of tuples where each tuple contains:
            - content (str or None): Raw content if parsing failed, None if successful
            - function_name (str or None): Name of the function to call
            - parameters (dict or None): Function parameters
        """
        try:
            data = eval(model_output, {}, {})
            return [(None, data["name"], data["parameters"])]
        except Exception:
            return [(model_output, None, None)]

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> Optional[Any]:
        """
        Extract tool calls from streaming output.

        Currently not supported for Llama3 model. This method raises
        a ValueError indicating that streaming tool call extraction is only
        available for specific model/backend combinations.

        Args:
            previous_text (List[str]): Previous text chunks from the stream.
            current_text (str): Current accumulated text.
            delta_text (str): New text delta in this chunk.

        Raises:
            ValueError: Always raised as streaming is not supported.

        Note:
            Llama3 model does not currently support streaming tool call
            extraction. Use extract_tool_calls() with complete output instead.
        """
        raise NotImplementedError(
            "Streaming support for tool calls is available only when using "
            "Qwen models with vLLM backend or GLM4-chat models without vLLM backend. "
            "Llama3 does not support streaming tool call extraction."
        )
