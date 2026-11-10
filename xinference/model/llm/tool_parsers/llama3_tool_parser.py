import ast
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from . import register_tool_parser
from .abstract_tool_parser import ToolParser
from .argument_stream import ArgumentStart, json_argument_start

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

        Parses the model output using safe parsing to extract tool call information.
        This method expects the output to be a valid JSON or Python literal dictionary format.

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
            # Try JSON first (most common LLM output format)
            data = json.loads(model_output)
        except (json.JSONDecodeError, TypeError):
            try:
                # Fall back to ast.literal_eval for Python literal formats
                # (e.g., True/False/None instead of true/false/null).
                # Unlike eval(), ast.literal_eval() only allows literal
                # structures and cannot execute arbitrary code.
                data = ast.literal_eval(model_output)
            except (ValueError, SyntaxError):
                return [(model_output, None, None)]
        try:
            return [(None, data["name"], data["parameters"])]
        except (KeyError, TypeError):
            return [(model_output, None, None)]

    @staticmethod
    def flush_streaming_content(text: str) -> Optional[str]:
        # A leading brace might be ordinary text or a JSON/Python tool call.
        # If no call was found, release the buffered content at the end.
        if not text.strip() or text.lstrip().startswith("{"):
            return text
        return None

    def locate_argument_start(self, text: str) -> Optional[ArgumentStart]:
        if text.rstrip().endswith("}"):
            return None
        return json_argument_start(text, 0)

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> List[Tuple]:
        if not current_text.strip():
            return []
        if not current_text.lstrip().startswith("{"):
            previous = previous_text[-1] if previous_text else ""
            return [(delta_text if previous.strip() else current_text, None, None)]
        previous = previous_text[-1] if previous_text else ""
        if (
            not current_text.rstrip().endswith("}")
            or current_text.strip() == previous.strip()
        ):
            return []
        result = self.extract_tool_calls(current_text)
        return result if result[0][1] else []
