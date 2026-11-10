import ast
import json
import logging
import tokenize
from io import StringIO
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

    def _leading_call(self, text: str) -> Tuple[int, List[Tuple]]:
        leading = len(text) - len(text.lstrip())
        candidate = text[leading:]
        if not candidate.startswith("{") or "}" not in candidate:
            return 0, []
        try:
            _, end = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError:
            # A Python literal can use single quotes and non-JSON scalars.
            # Tokenize only to find its boundary and let literal_eval validate.
            depth = 0
            try:
                for token in tokenize.generate_tokens(StringIO(candidate).readline):
                    if token.type != tokenize.OP:
                        continue
                    if token.string == "{":
                        depth += 1
                    elif token.string == "}":
                        depth -= 1
                        if depth == 0:
                            row, column = token.end
                            end = (
                                sum(
                                    len(line)
                                    for line in candidate.splitlines(keepends=True)[
                                        : row - 1
                                    ]
                                )
                                + column
                            )
                            break
                else:
                    return 0, []
            except (tokenize.TokenError, IndentationError):
                return 0, []
        result = self.extract_tool_calls(candidate[:end])
        _, name, arguments = result[0]
        if isinstance(name, str) and name and isinstance(arguments, dict):
            return leading + end, result
        return 0, []

    def flush_streaming_content(self, text: str) -> Optional[str]:
        if not text.strip():
            return text
        if text.lstrip().startswith("{") and not self._leading_call(text)[0]:
            return text
        return None

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> Optional[List[Tuple]]:
        if not delta_text or not current_text.strip():
            return None
        previous = previous_text[-1] if previous_text else ""
        if not current_text.lstrip().startswith("{"):
            return [(delta_text if previous.strip() else current_text, None, None)]

        previous_end, _ = self._leading_call(previous)
        if previous_end:
            return [(delta_text, None, None)]
        end, events = self._leading_call(current_text)
        if end and current_text[end:]:
            events.append((current_text[end:], None, None))
        return events or None
