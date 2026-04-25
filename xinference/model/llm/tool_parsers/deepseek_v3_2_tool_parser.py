import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

logger = logging.getLogger(__name__)


@register_tool_parser("deepseek-v3.2")
class DeepseekV3_2ToolParser(ToolParser):
    """
    Tool parser implementation for DeepSeek V3.2 model.

    Handles two formats used by DeepSeek V3.2 for tool calls:

    DSML format:
    <｜DSML｜function_calls>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>

    Plain format:
    <function_calls>
    <invoke name="get_weather">
    <parameter name="location" string="true">杭州</parameter>
    <parameter name="date" string="true">2024-01-16</parameter>
    </invoke>
    </function_calls>
    """

    # Prefix used in DSML-flavoured tags
    _DSML = "<｜DSML｜"

    def __init__(self):
        super().__init__()

        self.tool_call_start_tokens: List[str] = [
            f"{self._DSML}function_calls>",
            "<function_calls>",
        ]

        # Streaming state
        self.is_tool_call_started: bool = False
        self.current_tool_index: int = 0
        self._use_dsml: bool = True  # detected per-stream

        # Regex patterns — built dynamically after format detection
        self.tool_call_complete_regex: re.Pattern = re.compile(r"")  # placeholder
        self.invoke_complete_regex: re.Pattern = re.compile(r"")  # placeholder
        self.parameter_complete_regex: re.Pattern = re.compile(r"")  # placeholder

    def _build_regexes(self, dsml: bool) -> None:
        """Compile regex patterns for the detected format."""
        # For DSML mode, open_tag = "<｜DSML｜", close_body = "｜DSML｜"
        # For plain mode,  open_tag = "<",        close_body = ""
        if dsml:
            open_tag = re.escape(self._DSML)  # <｜DSML｜
            close_body = open_tag[1:]  # ｜DSML｜
        else:
            open_tag = "<"
            close_body = ""

        self.tool_call_complete_regex = re.compile(
            rf"{open_tag}function_calls>(.*?)</{close_body}function_calls>", re.DOTALL
        )
        self.invoke_complete_regex = re.compile(
            rf"{open_tag}invoke\s+name=\"([^\"]+)\"\s*>(.*?)</{close_body}invoke>",
            re.DOTALL,
        )
        _non_cap = "(?:"  # noqa: E231 — split to avoid flake8 false positive on "?:"
        self.parameter_complete_regex = re.compile(
            rf'{open_tag}parameter\s+name="([^"]+)"\s+string="{_non_cap}true|false)"\s*>'
            rf"(.*?)</{close_body}parameter>",
            re.DOTALL,
        )

    def _parse_invoke_params(self, invoke_str: str) -> dict:
        """Parse parameter name-value pairs from an invoke block."""
        param_dict = {}
        for param_name, param_val in self.parameter_complete_regex.findall(invoke_str):
            param_dict[param_name] = param_val
        return param_dict

    def _detect_format(self, text: str) -> bool:
        """Return True when DSML-prefixed format is detected, False for plain."""
        return f"{self._DSML}function_calls>" in text

    def extract_tool_calls(
        self, model_output: str
    ) -> List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        """
        Extract tool calls from complete model output.

        Args:
            model_output: The complete output string from the model.

        Returns:
            A list of tuples where each tuple contains:
            - content: Raw content if parsing failed, None if successful
            - function_name: Name of the function to call
            - parameters: Function parameters as dict
        """
        # Detect which format is present
        has_any = any(tok in model_output for tok in self.tool_call_start_tokens)
        if not has_any:
            return [(model_output, None, None)]

        dsml = self._detect_format(model_output)
        self._build_regexes(dsml)

        results: List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]] = (
            []
        )

        for tool_call_match in self.tool_call_complete_regex.findall(model_output):
            for invoke_name, invoke_content in self.invoke_complete_regex.findall(
                tool_call_match
            ):
                param_dict = self._parse_invoke_params(invoke_content)
                results.append((None, invoke_name, param_dict))

        if not results:
            return [(model_output, None, None)]

        return results

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> Optional[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        """
        Extract tool calls from streaming output.

        Buffers tokens until a complete invoke block is available, then
        parses and returns it in one shot.

        Args:
            previous_text: Previous text chunks from the stream.
            current_text: Current accumulated text.
            delta_text: New text delta in this chunk.

        Returns:
            None if waiting for more data, or a tuple of (content, function_name, parameters).
        """
        try:
            # First chunk of a new stream — reset state
            if not previous_text:
                self.is_tool_call_started = False
                self.current_tool_index = 0
                self._use_dsml = True

            # Detect tool-call region
            if not self.is_tool_call_started:
                matched_token = None
                for tok in self.tool_call_start_tokens:
                    if tok in current_text:
                        matched_token = tok
                        break

                if matched_token:
                    self.is_tool_call_started = True
                    self._use_dsml = self._detect_format(current_text)
                    self._build_regexes(self._use_dsml)
                else:
                    # Still in plain-text region — check for partial start tokens
                    for tok in self.tool_call_start_tokens:
                        for i in range(1, len(tok)):
                            if current_text.endswith(tok[:i]):
                                return None
                    return (delta_text, None, None) if delta_text else None

            # Inside tool-call region: check for newly completed invokes
            complete_invokes = self.invoke_complete_regex.findall(current_text)

            if len(complete_invokes) > self.current_tool_index:
                invoke_name, invoke_body = complete_invokes[self.current_tool_index]
                self.current_tool_index += 1
                param_dict = self._parse_invoke_params(invoke_body)
                return (None, invoke_name, param_dict)

            # Buffering — no complete invoke yet
            return None

        except Exception as e:
            logger.error("Error in DeepSeek V3.2 streaming tool call extraction: %s", e)
            raise
