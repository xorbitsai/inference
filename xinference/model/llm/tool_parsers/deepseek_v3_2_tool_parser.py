import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from . import register_tool_parser
from .abstract_tool_parser import ToolParser

logger = logging.getLogger(__name__)

ToolCall = Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]


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

        # One instance serves every request to the model, so nothing on it
        # may change per request. Compile the patterns of both formats once,
        # keyed by whether the text uses DSML tags.
        self._regexes = {dsml: self._build_regexes(dsml) for dsml in (True, False)}

    def _build_regexes(self, dsml: bool) -> Tuple[re.Pattern, re.Pattern, re.Pattern]:
        """Compile the tool call, invoke and parameter patterns of a format."""
        # For DSML mode, open_tag = "<｜DSML｜", close_body = "｜DSML｜"
        # For plain mode,  open_tag = "<",        close_body = ""
        if dsml:
            open_tag = re.escape(self._DSML)  # <｜DSML｜
            close_body = open_tag[1:]  # ｜DSML｜
        else:
            open_tag = "<"
            close_body = ""

        tool_call_regex = re.compile(
            rf"{open_tag}function_calls>(.*?)</{close_body}function_calls>", re.DOTALL
        )
        invoke_regex = re.compile(
            rf"{open_tag}invoke\s+name=\"([^\"]+)\"\s*>(.*?)</{close_body}invoke>",
            re.DOTALL,
        )
        parameter_regex = re.compile(
            rf'{open_tag}parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>'
            rf"(.*?)</{close_body}parameter>",
            re.DOTALL,
        )
        return tool_call_regex, invoke_regex, parameter_regex

    def _parse_invoke_params(
        self, invoke_str: str, parameter_regex: re.Pattern
    ) -> dict:
        """Parse parameter name-value pairs from an invoke block."""
        param_dict = {}
        for param_name, is_string, param_val in parameter_regex.findall(invoke_str):
            try:
                param_dict[param_name] = (
                    param_val if is_string == "true" else json.loads(param_val)
                )
            except json.JSONDecodeError:
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

        tool_call_regex, invoke_regex, parameter_regex = self._regexes[
            self._detect_format(model_output)
        ]

        results: List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]] = (
            []
        )

        content_start = 0
        for tool_call_match in tool_call_regex.finditer(model_output):
            invokes = invoke_regex.findall(tool_call_match.group(1))
            if not invokes:
                continue
            content = model_output[content_start : tool_call_match.start()]
            if content:
                results.append((content, None, None))
            for invoke_name, invoke_content in invokes:
                param_dict = self._parse_invoke_params(invoke_content, parameter_regex)
                results.append((None, invoke_name, param_dict))
            content_start = tool_call_match.end()

        if not results:
            return [(model_output, None, None)]

        content = model_output[content_start:]
        if content:
            results.append((content, None, None))
        return results

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> Optional[Union[ToolCall, List[ToolCall]]]:
        """
        Extract tool calls from streaming output.

        Buffers tokens until a complete invoke block is available, then
        parses and returns it in one shot. The parser is shared by
        concurrent requests, so progress is derived from this request's
        accumulated text and delta, never from instance state.

        Args:
            previous_text: Previous text chunks from the stream.
            current_text: Current accumulated text.
            delta_text: New text delta in this chunk.

        Returns:
            None if waiting for more data, a tuple of (content, function_name,
            parameters), or a list of them when one delta completes several
            invokes.
        """
        try:
            if not any(tok in current_text for tok in self.tool_call_start_tokens):
                # Still in plain-text region. Hold back a trailing partial
                # start token and release it once it turns out to be text.
                content = self._plain_text_delta(
                    current_text, delta_text, self.tool_call_start_tokens
                )
                return (content, None, None) if content else None

            # Inside tool-call region: return the invokes whose closing tag
            # arrived in this delta
            dsml = self._detect_format(current_text)
            _, invoke_regex, parameter_regex = self._regexes[dsml]
            previous_length = len(current_text) - len(delta_text)
            # Every invoke ends with its closing tag, so none can complete in
            # this delta unless the tag does. Skip the regex scan until then.
            invoke_close = f"</{self._DSML[1:]}invoke>" if dsml else "</invoke>"
            if (
                current_text.find(
                    invoke_close, max(previous_length - len(invoke_close) + 1, 0)
                )
                == -1
            ):
                return None
            results: List[ToolCall] = []
            for invoke in invoke_regex.finditer(current_text):
                if invoke.end() > previous_length:
                    invoke_name, invoke_body = invoke.groups()
                    param_dict = self._parse_invoke_params(invoke_body, parameter_regex)
                    results.append((None, invoke_name, param_dict))

            if not results:
                # Buffering — no invoke completed in this delta
                return None
            return results[0] if len(results) == 1 else results

        except Exception as e:
            logger.error("Error in DeepSeek V3.2 streaming tool call extraction: %s", e)
            raise
