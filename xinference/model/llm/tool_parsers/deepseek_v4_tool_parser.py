import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from . import register_tool_parser
from .abstract_tool_parser import ToolParser
from .argument_stream import ArgumentStart, dsml_argument_start
from .invoke_scan import invoke_end, iter_invokes

logger = logging.getLogger(__name__)

ToolCall = Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]


@register_tool_parser("deepseek-v4")
class DeepseekV4ToolParser(ToolParser):
    """
    Tool parser implementation for DeepSeek V4 model.

    Handles two formats used by DeepSeek V4 for tool calls:

    DSML format:
    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

    Plain format:
    <tool_calls>
    <invoke name="get_weather">
    <parameter name="location" string="true">杭州</parameter>
    <parameter name="date" string="true">2024-01-16</parameter>
    </invoke>
    </tool_calls>
    """

    # Prefix used in DSML-flavoured tags
    _DSML = "<｜DSML｜"

    def __init__(self):
        super().__init__()

        self.tool_call_start_tokens: List[str] = [
            f"{self._DSML}tool_calls>",
            "<tool_calls>",
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
            rf"{open_tag}tool_calls>(.*?)</{close_body}tool_calls>", re.DOTALL
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
        return f"{self._DSML}tool_calls>" in text

    def _wrapper_tags(self, dsml: bool) -> Tuple[str, str]:
        """Return the tags that open and close the tool calls block."""
        if dsml:
            return f"{self._DSML}tool_calls>", f"</{self._DSML[1:]}tool_calls>"
        return "<tool_calls>", "</tool_calls>"

    def _call_start(self, dsml: bool) -> str:
        return f"{self._DSML}invoke" if dsml else "<invoke"

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
        parameter_regex = self._regexes[dsml][2]
        wrapper_start, wrapper_end = self._wrapper_tags(dsml)

        results: List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]] = (
            []
        )

        content_start = 0
        position = 0
        while (block := model_output.find(wrapper_start, position)) != -1:
            body = block + len(wrapper_start)
            close = self.block_end(
                model_output,
                body,
                wrapper_end,
                self._call_start(dsml),
                lambda text, index: invoke_end(text, index, dsml, True),
            )
            if close == -1:
                break
            position = close + len(wrapper_end)
            invokes = list(iter_invokes(model_output, dsml, body, close, True))
            if not invokes:
                continue
            content = model_output[content_start:block]
            if content:
                results.append((content, None, None))
            for invoke_name, invoke_content, _ in invokes:
                param_dict = self._parse_invoke_params(invoke_content, parameter_regex)
                results.append((None, invoke_name, param_dict))
            content_start = position

        if not results:
            return [(model_output, None, None)]

        content = model_output[content_start:]
        if content:
            results.append((content, None, None))
        return results

    def locate_argument_start(self, text: str) -> Optional[ArgumentStart]:
        dsml = self._detect_format(text)
        return dsml_argument_start(
            text,
            "tool_calls",
            self._DSML[1:] if dsml else "",
            lambda body: self._parse_invoke_params(body, self._regexes[dsml][2]),
        )

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
            parameter_regex = self._regexes[dsml][2]
            previous_length = len(current_text) - len(delta_text)
            # An invoke completes when its closing tag arrives, or when the
            # start of the next invoke shows that a cut short JSON string
            # ended it. Skip the scan until one of them does.
            invoke_close = f"</{self._DSML[1:]}invoke>" if dsml else "</invoke>"
            invoke_start = self._call_start(dsml)
            # Text outside the tool call blocks is content, even when it
            # shares a chunk with a marker.
            wrapper_start, wrapper_end = self._wrapper_tags(dsml)
            content = self._text_outside_blocks(
                current_text,
                delta_text,
                wrapper_start,
                wrapper_end,
                self.tool_call_start_tokens,
                call_start=self._call_start(dsml),
                call_end=lambda text, index: invoke_end(text, index, dsml),
            )
            tail = max(
                previous_length - max(len(invoke_close), len(invoke_start)) + 1, 0
            )
            if (
                current_text.find(invoke_close, tail) == -1
                and current_text.find(invoke_start, tail) == -1
            ):
                return (content, None, None) if content else None
            results: List[ToolCall] = [(content, None, None)] if content else []
            # New invokes are counted, not located: a cut short invoke ends
            # before this delta, but is only known to be complete now.
            completed = len(list(iter_invokes(current_text[:previous_length], dsml)))
            for invoke_name, invoke_body, _ in list(iter_invokes(current_text, dsml))[
                completed:
            ]:
                param_dict = self._parse_invoke_params(invoke_body, parameter_regex)
                results.append((None, invoke_name, param_dict))

            if not results:
                # Buffering — no invoke completed in this delta
                return None
            return results[0] if len(results) == 1 else results

        except Exception as e:
            logger.error("Error in DeepSeek V4 streaming tool call extraction: %s", e)
            raise

    def flush_streaming_tool_calls(
        self, current_text: str
    ) -> List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        """
        Return what the finished stream adds once the text is known to be
        complete: a JSON string still open at the end was cut short, so the
        invokes it hid are complete, and so is the block that holds them.
        """
        if not any(tok in current_text for tok in self.tool_call_start_tokens):
            return []
        dsml = self._detect_format(current_text)
        parameter_regex = self._regexes[dsml][2]
        wrapper_start, wrapper_end = self._wrapper_tags(dsml)
        results: List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]] = (
            []
        )
        emitted = len(list(iter_invokes(current_text, dsml)))
        for invoke_name, invoke_body, _ in list(
            iter_invokes(current_text, dsml, final=True)
        )[emitted:]:
            param_dict = self._parse_invoke_params(invoke_body, parameter_regex)
            results.append((None, invoke_name, param_dict))
        if results:
            block = current_text.rfind(wrapper_start)
            close = self.block_end(
                current_text,
                block + len(wrapper_start),
                wrapper_end,
                self._call_start(dsml),
                lambda text, index: invoke_end(text, index, dsml, True),
            )
            if close != -1 and current_text[close + len(wrapper_end) :]:
                results.append((current_text[close + len(wrapper_end) :], None, None))
        return results
