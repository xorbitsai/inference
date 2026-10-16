from contextvars import ContextVar
from typing import Any, Dict, Iterable, List, Optional

# Tools of the request whose output is being parsed. One parser instance serves
# every request to a model, so the chat helpers set this around each parse call
# instead of storing the tools on the parser.
request_tools_var: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "request_tools", default=None
)


class ToolParser:
    """
    Abstract ToolParser class that should not be used directly. Provided
    properties and methods should be used in
    derived classes.
    """

    def extract_tool_calls(self, model_output: str):
        """
        Static method that should be implemented for extracting tool calls from
        a complete model-generated string.
        Used for non-streaming responses where we have the entire model response
        available before sending to the client.
        Static because it's stateless.
        """
        raise NotImplementedError(
            "AbstractToolParser.extract_tool_calls has not been implemented!"
        )

    def extract_tool_calls_streaming(
        self, previous_text, current_text: str, delta_text: str
    ):
        """
        Instance method that should be implemented for extracting tool calls
        from an incomplete response; for use when handling tool calls and
        streaming. Has to be an instance method because  it requires state -
        the current tokens/diffs, but also the information about what has
        previously been parsed and extracted (see constructor)
        """
        raise NotImplementedError(
            "AbstractToolParser.extract_tool_calls_streaming has not been "
            "implemented!"
        )

    @staticmethod
    def _is_string_parameter(function_name: Optional[str], parameter_name: str) -> bool:
        """
        Return True when the request's tools declare the parameter as
        ``"type": "string"``.

        Formats that write every argument as raw text cannot tell the string
        ``404`` from the number ``404``. Their parsers keep such values as text
        and decode the others as JSON.
        """
        for tool in request_tools_var.get() or ():
            function = tool.get("function", tool) if isinstance(tool, dict) else None
            if not isinstance(function, dict) or function.get("name") != function_name:
                continue
            parameters = function.get("parameters")
            properties = (
                parameters.get("properties") if isinstance(parameters, dict) else None
            )
            schema = (
                properties.get(parameter_name) if isinstance(properties, dict) else None
            )
            return isinstance(schema, dict) and schema.get("type") == "string"
        return False

    @staticmethod
    def _partial_marker_length(text: str, markers: Iterable[str]) -> int:
        """
        Return the length of the longest suffix of ``text`` that is a proper
        prefix of one of ``markers``, or 0 when there is none.
        """
        longest = 0
        for marker in markers:
            for length in range(min(len(text), len(marker) - 1), longest, -1):
                if text.endswith(marker[:length]):
                    longest = length
                    break
        return longest

    def _plain_text_delta(
        self, current_text: str, delta_text: str, markers: Iterable[str]
    ) -> Optional[str]:
        """
        Return the content that is safe to stream for this chunk while no
        marker has been seen yet, or None when there is nothing to send.

        A trailing piece of text that may be the start of a marker is held
        back. It is not part of ``delta_text`` on the next call, so it is
        released here, together with the new text, as soon as the stream
        shows it is not a marker.
        """
        markers = tuple(markers)
        previous_text = current_text[: len(current_text) - len(delta_text)]
        start = len(previous_text) - self._partial_marker_length(previous_text, markers)
        end = len(current_text) - self._partial_marker_length(current_text, markers)
        return current_text[start:end] or None
