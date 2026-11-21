from contextvars import ContextVar
from typing import Any, Dict, Iterable, List, NamedTuple, Optional

# Tools of the request whose output is being parsed. One parser instance serves
# every request to a model, so the chat helpers set this around each parse call
# instead of storing the tools on the parser.
request_tools_var: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "request_tools", default=None
)


class ToolArgumentsDelta(NamedTuple):
    """An already serialized fragment of OpenAI function arguments."""

    value: str


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

    def flush_streaming_content(self, text: str) -> Optional[str]:
        """Release any plain text withheld at the end of a stream."""
        return None

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

    def _text_outside_blocks(
        self,
        current_text: str,
        delta_text: str,
        start: str,
        end: str,
        held_markers: Optional[Iterable[str]] = None,
        call_start: Optional[str] = None,
        call_end: Optional[str] = None,
    ) -> Optional[str]:
        """
        Return the plain text this chunk adds outside ``start``...``end``
        blocks, or None when there is none.

        Text before a block, between blocks and after the last one is content,
        even when it shares a chunk with a marker. A trailing piece that may be
        the start of a marker (``held_markers``, default ``start``) is held
        back, as in ``_plain_text_delta``, and nothing is returned while a block
        is still open.

        ``end`` only closes a block outside a call. When ``call_start`` and
        ``call_end`` are given, a call that has started is skipped up to its
        ``call_end``, so an ``end`` marker written inside a tool argument does
        not close the block.
        """
        markers = tuple(held_markers or (start,))
        previous_length = len(current_text) - len(delta_text)
        floor = previous_length - self._partial_marker_length(
            current_text[:previous_length], markers
        )
        parts = []
        position = 0
        while (block := current_text.find(start, position)) != -1:
            parts.append(current_text[max(position, floor) : block])
            body_end = self._block_end(
                current_text, block + len(start), end, call_start, call_end
            )
            if body_end == -1:
                return "".join(parts) or None
            position = body_end + len(end)
        content_end = len(current_text) - self._partial_marker_length(
            current_text, markers
        )
        parts.append(current_text[max(position, floor) : content_end])
        return "".join(parts) or None

    @staticmethod
    def _block_end(
        text: str,
        position: int,
        end: str,
        call_start: Optional[str],
        call_end: Optional[str],
    ) -> int:
        """
        Return the index of the ``end`` marker that closes the block whose body
        starts at ``position``, or -1 while it is still open. Text between a
        ``call_start`` and its ``call_end`` is never searched for ``end``.
        """
        while True:
            closing = text.find(end, position)
            if not call_start or not call_end:
                return closing
            opening = text.find(call_start, position)
            if opening == -1 or (closing != -1 and closing < opening):
                return closing
            call_close = text.find(call_end, opening + len(call_start))
            if call_close == -1:
                return -1
            position = call_close + len(call_end)
