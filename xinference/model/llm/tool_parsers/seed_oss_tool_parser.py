import re

from . import register_tool_parser
from .minimax_tool_parser import MiniMaxToolParser


@register_tool_parser("seed-oss")
class SeedOssToolParser(MiniMaxToolParser):
    """
    Tool parser implementation for Seed-OSS.

    The block has the shape of MiniMax's with different markup:
    <seed:tool_call> holds <function=name> blocks, which hold
    <parameter=key>value</parameter> pairs.
    """

    def __init__(self):
        super().__init__()
        self.think_start_token = "<seed:think>"
        self.think_end_token = "</seed:think>"
        self.tool_call_start_token = "<seed:tool_call>"
        self.tool_call_end_token = "</seed:tool_call>"

        self.think_regex = re.compile(r"<seed:think>(.*?)</seed:think>", re.DOTALL)
        self.tool_call_complete_regex = re.compile(
            r"<seed:tool_call>(.*?)</seed:tool_call>", re.DOTALL
        )
        self.tool_call_regex = re.compile(
            r"<seed:tool_call>.*?</seed:tool_call>|<seed:tool_call>.*?$",
            re.DOTALL,
        )
        self.invoke_regex = re.compile(
            r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL
        )
        self.param_regex = re.compile(
            r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL
        )

    def _tool_markup(self, text: str) -> str:
        """Blank reasoning, which may still be open, so tags quoted in it stay text."""
        parts = []
        position = 0
        while True:
            think = text.find(self.think_start_token, position)
            if think == -1:
                break
            block = text.find(self.tool_call_start_token, position, think)
            if block != -1:
                # A tool call argument may quote the reasoning tag.
                close = text.find(self.tool_call_end_token, block)
                if close == -1:
                    break
                parts.append(text[position:close])
                position = close
                continue
            close = text.find(self.think_end_token, think)
            end = len(text) if close == -1 else close + len(self.think_end_token)
            parts.append(text[position:think])
            parts.append(" " * (end - think))
            position = end
        parts.append(text[position:])
        return "".join(parts)
