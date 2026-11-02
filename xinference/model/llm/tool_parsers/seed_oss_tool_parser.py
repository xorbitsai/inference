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
        self.content_regex = (
            r"(<(?:seed:think|seed:tool_call)>.*?</(?:seed:think|seed:tool_call)>)"
        )
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
