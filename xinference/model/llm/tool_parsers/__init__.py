from functools import wraps
from typing import Any, Callable, Dict, Type

# Global registry for tool parsers, mapping parser names to their classes
TOOL_PARSERS: Dict[str, Type[Any]] = {}


def register_tool_parser(name: str):
    """
    Decorator for registering ToolParser classes to the TOOL_PARSERS registry.

    This decorator allows tool parser classes to be automatically registered
    when they are defined, making them available for dynamic lookup.

    Args:
        name (str): The name to register the tool parser under. This should
                   typically match the model family name (e.g., "qwen", "glm4").

    Returns:
        Callable: The decorator function that registers the class.

    Example:
        @register_tool_parser("qwen")
        class QwenToolParser(ToolParser):
            def parse_tool_calls(self, text: str) -> List[ToolCall]:
                # Implementation for parsing Qwen model tool calls
                pass

    Note:
        The registered class should implement the ToolParser interface
        and provide methods for parsing tool calls from model outputs.
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        """
        The actual decorator that performs the registration.

        Args:
            cls: The tool parser class to register.

        Returns:
            The same class (unmodified) after registration.
        """
        TOOL_PARSERS[name] = cls
        return cls

    return decorator


# Import all tool parser modules to trigger decorator registration
# This ensures all tool parsers are automatically registered when this module is imported
from . import (
    deepseek_r1_tool_parser,
    deepseek_v3_tool_parser,
    glm4_tool_parser,
    llama3_tool_parser,
    minimax_tool_parser,
    qwen_tool_parser,
)
