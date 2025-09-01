from typing import Dict, Type, Any, Callable
from functools import wraps

TOOL_PARSERS: Dict[str, Type[Any]] = {}


def register_tool_parser(name: str):
    """
    装饰器用于注册 ToolParser 类到 TOOL_PARSERS 字典中
    
    Args:
        name: 注册的工具解析器名称
    
    Usage:
        @register_tool_parser("qwen")
        class QwenToolParser(ToolParser):
            pass
    """
    def decorator(cls: Type[Any]) -> Type[Any]:
        TOOL_PARSERS[name] = cls
        return cls
    return decorator


# 导入所有工具解析器以触发装饰器注册
from . import qwen_tool_parser, glm4_tool_parser, llama3_tool_parser, deepseek_v3_tool_parser, deepseek_r1_tool_parser