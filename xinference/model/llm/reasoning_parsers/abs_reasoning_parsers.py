from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Type, Union

from ....types import ChatCompletionChunkDelta, CompletionChoice


class ReasoningParser(ABC):
    """Abstract base class for reasoning content parsers."""

    def __init__(
        self,
        reasoning_start_tag: str = "<think>",
        reasoning_end_tag: str = "</think>",
    ):
        """Initialize the reasoning parser.

        Args:
            reasoning_start_tag (str, optional): Start tag for reasoning content. Defaults to "<think>".
            reasoning_end_tag (str, optional): End tag for reasoning content. Defaults to "</think>".
        """
        self.reasoning_start_tag = reasoning_start_tag
        self.reasoning_end_tag = reasoning_end_tag

    @abstractmethod
    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta: ChatCompletionChunkDelta,
    ) -> ChatCompletionChunkDelta:
        """Extract reasoning content from model output in a streaming fashion.

        Args:
            content (str): The model output content to parse.

        Yields:
            str: Extracted reasoning content chunks.
        """
        pass

    @abstractmethod
    def extract_reasoning_content(
        self, model_output: Union[str, CompletionChoice]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract reasoning content from model output.

        Args:
            content (str): The model output content to parse.

        Returns:
            Optional[str]: Extracted reasoning content, or None if no reasoning content found.
        """
        pass


class ReasoningParserManager:
    """Manager class for reasoning parsers."""

    _parsers: Dict[str, Type[ReasoningParser]] = {}

    @classmethod
    def register(cls, model_name: str, parser_cls: Type[ReasoningParser]) -> None:
        """Register a reasoning parser for a specific model.

        Args:
            model_name (str): The name of the model.
            parser_cls (Type[ReasoningParser]): The parser class to register.
        """
        cls._parsers[model_name] = parser_cls

    @classmethod
    def register_module(cls, model_name: str):
        """Decorator for registering a reasoning parser for a specific model.

        Args:
            model_name (str): The name of the model.

        Returns:
            Callable: The decorator function.
        """

        def _register(parser_cls: Type[ReasoningParser]) -> Type[ReasoningParser]:
            cls.register(model_name, parser_cls)
            return parser_cls

        return _register

    @classmethod
    def get_parser(cls, model_name: str) -> Optional[Type[ReasoningParser]]:
        """Get the registered parser for a specific model.

        Args:
            model_name (str): The name of the model.

        Returns:
            Optional[Type[ReasoningParser]]: The registered parser class, or None if not found.
        """
        return cls._parsers.get(model_name)
