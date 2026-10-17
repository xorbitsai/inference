# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Connect Xinference reasoning parsers to vLLM's structured decoding."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .. import reasoning_parser as parsers
from ..reasoning_parser import ReasoningParser


def configure_thinking_budget(
    model_config: Dict[str, Any], parser: Optional[ReasoningParser]
) -> None:
    if (
        parser is None
        or "reasoning_config" in model_config
        or not parser.reasoning_start_tag.startswith("<")
        or not parser.reasoning_end_tag
    ):
        return
    try:
        from vllm.config import ReasoningConfig
    except ImportError:
        return
    model_config["reasoning_config"] = ReasoningConfig(
        reasoning_start_str=parser.reasoning_start_tag,
        reasoning_end_str=parser.reasoning_end_tag,
    )


def create_reasoning_plugin(
    parser: ReasoningParser, cache_dir: str
) -> Optional[Tuple[str, str]]:
    """Write an immutable plugin that EngineCore can import under spawn."""
    if getattr(parsers, type(parser).__name__, None) is not type(parser):
        return None
    if not parser.reasoning_start_tag or not parser.reasoning_end_tag:
        return None
    # Harmony's channel names are not literal opening/closing text markers.
    # Preserve its specialized engine path rather than treating them as tags.
    if not parser.reasoning_start_tag.startswith("<"):
        return None
    config = parser.get_config()
    config["reasoning_content"] = True
    payload = {"parser_class": type(parser).__name__, "config": config}
    serialized = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    name = f"xinference_{digest}"
    directory = Path(cache_dir) / "vllm-reasoning"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.py"
    source = (
        "from xinference.model.llm.vllm.reasoning import register_reasoning_adapter\n"
        f"register_reasoning_adapter({name!r}, {payload!r})\n"
    )
    if not path.exists() or path.read_text() != source:
        fd, temporary = tempfile.mkstemp(dir=directory, suffix=".py")
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(source)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return name, str(path)


def configure_structured_reasoning(
    model_config: Dict[str, Any], parser: Optional[ReasoningParser], cache_dir: str
) -> None:
    if parser is None or any(
        key in model_config for key in ("reasoning_parser", "reasoning_parser_plugin")
    ):
        return
    configured = model_config.get("structured_outputs_config")
    if configured is not None and not isinstance(configured, dict):
        return
    structured = dict(configured or {})
    if "reasoning_parser" in structured or "reasoning_parser_plugin" in structured:
        return
    try:
        from vllm.config import StructuredOutputsConfig
    except ImportError:
        return
    # Older engines can continue ordinary generation without plugin support.
    if "reasoning_parser_plugin" not in getattr(
        StructuredOutputsConfig, "__dataclass_fields__", {}
    ):
        return
    plugin = create_reasoning_plugin(parser, cache_dir)
    if plugin is None:
        return
    name, path = plugin
    structured.update(reasoning_parser=name, reasoning_parser_plugin=path)
    model_config["structured_outputs_config"] = StructuredOutputsConfig(**structured)


def register_reasoning_adapter(name: str, payload: Dict[str, Any]) -> None:
    """Register a thin adapter in the actual engine process."""
    from vllm.reasoning import ReasoningParser as VLLMReasoningParser
    from vllm.reasoning import ReasoningParserManager

    parser_class = getattr(parsers, payload["parser_class"])

    class XinferenceReasoningAdapter(VLLMReasoningParser):
        def __init__(self, tokenizer: Any, *args: Any, **kwargs: Any):
            super().__init__(tokenizer, *args, **kwargs)
            self.parser = parser_class(**payload["config"])
            self.parser.bind_tokenizer(tokenizer)

        def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
            return self.parser.is_reasoning_end(input_ids)

        def is_reasoning_end_streaming(
            self, input_ids: Sequence[int], delta_ids: Iterable[int]
        ) -> bool:
            return self.parser.is_reasoning_end_streaming(input_ids, delta_ids)

        def extract_content_ids(self, input_ids: List[int]) -> List[int]:
            return self.parser.extract_content_ids(input_ids)

        def extract_reasoning(self, model_output: str, request: Any):
            return self.parser.extract_reasoning_content(model_output)

        def extract_reasoning_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Sequence[int],
            current_token_ids: Sequence[int],
            delta_token_ids: Sequence[int],
        ):
            try:
                from vllm.entrypoints.openai.engine.protocol import DeltaMessage
            except ImportError:
                from vllm.entrypoints.openai.protocol import DeltaMessage

            delta = self.parser.extract_reasoning_content_streaming(
                previous_text, current_text, delta_text
            )
            if not delta:
                return None
            field = (
                "reasoning"
                if "reasoning" in DeltaMessage.model_fields
                else "reasoning_content"
            )
            return DeltaMessage(
                **{
                    field: delta.get("reasoning_content"),
                    "content": delta.get("content"),
                }
            )

    ReasoningParserManager.register_module(name=name, module=XinferenceReasoningAdapter)
