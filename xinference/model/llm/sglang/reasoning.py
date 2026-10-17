# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Expose Xinference's existing parsers to SGLang reasoning grammars."""

import hashlib
import inspect
import json
import os
import re
import sys
import tempfile
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Dict, List, Optional

from ....constants import XINFERENCE_CACHE_DIR
from .. import reasoning_parser as parsers
from ..reasoning_parser import ReasoningParser

PLUGIN_NAME = "xinference_reasoning"
PARSER_NAME = re.compile(r"xinference_[0-9a-f]{64}\Z")


def _config_path(name: str) -> Path:
    return Path(XINFERENCE_CACHE_DIR) / "sglang-reasoning" / f"{name}.json"


def configure_structured_reasoning(
    model_config: Dict[str, Any],
    parser: Optional[ReasoningParser],
    stop: Optional[List[str]] = None,
) -> None:
    if parser is None or "reasoning_parser" in model_config:
        return
    if getattr(parsers, type(parser).__name__, None) is not type(parser):
        return
    if not parser.reasoning_start_tag or not parser.reasoning_end_tag:
        return
    if not parser.reasoning_start_tag.startswith("<"):
        return
    try:
        from sglang.srt.plugins import load_plugins  # noqa: F401
    except ImportError:
        return
    # The entry point must also be discoverable by spawned scheduler processes.
    if not any(
        ep.name == PLUGIN_NAME for ep in entry_points(group="sglang.srt.plugins")
    ):
        return
    config = parser.get_config()
    config["reasoning_content"] = True
    payload = {
        "parser_class": type(parser).__name__,
        "config": config,
        "stop": stop or [],
    }
    serialized = json.dumps(payload, sort_keys=True)
    name = "xinference_" + hashlib.sha256(serialized.encode()).hexdigest()
    path = _config_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(serialized)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    plugins = os.environ.get("SGLANG_PLUGINS")
    if plugins and PLUGIN_NAME not in plugins.split(","):
        os.environ["SGLANG_PLUGINS"] = f"{plugins},{PLUGIN_NAME}"
    register()
    model_config["reasoning_parser"] = name
    try:
        from sglang.srt.constrained.reasoner_grammar_backend import (
            ReasonerGrammarBackend,
        )
    except ImportError:
        return
    if (
        model_config.get("grammar_backend") in (None, "xgrammar")
        and "enable_strict_thinking"
        in inspect.signature(ReasonerGrammarBackend).parameters
    ):
        # Otherwise an EOS inside reasoning can bypass the JSON grammar entirely.
        model_config.setdefault("enable_strict_thinking", True)


def _detector_class(payload: Dict[str, Any]):
    from sglang.srt.parser.reasoning_parser import (
        BaseReasoningFormatDetector,
        StreamingParseResult,
    )

    parser_class = getattr(parsers, payload["parser_class"])
    if not isinstance(parser_class, type) or not issubclass(
        parser_class, ReasoningParser
    ):
        raise ValueError("Invalid Xinference reasoning parser class")

    class XinferenceDetector(BaseReasoningFormatDetector):
        def __init__(
            self,
            stream_reasoning: bool = True,
            force_reasoning: Optional[bool] = None,
            tokenizer: Any = None,
            **kwargs: Any,
        ):
            self.parser = parser_class(**payload["config"])
            detector_config: Dict[str, Any] = {
                "think_start_token": self.parser.reasoning_start_tag,
                "think_end_token": self.parser.reasoning_end_tag,
                "force_reasoning": bool(force_reasoning),
                "stream_reasoning": stream_reasoning,
            }
            if (
                "think_excluded_tokens"
                in inspect.signature(BaseReasoningFormatDetector).parameters
            ):
                excluded = []
                if tokenizer is not None:
                    end_ids = tokenizer.encode(
                        self.parser.reasoning_end_tag, add_special_tokens=False
                    )
                    candidates = [*payload.get("stop", [])]
                    if getattr(tokenizer, "eos_token", None):
                        candidates.append(tokenizer.eos_token)
                    for token in dict.fromkeys(candidates):
                        ids = tokenizer.encode(token, add_special_tokens=False)
                        if (
                            len(ids) == 1
                            and ids[0] not in end_ids
                            and ids[0] != getattr(tokenizer, "unk_token_id", None)
                        ):
                            excluded.append(token)
                detector_config["think_excluded_tokens"] = excluded
            super().__init__(**detector_config)
            self._text = ""
            if tokenizer is not None:
                self.parser.bind_tokenizer(tokenizer)

        def get_think_end_token_ids(self, tokenizer: Any):
            self.parser.bind_tokenizer(tokenizer)
            return list(self.parser._end_token_ids)

        def detect_and_parse(self, text: str):
            reasoning, content = self.parser.extract_reasoning_content(text)
            return StreamingParseResult(normal_text=content, reasoning_text=reasoning)

        def parse_streaming_increment(self, text: str):
            previous = self._text
            self._text += text
            delta = self.parser.extract_reasoning_content_streaming(
                previous, self._text, text
            )
            return StreamingParseResult(
                normal_text=delta.get("content"),
                reasoning_text=delta.get("reasoning_content"),
            )

    return XinferenceDetector


class _DetectorMap(dict):
    def get(self, name: str, default: Any = None):
        if name not in self and PARSER_NAME.fullmatch(name):
            path = _config_path(name)
            if path.is_file():
                serialized = path.read_text()
                expected = (
                    "xinference_" + hashlib.sha256(serialized.encode()).hexdigest()
                )
                if name != expected:
                    raise ValueError(
                        "Xinference reasoning parser configuration changed"
                    )
                self[name] = _detector_class(json.loads(serialized))
        return super().get(name, default)


def register() -> None:
    """Run in the Runtime parent and every spawned SGLang process."""
    from sglang.srt.parser.reasoning_parser import ReasoningParser as SGLangParser

    if not isinstance(SGLangParser.DetectorMap, _DetectorMap):
        SGLangParser.DetectorMap = _DetectorMap(SGLangParser.DetectorMap)
    # Runtime uses the CLI in some releases. Make its generated parser name
    # visible to argparse choices before the command line is validated.
    for argument in sys.argv:
        name = argument.split("=", 1)[-1]
        if PARSER_NAME.fullmatch(name):
            SGLangParser.DetectorMap.get(name)
