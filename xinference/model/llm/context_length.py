# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Engine-independent per-request context configuration for LLM launches."""

import os
from typing import Any, Dict, Optional

MAX_CONTEXT_LENGTH = 2**31 - 1
CONTEXT_LENGTH_KEYS = {
    "vllm": "max_model_len",
    "sglang": "context_length",
    "llama.cpp": "n_ctx",
    "transformers": "context_length",
    "mlx": "context_length",
    "lmdeploy": "session_len",
}


class ContextLengthExceeded(ValueError):
    """The request leaves no output capacity within its configured context."""


def validate_context_length(value: Optional[int]) -> Optional[int]:
    if value is not None and (
        type(value) is not int or not 1 <= value <= MAX_CONTEXT_LENGTH
    ):
        raise ValueError(
            f"context_length must be an integer between 1 and {MAX_CONTEXT_LENGTH}"
        )
    return value


def validate_context_length_launch(
    model_type: str, engine: Optional[str], config: dict
) -> None:
    context = validate_context_length(config.get("context_length"))
    if context is None:
        return
    if model_type.lower() != "llm":
        raise ValueError("context_length is only supported for LLM launches")
    if (engine or "").lower() not in CONTEXT_LENGTH_KEYS:
        raise ValueError(f"context_length is not supported by engine {engine}")


def map_context_length(engine: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Map only an explicit unified setting; retain legacy engine defaults.

    llama.cpp divides n_ctx among parallel slots. Resolve its host-dependent
    slot default on the worker, where the model instance is created.
    """
    result = dict(config)
    context = validate_context_length(result.pop("context_length", None))
    if context is None:
        return result
    validate_context_length_launch("LLM", engine, {"context_length": context})
    key = CONTEXT_LENGTH_KEYS[engine.lower()]
    native = context
    if engine.lower() == "llama.cpp":
        slots = result.get("n_parallel", min(8, os.cpu_count() or 1))
        if type(slots) is not int or slots <= 0:
            raise ValueError(
                "n_parallel must be a positive integer when context_length is set"
            )
        native *= slots
        if native > MAX_CONTEXT_LENGTH:
            raise ValueError(
                "context_length * n_parallel exceeds the llama.cpp context limit"
            )
        result["n_parallel"] = slots
    if (
        key in result
        and result[key] is not None
        and (type(result[key]) is not int or result[key] != native)
    ):
        raise ValueError(
            f"context_length conflicts with {key}; expected {native}, got {result[key]}"
        )
    result[key] = native
    return result
