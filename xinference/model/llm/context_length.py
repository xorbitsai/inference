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

import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

from ..._compat import BaseModel, Field

if TYPE_CHECKING:
    from .llm_family import LLMFamilyV2

MAX_CONTEXT_LENGTH = 2**31 - 1
CONTEXT_LENGTH_KEYS = {
    "vllm": "max_model_len",
    "sglang": "context_length",
    "llama.cpp": "n_ctx",
    "transformers": "context_length",
    "mlx": "context_length",
    "trt-llm": "max_seq_len",
}


class ContextLengthExtension(BaseModel):
    """A model's verified YaRN recipe, independent of its native default."""

    max_context_length: int = Field(gt=0)
    engines: List[Literal["vllm", "sglang"]]
    rope_parameters: Dict[str, Any]
    rope_parameters_path: List[str] = Field(min_items=1)


def _has_rope_override(value: Any, checkpoint: bool = False) -> bool:
    if not isinstance(value, dict):
        return False
    for key, item in value.items():
        if key in ("rope_parameters", "rope_scaling"):
            if not checkpoint:
                return True
            if isinstance(item, dict) and (
                item.get("rope_type", item.get("type", "default")) != "default"
                or item.get("factor", 1) != 1
            ):
                return True
        if isinstance(item, dict) and _has_rope_override(item, checkpoint):
            return True
    return False


def configure_auto_yarn(
    family: "LLMFamilyV2",
    engine: str,
    config: Dict[str, Any],
    model_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Derive YaRN only for an explicit extended context and a declared recipe.

    Return the required model-process environment flag separately, so the worker
    can set it before spawning the model without changing its own environment.
    Manual RoPE settings, including scaled checkpoint configs, remain authoritative.
    """
    result = dict(config)
    extension = getattr(family, "context_length_extension", None)
    native = getattr(family, "context_length", None)
    engine = engine.lower()
    if extension is None or engine not in extension.engines or not native:
        return result, None
    key = CONTEXT_LENGTH_KEYS[engine]
    requested = result.get("context_length") or result.get(key)
    if type(requested) is not int or requested <= native:
        return result, None
    # Validate aliases before any launch side effects.
    map_context_length(engine, result)
    override_key = "hf_overrides" if engine == "vllm" else "json_model_override_args"
    overrides = result.get(override_key) or {}
    if callable(overrides):
        return result, None
    if isinstance(overrides, str):
        try:
            overrides = json.loads(overrides)
        except ValueError as exc:
            raise ValueError(f"{override_key} must be a JSON object") from exc
    if not isinstance(overrides, dict):
        raise ValueError(f"{override_key} must be a JSON object")
    if "rope_scaling" in result or _has_rope_override(overrides):
        return result, None
    if model_path:
        try:
            checkpoint = json.loads((Path(model_path) / "config.json").read_text())
        except (OSError, ValueError):
            checkpoint = {}
        if _has_rope_override(checkpoint, checkpoint=True):
            return result, None
    if requested > extension.max_context_length:
        raise ValueError(
            f"{family.model_name} automatic YaRN supports context_length up to "
            f"{extension.max_context_length}; got {requested}"
        )
    overrides = deepcopy(overrides)
    target = overrides
    for part in extension.rope_parameters_path[:-1]:
        target = target.setdefault(part, {})
        if not isinstance(target, dict):
            raise ValueError(f"{override_key} has an invalid YaRN configuration path")
    parameters = dict(extension.rope_parameters)
    parameters["factor"] = float(math.ceil(requested / native))
    target[extension.rope_parameters_path[-1]] = parameters
    result[override_key] = overrides if engine == "vllm" else json.dumps(overrides)
    flag = (
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN"
        if engine == "vllm"
        else "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"
    )
    return result, flag


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
