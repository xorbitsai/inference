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
import re
from copy import deepcopy
from numbers import Integral
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
)

from ..._compat import BaseModel, Field, validator

if TYPE_CHECKING:
    from .llm_family import LLMFamilyV2

MAX_CONTEXT_LENGTH = 2**31 - 1
CONTEXT_LENGTH_EXCEEDED_DETAIL = (
    "Request exceeds the model's configured context length."
)
INVALID_MULTIMODAL_INPUT_DETAIL = "Invalid multimodal input or processing parameters."
MODEL_OPERATION_ERROR_KEY = "_xinference_error"
MODEL_OPERATION_ERROR_VERSION = 1
CONTEXT_LENGTH_KEYS = {
    "vllm": "max_model_len",
    "sglang": "context_length",
    "llama.cpp": "n_ctx",
    "transformers": "context_length",
    "mlx": "context_length",
    "trt-llm": "max_seq_len",
}

_VLLM_PROMPT_LENGTH_ERROR = re.compile(
    r"The (?:decoder|encoder) prompt \(length (?P<input_tokens>\d+)\)\s+"
    r"(?:is longer than|plus the number of requested output tokens "
    r"\(at least 1\) is longer than)\s+"
    r"the maximum model length of (?P<max_model_len>\d+)\."
)
_XINFERENCE_MULTIMODAL_CONTEXT_ERROR = re.compile(
    r"Multimodal input uses (?P<input_tokens>\d+) tokens, leaving no output "
    r"capacity within max_model_len=(?P<max_model_len>\d+)"
)
_MULTIMODAL_LIMIT_ERROR = re.compile(
    r"At most \d+ (?:image|video|audio)\(s\) may be provided in one "
    r"prompt; received \d+\."
)
_MULTIMODAL_OPERATIONS = frozenset(
    {"multimodal_preflight", "anthropic_prepare", "token_count"}
)
_KNOWN_MULTIMODAL_INPUT_ERROR_TYPES = frozenset(
    {
        ("PIL", "UnidentifiedImageError"),
        ("PIL.Image", "DecompressionBombError"),
    }
)


def normalize_runtime_positive_int(value: Any) -> Optional[int]:
    """Normalize a trusted engine/runtime integer to a JSON-safe int."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    normalized = int(value)
    return normalized if normalized > 0 else None


class ContextLengthExtension(BaseModel):
    """A model's verified YaRN recipe, independent of its native default."""

    max_context_length: int = Field(gt=0)
    engines: List[Literal["vllm", "sglang", "mlx", "llama.cpp"]] = Field(
        min_items=1, unique_items=True
    )
    rope_parameters: Dict[str, Any]
    rope_parameters_path: List[str] = Field(min_items=1)

    @validator("rope_parameters_path", each_item=True)
    def validate_rope_path(cls, value: str) -> str:
        if not value:
            raise ValueError("rope_parameters_path items must not be empty")
        return value

    @validator("rope_parameters")
    def validate_original_context(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        original = value.get("original_max_position_embeddings")
        if original is not None and (type(original) is not int or original <= 0):
            raise ValueError(
                "original_max_position_embeddings must be a positive integer"
            )
        return value


def _has_rope_override(value: Any, checkpoint: bool = False) -> bool:
    if not isinstance(value, dict):
        return False
    for key, item in value.items():
        if key in ("rope_parameters", "rope_scaling"):
            if not checkpoint:
                return True
            if isinstance(item, dict) and (
                item.get("rope_type", item.get("type"))
                in {"yarn", "linear", "dynamic", "longrope", "llama3"}
                or (type(item.get("factor")) in (int, float) and item["factor"] > 1)
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
    can propagate the resolved decision without changing its own environment.
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
    if engine == "llama.cpp" and not result.get("context_length"):
        slots = result.get("n_parallel", min(8, os.cpu_count() or 1))
        if type(requested) is int and type(slots) is int and slots > 0:
            requested //= slots
    if type(requested) is not int or requested <= native:
        return result, None
    # Validate aliases before any launch side effects.
    map_context_length(engine, result)
    if engine in ("mlx", "llama.cpp"):
        if _has_rope_override(result) or (
            engine == "llama.cpp"
            and any(k.startswith(("rope_", "yarn_")) for k in result)
        ):
            return result, None
        checkpoint = _read_checkpoint_config(family, engine, model_path)
        if _has_rope_override(checkpoint, checkpoint=True):
            return result, None
        parameters = _yarn_parameters(family, requested)
        if engine == "mlx":
            # Consumed by the MLX adapter, never written into the cached config.
            result["_xinference_auto_yarn"] = parameters
        else:
            result.update(
                rope_scaling_type="yarn",
                rope_freq_scale=1.0 / parameters["factor"],
                yarn_orig_ctx=parameters["original_max_position_embeddings"],
                yarn_ext_factor=1.0,
                yarn_beta_fast=parameters.get("beta_fast", 32.0),
                yarn_beta_slow=parameters.get("beta_slow", 1.0),
            )
        return result, None
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
    parameters = _yarn_parameters(family, requested)
    overrides = deepcopy(overrides)
    target = overrides
    for part in extension.rope_parameters_path[:-1]:
        target = target.setdefault(part, {})
        if not isinstance(target, dict):
            raise ValueError(f"{override_key} has an invalid YaRN configuration path")
    target[extension.rope_parameters_path[-1]] = parameters
    result[override_key] = overrides if engine == "vllm" else json.dumps(overrides)
    flag = (
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN"
        if engine == "vllm"
        else "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"
    )
    return result, flag


def _yarn_parameters(family: "LLMFamilyV2", requested: int) -> Dict[str, Any]:
    extension = family.context_length_extension
    assert extension is not None
    if requested > extension.max_context_length:
        raise ValueError(
            f"{family.model_name} automatic YaRN supports context_length up to "
            f"{extension.max_context_length}; got {requested}"
        )
    parameters = dict(extension.rope_parameters)
    original = parameters.setdefault(
        "original_max_position_embeddings", family.context_length
    )
    assert original is not None
    parameters["factor"] = float(math.ceil(requested / original))
    return parameters


def _read_checkpoint_config(
    family: "LLMFamilyV2", engine: str, model_path: Optional[str]
) -> Dict[str, Any]:
    if not model_path:
        return {}
    if engine == "llama.cpp":
        from .gguf_model_metadata import read_gguf_config, resolve_gguf_model_path

        path = resolve_gguf_model_path(model_path, family.model_specs[0])
        try:
            with open(path, "rb") as stream:
                config, _, _ = read_gguf_config(stream, limit=64 * 1024**2)
                return config
        except OSError:
            return {}
    try:
        return json.loads((Path(model_path) / "config.json").read_text())
    except (OSError, ValueError):
        return {}


class ContextLengthExceeded(ValueError):
    """The request leaves no output capacity within its configured context."""

    def __init__(
        self,
        message: str,
        *,
        input_tokens: Optional[int] = None,
        max_model_len: Optional[int] = None,
        stage: Optional[str] = None,
        diagnostic_code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.input_tokens = normalize_runtime_positive_int(input_tokens)
        self.max_model_len = normalize_runtime_positive_int(max_model_len)
        self.stage = stage
        self.diagnostic_code = diagnostic_code


def parse_vllm_prompt_length_error(message: str) -> Optional[Tuple[int, int]]:
    """Return prompt and context lengths for a known vLLM validation error.

    Keep this parser independent of the optional vLLM package so workers can
    normalize engine-specific exceptions before they cross an actor boundary,
    while API processes can defensively recognize actor-wrapped error text.
    """
    match = _VLLM_PROMPT_LENGTH_ERROR.search(message)
    if match is None:
        return None
    return int(match.group("input_tokens")), int(match.group("max_model_len"))


def _iter_exception_chain(exc: Exception, max_depth: int = 8) -> Iterator[Exception]:
    """Yield a bounded exception chain without allowing cycles."""

    pending = [exc]
    seen: Set[int] = set()
    while pending and len(seen) < max_depth:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        yield current
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if isinstance(context, Exception):
            pending.append(context)
        if isinstance(cause, Exception):
            pending.append(cause)


def _positive_int_attribute(exc: Exception, names: Tuple[str, ...]) -> Optional[int]:
    for name in names:
        try:
            value = getattr(exc, name, None)
        except Exception:
            continue
        normalized = normalize_runtime_positive_int(value)
        if normalized is not None:
            return normalized
    return None


def classify_vllm_context_error(
    exc: Exception, fallback_max_model_len: Optional[int] = None
) -> Optional[Tuple[int, int]]:
    """Recognize a context overflow from exact, worker-local evidence.

    This intentionally avoids broad keyword matching.  Optional-engine
    exception objects and their chains are inspected only inside the model
    worker and are never returned across the actor boundary.
    """

    valid_fallback = normalize_runtime_positive_int(fallback_max_model_len)
    for current in _iter_exception_chain(exc):
        try:
            message = str(current)
        except Exception:
            message = ""
        parsed = parse_vllm_prompt_length_error(message)
        if parsed is None:
            match = _XINFERENCE_MULTIMODAL_CONTEXT_ERROR.search(message)
            if match is not None:
                parsed = (
                    int(match.group("input_tokens")),
                    int(match.group("max_model_len")),
                )
        if parsed is not None:
            return parsed

        input_tokens = _positive_int_attribute(
            current,
            (
                "input_tokens",
                "prompt_tokens",
                "prompt_len",
                "prompt_length",
                "num_prompt_tokens",
            ),
        )
        max_model_len = _positive_int_attribute(
            current,
            (
                "max_model_len",
                "max_seq_len",
                "context_length",
                "model_max_length",
            ),
        )
        if isinstance(current, ContextLengthExceeded):
            max_model_len = max_model_len or valid_fallback
            if input_tokens is not None and max_model_len is not None:
                return input_tokens, max_model_len
            continue

        is_vllm_error = type(current).__module__.startswith("vllm")
        if not is_vllm_error or input_tokens is None:
            continue
        max_model_len = max_model_len or valid_fallback
        if max_model_len is not None and input_tokens >= max_model_len:
            return input_tokens, max_model_len
    return None


def _find_invalid_multimodal_input_error(
    exc: Exception,
) -> Optional[Exception]:
    """Find exact evidence that a failure came from client media input.

    Optional media libraries live in the model environment and must not be
    imported by the supervisor.  Match their stable module and class names
    instead, and inspect a bounded exception chain for wrappers added by media
    processors.  Do not classify broad ``OSError`` or ``RuntimeError`` types:
    they may represent filesystem, worker, or accelerator failures.
    """

    for current in _iter_exception_chain(exc):
        error_type = type(current)
        if current is exc and isinstance(current, ValueError):
            return current
        if (
            error_type.__module__ == "vllm.exceptions"
            and error_type.__name__ == "VLLMValidationError"
        ):
            return current
        if (
            error_type.__module__,
            error_type.__name__,
        ) in _KNOWN_MULTIMODAL_INPUT_ERROR_TYPES:
            return current
    return None


def build_model_operation_error(exc: Exception, operation: str) -> Dict[str, Any]:
    """Convert a model-side failure into an optional-engine-free RPC payload.

    Model workers may load vLLM from a per-model virtual environment while the
    supervisor intentionally does not.  Returning this plain-data envelope
    prevents xoscar from pickling an engine exception type, traceback, cause,
    or context across that process boundary.
    """

    message = str(exc)
    parsed = classify_vllm_context_error(exc)
    invalid_input_error = _find_invalid_multimodal_input_error(exc)

    error: Dict[str, Any] = {
        "version": MODEL_OPERATION_ERROR_VERSION,
        "operation": operation,
        "engine_error_type": type(exc).__name__,
    }
    limit_match = _MULTIMODAL_LIMIT_ERROR.search(message)
    if limit_match is not None:
        error.update(
            {
                "kind": "multimodal_limit_exceeded",
                "message": limit_match.group(0),
            }
        )
    elif isinstance(exc, ContextLengthExceeded) or parsed is not None:
        error.update(
            {
                "kind": "context_length_exceeded",
                "message": CONTEXT_LENGTH_EXCEEDED_DETAIL,
            }
        )
        if parsed is not None:
            error["input_tokens"], error["max_model_len"] = parsed
        stage = getattr(exc, "stage", None)
        diagnostic_code = getattr(exc, "diagnostic_code", None)
        if not isinstance(stage, str) and parsed is not None:
            stage = "multimodal_process"
        if not isinstance(diagnostic_code, str) and parsed is not None:
            diagnostic_code = "VLLM_PROMPT_LENGTH_REJECTED"
        if isinstance(stage, str):
            error["stage"] = stage
        if isinstance(diagnostic_code, str):
            error["diagnostic_code"] = diagnostic_code
    elif isinstance(exc, NotImplementedError):
        error.update(
            {
                "kind": "unsupported",
                "message": (
                    "The requested model operation is not supported by this "
                    "model backend."
                ),
            }
        )
    elif invalid_input_error is not None:
        error["engine_error_type"] = type(invalid_input_error).__name__
        error.update(
            {
                "kind": "invalid_multimodal_input",
                "message": INVALID_MULTIMODAL_INPUT_DETAIL,
            }
        )
        if operation in _MULTIMODAL_OPERATIONS:
            error.update(
                {
                    "stage": "multimodal_process",
                    "diagnostic_code": "VLLM_MM_PROCESS_REJECTED",
                }
            )
    elif type(exc).__module__.startswith("vllm"):
        # Fail closed for an unknown engine-specific exception.  It cannot be
        # treated as a compatibility-counting gap, because proceeding to a
        # streaming generation could commit HTTP 200 before the same engine
        # validation fails again.
        error.update(
            {
                "kind": "internal_error",
                "message": "The model backend rejected the multimodal request.",
            }
        )
        if operation in _MULTIMODAL_OPERATIONS:
            error.update(
                {
                    "stage": "multimodal_process",
                    "diagnostic_code": "MM_PREFLIGHT_INTERNAL_ERROR",
                }
            )
    else:
        error.update(
            {
                "kind": "internal_error",
                "message": "The requested model operation failed.",
            }
        )
        if operation in _MULTIMODAL_OPERATIONS:
            error.update(
                {
                    "stage": "multimodal_process",
                    "diagnostic_code": "MM_PREFLIGHT_INTERNAL_ERROR",
                }
            )
    return {MODEL_OPERATION_ERROR_KEY: error}


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
