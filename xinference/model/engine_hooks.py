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

"""Startup hooks for distribution-specific inference engines."""

import logging
from typing import Callable, Dict, List, MutableMapping

logger = logging.getLogger(__name__)

MODEL_TYPE_LLM = "LLM"
MODEL_TYPE_EMBEDDING = "embedding"
MODEL_TYPE_RERANK = "rerank"
_VALID_MODEL_TYPES = (
    MODEL_TYPE_LLM,
    MODEL_TYPE_EMBEDDING,
    MODEL_TYPE_RERANK,
)

EngineTable = MutableMapping[str, List[type]]
EngineRegistrationHook = Callable[[EngineTable], None]

_ENGINE_REGISTRATION_HOOKS: Dict[str, List[EngineRegistrationHook]] = {}


def _validate_engine_table(target: EngineTable) -> None:
    for engine, classes in target.items():
        if not isinstance(engine, str):
            raise TypeError("Engine names must be strings")
        if not isinstance(classes, list):
            raise TypeError(f"Engine {engine!r} classes must be a list")
        if not all(isinstance(engine_class, type) for engine_class in classes):
            raise TypeError(f"Engine {engine!r} entries must be classes")


def register_engine_registration_hook(
    model_type: str, hook: EngineRegistrationHook
) -> None:
    """Register a callback that contributes engines for one model type.

    The callback receives a mapping of engine names to lists of engine classes.
    Classes must implement the interface used by the corresponding model package;
    LLM configuration queries ``match``, while embedding and rerank configuration
    query ``match_json``.
    """
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(
            f"Invalid model type {model_type!r}; expected one of {_VALID_MODEL_TYPES!r}"
        )
    if not callable(hook):
        raise TypeError("Engine registration hook must be callable")

    hooks = _ENGINE_REGISTRATION_HOOKS.setdefault(model_type, [])
    if hook not in hooks:
        hooks.append(hook)


def run_engine_registration_hooks(model_type: str, target: EngineTable) -> None:
    """Run callbacks without letting one break or partially mutate bootstrap."""
    for hook in tuple(_ENGINE_REGISTRATION_HOOKS.get(model_type, ())):
        snapshot = {engine: list(classes) for engine, classes in target.items()}
        try:
            hook(target)
            _validate_engine_table(target)
        except Exception:
            target.clear()
            target.update(snapshot)
            logger.exception(
                "Failed to run %s engine registration hook %r", model_type, hook
            )
