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

EngineTable = MutableMapping[str, List[type]]
EngineRegistrationHook = Callable[[EngineTable], None]

_ENGINE_REGISTRATION_HOOKS: Dict[str, List[EngineRegistrationHook]] = {}


def register_engine_registration_hook(
    model_type: str, hook: EngineRegistrationHook
) -> None:
    """Register a callback that contributes engines for one model type."""
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
        except Exception:
            target.clear()
            target.update(snapshot)
            logger.exception(
                "Failed to run %s engine registration hook %r", model_type, hook
            )
