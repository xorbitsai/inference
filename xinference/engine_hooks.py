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

"""Pre-bootstrap hooks for distribution-specific inference engines.

This internal extension point currently covers LLM, embedding, and rerank
engines. Registration is process-local: distribution wiring must run in every
process before :mod:`xinference.model` initializes. Spawned processes do not
inherit callbacks registered in their parent process.

Each model type runs its callbacks once per process, during its first model
bootstrap. Later ``register_builtin_model()`` calls reuse the contributed entry
already stored in ``SUPPORTED_ENGINES`` and do not rerun callbacks. Callbacks
receive a temporary engine-table copy; retaining or mutating that copy after the
callback returns has no effect. Failed callbacks cannot partially mutate the
live table, but mutations they make to unrelated global state are outside this
module's rollback boundary.

This hook only contributes engine classes. Other engine-name allowlists, such
as the LLM GGUF format policy, remain independent constraints.
"""

import logging
from typing import Callable, Dict, List, MutableMapping, Set

logger = logging.getLogger(__name__)

__all__ = [
    "MODEL_TYPE_LLM",
    "MODEL_TYPE_EMBEDDING",
    "MODEL_TYPE_RERANK",
    "EngineTable",
    "EngineRegistrationHook",
    "register_engine_registration_hook",
]

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
_RAN_ENGINE_REGISTRATION_HOOKS: Set[str] = set()


def _validate_model_type(model_type: str) -> None:
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(
            f"Invalid model type {model_type!r}; expected one of {_VALID_MODEL_TYPES!r}"
        )


def _validate_engine_table(target: EngineTable) -> None:
    for engine, classes in target.items():
        if not isinstance(engine, str):
            raise TypeError("Engine names must be strings")
        if not isinstance(classes, list):
            raise TypeError(f"Engine {engine!r} classes must be a list")
        for engine_class in classes:
            if not isinstance(engine_class, type):
                raise TypeError(f"Engine {engine!r} entries must be classes")
            for method_name in ("match", "match_json"):
                method = getattr(engine_class, method_name, None)
                if not callable(method) or getattr(
                    method, "__isabstractmethod__", False
                ):
                    raise TypeError(
                        f"Engine {engine!r} classes must implement callable, "
                        f"non-abstract {method_name} methods"
                    )


def _copy_engine_table(target: EngineTable) -> Dict[str, List[type]]:
    return {engine: list(classes) for engine, classes in target.items()}


def _validate_preserved_engines(before: EngineTable, candidate: EngineTable) -> None:
    missing = [engine for engine in before if engine not in candidate]
    if missing:
        raise ValueError(
            "Engine registration hooks must not delete existing engines: "
            + ", ".join(repr(engine) for engine in missing)
        )


def _publish_engine_table(target: EngineTable, staged: EngineTable) -> None:
    original_lists = dict(target)
    for engine, classes in staged.items():
        if engine in original_lists:
            original_classes = original_lists[engine]
            original_classes[:] = classes
            target[engine] = original_classes
        else:
            target[engine] = list(classes)


def register_engine_registration_hook(
    model_type: str, hook: EngineRegistrationHook
) -> None:
    """Register a callback before model bootstrap starts in this process.

    The same callable object can be registered repeatedly without being added
    twice. Distinct callable objects are distinct hooks; callbacks should remain
    idempotent and should replace their distribution-owned entry instead of
    repeatedly extending it.

    Registration after the model type has bootstrapped is rejected because hooks
    are intentionally not rerun during runtime model-registry refreshes.
    """
    _validate_model_type(model_type)
    if not callable(hook):
        raise TypeError("Engine registration hook must be callable")
    if model_type in _RAN_ENGINE_REGISTRATION_HOOKS:
        raise RuntimeError(
            f"{model_type} engine hooks must be registered before model bootstrap"
        )

    hooks = _ENGINE_REGISTRATION_HOOKS.setdefault(model_type, [])
    if not any(existing is hook for existing in hooks):
        hooks.append(hook)


def _run_engine_registration_hooks(model_type: str, target: EngineTable) -> None:
    """Run the internal, once-per-process model-bootstrap callback phase."""
    _validate_model_type(model_type)
    if model_type in _RAN_ENGINE_REGISTRATION_HOOKS:
        return

    # Mark the phase complete even when no callback is registered. Otherwise a
    # callback registered after bootstrap could run during a later live reload.
    _RAN_ENGINE_REGISTRATION_HOOKS.add(model_type)
    hooks = tuple(_ENGINE_REGISTRATION_HOOKS.get(model_type, ()))
    if not hooks:
        return

    _validate_engine_table(target)
    staged = _copy_engine_table(target)
    for hook in hooks:
        candidate = _copy_engine_table(staged)
        try:
            hook(candidate)
            _validate_engine_table(candidate)
            # A callback may retain the table it received. Detach its accepted
            # result and validate the owned snapshot again before promotion so
            # later callback activity cannot mutate the staged state through
            # that retained reference.
            detached = _copy_engine_table(candidate)
            _validate_engine_table(detached)
            _validate_preserved_engines(staged, detached)
        except Exception:
            logger.exception(
                "Failed to run %s engine registration hook %r", model_type, hook
            )
        else:
            staged = detached

    _publish_engine_table(target, staged)
