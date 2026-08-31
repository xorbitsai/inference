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

import logging

import pytest

from .. import engine_hooks
from ..engine_hooks import (
    MODEL_TYPE_EMBEDDING,
    MODEL_TYPE_LLM,
    register_engine_registration_hook,
    run_engine_registration_hooks,
)


@pytest.fixture(autouse=True)
def isolated_hooks(monkeypatch):
    monkeypatch.setattr(engine_hooks, "_ENGINE_REGISTRATION_HOOKS", {})


def test_hook_contributes_to_one_model_type_after_builtins():
    class DummyModel:
        pass

    def hook(target):
        target["external"] = [DummyModel]

    register_engine_registration_hook(MODEL_TYPE_LLM, hook)

    llm_target = {"builtin": [DummyModel]}
    embedding_target = {}
    run_engine_registration_hooks(MODEL_TYPE_LLM, llm_target)
    run_engine_registration_hooks(MODEL_TYPE_EMBEDDING, embedding_target)

    assert list(llm_target) == ["builtin", "external"]
    assert embedding_target == {}


def test_registering_same_hook_twice_is_idempotent():
    calls = []

    def hook(_target):
        calls.append(1)

    register_engine_registration_hook(MODEL_TYPE_LLM, hook)
    register_engine_registration_hook(MODEL_TYPE_LLM, hook)
    run_engine_registration_hooks(MODEL_TYPE_LLM, {})

    assert calls == [1]


def test_broken_hook_rolls_back_and_does_not_stop_later_hooks(caplog):
    class DummyModel:
        pass

    def broken(target):
        target.clear()
        target["partial"] = [DummyModel]
        raise RuntimeError("boom")

    def healthy(target):
        target["healthy"] = [DummyModel]

    register_engine_registration_hook(MODEL_TYPE_LLM, broken)
    register_engine_registration_hook(MODEL_TYPE_LLM, healthy)

    target = {"builtin": [DummyModel]}
    with caplog.at_level(logging.ERROR):
        run_engine_registration_hooks(MODEL_TYPE_LLM, target)

    assert list(target) == ["builtin", "healthy"]
    assert "partial" not in target
    assert "Failed to run LLM engine registration hook" in caplog.text


def test_malformed_hook_output_rolls_back_and_does_not_stop_later_hooks(caplog):
    class DummyModel:
        pass

    def malformed(target):
        target["invalid"] = DummyModel

    def healthy(target):
        target["healthy"] = [DummyModel]

    register_engine_registration_hook(MODEL_TYPE_LLM, malformed)
    register_engine_registration_hook(MODEL_TYPE_LLM, healthy)

    target = {"builtin": [DummyModel]}
    with caplog.at_level(logging.ERROR):
        run_engine_registration_hooks(MODEL_TYPE_LLM, target)

    assert list(target) == ["builtin", "healthy"]
    assert "invalid" not in target
    assert "Engine 'invalid' classes must be a list" in caplog.text


def test_hook_must_be_callable():
    with pytest.raises(TypeError, match="must be callable"):
        register_engine_registration_hook(MODEL_TYPE_LLM, None)  # type: ignore[arg-type]


def test_model_type_must_be_valid():
    with pytest.raises(ValueError, match="Invalid model type"):
        register_engine_registration_hook("llm", lambda _target: None)
