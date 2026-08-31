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
import os
import subprocess
import sys
import textwrap

import pytest

from ... import engine_hooks
from ...engine_hooks import (
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


def test_failed_hook_restores_aliased_list_in_place(caplog):
    class BuiltinModel:
        pass

    class FailedModel:
        pass

    canonical_classes = [BuiltinModel]
    target = {"builtin": canonical_classes}

    def broken(table):
        table["builtin"].append(FailedModel)
        raise RuntimeError("boom")

    register_engine_registration_hook(MODEL_TYPE_LLM, broken)

    with caplog.at_level(logging.ERROR):
        run_engine_registration_hooks(MODEL_TYPE_LLM, target)

    assert target["builtin"] is canonical_classes
    assert canonical_classes == [BuiltinModel]

    # A later install rebinds the table to the canonical list. The failed class
    # must not reappear through that alias.
    target["builtin"] = canonical_classes
    assert FailedModel not in target["builtin"]


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


def test_hooks_can_register_before_model_bootstrap(tmp_path):
    script = textwrap.dedent(
        """
        from xinference.engine_hooks import (
            MODEL_TYPE_EMBEDDING,
            MODEL_TYPE_LLM,
            MODEL_TYPE_RERANK,
            register_engine_registration_hook,
        )
        import sys

        assert "xinference.model" not in sys.modules

        class ExternalModel:
            @classmethod
            def match(cls, *_args):
                return True

            @classmethod
            def match_json(cls, *_args):
                return True

        class FailedModel(ExternalModel):
            pass

        def contribute(target):
            target["external"] = [ExternalModel]

        def append_then_fail(target):
            target["vLLM"].append(FailedModel)
            raise RuntimeError("boom")

        register_engine_registration_hook(MODEL_TYPE_LLM, contribute)
        register_engine_registration_hook(MODEL_TYPE_LLM, append_then_fail)
        register_engine_registration_hook(MODEL_TYPE_EMBEDDING, contribute)
        register_engine_registration_hook(MODEL_TYPE_RERANK, contribute)

        import xinference.model
        from xinference.model.llm import register_builtin_model
        from xinference.model.embedding.embed_family import (
            EMBEDDING_ENGINES,
            SUPPORTED_ENGINES as EMBEDDING_SUPPORTED_ENGINES,
        )
        from xinference.model.llm.llm_family import (
            LLM_ENGINES,
            SUPPORTED_ENGINES as LLM_SUPPORTED_ENGINES,
            VLLM_CLASSES,
        )
        from xinference.model.rerank.rerank_family import (
            RERANK_ENGINES,
            SUPPORTED_ENGINES as RERANK_SUPPORTED_ENGINES,
        )

        assert list(LLM_SUPPORTED_ENGINES)[-1] == "external"
        assert list(EMBEDDING_SUPPORTED_ENGINES)[-1] == "external"
        assert list(RERANK_SUPPORTED_ENGINES)[-1] == "external"
        assert any("external" in engines for engines in LLM_ENGINES.values())
        assert any("external" in engines for engines in EMBEDDING_ENGINES.values())
        assert any("external" in engines for engines in RERANK_ENGINES.values())
        assert LLM_SUPPORTED_ENGINES["vLLM"] is VLLM_CLASSES
        assert FailedModel not in VLLM_CLASSES
        assert all(
            param["llm_class"] is not FailedModel
            for engines in LLM_ENGINES.values()
            for params in engines.values()
            for param in params
        )

        register_builtin_model()

        assert LLM_SUPPORTED_ENGINES["vLLM"] is VLLM_CLASSES
        assert FailedModel not in VLLM_CLASSES
        assert all(
            param["llm_class"] is not FailedModel
            for engines in LLM_ENGINES.values()
            for params in engines.values()
            for param in params
        )
        """
    )
    env = os.environ.copy()
    env["XINFERENCE_HOME"] = str(tmp_path)
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        cwd=repo_root,
        env=env,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
