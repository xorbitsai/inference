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
from abc import abstractmethod
from pathlib import Path

import pytest

from xinference import engine_hooks
from xinference.engine_hooks import MODEL_TYPE_EMBEDDING, MODEL_TYPE_LLM
from xinference.engine_hooks import (
    _run_engine_registration_hooks as run_engine_registration_hooks,
)
from xinference.engine_hooks import register_engine_registration_hook


class DummyModel:
    @classmethod
    def match(cls, *_args):
        return True

    @classmethod
    def match_json(cls, *_args):
        return True


@pytest.fixture(autouse=True)
def isolated_hooks(monkeypatch):
    monkeypatch.setattr(engine_hooks, "_ENGINE_REGISTRATION_HOOKS", {})
    monkeypatch.setattr(engine_hooks, "_RAN_ENGINE_REGISTRATION_HOOKS", set())


def test_hook_contributes_to_one_model_type_after_builtins():
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


def test_distinct_callables_are_distinct_hooks():
    calls = []

    def make_hook(label):
        def hook(_target):
            calls.append(label)

        return hook

    register_engine_registration_hook(MODEL_TYPE_LLM, make_hook("first"))
    register_engine_registration_hook(MODEL_TYPE_LLM, make_hook("second"))
    run_engine_registration_hooks(MODEL_TYPE_LLM, {})

    assert calls == ["first", "second"]


def test_hook_equality_is_not_evaluated_during_registration():
    calls = []

    class Hook:
        def __init__(self, label):
            self._label = label

        def __call__(self, _target):
            calls.append(self._label)

        def __eq__(self, _other):
            raise RuntimeError("equality must not be evaluated")

    register_engine_registration_hook(MODEL_TYPE_LLM, Hook("first"))
    register_engine_registration_hook(MODEL_TYPE_LLM, Hook("second"))
    run_engine_registration_hooks(MODEL_TYPE_LLM, {})

    assert calls == ["first", "second"]


def test_hooks_run_only_once_per_process():
    calls = []

    def hook(target):
        calls.append(1)
        target["external"] = [DummyModel]

    register_engine_registration_hook(MODEL_TYPE_LLM, hook)
    target = {}
    run_engine_registration_hooks(MODEL_TYPE_LLM, target)
    run_engine_registration_hooks(MODEL_TYPE_LLM, target)

    assert calls == [1]
    assert target == {"external": [DummyModel]}


def test_zero_hooks_skip_validation_and_close_registration_phase():
    invalid_target = {"invalid": object()}

    run_engine_registration_hooks(MODEL_TYPE_LLM, invalid_target)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="before model bootstrap"):
        register_engine_registration_hook(MODEL_TYPE_LLM, lambda _target: None)


def test_broken_hook_rolls_back_and_does_not_stop_later_hooks(caplog):
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


def test_hook_cannot_delete_an_existing_engine(caplog):
    def destructive(target):
        del target["builtin"]

    def healthy(target):
        target["healthy"] = [DummyModel]

    register_engine_registration_hook(MODEL_TYPE_LLM, destructive)
    register_engine_registration_hook(MODEL_TYPE_LLM, healthy)

    target = {"builtin": [DummyModel]}
    with caplog.at_level(logging.ERROR):
        run_engine_registration_hooks(MODEL_TYPE_LLM, target)

    assert list(target) == ["builtin", "healthy"]
    assert "must not delete existing engines" in caplog.text


def test_successful_hook_preserves_existing_list_identity():
    class AdditionalModel(DummyModel):
        pass

    canonical_classes = [DummyModel]
    target = {"builtin": canonical_classes}

    def contribute(table):
        table["builtin"].append(AdditionalModel)

    register_engine_registration_hook(MODEL_TYPE_LLM, contribute)
    run_engine_registration_hooks(MODEL_TYPE_LLM, target)

    assert target["builtin"] is canonical_classes
    assert canonical_classes == [DummyModel, AdditionalModel]


def test_published_table_is_detached_from_retained_staging_copy():
    retained = []

    def contribute(table):
        table["external"] = [DummyModel]
        retained.append(table)

    target = {}
    register_engine_registration_hook(MODEL_TYPE_LLM, contribute)
    run_engine_registration_hooks(MODEL_TYPE_LLM, target)

    retained[0]["external"].clear()

    assert target == {"external": [DummyModel]}


def test_failed_hook_keeps_aliased_list_untouched(caplog):
    class BuiltinModel(DummyModel):
        pass

    class FailedModel(DummyModel):
        pass

    canonical_classes = [BuiltinModel]
    target = {"builtin": canonical_classes}
    appended = []

    def broken(table):
        assert table["builtin"] is not canonical_classes
        table["builtin"].append(FailedModel)
        appended.append(True)
        raise RuntimeError("boom")

    register_engine_registration_hook(MODEL_TYPE_LLM, broken)

    with caplog.at_level(logging.ERROR):
        run_engine_registration_hooks(MODEL_TYPE_LLM, target)

    assert target["builtin"] is canonical_classes
    assert canonical_classes == [BuiltinModel]
    assert appended == [True]

    # A later install can rebind the table to the same canonical list without
    # making the failed class reappear through that alias.
    target["builtin"] = canonical_classes
    assert FailedModel not in target["builtin"]


def test_malformed_hook_output_rolls_back_and_does_not_stop_later_hooks(caplog):
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


def test_hook_with_invalid_engine_class_rolls_back(caplog):
    class MissingMatchMethods:
        pass

    def malformed(target):
        target["invalid"] = [MissingMatchMethods]

    def healthy(target):
        target["healthy"] = [DummyModel]

    register_engine_registration_hook(MODEL_TYPE_LLM, malformed)
    register_engine_registration_hook(MODEL_TYPE_LLM, healthy)

    target = {"builtin": [DummyModel]}
    with caplog.at_level(logging.ERROR):
        run_engine_registration_hooks(MODEL_TYPE_LLM, target)

    assert list(target) == ["builtin", "healthy"]
    assert "invalid" not in target
    assert "must implement callable, non-abstract match methods" in caplog.text


def test_hook_with_unoverridden_abstract_method_rolls_back(caplog):
    class AbstractMatchJsonModel(DummyModel):
        @classmethod
        @abstractmethod
        def match_json(cls, *_args):
            raise NotImplementedError

    def malformed(target):
        target["invalid"] = [AbstractMatchJsonModel]

    register_engine_registration_hook(MODEL_TYPE_LLM, malformed)

    target = {"builtin": [DummyModel]}
    with caplog.at_level(logging.ERROR):
        run_engine_registration_hooks(MODEL_TYPE_LLM, target)

    assert target == {"builtin": [DummyModel]}
    assert "non-abstract match_json methods" in caplog.text


def test_hook_must_be_callable():
    with pytest.raises(TypeError, match="must be callable"):
        register_engine_registration_hook(MODEL_TYPE_LLM, None)  # type: ignore[arg-type]


def test_model_type_must_be_valid():
    with pytest.raises(ValueError, match="Invalid model type"):
        register_engine_registration_hook("llm", lambda _target: None)
    with pytest.raises(ValueError, match="Invalid model type"):
        run_engine_registration_hooks("llm", {})


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

        contribute_calls = []
        append_then_fail_calls = []

        def contribute(target):
            contribute_calls.append(True)
            target["external"] = [ExternalModel]

        def append_then_fail(target):
            target["vLLM"].append(FailedModel)
            append_then_fail_calls.append(True)
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
        assert contribute_calls == [True, True, True]
        assert append_then_fail_calls == [True]
        assert LLM_SUPPORTED_ENGINES["vLLM"] is VLLM_CLASSES
        assert FailedModel not in VLLM_CLASSES
        assert all(
            param["llm_class"] is not FailedModel
            for engines in LLM_ENGINES.values()
            for params in engines.values()
            for param in params
        )

        register_builtin_model()

        assert contribute_calls == [True, True, True]
        assert append_then_fail_calls == [True]
        assert LLM_SUPPORTED_ENGINES["external"] == [ExternalModel]
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
    repo_root = str(Path(__file__).resolve().parents[2])
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root, current_pythonpath] if current_pythonpath else [repo_root]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        cwd=repo_root,
        env=env,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
