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
# CI: linux

from unittest.mock import Mock

import pytest

from .. import custom, llm_family
from ..llm_family import LLMFamilyV2


def _family(name, *, parent=None, architectures=None):
    return LLMFamilyV2(
        version=2,
        model_name=name,
        model_family=parent,
        architectures=architectures,
        model_lang=["en"],
        model_ability=["chat"],
        model_specs=[],
    )


@pytest.mark.parametrize("architectures", [None, ["ExplicitForCausalLM"]])
def test_own_architectures_do_not_query_custom_registry(monkeypatch, architectures):
    lookup = Mock(side_effect=AssertionError("custom registry must not be queried"))
    monkeypatch.setattr(custom, "get_user_defined_llm_families", lookup)
    family = _family("standalone", architectures=architectures)
    assert family._resolve_architectures() == architectures
    lookup.assert_not_called()


def test_explicit_architectures_take_precedence_over_parent(monkeypatch):
    monkeypatch.setattr(
        llm_family,
        "BUILTIN_LLM_FAMILIES",
        [_family("parent", architectures=["ParentForCausalLM"])],
    )
    lookup = Mock(side_effect=AssertionError("custom registry must not be queried"))
    monkeypatch.setattr(custom, "get_user_defined_llm_families", lookup)
    family = _family("custom", parent="parent", architectures=["ExplicitForCausalLM"])
    assert family.has_architecture("ExplicitForCausalLM")
    assert not family.has_architecture("ParentForCausalLM")
    lookup.assert_not_called()


@pytest.mark.parametrize("architectures", [None, [], ["BuiltinForCausalLM"]])
def test_builtin_family_does_not_query_custom_registry(monkeypatch, architectures):
    monkeypatch.setattr(
        llm_family,
        "BUILTIN_LLM_FAMILIES",
        [_family("parent", architectures=architectures)],
    )
    lookup = Mock(side_effect=AssertionError("builtin lookup must remain lazy"))
    monkeypatch.setattr(custom, "get_user_defined_llm_families", lookup)
    assert _family("custom", parent="parent")._resolve_architectures() == architectures
    lookup.assert_not_called()


@pytest.mark.parametrize("reverse", [False, True])
def test_custom_parent_architectures_are_resolved(monkeypatch, reverse):
    parent = _family("custom-parent", architectures=["CustomForCausalLM"])
    unrelated = _family("unrelated", architectures=["OtherForCausalLM"])
    families = [unrelated, parent] if reverse else [parent, unrelated]
    monkeypatch.setattr(llm_family, "BUILTIN_LLM_FAMILIES", [unrelated])
    monkeypatch.setattr(custom, "get_user_defined_llm_families", lambda: families)
    child = _family("custom-child", parent="custom-parent")
    assert child.has_architecture("CustomForCausalLM")
    assert not child.has_architecture("OtherForCausalLM")
    assert child.matches_supported_architectures(["CustomForCausalLM"])
    assert not child.matches_supported_architectures(["OtherForCausalLM"])


@pytest.mark.parametrize("architectures", [None, []])
def test_custom_parent_without_architectures(monkeypatch, architectures):
    monkeypatch.setattr(llm_family, "BUILTIN_LLM_FAMILIES", [])
    parent = _family("custom-parent", architectures=architectures)
    monkeypatch.setattr(custom, "get_user_defined_llm_families", lambda: [parent])
    child = _family("custom-child", parent="custom-parent")
    assert child._resolve_architectures() == architectures
    assert not child.has_architecture("CustomForCausalLM")
    assert not child.matches_supported_architectures(["CustomForCausalLM"])


def test_custom_parent_lookup_reflects_registry_changes(monkeypatch):
    monkeypatch.setattr(llm_family, "BUILTIN_LLM_FAMILIES", [])
    families = []
    monkeypatch.setattr(
        custom, "get_user_defined_llm_families", lambda: families.copy()
    )
    child = _family("custom-child", parent="custom-parent")
    assert child._resolve_architectures() is None
    families.append(_family("custom-parent", architectures=["CustomForCausalLM"]))
    assert child.has_architecture("CustomForCausalLM")
    families.clear()
    assert child._resolve_architectures() is None


@pytest.mark.parametrize("registered_parent", [False, True])
def test_registration_can_resolve_custom_architectures(monkeypatch, registered_parent):
    from ... import llm
    from ...custom import RegistryManager
    from ..llm_family import LlamaCppLLMSpecV2

    monkeypatch.setattr(custom, "UD_LLM_FAMILIES", [])
    monkeypatch.setattr(llm_family, "BUILTIN_LLM_FAMILIES", [])
    registry = custom.LLMModelRegistry()
    monkeypatch.setattr(RegistryManager, "_instances", {"llm": registry})

    class BoundedLock:
        # Preserve the registry's real lock semantics, but fail instead of
        # hanging the entire test process when registration re-enters it.
        def __init__(self, lock):
            self.lock = lock

        def __enter__(self):
            assert self.lock.acquire(timeout=1), "registry lookup deadlocked"

        def __exit__(self, *_):
            self.lock.release()

    registry.lock = BoundedLock(registry.lock)
    expected = ["CustomForCausalLM"] if registered_parent else None
    checked = []

    class MatchingEngine:
        @classmethod
        def match(cls, family, spec, quantization):
            checked.append(family.model_name)
            assert family._resolve_architectures() == expected
            return family.has_architecture("CustomForCausalLM")

    engines = {}
    monkeypatch.setattr(llm, "SUPPORTED_ENGINES", {"llama.cpp": [MatchingEngine]})
    monkeypatch.setattr(llm, "LLM_ENGINES", engines)
    if registered_parent:
        custom.register_llm(
            _family("custom-parent", architectures=expected), persist=False
        )

    child = _family("custom-child", parent="custom-parent")
    child.model_specs = [
        LlamaCppLLMSpecV2(
            model_format="ggufv2",
            model_size_in_billions=1,
            quantization="Q4_0",
            model_file_name_template="unused.gguf",
        )
    ]
    custom.register_llm(child, persist=False)

    assert checked == ["custom-child"]
    assert ("llama.cpp" in engines["custom-child"]) is registered_parent
    assert registry.get_custom_models()[-1] is child
