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

import importlib
from types import SimpleNamespace

import pytest

from ..utils import _clean_engine_param


@pytest.mark.parametrize("kind", ["image", "video", "audio", "world"])
@pytest.mark.parametrize("with_defaults", [False, True])
def test_engine_catalog_exposes_model_defaults_without_sharing_mutable_values(
    monkeypatch, kind, with_defaults
):
    module = importlib.import_module(f"xinference.model.{kind}.engine_family")
    registry = {}
    monkeypatch.setattr(module, f"{kind.upper()}_ENGINES", registry)

    class Engine:
        @classmethod
        def match(cls, family):
            return True

    monkeypatch.setattr(module, "SUPPORTED_ENGINES", {"ExampleEngine": [Engine]})
    defaults = (
        {
            "torch_dtype": "bfloat16",
            "enabled": False,
            "nested": {"layers": [1, 2]},
            "optional": None,
        }
        if with_defaults
        else {}
    )
    family = SimpleNamespace(
        model_name="example-model",
        model_format="pytorch",
        default_model_config=defaults,
        runtime_precisions=None,
    )
    module.generate_engine_config_by_model_name(family)
    param = registry[family.model_name]["ExampleEngine"][0]
    cleaned = _clean_engine_param(param, f"{kind}_class")
    assert f"{kind}_class" not in cleaned
    if with_defaults:
        assert cleaned["default_model_config"] == defaults
        cleaned["default_model_config"]["nested"]["layers"].append(3)
        assert defaults["nested"]["layers"] == [1, 2]
    else:
        assert "default_model_config" not in cleaned
