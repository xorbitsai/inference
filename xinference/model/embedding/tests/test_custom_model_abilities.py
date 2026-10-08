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

import json
from unittest.mock import Mock

import pytest

from .... import constants
from ...._compat import ValidationError
from ... import custom as custom_registry
from ... import embedding as model_module
from ..custom import CustomEmbeddingModelFamilyV2


@pytest.fixture
def model_spec():
    return {
        "version": 2,
        "model_name": "legacy-custom-embedding",
        "dimensions": 384,
        "max_tokens": 512,
        "language": ["en"],
        "model_specs": [
            {
                "model_format": "pytorch",
                "model_id": "test/custom-embedding",
                "model_revision": "main",
                "quantization": "none",
            }
        ],
    }


@pytest.mark.parametrize(
    "abilities",
    [
        ["vision", "video", "audio"],
        ["embed_vision", "embed_video", "embed_audio"],
        ["vision", "embed_video", "audio"],
    ],
)
def test_load_persisted_custom_abilities(tmp_path, monkeypatch, model_spec, abilities):
    model_spec["model_ability"] = abilities
    registration_dir = tmp_path / "v2" / "embedding"
    registration_dir.mkdir(parents=True)
    (registration_dir / "custom.json").write_text(json.dumps(model_spec))
    monkeypatch.setattr(constants, "XINFERENCE_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(
        model_module, "XINFERENCE_MODEL_DIR", str(tmp_path), raising=False
    )
    monkeypatch.setattr(custom_registry, "migrate_from_v1_to_v2", Mock())
    register = Mock()
    monkeypatch.setattr(model_module, "register_embedding", register)

    model_module.register_custom_model()

    register.assert_called_once()
    family = register.call_args.args[0]
    assert isinstance(family, CustomEmbeddingModelFamilyV2)
    expected = ["embed_vision", "embed_video", "embed_audio"]
    assert family.model_ability == expected
    assert family.to_description()["model_ability"] == ["embed", *expected]
    assert json.loads(family.json())["model_ability"] == expected
    assert register.call_args.kwargs == {"persist": False}
    assert model_spec["model_ability"] == abilities


@pytest.mark.parametrize("ability", ["unknown", "rerank_vision"])
def test_custom_abilities_reject_unknown_values(model_spec, ability):
    model_spec["model_ability"] = [ability]
    with pytest.raises(ValidationError, match="model_ability"):
        CustomEmbeddingModelFamilyV2.parse_obj(model_spec)


def test_custom_abilities_default_to_empty(model_spec):
    family = CustomEmbeddingModelFamilyV2.parse_obj(model_spec)
    assert family.model_ability == []
    assert family.to_description()["model_ability"] == ["embed"]


def test_custom_abilities_do_not_mutate_input(model_spec):
    abilities = ["vision", "embed_video", "audio"]
    model_spec["model_ability"] = abilities
    CustomEmbeddingModelFamilyV2.parse_obj(model_spec)
    assert abilities == ["vision", "embed_video", "audio"]
