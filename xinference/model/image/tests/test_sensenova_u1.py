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

from types import SimpleNamespace

import pytest

from .. import sensenova_u1


def _make_model():
    model_spec = SimpleNamespace(
        model_ability=["text2image", "image2image"],
        default_generate_config={},
    )
    return sensenova_u1.SenseNovaU1Model("test", "/path", model_spec)


@pytest.mark.parametrize("kwargs", [{}, {"seed": None}, {"seed": -1}])
def test_generate_config_randomizes_default_seed(monkeypatch, kwargs):
    monkeypatch.setattr(sensenova_u1.random, "randint", lambda lower, upper: 1234)

    config = _make_model()._get_generate_config(kwargs)

    assert config["seed"] == 1234


@pytest.mark.parametrize("seed", [0, 1, 1234])
def test_generate_config_preserves_non_negative_seed(seed):
    config = _make_model()._get_generate_config({"seed": seed})

    assert config["seed"] == seed
