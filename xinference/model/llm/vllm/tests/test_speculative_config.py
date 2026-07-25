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

import pytest
from packaging import version


def _model():
    from ..core import VLLMModel

    model = object.__new__(VLLMModel)
    model.model_uid = "test-model-0"
    return model


def test_drafter_becomes_speculative_config(monkeypatch):
    from .. import core

    monkeypatch.setattr(core, "VLLM_VERSION", version.parse("0.22.0"))
    model_config = {
        "draft_model_path": "/cache/gemma-4-draft",
        "gpu_memory_utilization": 0.9,
    }

    _model()._apply_draft_model(model_config)

    assert model_config["speculative_config"] == {
        "method": "mtp",
        "model": "/cache/gemma-4-draft",
        "num_speculative_tokens": 1,
    }
    # the engine-neutral options must not reach AsyncEngineArgs
    assert "draft_model_path" not in model_config
    assert "num_speculative_tokens" not in model_config
    assert model_config["gpu_memory_utilization"] == 0.9


def test_num_speculative_tokens_is_honored(monkeypatch):
    from .. import core

    monkeypatch.setattr(core, "VLLM_VERSION", version.parse("0.22.0"))
    # the Web UI submits additional parameters as strings
    model_config = {"draft_model_path": "/cache/draft", "num_speculative_tokens": "3"}

    _model()._apply_draft_model(model_config)

    assert model_config["speculative_config"]["num_speculative_tokens"] == 3


def test_explicit_speculative_config_wins(monkeypatch):
    from .. import core

    monkeypatch.setattr(core, "VLLM_VERSION", version.parse("0.22.0"))
    explicit = {"method": "ngram", "num_speculative_tokens": 5}
    model_config = {"draft_model_path": "/cache/draft", "speculative_config": explicit}

    _model()._apply_draft_model(model_config)

    assert model_config["speculative_config"] is explicit
    assert "draft_model_path" not in model_config


def test_old_vllm_is_rejected(monkeypatch):
    # 0.21 treats an assistant checkpoint as a generic draft model and fails to
    # initialize against a multimodal target; fail with a readable message
    # instead of letting vLLM crash deep inside engine startup.
    from .. import core

    monkeypatch.setattr(core, "VLLM_VERSION", version.parse("0.21.0"))
    model_config = {"draft_model_path": "/cache/draft"}

    with pytest.raises(ValueError, match="needs vllm>=0.22.0"):
        _model()._apply_draft_model(model_config)


def test_no_drafter_is_a_noop(monkeypatch):
    from .. import core

    monkeypatch.setattr(core, "VLLM_VERSION", version.parse("0.22.0"))
    model_config = {"gpu_memory_utilization": 0.9}

    _model()._apply_draft_model(model_config)

    assert model_config == {"gpu_memory_utilization": 0.9}
