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


from types import SimpleNamespace

import pytest

from ..context_length import (
    MAX_CONTEXT_LENGTH,
    map_context_length,
    validate_context_length,
    validate_context_length_launch,
)


@pytest.mark.parametrize(
    "engine,key",
    [
        ("vLLM", "max_model_len"),
        ("SGLang", "context_length"),
        ("Transformers", "context_length"),
        ("MLX", "context_length"),
        ("TRT-LLM", "max_seq_len"),
        ("llama.cpp", "n_ctx"),
    ],
)
def test_factory_passes_unified_context_to_each_engine(monkeypatch, engine, key):
    from .. import llm_family
    from ..core import create_llm_model_instance

    spec = SimpleNamespace(
        model_format="pytorch", model_size_in_billions=4, quantization="none"
    )
    family = SimpleNamespace(model_name="test", model_specs=[spec])

    class DummyModel:
        def __init__(self, uid, family, path, config):
            self.config = config

    monkeypatch.setattr(llm_family, "match_llm", lambda *args: family)
    monkeypatch.setattr(
        llm_family, "check_engine_by_spec_parameters", lambda *args: DummyModel
    )
    kwargs = {"context_length": 32768, "enable_virtual_env": False}
    if engine == "llama.cpp":
        kwargs["n_parallel"] = 3
    model = create_llm_model_instance(
        "test-0", "test", engine, model_path="/already-cached", **kwargs
    )
    assert model.config[key] == 32768 * (3 if engine == "llama.cpp" else 1)
    assert ("context_length" in model.config) == (key == "context_length")
    assert kwargs["context_length"] == 32768


@pytest.mark.parametrize(
    "engine,key",
    [("vLLM", "max_model_len"), ("llama.cpp", "n_ctx")],
)
def test_native_context_defaults_and_explicit_aliases_are_preserved(engine, key):
    config = {key: 8192, "temperature": 0.5}
    assert map_context_length(engine, config) == config
    assert map_context_length(engine, {"context_length": None, **config}) == config
    matching = {"context_length": 8192, **config}
    if engine == "llama.cpp":
        matching["n_parallel"] = 1
    assert map_context_length(engine, matching)[key] == 8192
    with pytest.raises(ValueError, match="conflicts"):
        map_context_length(engine, {**matching, "context_length": 4096})


@pytest.mark.parametrize("cpus,slots", [(None, 1), (2, 2), (32, 8)])
def test_llama_cpp_maps_per_request_context_to_slot_capacity(monkeypatch, cpus, slots):
    monkeypatch.setattr("os.cpu_count", lambda: cpus)
    config = map_context_length("llama.cpp", {"context_length": 32768})
    assert config == {"n_ctx": 32768 * slots, "n_parallel": slots}
    with pytest.raises(ValueError, match="exceeds"):
        map_context_length(
            "llama.cpp", {"context_length": MAX_CONTEXT_LENGTH, "n_parallel": 2}
        )


@pytest.mark.parametrize(
    "value", [True, False, 0, -1, 1.5, "32768", MAX_CONTEXT_LENGTH + 1]
)
def test_invalid_context_is_rejected(value):
    with pytest.raises(ValueError, match="context_length"):
        validate_context_length(value)


def test_invalid_context_and_aliases_fail_before_discovery_or_download(monkeypatch):
    from .. import llm_family
    from ..core import create_llm_model_instance

    def no_discovery(*args):
        raise AssertionError("invalid context must fail before model discovery")

    monkeypatch.setattr(llm_family, "match_llm", no_discovery)
    with pytest.raises(ValueError, match="conflicts"):
        create_llm_model_instance(
            "test", "test", "vLLM", context_length=8192, max_model_len=4096
        )
    with pytest.raises(ValueError, match="context_length"):
        create_llm_model_instance("test", "test", "vLLM", context_length=0)
    with pytest.raises(ValueError, match="positive integer"):
        create_llm_model_instance(
            "test", "test", "llama.cpp", context_length=8192, n_parallel=False
        )


def test_context_only_applies_to_supported_llm_engines():
    with pytest.raises(ValueError, match="only supported for LLM"):
        validate_context_length_launch(
            "embedding", "sentence_transformers", {"context_length": 32768}
        )
    with pytest.raises(ValueError, match="not supported"):
        map_context_length("custom", {"context_length": 32768})
    assert map_context_length("custom", {"native_option": 1}) == {"native_option": 1}


def test_mlx_runtime_override_wins_over_nested_architecture_context():
    from ..utils import get_context_length_from_config

    config = {
        "max_position_embeddings": 262144,
        "text_config": {"max_position_embeddings": 262144},
    }
    assert get_context_length_from_config(config) == 262144
    assert get_context_length_from_config(config, context_length=32768) == 32768
    assert config["text_config"]["max_position_embeddings"] == 262144
    with pytest.raises(ValueError):
        get_context_length_from_config(config, context_length=True)
