# Copyright 2022-2026 XProbe Inc.
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

from unittest.mock import patch

import pytest

from .. import BUILTIN_AUDIO_MODELS, _install
from ..core import create_audio_model_instance
from ..engine import TransformersQwen3ASRAudioModel, VLLMQwen3ASRAudioModel
from ..engine import platform as engine_platform
from ..engine import register_builtin_audio_engines
from ..engine_family import (
    AUDIO_ENGINES,
    check_engine_by_model_name_and_engine,
    generate_engine_config_by_model_name,
)
from ..funasr import FunASRModel
from ..whisper import WhisperModel


@pytest.fixture(scope="module", autouse=True)
def setup_builtin_models():
    _install()


def _get_spec(model_name: str):
    return BUILTIN_AUDIO_MODELS[model_name][0]


def _register_all_engines():
    register_builtin_audio_engines()
    for model_specs in BUILTIN_AUDIO_MODELS.values():
        for model_spec in model_specs:
            generate_engine_config_by_model_name(model_spec)


@pytest.fixture
def linux_cuda_engines():
    engine_mod = __import__(
        VLLMQwen3ASRAudioModel.__module__, fromlist=["has_cuda_device"]
    )
    old_engines = {k: dict(v) for k, v in AUDIO_ENGINES.items()}
    with patch.object(engine_platform, "system", return_value="Linux"), patch.object(
        engine_mod, "has_cuda_device", return_value=True
    ):
        AUDIO_ENGINES.clear()
        _register_all_engines()
        yield
    AUDIO_ENGINES.clear()
    AUDIO_ENGINES.update(old_engines)


def test_qwen3_asr_registers_transformers_engine():
    assert "Qwen3-ASR-0.6B" in AUDIO_ENGINES
    assert "transformers" in AUDIO_ENGINES["Qwen3-ASR-0.6B"]
    # default engine is the first registered one
    assert next(iter(AUDIO_ENGINES["Qwen3-ASR-0.6B"])) == "transformers"


def test_non_engine_families_not_registered():
    assert "whisper-large-v3" not in AUDIO_ENGINES


def test_qwen3_asr_vllm_engine_on_linux_cuda(linux_cuda_engines):
    for model_name in ("Qwen3-ASR-0.6B", "Qwen3-ASR-1.7B"):
        assert sorted(AUDIO_ENGINES[model_name]) == ["transformers", "vLLM"]
        cls = check_engine_by_model_name_and_engine("vLLM", model_name)
        assert cls is VLLMQwen3ASRAudioModel
        # engine name is case-insensitive
        cls = check_engine_by_model_name_and_engine("vllm", model_name)
        assert cls is VLLMQwen3ASRAudioModel


def test_vllm_engine_not_matched_without_cuda():
    engine_mod = __import__(
        VLLMQwen3ASRAudioModel.__module__, fromlist=["has_cuda_device"]
    )
    with patch.object(engine_platform, "system", return_value="Linux"), patch.object(
        engine_mod, "has_cuda_device", return_value=False
    ):
        assert VLLMQwen3ASRAudioModel.match(_get_spec("Qwen3-ASR-0.6B")) is False
    with patch.object(engine_platform, "system", return_value="Darwin"), patch.object(
        engine_mod, "has_cuda_device", return_value=True
    ):
        assert VLLMQwen3ASRAudioModel.match(_get_spec("Qwen3-ASR-0.6B")) is False


def test_create_audio_model_instance_default_engine():
    model = create_audio_model_instance(
        "uid",
        "Qwen3-ASR-0.6B",
        model_path="/fake/path",
        enable_virtual_env=False,
    )
    assert isinstance(model, TransformersQwen3ASRAudioModel)


def test_create_audio_model_instance_vllm_engine(linux_cuda_engines):
    model = create_audio_model_instance(
        "uid",
        "Qwen3-ASR-0.6B",
        model_path="/fake/path",
        model_engine="vLLM",
        enable_virtual_env=False,
    )
    assert isinstance(model, VLLMQwen3ASRAudioModel)


def test_create_audio_model_instance_legacy_dispatch():
    model = create_audio_model_instance(
        "uid",
        "whisper-large-v3",
        model_path="/fake/path",
        enable_virtual_env=False,
    )
    assert isinstance(model, WhisperModel)

    # model_engine on a legacy family is ignored with a warning
    model = create_audio_model_instance(
        "uid",
        "SenseVoiceSmall",
        model_path="/fake/path",
        model_engine="transformers",
        enable_virtual_env=False,
    )
    assert isinstance(model, FunASRModel)


def test_builtin_specs_have_vllm_virtualenv_marker():
    for model_name in ("Qwen3-ASR-0.6B", "Qwen3-ASR-1.7B"):
        for spec in BUILTIN_AUDIO_MODELS[model_name]:
            packages = spec.virtualenv.packages if spec.virtualenv else []
            assert any(
                "qwen-asr[vllm]" in pkg and '#engine# == "vLLM"' in pkg
                for pkg in packages
            ), f"{model_name} misses qwen-asr[vllm] virtualenv marker"
