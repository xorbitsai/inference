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

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ...cache_manager import CacheManager
from ...utils import (
    get_engine_params_by_name,
    get_engine_params_by_name_with_virtual_env,
)
from .. import (
    BUILTIN_AUDIO_MODELS,
    _audio_model_variant_identity,
    _install,
    _normalize_legacy_audio_model,
    load_model_family_from_json,
)
from .. import platform as audio_platform
from .. import sys as audio_sys
from ..core import create_audio_model_instance, resolve_audio_model_name_and_engine
from ..engine import (
    MLXAudioSTTEngineModel,
    MLXAudioTTSEngineModel,
    MLXF5TTSAudioModel,
    MLXKokoroAudioModel,
    MLXWhisperAudioModel,
    PyTorchF5TTSAudioModel,
    PyTorchFunASRAudioModel,
    PyTorchKokoroAudioModel,
    PyTorchMeloTTSAudioModel,
    PyTorchQwen3TTSAudioModel,
    PyTorchVoxCPMAudioModel,
    TransformersQwen3ASRAudioModel,
    TransformersWhisperAudioModel,
    VLLMQwen3ASRAudioModel,
)
from ..engine import platform as engine_platform
from ..engine import register_builtin_audio_engines
from ..engine_family import (
    AUDIO_ENGINES,
    check_engine_by_model_name_and_engine,
    generate_engine_config_by_model_name,
    get_supported_engines_for_model,
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
    with (
        patch.object(engine_platform, "system", return_value="Linux"),
        patch.object(engine_mod, "has_cuda_device", return_value=True),
    ):
        AUDIO_ENGINES.clear()
        _register_all_engines()
        yield
    AUDIO_ENGINES.clear()
    AUDIO_ENGINES.update(old_engines)


@pytest.fixture
def apple_mlx_engines():
    model_names = (
        "whisper-tiny",
        "F5-TTS",
        "Kokoro-82M",
        "SenseVoiceSmall",
        "Fun-ASR-Nano-2512",
        "Qwen3-ASR-0.6B",
        "Qwen3-ASR-1.7B",
        "Qwen3-TTS-12Hz-0.6B-Base",
        "Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "MeloTTS-English",
        "MeloTTS-English-v3",
        "VoxCPM2",
    )
    old_models = {name: BUILTIN_AUDIO_MODELS.get(name) for name in model_names}
    old_engines = {k: dict(v) for k, v in AUDIO_ENGINES.items()}
    with (
        patch.object(audio_sys, "platform", "darwin"),
        patch.object(audio_platform, "processor", return_value="arm"),
        patch.object(engine_platform, "system", return_value="Darwin"),
    ):
        models = {}
        load_model_family_from_json("model_spec.json", models)
        for name in model_names:
            BUILTIN_AUDIO_MODELS[name] = models[name]
        AUDIO_ENGINES.clear()
        _register_all_engines()
        yield models
    for name, specs in old_models.items():
        if specs is None:
            BUILTIN_AUDIO_MODELS.pop(name, None)
        else:
            BUILTIN_AUDIO_MODELS[name] = specs
    AUDIO_ENGINES.clear()
    AUDIO_ENGINES.update(old_engines)


def test_qwen3_asr_registers_transformers_engine():
    assert "Qwen3-ASR-0.6B" in AUDIO_ENGINES
    assert "transformers" in AUDIO_ENGINES["Qwen3-ASR-0.6B"]
    # default engine is the first registered one
    assert next(iter(AUDIO_ENGINES["Qwen3-ASR-0.6B"])) == "transformers"


def test_audio_engine_families_registered():
    assert "transformers" in AUDIO_ENGINES["whisper-large-v3"]
    assert "PyTorch" in AUDIO_ENGINES["F5-TTS"]
    assert "PyTorch" in AUDIO_ENGINES["Kokoro-82M"]
    assert "PyTorch" in AUDIO_ENGINES["SenseVoiceSmall"]


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
    with (
        patch.object(engine_platform, "system", return_value="Linux"),
        patch.object(engine_mod, "has_cuda_device", return_value=False),
    ):
        assert VLLMQwen3ASRAudioModel.match(_get_spec("Qwen3-ASR-0.6B")) is False
    with (
        patch.object(engine_platform, "system", return_value="Darwin"),
        patch.object(engine_mod, "has_cuda_device", return_value=True),
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


def test_invalid_audio_engine_is_rejected_before_download():
    with patch.object(CacheManager, "cache") as cache:
        with pytest.raises(ValueError, match="cannot be run on engine"):
            create_audio_model_instance(
                "uid",
                "whisper-tiny",
                model_engine="not-an-engine",
                enable_virtual_env=False,
            )
    cache.assert_not_called()


def test_minimax_music3_without_cuda_is_rejected_before_download():
    engine_mod = __import__(
        register_builtin_audio_engines.__module__, fromlist=["has_cuda_device"]
    )
    with (
        patch.object(engine_mod, "has_cuda_device", return_value=False),
        patch.dict(AUDIO_ENGINES, {}, clear=True),
        patch.object(CacheManager, "cache") as cache,
    ):
        register_builtin_audio_engines()
        for model_spec in BUILTIN_AUDIO_MODELS["MiniMax-Music3"]:
            generate_engine_config_by_model_name(model_spec)
        with pytest.raises(ValueError, match="requires an NVIDIA CUDA device"):
            create_audio_model_instance(
                "uid",
                "MiniMax-Music3",
                enable_virtual_env=False,
            )
    cache.assert_not_called()


def test_consolidated_mlx_specs_and_legacy_aliases(apple_mlx_engines):
    models = apple_mlx_engines
    assert "whisper-tiny-mlx" not in models
    assert "F5-TTS-MLX" not in models
    assert "Kokoro-82M-MLX" not in models

    expected_engines = {
        "whisper-tiny": ["transformers", "MLX"],
        "F5-TTS": ["PyTorch", "MLX"],
        "Kokoro-82M": ["PyTorch", "MLX"],
        "SenseVoiceSmall": ["PyTorch", "MLX"],
        "Fun-ASR-Nano-2512": ["PyTorch", "MLX"],
        "Qwen3-ASR-0.6B": ["transformers", "MLX"],
        "Qwen3-TTS-12Hz-0.6B-Base": ["PyTorch", "MLX"],
        "MeloTTS-English": ["PyTorch", "MLX"],
        "VoxCPM2": ["PyTorch", "MLX"],
    }
    for model_name, engines in expected_engines.items():
        assert list(AUDIO_ENGINES[model_name]) == engines

    assert resolve_audio_model_name_and_engine("whisper-tiny-mlx") == (
        "whisper-tiny",
        "MLX",
    )
    assert resolve_audio_model_name_and_engine("F5-TTS-MLX") == (
        "F5-TTS",
        "MLX",
    )
    with pytest.raises(ValueError, match="cannot be launched"):
        resolve_audio_model_name_and_engine("Kokoro-82M-MLX", "PyTorch")


def test_whisper_mlx_virtualenv_overrides_incompatible_numba(apple_mlx_engines):
    whisper_mlx_specs = [
        spec
        for specs in apple_mlx_engines.values()
        for spec in specs
        if spec.model_family == "whisper" and spec.engine == "MLX"
    ]

    assert whisper_mlx_specs
    for spec in whisper_mlx_specs:
        assert spec.virtualenv is not None
        assert 'numba>=0.64.0 ; #engine# == "MLX"' in spec.virtualenv.packages
        assert '#system_numpy# ; #engine# == "MLX"' in spec.virtualenv.packages


def test_create_consolidated_audio_engines(apple_mlx_engines):
    cases = [
        ("whisper-tiny", None, TransformersWhisperAudioModel, "openai/whisper-tiny"),
        ("whisper-tiny", "mlx", MLXWhisperAudioModel, "mlx-community/whisper-tiny"),
        ("whisper-tiny-mlx", None, MLXWhisperAudioModel, "mlx-community/whisper-tiny"),
        ("F5-TTS", None, PyTorchF5TTSAudioModel, "SWivid/F5-TTS"),
        ("F5-TTS-MLX", None, MLXF5TTSAudioModel, "lucasnewman/f5-tts-mlx"),
        ("Kokoro-82M", None, PyTorchKokoroAudioModel, "hexgrad/Kokoro-82M"),
        ("Kokoro-82M-MLX", None, MLXKokoroAudioModel, "prince-canuma/Kokoro-82M"),
        (
            "SenseVoiceSmall",
            "PyTorch",
            PyTorchFunASRAudioModel,
            "FunAudioLLM/SenseVoiceSmall",
        ),
        (
            "SenseVoiceSmall",
            "mlx",
            MLXAudioSTTEngineModel,
            "mlx-community/SenseVoiceSmall",
        ),
        (
            "Qwen3-TTS-12Hz-0.6B-Base",
            None,
            PyTorchQwen3TTSAudioModel,
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        ),
        (
            "Qwen3-TTS-12Hz-0.6B-Base",
            "MLX",
            MLXAudioTTSEngineModel,
            "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        ),
        (
            "MeloTTS-English",
            None,
            PyTorchMeloTTSAudioModel,
            "myshell-ai/MeloTTS-English",
        ),
        (
            "VoxCPM2",
            None,
            PyTorchVoxCPMAudioModel,
            "openbmb/VoxCPM2",
        ),
    ]
    for model_name, engine, expected_class, expected_model_id in cases:
        model = create_audio_model_instance(
            "uid",
            model_name,
            model_path="/fake/path",
            model_engine=engine,
            enable_virtual_env=False,
        )
        assert isinstance(model, expected_class)
        assert model.model_family.model_id == expected_model_id


def test_audio_engine_variants_keep_separate_cache_paths(apple_mlx_engines):
    specs = apple_mlx_engines["whisper-tiny"]
    transformers_spec = next(spec for spec in specs if spec.engine == "transformers")
    mlx_spec = next(spec for spec in specs if spec.engine == "MLX")

    assert CacheManager(transformers_spec).get_cache_dir().endswith("/whisper-tiny")
    assert CacheManager(mlx_spec).get_cache_dir().endswith("/whisper-tiny-mlx")
    assert transformers_spec.to_version_info()["model_version"] == "whisper-tiny"
    assert mlx_spec.to_version_info()["model_version"] == "whisper-tiny-mlx"


def test_audio_engine_discovery_filters_unrelated_engines(apple_mlx_engines):
    models = apple_mlx_engines
    assert set(get_supported_engines_for_model(models["F5-TTS"])) == {
        "PyTorch",
        "MLX",
    }
    assert set(get_supported_engines_for_model(models["whisper-tiny"])) == {
        "transformers",
        "MLX",
    }


def test_downloaded_audio_registry_updates_variants_independently(tmp_path):
    from ...utils import install_models_with_merge

    model_spec_path = Path(__file__).parents[1] / "model_spec.json"
    built_in_data = json.loads(model_spec_path.read_text())
    downloaded_transformers = dict(built_in_data[0])
    downloaded_transformers["updated_at"] += 1

    downloaded_dir = tmp_path / "v2" / "builtin" / "audio"
    downloaded_dir.mkdir(parents=True)
    (downloaded_dir / "audio_models.json").write_text(
        json.dumps([downloaded_transformers])
    )

    models = {}
    with (
        patch.object(audio_sys, "platform", "darwin"),
        patch.object(audio_platform, "processor", return_value="arm"),
        patch("xinference.constants.XINFERENCE_MODEL_DIR", str(tmp_path)),
    ):
        install_models_with_merge(
            models,
            "model_spec.json",
            "audio",
            "audio_models.json",
            lambda: True,
            load_model_family_from_json,
            model_identity_func=_audio_model_variant_identity,
            model_normalize_func=_normalize_legacy_audio_model,
        )

    whisper_specs = models["whisper-tiny"]
    assert {spec.engine for spec in whisper_specs} == {"transformers", "MLX"}
    assert {
        spec.updated_at for spec in whisper_specs if spec.engine == "transformers"
    } == {downloaded_transformers["updated_at"]}
    assert {spec.cache_name for spec in whisper_specs if spec.engine == "MLX"} == {
        "whisper-tiny-mlx"
    }


@pytest.mark.parametrize("timestamp_delta", [-1, 0, 1])
def test_downloaded_legacy_audio_registry_migrates_default_variant(
    tmp_path, timestamp_delta
):
    from ...utils import install_models_with_merge

    model_spec_path = Path(__file__).parents[1] / "model_spec.json"
    built_in_data = json.loads(model_spec_path.read_text())
    built_in_transformers = built_in_data[0]
    downloaded_legacy = dict(built_in_transformers)
    downloaded_legacy.pop("engine")
    downloaded_legacy.pop("model_format")
    downloaded_legacy.pop("cache_name", None)
    downloaded_legacy["updated_at"] += timestamp_delta

    downloaded_dir = tmp_path / "v2" / "builtin" / "audio"
    downloaded_dir.mkdir(parents=True)
    (downloaded_dir / "audio_models.json").write_text(json.dumps([downloaded_legacy]))

    models = {}
    with (
        patch.object(audio_sys, "platform", "darwin"),
        patch.object(audio_platform, "processor", return_value="arm"),
        patch("xinference.constants.XINFERENCE_MODEL_DIR", str(tmp_path)),
    ):
        install_models_with_merge(
            models,
            "model_spec.json",
            "audio",
            "audio_models.json",
            lambda: True,
            load_model_family_from_json,
            model_identity_func=_audio_model_variant_identity,
            model_normalize_func=_normalize_legacy_audio_model,
        )

    whisper_specs = models["whisper-tiny"]
    transformers_specs = [
        spec for spec in whisper_specs if spec.engine == "transformers"
    ]
    assert len(transformers_specs) == 1
    assert all(spec.engine is not None for spec in whisper_specs)
    assert transformers_specs[0].model_format == "pytorch"
    assert transformers_specs[0].updated_at == max(
        built_in_transformers["updated_at"], downloaded_legacy["updated_at"]
    )
    assert bool(getattr(transformers_specs[0], "is_builtin", False)) is (
        timestamp_delta <= 0
    )


def test_audio_engine_api_returns_variant_formats(apple_mlx_engines):
    with (
        patch.object(PyTorchF5TTSAudioModel, "check_lib", return_value=True),
        patch.object(MLXF5TTSAudioModel, "check_lib", return_value=True),
    ):
        params = get_engine_params_by_name("audio", "F5-TTS", enable_virtual_env=False)

    assert params == {
        "PyTorch": [{"model_name": "F5-TTS", "model_format": "pytorch"}],
        "MLX": [{"model_name": "F5-TTS", "model_format": "mlx"}],
    }


def test_mlx_audio_engine_uses_virtualenv_when_dependency_is_missing(
    apple_mlx_engines,
):
    with (
        patch.object(PyTorchQwen3TTSAudioModel, "check_lib", return_value=True),
        patch.object(
            MLXAudioTTSEngineModel,
            "check_lib",
            return_value=(False, "mlx-audio is not installed"),
        ),
    ):
        params = get_engine_params_by_name_with_virtual_env(
            "audio", "Qwen3-TTS-12Hz-0.6B-Base", enable_virtual_env=True
        )

    assert params["PyTorch"] == [
        {
            "model_name": "Qwen3-TTS-12Hz-0.6B-Base",
            "model_format": "pytorch",
        }
    ]
    assert params["MLX"][0]["model_format"] == "mlx"
    assert params["MLX"][0]["virtualenv_required"] is True


@pytest.mark.parametrize(
    ("model_name", "default_engine_class", "mlx_engine_class"),
    [
        ("whisper-tiny", TransformersWhisperAudioModel, MLXWhisperAudioModel),
        ("F5-TTS", PyTorchF5TTSAudioModel, MLXF5TTSAudioModel),
        ("Kokoro-82M", PyTorchKokoroAudioModel, MLXKokoroAudioModel),
    ],
)
def test_legacy_mlx_audio_engines_use_virtualenv_when_dependency_is_missing(
    apple_mlx_engines, model_name, default_engine_class, mlx_engine_class
):
    with (
        patch.object(default_engine_class, "check_lib", return_value=True),
        patch.object(
            mlx_engine_class,
            "check_lib",
            return_value=(False, "MLX dependency is not installed"),
        ),
    ):
        params = get_engine_params_by_name_with_virtual_env(
            "audio", model_name, enable_virtual_env=True
        )

    assert params["MLX"][0]["virtualenv_required"] is True


def test_mlx_audio_specs_pin_isolated_runtime(apple_mlx_engines):
    models = apple_mlx_engines
    for model_name in (
        "SenseVoiceSmall",
        "Fun-ASR-Nano-2512",
        "Qwen3-ASR-0.6B",
        "Qwen3-TTS-12Hz-0.6B-Base",
        "MeloTTS-English",
        "VoxCPM2",
    ):
        mlx_spec = next(spec for spec in models[model_name] if spec.engine == "MLX")
        packages = mlx_spec.virtualenv.packages
        assert any("mlx-audio" in package for package in packages)
        assert any('#engine# == "MLX"' in package for package in packages)
        assert all("#system_" not in package for package in packages)


@pytest.mark.asyncio
async def test_audio_catalog_groups_engine_variants(apple_mlx_engines):
    from ....core.worker import WorkerActor

    worker = WorkerActor.__new__(WorkerActor)
    registrations = await worker.list_model_registrations("audio", detailed=True)
    whisper_entries = [
        item for item in registrations if item["model_name"] == "whisper-tiny"
    ]

    assert len(whisper_entries) == 1
    specs = whisper_entries[0]["model_specs"]
    assert {(spec["model_engine"], spec["model_format"]) for spec in specs} == {
        ("transformers", "pytorch"),
        ("MLX", "mlx"),
    }


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

    # The fallback remains correct if the derived engine registry has not been
    # initialized yet; a non-empty ``transformers`` metadata value is not MLX.
    with patch.dict(AUDIO_ENGINES, {}, clear=True):
        model = create_audio_model_instance(
            "uid",
            "whisper-large-v3",
            model_path="/fake/path",
            enable_virtual_env=False,
        )
    assert isinstance(model, WhisperModel)

    # Families that still do not have an engine registry retain legacy dispatch.
    model = create_audio_model_instance(
        "uid", "paraformer-zh", model_path="/fake/path", enable_virtual_env=False
    )
    assert isinstance(model, FunASRModel)


def test_builtin_specs_have_vllm_virtualenv_marker():
    for model_name in ("Qwen3-ASR-0.6B", "Qwen3-ASR-1.7B"):
        spec = next(
            spec
            for spec in BUILTIN_AUDIO_MODELS[model_name]
            if spec.engine == "transformers"
        )
        packages = spec.virtualenv.packages if spec.virtualenv else []
        assert any(
            "qwen-asr[vllm]" in pkg and '#engine# == "vLLM"' in pkg for pkg in packages
        ), f"{model_name} misses qwen-asr[vllm] virtualenv marker"
