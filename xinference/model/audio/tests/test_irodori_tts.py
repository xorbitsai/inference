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

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from .. import _install
from .. import irodori_tts as irodori_tts_module
from .. import load_model_family_from_json
from ..core import create_audio_model_instance, match_audio
from ..irodori_tts import IrodoriTTSModel


def _model_spec(model_file_name="model.safetensors"):
    return SimpleNamespace(
        model_name="Irodori-TTS-v4.1-Small",
        model_family="Irodori-TTS",
        model_ability=[
            "text2audio",
            "text2audio_voice_design",
            "text2audio_voice_cloning",
            "text2audio_emotion_control",
        ],
        model_file_name=model_file_name,
    )


class _FakeRuntimeKey:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeSamplingRequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeRuntime:
    def __init__(self):
        self.requests = []
        self.prompt_bytes = None

    def synthesize(self, request):
        self.requests.append(request)
        if getattr(request, "ref_wav", None):
            self.prompt_bytes = Path(request.ref_wav).read_bytes()
        return SimpleNamespace(sample_rate=48000, audio="audio")


def test_irodori_catalog_registers_quantization_variants():
    torchao_x86_64_package = 'torchao>=0.16,<0.17 ; platform_machine == "x86_64"'
    torchao_aarch64_package = (
        "torchao @ https://files.pythonhosted.org/packages/d0/3d/"
        "0c5a5833a135a045510e06c06b3d4cf316b06d59415bc21e0b021a000cc8/"
        'torchao-0.16.0-py3-none-any.whl ; platform_machine == "aarch64"'
    )
    models = {}
    load_model_family_from_json("model_spec.json", models)

    expected_model_names = {
        "Irodori-TTS-v4.1-Small",
        "Irodori-TTS-v4.1-Anime",
    }
    assert expected_model_names <= models.keys()

    expected_quantizations = {
        "none",
        "INT8-Weight-Only",
        "INT8-Dynamic",
        "INT4-Weight-Only",
        "Float8-Weight-Only",
        "Float8-Dynamic",
    }
    expected_model_ids = {
        "Irodori-TTS-v4.1-Small": "Aratako/Irodori-TTS-v4.1-Small",
        "Irodori-TTS-v4.1-Anime": "phasefield-audio/Irodori-TTS-v4.1-Anime",
    }
    expected_quantized_model_ids = {
        "Irodori-TTS-v4.1-Small": "Aratako/Irodori-TTS-v4.1-Small-Quantized",
        "Irodori-TTS-v4.1-Anime": "phasefield-audio/Irodori-TTS-v4.1-Anime",
    }

    for model_name in expected_model_names:
        specs = models[model_name]
        assert {spec.quantization for spec in specs} == expected_quantizations
        assert {spec.model_hub for spec in specs} == {"huggingface", "modelscope"}
        assert {spec.model_revision for spec in specs} == {"main", "master"}
        assert {spec.model_family for spec in specs} == {"Irodori-TTS"}
        assert {tuple(spec.model_lang or []) for spec in specs} == {("JA",)}
        assert all(spec.virtualenv is not None for spec in specs)
        assert all(
            not any(
                package.startswith("irodori-tts @")
                for package in spec.virtualenv.packages
            )
            for spec in specs
        )
        assert all(
            {
                "argbind>=0.3.7",
                "descript-audiotools>=0.7.2",
                "einops",
                "protobuf==3.19.6",
                "tqdm==4.66.5",
            }.issubset(spec.virtualenv.packages)
            for spec in specs
        )
        assert all(
            not any(
                package.startswith("dacvae @") or package.startswith("silentcipher")
                for package in spec.virtualenv.packages
            )
            for spec in specs
        )
        assert all("#system_torchcodec#" in spec.virtualenv.packages for spec in specs)
        for quantization in expected_quantizations:
            quantized_specs = [
                spec for spec in specs if spec.quantization == quantization
            ]
            assert {spec.model_hub for spec in quantized_specs} == {
                "huggingface",
                "modelscope",
            }
            if quantization == "none":
                assert all(
                    spec.model_id == expected_model_ids[model_name]
                    and spec.model_file_name == "model.safetensors"
                    and torchao_x86_64_package not in spec.virtualenv.packages
                    and torchao_aarch64_package not in spec.virtualenv.packages
                    for spec in quantized_specs
                )
            else:
                assert all(
                    spec.model_id == expected_quantized_model_ids[model_name]
                    and spec.model_file_name
                    == f"{quantization.lower()}/model.safetensors"
                    and torchao_x86_64_package in spec.virtualenv.packages
                    and torchao_aarch64_package in spec.virtualenv.packages
                    for spec in quantized_specs
                )


def test_irodori_uses_vendored_source(monkeypatch):
    vendor_root = Path(irodori_tts_module._IRODORI_VENDOR_ROOT)
    dacvae_vendor_root = Path(irodori_tts_module._DACVAE_VENDOR_ROOT)
    assert (vendor_root / "irodori_tts" / "inference_runtime.py").is_file()
    assert (vendor_root / "irodori_tts" / "LICENSE").is_file()
    assert (dacvae_vendor_root / "dacvae" / "__init__.py").is_file()
    assert (dacvae_vendor_root / "LICENSE").is_file()

    paths = list(sys.path)
    paths.append(str(vendor_root))
    paths.append(str(dacvae_vendor_root))
    monkeypatch.setattr(sys, "path", paths)

    irodori_tts_module._ensure_vendored_irodori_source()

    assert sys.path[0] == str(dacvae_vendor_root)
    assert sys.path[1] == str(vendor_root)
    assert sys.path.count(str(vendor_root)) == 1
    assert sys.path.count(str(dacvae_vendor_root)) == 1


def test_irodori_runtime_disables_silentcipher_watermarking():
    runtime_source = (
        Path(irodori_tts_module._IRODORI_VENDOR_ROOT)
        / "irodori_tts"
        / "inference_runtime.py"
    ).read_text()

    assert "SilentCipherWatermarker" not in runtime_source
    assert "silentcipher_watermark" not in runtime_source


def test_irodori_load_uses_selected_checkpoint_and_runtime_options(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "int8-weight-only" / "model.safetensors"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    runtime = _FakeRuntime()
    captured = {}

    class FakeInferenceRuntime:
        @staticmethod
        def from_key(key):
            captured["key"] = key
            return runtime

    monkeypatch.setattr(
        irodori_tts_module,
        "_load_irodori_runtime_components",
        lambda: (
            FakeInferenceRuntime,
            _FakeRuntimeKey,
            _FakeSamplingRequest,
            lambda: "cpu",
        ),
    )
    model = IrodoriTTSModel(
        "irodori",
        str(tmp_path),
        _model_spec("int8-weight-only/model.safetensors"),
        device="cuda:1",
        codec_device="cpu",
        model_precision="bf16",
        codec_precision="fp32",
        compile_model=True,
    )

    model.load()

    assert captured["key"].checkpoint == str(checkpoint)
    assert captured["key"].model_device == "cuda:1"
    assert captured["key"].codec_device == "cpu"
    assert captured["key"].model_precision == "bf16"
    assert captured["key"].compile_model is True


def test_irodori_speech_maps_caption_and_cleans_uploaded_reference(
    tmp_path, monkeypatch
):
    runtime = _FakeRuntime()
    model = IrodoriTTSModel("irodori", str(tmp_path), _model_spec())
    model._runtime = runtime
    model._sampling_request_cls = _FakeSamplingRequest
    captured = {}
    monkeypatch.setattr(
        irodori_tts_module,
        "_audio_to_bytes",
        lambda response_format, sample_rate, audio: captured.update(
            response_format=response_format, sample_rate=sample_rate, audio=audio
        )
        or b"encoded-audio",
    )

    result = model.speech(
        input="こんにちは。",
        voice="",
        response_format="wav",
        prompt_speech=b"reference-audio",
        instruct="落ち着いた女性の声",
        num_steps=6,
        seed=7,
    )

    request = runtime.requests[-1]
    assert result == b"encoded-audio"
    assert runtime.prompt_bytes == b"reference-audio"
    assert request.caption == "落ち着いた女性の声"
    assert request.no_ref is False
    assert request.num_steps == 6
    assert request.seed == 7
    assert not Path(request.ref_wav).exists()
    assert captured == {
        "response_format": "wav",
        "sample_rate": 48000,
        "audio": "audio",
    }


def test_irodori_speech_defaults_to_reference_free_voice_design(tmp_path, monkeypatch):
    runtime = _FakeRuntime()
    model = IrodoriTTSModel("irodori", str(tmp_path), _model_spec())
    model._runtime = runtime
    model._sampling_request_cls = _FakeSamplingRequest
    monkeypatch.setattr(irodori_tts_module, "_audio_to_bytes", lambda *args: b"ok")

    model.speech(input="こんにちは。", voice=None, prompt_text="明るいアニメ声")

    request = runtime.requests[-1]
    assert request.no_ref is True
    assert request.caption == "明るいアニメ声"


def test_irodori_rejects_streaming_and_conflicting_references(tmp_path):
    model = IrodoriTTSModel("irodori", str(tmp_path), _model_spec())
    model._runtime = _FakeRuntime()
    model._sampling_request_cls = _FakeSamplingRequest

    with pytest.raises(ValueError, match="does not support streaming"):
        model.speech(input="こんにちは。", voice=None, stream=True)
    with pytest.raises(ValueError, match="cannot be combined"):
        model.speech(
            input="こんにちは。",
            voice=None,
            prompt_speech=b"reference-audio",
            ref_embed="speaker.safetensors",
        )


def test_audio_factory_creates_irodori_model():
    _install()

    model = create_audio_model_instance(
        "irodori",
        "Irodori-TTS-v4.1-Anime",
        model_path="/fake/path",
        enable_virtual_env=False,
        quantization="INT4-Weight-Only",
    )

    assert isinstance(model, IrodoriTTSModel)
    assert model.model_family.model_file_name == "int4-weight-only/model.safetensors"
    assert match_audio("Irodori-TTS-v4.1-Anime").model_file_name == "model.safetensors"
    with pytest.raises(ValueError, match="does not support quantization"):
        match_audio("Irodori-TTS-v4.1-Anime", quantization="invalid")
