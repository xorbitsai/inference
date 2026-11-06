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

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch

from .... import constants
from ....core.exceptions import InvalidModelRequestError
from ....core.utils import (
    apply_engine_virtualenv_settings,
    filter_virtualenv_packages_by_markers,
)
from ...cache_manager import CacheManager
from .. import audio8_tts, audio8_tts_sglang
from .. import core as audio_core
from .. import engine, utils
from ..audio8_tts import Audio8TTSModel
from ..audio8_tts_sglang import SGLangOmniAudio8TTSModel
from ..core import create_audio_model_instance, match_audio
from ..engine import SGLangOmniAudio8TTSAudioModel, TransformersAudio8TTSAudioModel
from ..engine_family import AUDIO_ENGINES, generate_engine_config_by_model_name

MODEL = "Audio8-TTS-Realistic-0.6b"


@pytest.mark.parametrize("backend", ["transformers", "SGLang-Omni"])
def test_catalog_and_dispatch(monkeypatch, tmp_path, backend):
    monkeypatch.setattr(constants, "XINFERENCE_TRUST_REMOTE_CODE", True)
    hub, revision = "huggingface", "main"
    monkeypatch.setattr(engine, "has_cuda_device", lambda: True)
    monkeypatch.setattr(engine.platform, "system", lambda: "Linux")
    monkeypatch.setitem(AUDIO_ENGINES, MODEL, {})
    shared_spec = match_audio(MODEL, hub)
    generate_engine_config_by_model_name(shared_spec)
    assert set(AUDIO_ENGINES[MODEL]) == {"transformers", "SGLang-Omni"}
    spec = match_audio(MODEL, hub, model_engine=backend)
    assert spec is shared_spec
    assert spec.model_hub == hub
    assert spec.model_id == f"Edge0/{MODEL}"
    assert spec.model_revision == revision
    assert spec.engine == "transformers"
    assert len(spec.model_lang) == 23
    assert "text2audio_voice_cloning" in spec.model_ability
    model = create_audio_model_instance(
        "audio8",
        MODEL,
        hub,
        model_path=str(tmp_path),
        model_engine=backend.lower(),
        enable_virtual_env=False,
    )
    cls = (
        TransformersAudio8TTSAudioModel
        if backend == "transformers"
        else SGLangOmniAudio8TTSAudioModel
    )
    assert isinstance(model, cls)
    assert model._model_spec.model_hub == hub
    assert model._model_spec.model_engine == backend
    assert TransformersAudio8TTSAudioModel.match(shared_spec)
    assert SGLangOmniAudio8TTSAudioModel.match(shared_spec)
    assert (
        CacheManager(spec).get_cache_dir() == CacheManager(shared_spec).get_cache_dir()
    )
    default = create_audio_model_instance(
        "audio8",
        MODEL,
        hub,
        model_path=str(tmp_path),
        enable_virtual_env=False,
    )
    assert isinstance(default, TransformersAudio8TTSAudioModel)


@pytest.fixture
def audio8_spec(monkeypatch):
    monkeypatch.setattr(constants, "XINFERENCE_TRUST_REMOTE_CODE", False)
    monkeypatch.setattr(engine, "has_cuda_device", lambda: True)
    monkeypatch.setattr(engine.platform, "system", lambda: "Linux")
    monkeypatch.setitem(AUDIO_ENGINES, MODEL, {})
    spec = match_audio(MODEL, "huggingface").copy(deep=True)
    generate_engine_config_by_model_name(spec)
    monkeypatch.setattr(audio_core, "match_audio", Mock(return_value=spec))
    return spec


@pytest.mark.parametrize("backend", ["transformers", "SGLang-Omni"])
@pytest.mark.parametrize("builtin", [False, True])
def test_explicit_audio8_path_requires_operator_trust(
    audio8_spec, monkeypatch, tmp_path, backend, builtin
):
    audio8_spec.is_builtin = builtin
    cache = Mock()
    monkeypatch.setattr(CacheManager, "cache", cache)
    with pytest.raises(ValueError, match="XINFERENCE_TRUST_REMOTE_CODE=1"):
        create_audio_model_instance(
            "audio8",
            MODEL,
            model_path=str(tmp_path),
            model_engine=backend,
            enable_virtual_env=False,
            trust_remote_code=True,
            is_builtin=True,
        )
    cache.assert_not_called()
    assert audio8_spec.is_builtin is builtin


@pytest.mark.parametrize("backend", ["transformers", "SGLang-Omni"])
def test_custom_audio8_rejected_before_download(audio8_spec, monkeypatch, backend):
    audio8_spec.is_builtin = False
    cache = Mock()
    monkeypatch.setattr(CacheManager, "cache", cache)
    with pytest.raises(ValueError, match="XINFERENCE_TRUST_REMOTE_CODE=1"):
        create_audio_model_instance(
            "audio8", MODEL, model_engine=backend, enable_virtual_env=False
        )
    cache.assert_not_called()


@pytest.mark.parametrize("model_class", [Audio8TTSModel, SGLangOmniAudio8TTSModel])
def test_untrusted_audio8_load_rejected_before_model_code(
    audio8_spec, monkeypatch, tmp_path, model_class
):
    audio8_spec.is_builtin = False
    check_version = Mock(side_effect=AssertionError("loader must not run"))
    monkeypatch.setattr(audio8_tts, "_check_transformers_version", check_version)
    monkeypatch.setattr(audio8_tts_sglang, "_check_transformers_version", check_version)
    model = model_class("audio8", str(tmp_path), audio8_spec)
    with pytest.raises(ValueError, match="XINFERENCE_TRUST_REMOTE_CODE=1"):
        model.load()
    check_version.assert_not_called()
    if isinstance(model, SGLangOmniAudio8TTSModel):
        assert model._loop is None
        assert model._thread is None


@pytest.mark.parametrize("backend", ["transformers", "SGLang-Omni"])
@pytest.mark.parametrize("builtin", [False, True])
def test_operator_can_use_audio8_explicit_path(
    audio8_spec, monkeypatch, tmp_path, backend, builtin
):
    audio8_spec.is_builtin = builtin
    monkeypatch.setattr(constants, "XINFERENCE_TRUST_REMOTE_CODE", True)
    model = create_audio_model_instance(
        "audio8",
        MODEL,
        model_path=str(tmp_path),
        model_engine=backend,
        enable_virtual_env=False,
    )
    assert model.model_family.is_builtin is False
    assert audio8_spec.is_builtin is builtin


@pytest.mark.parametrize("backend", ["transformers", "SGLang-Omni"])
def test_builtin_audio8_cache_requires_no_opt_in(
    audio8_spec, monkeypatch, tmp_path, backend
):
    assert audio8_spec.is_builtin is True
    cache = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(CacheManager, "cache", cache)
    model = create_audio_model_instance(
        "audio8", MODEL, model_engine=backend, enable_virtual_env=False
    )
    cache.assert_called_once()
    assert model.model_family.is_builtin is True


def test_single_spec_and_engine_dependency_isolation():
    from packaging.requirements import Requirement
    from xoscar.virtualenv import VirtualEnvManager

    catalog = json.loads(
        (Path(__file__).parents[1] / "models" / f"{MODEL}.json").read_text()
    )
    assert len(catalog) == 1
    assert set(catalog[0]["model_src"]) == {"huggingface"}
    spec = match_audio(MODEL, "huggingface")
    assert spec.virtualenv is not None
    assert spec.virtualenv.extra_index_url is None
    resolved = {}
    for backend in ("transformers", "SGLang-Omni"):
        packages = filter_virtualenv_packages_by_markers(
            spec.virtualenv.packages, model_engine=backend, cuda_version="13.0"
        )
        resolved[backend] = [
            Requirement(package)
            for package in VirtualEnvManager.process_packages(packages)
        ]
        settings = spec.virtualenv.copy(deep=True)
        apply_engine_virtualenv_settings(settings, backend)
        if backend == "transformers":
            assert settings.extra_index_url is None
        else:
            assert settings.extra_index_url == [
                "https://download.pytorch.org/whl/cu128"
            ]
            assert settings.index_strategy == "unsafe-best-match"
    transformer_requirements = resolved["transformers"]
    assert not any(
        req.name in {"sglang", "sglang-omni"} for req in transformer_requirements
    )
    assert any(
        req.name == "torch" and req.specifier.contains(torch.__version__)
        for req in transformer_requirements
    )
    assert not any("2.9.1+cu128" in str(req) for req in transformer_requirements)
    omni_requirements = resolved["SGLang-Omni"]
    torch_requirements = [req for req in omni_requirements if req.name == "torch"]
    assert len(torch_requirements) == 1
    assert str(torch_requirements[0].specifier) == "==2.9.1+cu128"
    assert any(req.name == "sglang-omni" and req.url for req in omni_requirements)
    assert any(
        req.name == "transformers" and str(req.specifier) == "==4.57.1"
        for req in omni_requirements
    )


def test_engine_platform_and_dependencies(monkeypatch):
    spec = match_audio(MODEL, "huggingface", model_engine="SGLang-Omni")
    monkeypatch.setattr(engine.platform, "system", lambda: "Linux")
    monkeypatch.setattr(engine, "has_cuda_device", lambda: False)
    assert not SGLangOmniAudio8TTSAudioModel.match(spec)
    monkeypatch.setattr(engine, "has_cuda_device", lambda: True)
    assert SGLangOmniAudio8TTSAudioModel.match(spec)
    monkeypatch.setattr(engine.platform, "system", lambda: "Darwin")
    assert not SGLangOmniAudio8TTSAudioModel.match(spec)
    catalog = json.loads(
        (Path(__file__).parents[1] / "models" / f"{MODEL}.json").read_text()
    )
    assert all(set(variant["model_src"]) == {"huggingface"} for variant in catalog)
    assert (
        'transformers>=4.57.0,<5 ; #engine# == "transformers"'
        in catalog[0]["virtualenv"]["packages"]
    )
    packages = catalog[0]["virtualenv"]["packages"]
    assert any("68a572348837f7b004857b4b07993c20ade4c017" in p for p in packages)
    assert 'sglang==0.5.8 ; #engine# == "SGLang-Omni"' in packages
    assert 'torch==2.9.1+cu128 ; #engine# == "SGLang-Omni"' in packages


@pytest.mark.parametrize(
    "device,dtype", [("cpu", torch.float32), ("cuda:1", torch.bfloat16)]
)
def test_load_selected_local_snapshot(monkeypatch, tmp_path, device, dtype):
    hub = "huggingface"
    calls = []
    runtime = SimpleNamespace()
    runtime.eval = lambda: runtime
    runtime.to = lambda target: calls.append(("device", str(target))) or runtime

    def load_processor(path, **kwargs):
        calls.append(("processor", path, kwargs))
        return object()

    def load_model(path, **kwargs):
        calls.append(("model", path, kwargs))
        return runtime

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            __version__="4.57.1",
            AutoModel=SimpleNamespace(from_pretrained=load_model),
            AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    (tmp_path / "codec.pth").touch()
    model = Audio8TTSModel(
        "audio8", str(tmp_path), match_audio(MODEL, hub), device=device
    )
    model.load()
    assert calls[0] == (
        "processor",
        str(tmp_path),
        {"trust_remote_code": True, "local_files_only": True},
    )
    assert calls[1] == (
        "model",
        str(tmp_path),
        {"trust_remote_code": True, "local_files_only": True, "dtype": dtype},
    )
    assert calls[2] == ("device", device)


@pytest.mark.parametrize("version", ["4.56.2", "5.0.0", "5.13.1"])
def test_reject_incompatible_transformers(monkeypatch, version):
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(__version__=version)
    )
    with pytest.raises(ImportError, match="transformers>=4.57.0,<5"):
        audio8_tts._check_transformers_version()


@pytest.fixture
def loaded_model(monkeypatch, tmp_path):
    calls = {}
    model = Audio8TTSModel(
        "audio8", str(tmp_path), match_audio(MODEL, "huggingface"), device="cpu"
    )

    def processor(**kwargs):
        calls["processor"] = kwargs
        if "reference_audio" in kwargs:
            path = Path(kwargs["reference_audio"][0])
            calls["reference_path"] = path
            calls["reference_bytes"] = path.read_bytes()
        return {"input_ids": torch.tensor([[1, 2]])}

    def generate(**kwargs):
        calls["generate"] = kwargs
        assert not torch.is_grad_enabled()
        return SimpleNamespace(codes=torch.ones(1, 10, 2, dtype=torch.long))

    def encode(fmt, rate, audio):
        calls["encoded"] = (fmt, rate, audio)
        return b"encoded-audio"

    model._processor = processor
    model._model = SimpleNamespace(
        generate=generate,
        decode_audio=lambda codes: (
            torch.tensor([[0.1, -0.2, 0.9]]),
            torch.tensor([2]),
        ),
        config=SimpleNamespace(codec_sample_rate=44100),
    )
    monkeypatch.setattr(model, "_encode_audio", encode)
    return model, calls


@pytest.mark.parametrize("clone", [False, True])
def test_speech_and_reference_cleanup(loaded_model, clone):
    model, calls = loaded_model
    kwargs = (
        {"prompt_speech": b"reference", "prompt_text": "Exact transcript."}
        if clone
        else {}
    )
    assert (
        model.speech(
            " Hello ",
            None,
            "wav",
            seed=42,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            top_k=30,
            do_sample=False,
            language="en",
            **kwargs,
        )
        == b"encoded-audio"
    )
    assert calls["processor"]["text"] == ["Hello"]
    assert calls["generate"]["max_new_tokens"] == 128
    assert calls["generate"]["do_sample"] is False
    assert calls["generate"]["return_dict_in_generate"] is True
    assert calls["generate"]["generator"].initial_seed() == 42
    fmt, rate, audio = calls["encoded"]
    assert (fmt, rate, tuple(audio.shape)) == ("wav", 44100, (1, 2))
    if clone:
        assert calls["reference_bytes"] == b"reference"
        assert calls["processor"]["reference_text"] == ["Exact transcript."]
        assert not calls["reference_path"].exists()
    else:
        assert "reference_audio" not in calls["processor"]


def test_reference_cleanup_on_error(loaded_model):
    model, calls = loaded_model

    def fail(**kwargs):
        raise RuntimeError("generation failed")

    model._model.generate = fail
    with pytest.raises(RuntimeError, match="generation failed"):
        model.speech(
            "Hello", None, prompt_speech=b"reference", prompt_text="Transcript"
        )
    assert not calls["reference_path"].exists()


@pytest.mark.parametrize("model_class", [Audio8TTSModel, SGLangOmniAudio8TTSModel])
@pytest.mark.parametrize("response_format", [None, "wav", "flac", "ogg", "pcm"])
def test_speech_really_encodes_without_torchcodec(
    monkeypatch, tmp_path, model_class, response_format
):
    import soundfile

    monkeypatch.setitem(sys.modules, "torchcodec", None)
    monkeypatch.setitem(sys.modules, "torchcodec.encoders", None)
    sample_rate = 44100
    samples = 0.2 * torch.sin(2 * torch.pi * 440 * torch.arange(11025) / sample_rate)
    model = model_class("audio8", str(tmp_path), match_audio(MODEL, "huggingface"))
    generate = Mock(return_value=(samples.unsqueeze(0), sample_rate))
    monkeypatch.setattr(model, "_generate", generate)
    kwargs = {} if response_format is None else {"response_format": response_format}
    encoded = model.speech("Hello", None, **kwargs)
    assert isinstance(encoded, bytes) and encoded
    read_options = (
        {
            "format": "RAW",
            "subtype": "PCM_16",
            "endian": "LITTLE",
            "channels": 1,
            "samplerate": sample_rate,
        }
        if response_format == "pcm"
        else {}
    )
    with soundfile.SoundFile(io.BytesIO(encoded), **read_options) as audio:
        assert audio.samplerate == sample_rate
        assert audio.channels == 1
        if response_format is None:
            assert audio.format == "MP3"
        elif response_format == "ogg":
            assert audio.format == "OGG"
            assert audio.subtype == "VORBIS"
        decoded = audio.read(dtype="float32")
    assert len(decoded) >= len(samples)
    assert np.sqrt(np.mean(decoded**2)) > 0.01
    if response_format == "ogg":
        assert np.corrcoef(decoded[: len(samples)], samples.numpy())[0, 1] > 0.99
    elif response_format is not None:
        np.testing.assert_allclose(decoded, samples.numpy(), atol=1 / 32768)
    generate.assert_called_once()


@pytest.mark.parametrize("model_class", [Audio8TTSModel, SGLangOmniAudio8TTSModel])
@pytest.mark.parametrize("top_k", [0, -1, False, True, 1.5])
def test_top_k_must_be_positive_before_generation(
    monkeypatch, tmp_path, model_class, top_k
):
    model = model_class("audio8", str(tmp_path), match_audio(MODEL, "huggingface"))
    generate = Mock(side_effect=AssertionError("backend must not run"))
    monkeypatch.setattr(model, "_generate", generate)
    with pytest.raises(
        InvalidModelRequestError, match="top_k must be a positive integer"
    ):
        model.speech("Hello", None, top_k=top_k)
    generate.assert_not_called()


@pytest.mark.parametrize("model_class", [Audio8TTSModel, SGLangOmniAudio8TTSModel])
def test_top_k_one_is_forwarded_unchanged(model_class, tmp_path):
    model = model_class("audio8", str(tmp_path), match_audio(MODEL, "huggingface"))
    assert model._generation_options({"top_k": 1})["top_k"] == 1


def test_other_formats_keep_existing_encoder(monkeypatch):
    encode = Mock(return_value=b"encoded-aac")
    monkeypatch.setattr(utils, "audio_to_bytes", encode)
    audio = torch.zeros((1, 4410))
    assert Audio8TTSModel._encode_audio("aac", 44100, audio) == b"encoded-aac"
    encode.assert_called_once_with("aac", 44100, audio)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"prompt_speech": b"audio"}, "provided together"),
        ({"prompt_text": "text"}, "provided together"),
        ({"prompt_speech": b"", "prompt_text": "text"}, "non-empty audio bytes"),
        ({"prompt_speech": b"audio", "prompt_text": " "}, "non-empty text"),
        ({"stream": True}, "streaming"),
        ({"speed": 1.2}, "speed=1.0"),
        ({"max_new_tokens": 0}, "positive integer"),
        ({"top_p": 0}, "top_p"),
        ({"top_k": -1}, "top_k"),
        ({"temperature": 0}, "temperature"),
        ({"do_sample": "false"}, "boolean"),
        ({"language": "unknown"}, "language"),
        ({"unknown": 1}, "Unsupported Audio8-TTS arguments"),
    ],
)
def test_invalid_request(loaded_model, kwargs, error):
    model, calls = loaded_model
    with pytest.raises(ValueError, match=error):
        model.speech("Hello", None, **kwargs)
    assert "generate" not in calls


def test_sglang_loop_request_and_stop(monkeypatch, tmp_path):
    monkeypatch.setattr(audio8_tts_sglang, "_check_transformers_version", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        Audio8TTSModel,
        "_encode_audio",
        staticmethod(lambda fmt, rate, audio: (fmt, rate, audio.numpy())),
    )
    requests = []

    async def submit(request_id, request):
        requests.append(request)
        if "references" in request.inputs:
            path = Path(request.inputs["references"][0]["audio_path"])
            assert path.read_bytes() == b"reference"
        return {
            "audio_waveform": np.array([0.1, -0.2], dtype=np.float32).tobytes(),
            "sample_rate": 44100,
        }

    runner = SimpleNamespace(
        coordinator=SimpleNamespace(submit=submit, abort=AsyncMock()), stop=AsyncMock()
    )

    async def start(self):
        self._runner = runner

    monkeypatch.setattr(SGLangOmniAudio8TTSModel, "_start", start)
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.proto",
        SimpleNamespace(OmniRequest=lambda **kw: SimpleNamespace(**kw)),
    )
    model = SGLangOmniAudio8TTSModel(
        "audio8",
        str(tmp_path),
        match_audio(MODEL, "huggingface", model_engine="SGLang-Omni"),
    )
    model.load()
    thread, loop = model._thread, model._loop
    try:
        fmt, rate, audio = model.speech(
            "Hello",
            None,
            "pcm",
            do_sample=False,
            prompt_speech=b"reference",
            prompt_text="Transcript",
        )
        assert (fmt, rate, audio.shape) == ("pcm", 44100, (1, 2))
        assert requests[0].params["temperature"] == 0
        assert "do_sample" not in requests[0].params
        assert not Path(requests[0].inputs["references"][0]["audio_path"]).exists()
        with pytest.raises(ValueError, match="does not support seed"):
            model.speech("Hello", None, seed=42)
    finally:
        model.stop()
    model.stop()
    runner.stop.assert_awaited_once()
    assert not thread.is_alive()
    assert loop.is_closed()


def test_sglang_failed_start_cleans_loop(monkeypatch, tmp_path):
    monkeypatch.setattr(audio8_tts_sglang, "_check_transformers_version", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    async def fail(self):
        raise RuntimeError("startup failed")

    monkeypatch.setattr(SGLangOmniAudio8TTSModel, "_start", fail)
    model = SGLangOmniAudio8TTSModel(
        "audio8", str(tmp_path), match_audio(MODEL, "huggingface")
    )
    with pytest.raises(RuntimeError, match="startup failed"):
        model.load()
    assert model._loop is None
    assert model._thread is None


def test_sglang_requires_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(audio8_tts_sglang, "_check_transformers_version", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = SGLangOmniAudio8TTSModel(
        "audio8", str(tmp_path), match_audio(MODEL, "huggingface"), device="cpu"
    )
    with pytest.raises(ValueError, match="NVIDIA CUDA"):
        model.load()
