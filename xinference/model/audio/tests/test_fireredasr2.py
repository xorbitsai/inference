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

import importlib.util
import io
import sys
import wave
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from .... import constants
from ....core.exceptions import InvalidAudioInputError
from ....core.utils import filter_virtualenv_packages_by_markers
from ...cache_manager import CacheManager
from ...utils import (
    get_engine_params_by_name,
    get_engine_params_by_name_with_virtual_env,
)
from .. import fireredasr2
from ..core import create_audio_model_instance, match_audio
from ..engine import PyTorchFireRedASR2AudioModel
from ..engine_family import AUDIO_ENGINES
from ..fireredasr2 import FireRedASR2Model


@pytest.mark.parametrize("model_name", ["FireRedASR2-AED", "FireRedASR2-LLM"])
@pytest.mark.parametrize(
    "hub, owner, revision",
    [("huggingface", "FireRedTeam", "main"), ("modelscope", "xukaituo", "master")],
)
def test_fireredasr2_registration(
    monkeypatch, tmp_path, model_name, hub, owner, revision
):
    spec = match_audio(model_name, download_hub=hub)
    assert spec.model_hub == hub
    assert spec.model_id == f"{owner}/{model_name}"
    assert spec.model_revision == revision
    assert spec.model_ability == ["audio2text"]
    assert spec.model_lang == ["zh", "en"]
    assert spec.multilingual
    assert set(AUDIO_ENGINES[model_name]) == {"PyTorch"}
    assert spec.virtualenv is not None
    assert not any(
        "fireredasr2s" in package.lower() or "git+" in package.lower()
        for package in spec.virtualenv.packages
    )

    cached_specs = []

    def cache(manager):
        cached_specs.append(manager._model_family)
        return str(tmp_path)

    monkeypatch.setattr(CacheManager, "cache", cache)
    model = create_audio_model_instance(
        f"{model_name}-test", model_name, download_hub=hub, enable_virtual_env=False
    )
    assert cached_specs == [spec]
    assert isinstance(model, FireRedASR2Model)


@pytest.mark.parametrize("model_name", ["FireRedASR2-AED", "FireRedASR2-LLM"])
def test_fireredasr2_can_cache_without_ffmpeg(monkeypatch, tmp_path, model_name):
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg.binaries", None)
    cache = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(CacheManager, "cache", cache)

    model = create_audio_model_instance(
        model_name,
        model_name,
        download_hub="modelscope",
        enable_virtual_env=True,
        _xinference_cache_only=True,
    )
    cache.assert_called_once()
    assert isinstance(model, FireRedASR2Model)


@pytest.mark.parametrize("model_name", ["FireRedASR2-AED", "FireRedASR2-LLM"])
@pytest.mark.parametrize(
    "missing_lib, requirement",
    [("kaldiio", "kaldiio==2.18.1"), ("imageio_ffmpeg", "imageio-ffmpeg==0.8.2")],
)
def test_fireredasr2_discovery_can_install_missing_dependencies(
    monkeypatch, model_name, missing_lib, requirement
):
    find_spec = importlib.util.find_spec

    def find_spec_without_dependency(name, *args, **kwargs):
        if name == missing_lib:
            return None
        if name in PyTorchFireRedASR2AudioModel.required_libs:
            return SimpleNamespace()
        return find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec_without_dependency)

    strict = get_engine_params_by_name("audio", model_name, enable_virtual_env=False)
    assert strict == {"PyTorch": f"Library '{missing_lib}' is not installed"}

    installable = get_engine_params_by_name_with_virtual_env(
        "audio", model_name, enable_virtual_env=True
    )
    assert set(installable) == {"PyTorch"}
    assert isinstance(installable["PyTorch"], list)
    (engine,) = installable["PyTorch"]
    assert engine["model_name"] == model_name
    assert engine["model_format"] == "pytorch"
    assert engine["virtualenv_required"] is True
    assert strict["PyTorch"] in engine["virtualenv_reason"]

    spec = match_audio(model_name, "modelscope")
    packages = filter_virtualenv_packages_by_markers(
        spec.virtualenv.packages, model_engine="PyTorch", cuda_version=None
    )
    assert requirement in packages


def test_explicit_model_path_requires_trust(monkeypatch, tmp_path):
    monkeypatch.setattr(constants, "XINFERENCE_TRUST_REMOTE_CODE", False)
    monkeypatch.setitem(sys.modules, "xinference.thirdparty.fireredasr2", None)
    name = "FireRedASR2-AED"
    spec = match_audio(name, download_hub="huggingface")

    with pytest.raises(ValueError, match="XINFERENCE_TRUST_REMOTE_CODE=1"):
        model = create_audio_model_instance(
            "firered-test",
            name,
            download_hub="huggingface",
            model_path=str(tmp_path),
            enable_virtual_env=False,
        )
        model.load()

    assert spec.is_builtin


def _stereo_wav(seconds, sample_rate=8000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x10\x00\x10" * int(seconds * sample_rate))
    return buffer.getvalue()


@pytest.mark.parametrize("bundled_binaries", [0, 1, 2])
def test_ffmpeg_resolver_uses_only_wheel_binary(
    monkeypatch, tmp_path, bundled_binaries
):
    binaries = tmp_path / "binaries"
    binaries.mkdir()
    (binaries / "README.md").touch()
    for index in range(bundled_binaries):
        binary = binaries / f"ffmpeg-test-{index}"
        binary.touch(mode=0o755)

    system_ffmpeg = tmp_path / "ffmpeg"
    system_ffmpeg.write_text("#!/bin/sh\nexit 0\n")
    system_ffmpeg.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", str(system_ffmpeg))
    find_resources = Mock(return_value=binaries)
    monkeypatch.setattr(fireredasr2, "files", find_resources)

    if bundled_binaries == 1:
        assert fireredasr2._get_ffmpeg_executable() == str(
            (binaries / "ffmpeg-test-0").resolve()
        )
    else:
        with pytest.raises(RuntimeError, match="imageio-ffmpeg wheel"):
            fireredasr2._get_ffmpeg_executable()
    find_resources.assert_called_once_with("imageio_ffmpeg.binaries")


@pytest.fixture
def bundled_ffmpeg(monkeypatch, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", str(tmp_path / "missing-ffmpeg"))


@pytest.mark.usefixtures("bundled_ffmpeg")
@pytest.mark.parametrize(
    "variant, text, response_format, device",
    [
        ("aed", "你好世界", "json", "cpu"),
        ("llm", "Hello world", "verbose_json", "cuda:0"),
    ],
)
def test_fireredasr2_transcription(
    monkeypatch, tmp_path, variant, text, response_format, device
):
    upstream = ModuleType("xinference.thirdparty.fireredasr2")
    upstream.FireRedAsr2Config = SimpleNamespace
    recognizer = Mock()
    recognizer.transcribe.return_value = [{"text": text}]
    upstream.FireRedAsr2 = SimpleNamespace(
        from_pretrained=Mock(return_value=recognizer)
    )
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    cuda_context = Mock(side_effect=lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "device", cuda_context)

    spec = match_audio(f"FireRedASR2-{variant.upper()}", download_hub="huggingface")
    model = FireRedASR2Model("firered-test", str(tmp_path), spec, device=device)
    model.load()
    asr_type, model_path, config = upstream.FireRedAsr2.from_pretrained.call_args.args
    assert (asr_type, model_path) == (variant, str(tmp_path))
    assert config.use_gpu == device.startswith("cuda")
    assert config.use_half is False

    result = model.transcriptions(_stereo_wav(1), response_format=response_format)
    assert result == (
        {"text": text}
        if response_format == "json"
        else {
            "task": "transcribe",
            "text": text,
            "duration": 1.0,
            "words": [],
            "segments": [],
        }
    )
    recognizer.transcribe.assert_called_once()
    utterance_ids, [(sample_rate, samples)] = recognizer.transcribe.call_args.args
    assert utterance_ids == ["audio"]
    assert sample_rate == 16000
    assert samples.dtype == np.int16
    assert samples.shape == (16000,)
    assert samples[8000] > 0
    if config.use_gpu:
        assert [call.args for call in cuda_context.call_args_list] == [
            (torch.device(device),),
            (torch.device(device),),
        ]


@pytest.mark.usefixtures("bundled_ffmpeg")
@pytest.mark.parametrize("variant, limit", [("AED", 60), ("LLM", 40)])
def test_fireredasr2_duration_limit(tmp_path, variant, limit):
    spec = match_audio(f"FireRedASR2-{variant}", download_hub="huggingface")
    model = FireRedASR2Model("firered-test", str(tmp_path), spec)

    assert model._decode_audio(_stereo_wav(limit)).size == limit * 16000
    with pytest.raises(InvalidAudioInputError, match=rf"up to {limit} seconds"):
        model._decode_audio(_stereo_wav(limit + 0.125))
