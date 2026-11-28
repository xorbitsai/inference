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

import io
import shutil
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
from ...cache_manager import CacheManager
from ..core import create_audio_model_instance, match_audio
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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
@pytest.mark.parametrize("variant, limit", [("AED", 60), ("LLM", 40)])
def test_fireredasr2_duration_limit(tmp_path, variant, limit):
    spec = match_audio(f"FireRedASR2-{variant}", download_hub="huggingface")
    model = FireRedASR2Model("firered-test", str(tmp_path), spec)

    assert model._decode_audio(_stereo_wav(limit)).size == limit * 16000
    with pytest.raises(InvalidAudioInputError, match=rf"up to {limit} seconds"):
        model._decode_audio(_stereo_wav(limit + 0.125))
