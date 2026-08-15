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

import os
import sys
import wave
from io import BytesIO
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ..mlx_audio import MLXAudioSTTModel, MLXAudioTTSModel


def _model_spec(model_name, model_family, default_transcription_config=None):
    return SimpleNamespace(
        model_name=model_name,
        model_family=model_family,
        model_ability=(
            ["audio2text"]
            if model_family in ("qwen3_asr", "funasr")
            else ["text2audio"]
        ),
        default_transcription_config=default_transcription_config,
    )


def _mock_mlx_audio_decoder(monkeypatch, expected_sample_rate=16000):
    mlx_audio = ModuleType("mlx_audio")
    mlx_audio.__path__ = []
    audio_io = ModuleType("mlx_audio.audio_io")

    def read(file, **kwargs):
        assert isinstance(file, BytesIO)
        assert file.read() == b"audio"
        assert kwargs == {
            "dtype": "float32",
            "sample_rate": expected_sample_rate,
            "nchannels": 1,
        }
        return np.array([0.0, 0.25], dtype=np.float32), expected_sample_rate

    audio_io.read = read
    monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.audio_io", audio_io)


def test_mlx_audio_stt_maps_qwen_arguments_and_verbose_result(monkeypatch):
    _mock_mlx_audio_decoder(monkeypatch)

    class FakeModel:
        kwargs = None
        sample_rate = 16000

        def generate(self, audio, **kwargs):
            np.testing.assert_array_equal(
                audio, np.array([0.0, 0.25], dtype=np.float32)
            )
            self.kwargs = kwargs
            return SimpleNamespace(
                text="hello",
                language="English",
                segments=[{"text": "hello", "start": 0.0, "end": 0.5}],
            )

    model = MLXAudioSTTModel(
        "uid",
        "/fake/path",
        _model_spec("Qwen3-ASR-0.6B", "qwen3_asr"),
    )
    model._model = FakeModel()

    result = model.transcriptions(
        b"audio",
        language="en",
        prompt="Xinference",
        response_format="verbose_json",
        temperature=0.2,
        top_p=0.9,
    )

    assert result == {
        "task": "transcribe",
        "language": "English",
        "text": "hello",
        "segments": [{"text": "hello", "start": 0.0, "end": 0.5}],
    }
    assert model._model.kwargs == {
        "language": "English",
        "temperature": 0.2,
        "system_prompt": "Xinference",
        "top_p": 0.9,
    }


def test_mlx_audio_stt_maps_fun_asr_prompt_to_context(monkeypatch):
    _mock_mlx_audio_decoder(monkeypatch, expected_sample_rate=8000)

    class FakeModel:
        kwargs = None
        config = SimpleNamespace(frontend_conf=SimpleNamespace(fs=8000))

        def generate(self, audio, **kwargs):
            np.testing.assert_array_equal(
                audio, np.array([0.0, 0.25], dtype=np.float32)
            )
            self.kwargs = kwargs
            return {"text": "你好"}

    model = MLXAudioSTTModel(
        "uid",
        "/fake/path",
        _model_spec("Fun-ASR-Nano-2512", "funasr"),
    )
    model._model = FakeModel()

    assert model.transcriptions(b"audio", prompt="稀有词") == {"text": "你好"}
    assert model._model.kwargs["context"] == "稀有词"


def test_mlx_audio_stt_falls_back_to_path_for_mpeg2_mp3(monkeypatch):
    mlx_audio = ModuleType("mlx_audio")
    mlx_audio.__path__ = []
    audio_io = ModuleType("mlx_audio.audio_io")
    temp_path = None

    def read(file, **kwargs):
        nonlocal temp_path
        if isinstance(file, BytesIO):
            raise ValueError("Unable to detect audio format from bytes")

        temp_path = file
        with open(file, "rb") as audio_file:
            assert audio_file.read() == b"\xff\xf3audio"
        assert kwargs == {
            "dtype": "float32",
            "sample_rate": 16000,
            "nchannels": 1,
        }
        return np.array([0.0, 0.25], dtype=np.float32), 16000

    audio_io.read = read
    monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.audio_io", audio_io)

    model = MLXAudioSTTModel(
        "uid",
        "/fake/path",
        _model_spec("Qwen3-ASR-0.6B", "qwen3_asr"),
    )
    model._model = SimpleNamespace(sample_rate=16000)

    np.testing.assert_array_equal(
        model._decode_audio(b"\xff\xf3audio"),
        np.array([0.0, 0.25], dtype=np.float32),
    )
    assert temp_path is not None
    assert not os.path.exists(temp_path)


def test_mlx_audio_tts_qwen_voice_clone_and_wav_output():
    class FakeModel:
        kwargs = None

        def generate(self, **kwargs):
            assert os.path.exists(kwargs["ref_audio"])
            self.kwargs = kwargs
            yield SimpleNamespace(
                audio=np.array([0.0, 0.25], dtype=np.float32), sample_rate=24000
            )
            yield SimpleNamespace(
                audio=np.array([-0.25, 0.0], dtype=np.float32), sample_rate=24000
            )

    model = MLXAudioTTSModel(
        "uid",
        "/fake/path",
        _model_spec("Qwen3-TTS-12Hz-0.6B-Base", "qwen3_tts"),
    )
    model._model = FakeModel()

    result = model.speech(
        "hello",
        "Chelsie",
        response_format="wav",
        prompt_speech=b"reference",
        prompt_text="reference text",
        language="English",
    )

    with wave.open(BytesIO(result), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnframes() == 4
    assert model._model.kwargs["text"] == "hello"
    assert model._model.kwargs["voice"] == "Chelsie"
    assert model._model.kwargs["lang_code"] == "English"
    assert model._model.kwargs["max_tokens"] == 4096
    assert model._model.kwargs["ref_text"] == "reference text"
    assert not os.path.exists(model._model.kwargs["ref_audio"])


def test_mlx_audio_tts_qwen_requires_reference_text():
    model = MLXAudioTTSModel(
        "uid",
        "/fake/path",
        _model_spec("Qwen3-TTS-12Hz-0.6B-Base", "qwen3_tts"),
    )
    model._model = object()

    with pytest.raises(ValueError, match="prompt_text is required"):
        model.speech("hello", "", prompt_speech=b"reference")


def test_mlx_audio_tts_qwen_splits_and_joins_long_text():
    class FakeModel:
        kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            segments = kwargs["text"].split(kwargs["split_pattern"])
            for index, _segment in enumerate(segments):
                yield SimpleNamespace(
                    audio=np.full(100, 0.5, dtype=np.float32),
                    sample_rate=1000,
                    segment_idx=index,
                )

    model = MLXAudioTTSModel(
        "uid",
        "/fake/path",
        _model_spec("Qwen3-TTS-12Hz-0.6B-CustomVoice", "qwen3_tts"),
    )
    model._model = FakeModel()

    result = model.speech(
        "第一句话。第二句话！",
        "Chelsie",
        response_format="wav",
        max_tokens=2048,
    )

    assert model._model.kwargs["text"] == "第一句话。\n第二句话！"
    assert model._model.kwargs["split_pattern"] == "\n"
    assert model._model.kwargs["max_tokens"] == 2048
    with wave.open(BytesIO(result), "rb") as wav_file:
        # Two 100-sample segments plus a 280 ms sentence pause.
        assert wav_file.getnframes() == 480


def test_mlx_audio_tts_qwen_splits_overlong_sentence_at_clause():
    text = "甲" * 50 + "，" + "乙" * 50

    assert MLXAudioTTSModel._split_qwen_text(text) == ["甲" * 50 + "，", "乙" * 50]


def test_mlx_audio_tts_voxcpm_maps_reference_and_instruction():
    model = MLXAudioTTSModel("uid", "/fake/path", _model_spec("VoxCPM2", "VoxCPM"))
    temp_files = []
    try:
        kwargs = model._build_generation_kwargs(
            "hello",
            "A calm narrator",
            1.0,
            {"prompt_speech": b"reference", "prompt_text": "reference text"},
            temp_files,
        )
        assert kwargs["ref_audio"] == kwargs["prompt_audio"]
        assert kwargs["prompt_text"] == "reference text"
        assert kwargs["instruct"] == "A calm narrator"
    finally:
        for temp_file in temp_files:
            os.unlink(temp_file)
