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


def test_mlx_audio_tts_fish_maps_reference_and_sampling_args(monkeypatch):
    mlx_audio = ModuleType("mlx_audio")
    mlx_audio.__path__ = []
    mlx_audio_utils = ModuleType("mlx_audio.utils")
    reference_audio = np.array([0.0, 0.25], dtype=np.float32)
    reference_path = None

    def load_audio(path, *, sample_rate):
        nonlocal reference_path
        reference_path = path
        with open(path, "rb") as audio_file:
            assert audio_file.read() == b"reference"
        assert sample_rate == 44100
        return reference_audio

    mlx_audio_utils.load_audio = load_audio
    monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.utils", mlx_audio_utils)

    class FakeModel:
        kwargs = None
        sample_rate = 44100

        def generate(self, **kwargs):
            self.kwargs = kwargs
            yield SimpleNamespace(
                audio=np.array([0.0, 0.25], dtype=np.float32), sample_rate=44100
            )
            yield SimpleNamespace(
                audio=np.array([-0.25, 0.0], dtype=np.float32), sample_rate=44100
            )

    model = MLXAudioTTSModel(
        "uid",
        "/fake/path",
        _model_spec("FishAudio-S2-Pro", "FishAudio"),
    )
    model._model = FakeModel()

    result = model.speech(
        "hello",
        "",
        response_format="wav",
        speed=1.1,
        prompt_speech=b"reference",
        reference_text="reference text",
        instruct_text="Speak calmly",
        max_new_tokens=2048,
    )

    with wave.open(BytesIO(result), "rb") as wav_file:
        assert wav_file.getframerate() == 44100
        assert wav_file.getnframes() == 4
    assert model._model.kwargs["text"] == "hello"
    assert model._model.kwargs["speed"] == 1.1
    assert model._model.kwargs["ref_audio"] is reference_audio
    assert model._model.kwargs["ref_text"] == "reference text"
    assert model._model.kwargs["instruct"] == "Speak calmly"
    assert model._model.kwargs["max_tokens"] == 2048
    assert "max_new_tokens" not in model._model.kwargs
    assert reference_path is not None
    assert not os.path.exists(reference_path)


@pytest.mark.parametrize(
    "caption_key", ["caption", "instruct", "instruction", "prompt_text"]
)
@pytest.mark.parametrize("reference_key", [None, "prompt_speech", "reference_speech"])
def test_mlx_audio_irodori_voice_design_and_cloning(
    monkeypatch, caption_key, reference_key
):
    from .. import utils

    def consume_seed(kwargs):
        return kwargs.pop("seed", None)

    monkeypatch.setattr(utils, "apply_mlx_audio_seed", consume_seed)
    captured = {}

    class FakeModel:
        def generate(
            self, *, text, caption, rng_seed, seconds, num_steps, ref_audio=None
        ):
            assert text == "こんにちは"
            assert caption == "穏やかな声"
            assert rng_seed == 42
            assert seconds == 3
            assert num_steps == 8
            if reference_key:
                captured["path"] = ref_audio
                with open(ref_audio, "rb") as audio_file:
                    assert audio_file.read() == b"reference"
            else:
                assert ref_audio is None
            yield SimpleNamespace(
                audio=np.zeros(480, dtype=np.float32), sample_rate=48000
            )

    model = MLXAudioTTSModel(
        "uid", "/fake/path", _model_spec("Irodori-TTS-v4.1-Small", "Irodori-TTS")
    )
    model._model = FakeModel()
    kwargs = {caption_key: "穏やかな声", "seed": 42, "seconds": 3, "num_steps": 8}
    if reference_key:
        kwargs[reference_key] = b"reference"
    result = model.speech("こんにちは", "", response_format="wav", **kwargs)
    with wave.open(BytesIO(result), "rb") as wav_file:
        assert wav_file.getframerate() == 48000
        assert wav_file.getnframes() == 480
    if reference_key:
        assert not os.path.exists(captured["path"])


@pytest.mark.parametrize(
    "instruction_key", [None, "instruct", "instruction", "prompt_text"]
)
@pytest.mark.parametrize("reference_key", [None, "prompt_speech", "reference_speech"])
def test_mlx_breeze_arguments_and_reference_cleanup(
    monkeypatch, instruction_key, reference_key
):
    from .. import utils

    monkeypatch.setattr(
        utils, "apply_mlx_audio_seed", lambda kwargs: kwargs.pop("seed", None)
    )
    captured = {}

    class FakeModel:
        def generate(
            self, *, text, voice, instruct, cfg_scale, ref_audio, ref_text, max_tokens
        ):
            assert text == "Hello"
            assert voice == "S0"
            expected_instruction = (
                "Calm voice" if instruction_key else "Speak clearly and naturally."
            )
            if reference_key and instruction_key == "prompt_text":
                expected_instruction = "Speak clearly and naturally."
            assert instruct == expected_instruction
            assert cfg_scale == 4
            assert max_tokens == 99
            if reference_key:
                assert ref_text == "Reference transcript"
                captured["path"] = ref_audio
                with open(ref_audio, "rb") as audio_file:
                    assert audio_file.read() == b"reference"
            else:
                assert ref_audio is None and ref_text is None
            yield SimpleNamespace(
                audio=np.zeros(240, dtype=np.float32), sample_rate=24000
            )

    model = MLXAudioTTSModel(
        "uid", "/fake", _model_spec("Breeze-TTS-2", "Breeze-TTS-2")
    )
    model._model = FakeModel()
    kwargs = {"seed": 42, "guidance_scale": 4, "max_new_tokens": 99}
    if instruction_key:
        kwargs[instruction_key] = "Calm voice"
    if reference_key:
        kwargs.update(
            {reference_key: b"reference", "prompt_text": "Reference transcript"}
        )
    result = model.speech("Hello", "alloy", response_format="wav", **kwargs)
    with wave.open(BytesIO(result), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnframes() == 240
    if reference_key:
        assert not os.path.exists(captured["path"])


def test_mlx_breeze_parameter_precedence_and_validation():
    model = MLXAudioTTSModel(
        "uid", "/fake", _model_spec("Breeze-TTS-2", "Breeze-TTS-2")
    )
    kwargs = model._build_generation_kwargs(
        "Hello",
        "S1",
        1.0,
        {
            "instruct": "Calm",
            "instruction": "Ignored",
            "cfg_scale": 2,
            "guidance_scale": 4,
            "max_tokens": 100,
            "max_new_tokens": 200,
        },
        [],
    )
    assert kwargs == {
        "text": "Hello",
        "voice": "S1",
        "instruct": "Calm",
        "cfg_scale": 2.0,
        "ref_audio": None,
        "ref_text": None,
        "max_tokens": 100,
    }
    for scale in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="cfg_scale"):
            model._build_generation_kwargs("Hello", "", 1.0, {"cfg_scale": scale}, [])


def test_mlx_breeze_rejects_missing_transcript_and_cleans_temp_audio(monkeypatch):
    model = MLXAudioTTSModel(
        "uid", "/fake", _model_spec("Breeze-TTS-2", "Breeze-TTS-2")
    )
    model._model = object()
    paths = []
    save = model._save_temp_audio

    def capture(audio):
        path = save(audio)
        paths.append(path)
        return path

    monkeypatch.setattr(model, "_save_temp_audio", capture)
    with pytest.raises(ValueError, match="transcript"):
        model.speech("Hello", "", prompt_speech=b"reference")
    assert paths and all(not os.path.exists(path) for path in paths)
    with pytest.raises(RuntimeError, match="Streaming"):
        model.speech("Hello", "", stream=True)


def test_mlx_audio_tts_qwen_splits_and_joins_long_text():
    class FakeModel:
        calls = None

        def generate(self, **kwargs):
            if self.calls is None:
                self.calls = []
            self.calls.append(kwargs)
            yield SimpleNamespace(
                audio=np.full(100, 0.5, dtype=np.float32), sample_rate=1000
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

    assert [call["text"] for call in model._model.calls] == [
        "第一句话。",
        "第二句话！",
    ]
    assert all(call["split_pattern"] is None for call in model._model.calls)
    assert all(call["max_tokens"] == 2048 for call in model._model.calls)
    with wave.open(BytesIO(result), "rb") as wav_file:
        # Two 100-sample segments plus a 280 ms sentence pause.
        assert wav_file.getnframes() == 480


def test_mlx_audio_tts_qwen_voice_design_splits_independently():
    class FakeModel:
        calls = None

        def generate(self, **kwargs):
            if self.calls is None:
                self.calls = []
            self.calls.append(kwargs)
            yield SimpleNamespace(
                audio=np.full(40, 0.5, dtype=np.float32), sample_rate=1000
            )

    model = MLXAudioTTSModel(
        "uid",
        "/fake/path",
        _model_spec("Qwen3-TTS-12Hz-1.7B-VoiceDesign", "qwen3_tts"),
    )
    model._model = FakeModel()

    result = model.speech(
        "第一句话。第二句话！",
        "",
        response_format="wav",
        instruct="A calm narrator",
    )

    assert [call["text"] for call in model._model.calls] == [
        "第一句话。",
        "第二句话！",
    ]
    assert all(call["instruct"] == "A calm narrator" for call in model._model.calls)
    with wave.open(BytesIO(result), "rb") as wav_file:
        assert wav_file.getnframes() == 360


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
