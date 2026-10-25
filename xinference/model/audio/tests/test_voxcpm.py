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
import wave
from io import BytesIO

import numpy as np

from xinference.model.audio.voxcpm import VoxCPMModel


class _Spec:
    model_ability = ["text2audio", "text2audio_zero_shot", "text2audio_voice_cloning"]


class _FakeTTS:
    sample_rate = 48000


class _FakeModel:
    def __init__(self):
        self.tts_model = _FakeTTS()
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return np.zeros(32, dtype="float32")

    def generate_streaming(self, **kwargs):
        self.kwargs = kwargs
        yield np.zeros(16, dtype="float32")


class _FakeMultiChunkModel(_FakeModel):
    def generate_streaming(self, **kwargs):
        self.kwargs = kwargs
        for _ in range(2):
            yield np.zeros(16, dtype="float32")


def test_voxcpm_maps_voice_to_control_instruction():
    model = VoxCPMModel("uid", "/tmp/model", _Spec())
    model._model = _FakeModel()

    out = model.speech("hello", voice="warm narrator", response_format="pcm")

    assert len(out) == 64
    assert model._model.kwargs["text"] == "(warm narrator)hello"
    assert model._model.kwargs["prompt_wav_path"] is None
    assert model._model.kwargs["reference_wav_path"] is None


def test_voxcpm_prompt_speech_uses_temp_reference_file():
    model = VoxCPMModel("uid", "/tmp/model", _Spec())
    model._model = _FakeModel()

    out = model.speech(
        "hello",
        voice="alloy",
        response_format="wav",
        prompt_speech=b"fake wav bytes",
    )

    temp_path = model._model.kwargs["reference_wav_path"]

    with wave.open(BytesIO(out), "rb") as wav_file:
        assert wav_file.getframerate() == 48000
        assert wav_file.getnchannels() == 1
        assert wav_file.getnframes() == 32
    assert temp_path is not None
    assert not os.path.exists(temp_path)


def test_voxcpm_pcm_streaming_cleans_up_temp_files():
    model = VoxCPMModel("uid", "/tmp/model", _Spec())
    model._model = _FakeModel()

    stream = model.speech(
        "hello",
        voice="alloy",
        response_format="pcm",
        stream=True,
        prompt_speech=b"fake wav bytes",
        prompt_text="reference text",
    )

    chunks = list(stream)
    temp_path = model._model.kwargs["reference_wav_path"]

    assert len(chunks) == 1
    assert len(chunks[0]) == 32
    assert model._model.kwargs["prompt_wav_path"] == temp_path
    assert temp_path is not None
    assert not os.path.exists(temp_path)


def test_voxcpm_pcm_streaming_close_cleans_up_unconsumed_temp_files(
    monkeypatch, tmp_path
):
    temp_path = tmp_path / "prompt.wav"
    monkeypatch.setattr(
        VoxCPMModel,
        "_save_temp_audio",
        staticmethod(lambda audio: temp_path.write_bytes(audio) and str(temp_path)),
    )

    model = VoxCPMModel("uid", "/tmp/model", _Spec())
    model._model = _FakeModel()

    stream = model.speech(
        "hello",
        voice="alloy",
        response_format="pcm",
        stream=True,
        prompt_speech=b"fake wav bytes",
        prompt_text="reference text",
    )

    assert os.path.exists(temp_path)

    stream.close()

    assert not os.path.exists(temp_path)


def test_voxcpm_pcm_streaming_close_cleans_up_partially_consumed_temp_files(
    monkeypatch, tmp_path
):
    temp_path = tmp_path / "prompt.wav"
    monkeypatch.setattr(
        VoxCPMModel,
        "_save_temp_audio",
        staticmethod(lambda audio: temp_path.write_bytes(audio) and str(temp_path)),
    )

    model = VoxCPMModel("uid", "/tmp/model", _Spec())
    model._model = _FakeMultiChunkModel()

    stream = model.speech(
        "hello",
        voice="alloy",
        response_format="pcm",
        stream=True,
        prompt_speech=b"fake wav bytes",
        prompt_text="reference text",
    )
    first_chunk = next(stream)

    assert len(first_chunk) == 32
    assert model._model.kwargs["reference_wav_path"] == str(temp_path)
    assert os.path.exists(temp_path)

    stream.close()

    assert not os.path.exists(temp_path)
