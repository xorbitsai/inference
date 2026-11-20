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
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ....core.exceptions import InvalidModelRequestError
from ..zonos3_mini_sglang import SGLANG_MODEL_FILES, SGLangOmniZonos3MiniModel

SAMPLE = np.array([0.5, -0.5], dtype=np.float32)
WAVEFORM = {"audio_waveform": SAMPLE.tobytes(), "sample_rate": 44100}


@pytest.fixture
def sglang_model(monkeypatch, tmp_path):
    events = []

    class Config:
        def __init__(self, model_path):
            self.model_path = model_path
            self.stages = [SimpleNamespace(gpu=0) for _ in range(3)]

    class StreamMessage(SimpleNamespace):
        pass

    class Runner:
        def __init__(self, config):
            self.config = config
            self.coordinator = self

        async def start(self, timeout):
            events.append(("start", self.config))

        async def submit(self, request_id, request):
            events.append(("submit", request))
            return WAVEFORM

        async def stream(self, request_id, request):
            events.append(("stream", request))
            try:
                yield StreamMessage(modality="audio", chunk=WAVEFORM)
                yield SimpleNamespace(result=WAVEFORM)
            finally:
                events.append(("close", request_id))

        async def stop(self):
            events.append(("stop", self.config))

    for name, members in {
        "sglang_omni.models.zonos3.config": {"Zonos3PipelineConfig": Config},
        "sglang_omni.pipeline.mp_runner": {"MultiProcessPipelineRunner": Runner},
        "sglang_omni.proto": {
            "OmniRequest": SimpleNamespace,
            "StreamMessage": StreamMessage,
        },
        "torch": {
            "device": lambda value: SimpleNamespace(type="cuda", index=2),
            "cuda": SimpleNamespace(is_available=lambda: True),
            "version": SimpleNamespace(cuda="13.0"),
        },
    }.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    # Exercise the launch path while keeping test coverage on every CI Python.
    monkeypatch.setattr(
        "xinference.model.audio.zonos3_mini_sglang.sys",
        SimpleNamespace(platform="linux", version_info=(3, 12)),
    )

    for name in SGLANG_MODEL_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    voice_codes = tmp_path / "gguf/voices/English/Savannah_en-American_F_1.codes.npy"
    voice_codes.parent.mkdir(parents=True)
    np.save(voice_codes, np.arange(10).reshape(10, 1))
    model = SGLangOmniZonos3MiniModel("uid", str(tmp_path), SimpleNamespace())
    try:
        yield model, events
    finally:
        model.stop()


def test_sglang_uses_local_checkpoint_and_preset_codes(sglang_model, tmp_path):
    model, events = sglang_model
    model.load()
    config = events[0][1]
    assert config.model_path == str(tmp_path)
    assert [stage.gpu for stage in config.stages] == [2, 2, 2]
    assert model.speech("hello", "alloy", "pcm", seed=42) == b"\xff?\x01\xc0"
    request = events[1][1]
    assert request.params == {"seed": 42, "stream": False}
    assert request.inputs == {
        "text": "hello",
        "references": [{"vq_codes": [[code] for code in range(10)]}],
    }
    model.stop()
    assert events[-1][0] == "stop"
    assert model._loop is None


def test_sglang_stream_forwards_reference_and_closes_runner_stream(sglang_model):
    import soundfile

    model, events = sglang_model
    model.load()
    sample = io.BytesIO()
    soundfile.write(sample, np.zeros(44100), 44100, format="wav")
    stream = model.speech(
        "hello", None, "pcm", stream=True, prompt_speech=sample.getvalue()
    )
    assert next(stream) == b"\xff?\x01\xc0"
    request = events[-1][1]
    reference = Path(request.inputs["references"][0]["audio_path"])
    assert reference.is_file()
    assert request.params == {"stream": True}
    stream.close()
    assert events[-1][0] == "close"
    assert not reference.exists()

    assert list(model.speech("hello", None, "pcm", stream=True)) == [b"\xff?\x01\xc0"]


def test_sglang_rejects_incomplete_local_checkpoint(sglang_model, tmp_path):
    model, events = sglang_model
    (tmp_path / "autoencoder/teacher/config.json").unlink()
    with pytest.raises(ValueError, match="autoencoder/teacher/config.json"):
        model.load()
    assert not events


def test_sglang_failed_start_releases_runtime_thread(sglang_model, monkeypatch):
    model, _ = sglang_model

    async def fail(self, timeout):
        raise RuntimeError("Pipeline failed to start.")

    runner = sys.modules["sglang_omni.pipeline.mp_runner"].MultiProcessPipelineRunner
    monkeypatch.setattr(runner, "start", fail)
    with pytest.raises(RuntimeError, match="Pipeline failed to start"):
        model.load()
    assert model._loop is None
    assert model._thread is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"language": "en"}, "explicit language"),
        ({"top_p": 0}, "top_p must be positive"),
        ({"max_new_tokens": 8192}, "must be below 8192"),
    ],
)
def test_sglang_invalid_options_rejected_before_stream(sglang_model, kwargs, message):
    model, events = sglang_model
    with pytest.raises(InvalidModelRequestError, match=message):
        model.speech("hello", None, "pcm", stream=True, **kwargs)
    assert not events
