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

"""Optional installed-SDK input checks; no weights, downloads or GPU decoding."""

from importlib import import_module
from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ....core.realtime.protocol import RealtimeLimits
from ..funasr_realtime import PCM_FORMAT, NanoRealtimeSession


@pytest.fixture
def installed_sdk():
    try:
        nano = import_module("funasr.models.fun_asr_nano.model")
        auto = import_module("funasr.auto.auto_model")
    except ImportError:
        pytest.skip("Optional FunASR Nano SDK or its import dependencies are missing")
    return SimpleNamespace(nano=nano.FunASRNano, auto=auto.AutoModel, module=nano)


def automodel_without_weights(auto_class, model, **defaults):
    # Bypass only construction/downloads. Keep real inference, configuration
    # reset/merge and batching, including the installed SDK's CPU cleanup path.
    auto = auto_class.__new__(auto_class)
    baseline = {
        "device": "cpu",
        "disable_pbar": True,
        "ncpu": torch.get_num_threads(),
        **defaults,
    }
    auto.kwargs = dict(baseline)
    auto._base_kwargs_map = {"kwargs": dict(baseline)}
    auto.model = model
    return auto


def test_installed_nano_sdk_loads_adapter_tensor_through_automodel(
    installed_sdk, monkeypatch
):
    features = []
    effective_kwargs = []

    class Tokenizer:
        def encode(self, text):
            return [1, 2, 3]

    class Frontend:
        fs = 16000
        frame_shift = 10
        lfr_n = 1

        def __call__(self, audio, lengths, **kwargs):
            assert audio.shape == (1, 16000)
            assert audio.dtype == torch.float32
            assert audio.device.type == "cpu"
            torch.testing.assert_close(audio, torch.full_like(audio, 0.25))
            features.append(audio.clone())
            return torch.zeros((1, 10, 80)), torch.tensor([10])

    class LoaderOnlyNano(installed_sdk.nano):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.use_low_frame_rate = False
            self.register_parameter("cpu_marker", torch.nn.Parameter(torch.zeros(1)))

        def inference_llm(
            self,
            data_in,
            data_lengths=None,
            key=None,
            tokenizer=None,
            frontend=None,
            **kwargs,
        ):
            # Real Nano inference has already built the prompt/ChatML. Exercise
            # real template/loading methods, stopping before encoder/LLM work.
            effective_kwargs.append(dict(kwargs))
            assert len(data_in) == 1
            assert data_in[0][1]["audio"].shape == (16000,)
            contents = self.data_template(data_in[0])
            metadata = {}
            output = self.data_load_speech(
                contents, tokenizer, frontend, meta_data=metadata, **kwargs
            )
            assert output["speech"].shape == (1, 10, 80)
            assert output["speech_lengths"].tolist() == [[10]]
            return [{"text": "sdk input loaded"}], metadata

    frontend = Frontend()
    # Protocol input and target frontend are 16 kHz; do not inject audio_fs.
    loader = installed_sdk.module.load_audio_text_image_video
    assert signature(loader).parameters["audio_fs"].default == frontend.fs
    auto = automodel_without_weights(
        installed_sdk.auto, LoaderOnlyNano(), tokenizer=Tokenizer(), frontend=frontend
    )
    # A stale runtime value must be reset by the real AutoModel implementation.
    # This is distinct from fs in the persistent model baseline (tested below).
    auto.kwargs["fs"] = 8000
    defaults = {"fs": 16000}
    # Construct the real session so new runtime/options/cancellation state
    # cannot silently diverge from this installed-SDK contract test.
    dynamic = import_module("funasr.models.fsmn_vad_streaming.dynamic_vad")
    monkeypatch.setattr(dynamic, "DynamicStreamingVAD", lambda model: None)
    auto.vad_model = SimpleNamespace(vad_opts={})
    auto.vad_kwargs = {}
    owner = SimpleNamespace(
        _model=auto,
        _model_spec=SimpleNamespace(default_transcription_config=defaults),
    )
    config = {
        "task": "audio.transcription",
        "input": {"audio": PCM_FORMAT},
        "parameters": {"language": "zh", "hotwords": ["测试"]},
    }
    runtime = SimpleNamespace(
        limits=RealtimeLimits(), generation="sdk", record_event=lambda *a, **kw: None
    )
    session = NanoRealtimeSession(owner, config, runtime)
    session._audio = np.full(16000, 0.25, dtype=np.float32)
    for _ in range(2):
        assert session._decode(0, 16000) == "sdk input loaded"
    assert len(features) == len(effective_kwargs) == 2
    for kwargs in effective_kwargs:
        assert "fs" not in kwargs
        assert "audio_fs" not in kwargs
        assert kwargs["cache"] == {}
        assert kwargs["batch_size"] == 1
        assert kwargs["language"] == "中文"
        assert kwargs["hotwords"] == ["测试"]
    assert effective_kwargs[0]["cache"] is not effective_kwargs[1]["cache"]
    assert "fs" not in auto._base_kwargs_map["kwargs"]
    assert defaults == {"fs": 16000}


def test_installed_automodel_preserves_model_level_fs(installed_sdk):
    calls = []

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("cpu_marker", torch.nn.Parameter(torch.zeros(1)))

        def inference(self, data_in, **kwargs):
            calls.append(kwargs)
            return [{"text": "config observed"}], {"batch_data_time": 1}

    auto = automodel_without_weights(installed_sdk.auto, RecordingModel(), fs=16000)
    auto.inference(torch.zeros(123), cache={}, batch_size=1)
    # Request-local filtering does not remove a persistent model-level fs.
    # This test stops at recording: don't require a specific downstream SDK bug
    # or mutate shared configs to hide it. Check deployed model configs separately.
    assert calls[0]["fs"] == 16000
    assert auto._base_kwargs_map["kwargs"]["fs"] == 16000
