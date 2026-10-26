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


@pytest.fixture
def installed_vad_owner(monkeypatch):
    try:
        sdk = import_module("funasr.models.fsmn_vad_streaming.model")
        frontend = import_module("funasr.frontends.wav_frontend")
        auto = import_module("funasr.auto.auto_model")
        import_module("funasr.models.fsmn_vad_streaming.dynamic_vad")
    except ImportError:
        pytest.skip("Optional FunASR streaming VAD SDK dependencies are missing")

    from ..funasr import FunASRModel

    def compute_scores(self, feats, cache):
        # Substitute only neural scores, deterministically from the real
        # frontend's score-aligned waveform. Keep real decibels and detection.
        stats = cache["stats"]
        frames = stats.waveform.reshape(-1).unfold(0, 400, 160)
        speech = frames.square().sum(dim=-1) > 1e-6
        noise = torch.where(speech, 0.01, 0.99)
        scores = torch.stack((noise, 1 - noise), dim=-1).unsqueeze(0)
        assert scores.shape[1] == feats.shape[1]
        self.vad_opts.nn_eval_block_size = scores.shape[1]
        stats.frm_cnt += scores.shape[1]
        stats.scores = (
            scores if stats.scores is None else torch.cat((stats.scores, scores), dim=1)
        )

    monkeypatch.setattr(sdk.FsmnVADStreaming, "ComputeScores", compute_scores)

    vad = sdk.FsmnVADStreaming.__new__(sdk.FsmnVADStreaming)
    torch.nn.Module.__init__(vad)
    vad.vad_opts = sdk.VADXOptions()
    vad.register_parameter("cpu_marker", torch.nn.Parameter(torch.zeros(1)))
    model = automodel_without_weights(auto.AutoModel, torch.nn.Identity())
    model.vad_model = vad
    model.vad_kwargs = {
        "device": "cpu",
        "disable_pbar": True,
        "frontend": frontend.WavFrontendOnline(
            frame_length=25, frame_shift=10, lfr_m=5, lfr_n=1, dither=0.0
        ),
    }
    waveforms = []
    inference = model.inference

    def decode(input, **kwargs):
        if "model" in kwargs:
            return inference(input, **kwargs)
        waveforms.append(input.clone())
        return [{"text": "speech"}]

    model.inference = decode
    owner = FunASRModel(
        "nano", "/unused", SimpleNamespace(default_transcription_config={})
    )
    owner._model = model
    owner._realtime_ready = True
    return SimpleNamespace(owner=owner, waveforms=waveforms)


@pytest.mark.parametrize("samples", [0, 1, 123, 399, 400, 960, 15360, 16000, 16123])
async def test_installed_vad_finalization_over_websocket(installed_vad_owner, samples):
    import aiohttp

    from ....api.tests.test_realtime_api import serve_gateway

    pcm = np.full(samples, 1000, dtype="<i2")
    config = {
        "event": "session.start",
        "protocol_version": 1,
        "task": "audio.transcription",
        "input": {"audio": PCM_FORMAT},
    }
    async with serve_gateway(backend_model=installed_vad_owner.owner) as service:
        async with aiohttp.ClientSession() as client:
            async with client.ws_connect(service.url) as ws:
                await ws.send_json(config)
                assert (await ws.receive_json(timeout=5))["event"] == "session.started"
                raw = pcm.tobytes()
                for start in range(0, len(raw), 3199):
                    await ws.send_bytes(raw[start : start + 3199])
                await ws.send_json({"event": "input.end"})
                results = []
                while True:
                    event = await ws.receive_json(timeout=5)
                    assert event["event"] != "error", event
                    if event["event"] == "session.closed":
                        assert event["reason"] == "completed"
                        break
                    results.append(event["data"])
    assert sum(result["is_final"] for result in results) == 1
    final = results[-1]
    assert final["duration_ms"] == samples * 1000 // 16000
    if samples >= 15360:
        assert final["text"] == "speech"
        expected = torch.from_numpy(pcm.astype(np.float32) / 32768.0)
        torch.testing.assert_close(installed_vad_owner.waveforms[-1], expected)
    else:
        # A sub-frame utterance may not meet the real VAD onset threshold. If
        # recognized, only accepted PCM, never synthetic padding, reaches ASR.
        assert all(
            len(waveform) <= samples for waveform in installed_vad_owner.waveforms
        )
    if not samples:
        assert final["text"] == ""
        assert not installed_vad_owner.waveforms


@pytest.mark.parametrize(
    "silence_samples,flush_start", [(31040, 46080), (32000, 47040)]
)
async def test_installed_vad_silence_boundary_not_extended(
    installed_vad_owner, silence_samples, flush_start
):
    from ....core.realtime.runtime import RealtimeRuntime

    runtime = RealtimeRuntime(installed_vad_owner.owner, "nano", "nano-0")
    session = NanoRealtimeSession(
        installed_vad_owner.owner,
        {"task": "audio.transcription", "input": {"audio": PCM_FORMAT}},
        runtime,
    )
    try:
        # Cover a natural endpoint returned during the final call as well as
        # one already committed before that call. Neither may be lengthened.
        pcm = np.concatenate(
            (np.full(16000, 1000, dtype="<i2"), np.zeros(silence_samples, dtype="<i2"))
        )
        for start in range(0, flush_start, 960):
            session._feed(pcm[start : start + 960].tobytes())
        assert bool(session._sentences) == (flush_start == 47040)
        final = session._feed(pcm[flush_start:].tobytes(), final=True)
        assert len(final["sentences"]) == 1
        assert final["sentences"][0]["end"] == 1160
        assert final["duration_ms"] == len(pcm) * 1000 // 16000
        expected = torch.from_numpy(pcm[:18560].astype(np.float32) / 32768.0)
        torch.testing.assert_close(installed_vad_owner.waveforms[-1], expected)
    finally:
        await session.close("test")
        await runtime.stop()


async def test_installed_vad_forced_cut_preserves_tail(installed_vad_owner):
    from ....core.realtime.runtime import RealtimeRuntime

    runtime = RealtimeRuntime(installed_vad_owner.owner, "nano", "nano-0")
    session = NanoRealtimeSession(
        installed_vad_owner.owner,
        {"task": "audio.transcription", "input": {"audio": PCM_FORMAT}},
        runtime,
    )
    session.max_segment_samples = 3840
    session.overlap_samples = 640
    pcm = np.full(16123, 1000, dtype="<i2")
    try:
        await session.receive_input(pcm.tobytes())
        await session.finish_input()
        final = [result async for result in session.iter_outputs()][-1]
        assert len(final["sentences"]) == 5
        assert final["sentences"][-1]["end"] == len(pcm) * 1000 // 16000
        expected = torch.from_numpy(pcm[15360 - 640 :].astype(np.float32) / 32768.0)
        torch.testing.assert_close(installed_vad_owner.waveforms[-1], expected)
        assert all(len(waveform) <= 4480 for waveform in installed_vad_owner.waveforms)
    finally:
        await session.close("test")
        await runtime.stop()
