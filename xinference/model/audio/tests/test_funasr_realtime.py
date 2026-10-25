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


import asyncio
import copy
import sys
from importlib import import_module
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import pytest_asyncio
import torch

from ....core.realtime.protocol import RealtimeError
from ....core.realtime.runtime import RealtimeRuntime
from ..funasr import FunASRModel
from ..funasr_realtime import PCM_FORMAT, NanoRealtimeSession, validate_config

CONFIG = {
    "event": "session.start",
    "protocol_version": 1,
    "task": "audio.transcription",
    "input": {"audio": PCM_FORMAT},
    "parameters": {"language": "zh", "hotwords": ["测试"]},
}


class FakeDynamicVAD:
    def __init__(self, model):
        self.model = model
        self.current_speech_start = None
        self.total = 0

    def feed(self, audio, is_final=False):
        start = self.total
        self.total += len(audio)
        speech = bool((audio.abs() > 0.001).any())
        if speech and self.current_speech_start is None:
            self.current_speech_start = start // 16
        if (is_final or not speech) and self.current_speech_start is not None:
            segment = [self.current_speech_start, self.total // 16]
            self.current_speech_start = None
            return [segment]
        return []

    def reset(self):
        self.current_speech_start = None


@pytest_asyncio.fixture
async def nano(monkeypatch):
    dynamic = ModuleType("funasr.models.fsmn_vad_streaming.dynamic_vad")
    dynamic.DynamicStreamingVAD = FakeDynamicVAD
    postprocess = ModuleType("funasr.utils.postprocess_utils")
    postprocess.rich_transcription_postprocess = lambda text: text
    monkeypatch.setitem(sys.modules, dynamic.__name__, dynamic)
    monkeypatch.setitem(sys.modules, postprocess.__name__, postprocess)
    owner = FunASRModel(
        "nano",
        "/unused",
        SimpleNamespace(
            default_transcription_config={}, model_name="Fun-ASR-Nano-2512"
        ),
    )
    calls = []
    waveforms = []

    def inference(input, **kwargs):
        # Nano's real input builder accepts paths/Tensors, not NumPy arrays.
        assert isinstance(input, torch.Tensor)
        assert input.dtype == torch.float32
        assert input.ndim == 1
        assert input.device.type == "cpu"
        assert input.is_contiguous()
        assert not input.requires_grad
        assert "fs" not in kwargs
        calls.append((len(input), kwargs))
        waveforms.append(input.clone())
        return [{"text": f"识别{len(input)}"}]

    owner._model = SimpleNamespace(
        inference=inference, vad_model=SimpleNamespace(vad_opts={}), vad_kwargs={}
    )
    owner._realtime_ready = True
    runtime = RealtimeRuntime(owner, "nano", "nano-0")
    session = NanoRealtimeSession(owner, CONFIG, runtime)
    try:
        yield SimpleNamespace(
            owner=owner,
            runtime=runtime,
            session=session,
            calls=calls,
            waveforms=waveforms,
        )
    finally:
        await session.close("test")
        await runtime.stop()


def speech(samples):
    return np.full(samples, 1000, dtype="<i2").tobytes()


async def collect(session):
    return [data async for data in session.iter_outputs()]


async def test_irregular_frames_partial_and_unique_final(nano):
    session = nano.session
    outputs = asyncio.create_task(collect(session))
    pcm = speech(16000 + 123)
    for start in range(0, len(pcm), 3199):
        await session.receive_input(pcm[start : start + 3199])
    assert session._total >= 15360
    assert nano.calls  # Partial inference happened before input.end.
    await session.finish_input()
    data = await outputs
    assert sum(item["is_final"] for item in data) == 1
    assert any(item["partial"] for item in data[:-1])
    assert data[-1]["duration_ms"] == 16123 * 1000 // 16000
    assert "16123" in data[-1]["text"]  # Tail included in final decoding.
    assert nano.calls[-1][1]["hotwords"] == ["测试"]
    assert nano.calls[-1][1]["language"] == "中文"


async def test_silence_and_empty_input(nano):
    await nano.session.receive_input(bytes(3200))
    await nano.session.finish_input()
    data = await collect(nano.session)
    assert data[-1]["text"] == ""
    assert not nano.calls


async def test_short_tail(nano):
    await nano.session.receive_input(speech(123))
    await nano.session.finish_input()
    data = await collect(nano.session)
    assert data[-1]["text"] == "识别123"
    assert nano.calls[-1][0] == 123


async def test_vad_commit_uses_asr_parameter_contract(nano):
    await nano.session.receive_input(speech(960))
    await nano.session.receive_input(bytes(960 * 2))
    assert nano.calls  # Decoded at VAD end, before input.end.
    assert len(nano.session._sentences) == 1
    await nano.session.finish_input()
    data = await collect(nano.session)
    assert sum(item["is_final"] for item in data) == 1
    assert len(data[-1]["sentences"]) == 1
    assert all("fs" not in kwargs for _, kwargs in nano.calls)


@pytest.mark.parametrize("samples", [1, 123, 961, 16123])
async def test_asr_tensor_preserves_pcm_values_and_tail(nano, samples):
    pcm = np.resize(
        np.array([-32768, -12345, -1, 0, 1, 12345, 32767], dtype="<i2"),
        samples,
    )
    outputs = asyncio.create_task(collect(nano.session))
    try:
        raw = pcm.tobytes()
        for start in range(0, len(raw), 3199):
            await nano.session.receive_input(raw[start : start + 3199])
        await nano.session.finish_input()
        data = await outputs
        expected = torch.from_numpy(pcm.astype(np.float32) / 32768.0)
        torch.testing.assert_close(nano.waveforms[-1], expected, rtol=0, atol=0)
        assert data[-1]["duration_ms"] == samples * 1000 // 16000
        assert sum(item["is_final"] for item in data) == 1
        if samples > nano.session.partial_samples:
            assert any(item["partial"] for item in data[:-1])
            assert len(nano.waveforms) >= 2
    finally:
        if not outputs.done():
            outputs.cancel()
        await asyncio.gather(outputs, return_exceptions=True)


async def test_asr_tensor_does_not_share_session_audio(nano, monkeypatch):
    original_inference = nano.owner._model.inference

    def mutate_input(input, **kwargs):
        result = original_inference(input, **kwargs)
        input.zero_()
        return result

    monkeypatch.setattr(nano.owner._model, "inference", mutate_input)
    outputs = asyncio.create_task(collect(nano.session))
    try:
        await nano.session.receive_input(speech(16000))
        expected = np.full(nano.session._total, 1000 / 32768.0, dtype=np.float32)
        np.testing.assert_array_equal(nano.session._audio, expected)
        await nano.session.finish_input()
        await outputs
        torch.testing.assert_close(
            nano.waveforms[-1], torch.full((16000,), 1000 / 32768.0), rtol=0, atol=0
        )
        assert len(nano.waveforms) >= 2  # Both Partial and Final were decoded.
    finally:
        if not outputs.done():
            outputs.cancel()
        await asyncio.gather(outputs, return_exceptions=True)


async def test_empty_decode_does_not_call_asr(nano):
    assert nano.session._decode(0, 0) == ""
    assert not nano.calls


async def test_asr_kwargs_accept_explicit_downstream_sample_rate(nano, monkeypatch):
    original_inference = nano.owner._model.inference
    loaded = []

    def loader(audio, fs=16000, **kwargs):
        loaded.append(fs)
        return original_inference(audio, **kwargs)

    def inference(input, **kwargs):
        # Match Nano's explicit frontend.fs plus forwarded kwargs. Do not filter
        # here: the old adapter must fail at this duplicate-keyword boundary.
        return loader(input, fs=16000, **kwargs)

    monkeypatch.setattr(nano.owner._model, "inference", inference)
    nano.session._audio = np.full(123, 0.1, dtype=np.float32)
    assert nano.session._decode(0, 123) == "识别123"
    assert loaded == [16000]


@pytest.mark.parametrize("default_fs", [None, 8000, 16000])
async def test_asr_default_kwargs_filtered_without_shared_mutation(nano, default_fs):
    defaults = {
        "cache": {"previous_request": True},
        "batch_size": 8,
        "language": "English",
        "hotwords": ["default"],
        "beam_size": 2,
    }
    if default_fs is not None:
        defaults["fs"] = default_fs
    original = copy.deepcopy(defaults)
    nano.owner._model_spec.default_transcription_config = defaults
    nano.session._audio = np.full(123, 0.1, dtype=np.float32)
    nano.session._decode(0, 123)
    first = nano.calls[-1][1]
    first["cache"]["decoder_state"] = True
    nano.session._decode(0, 123)
    second = nano.calls[-1][1]
    assert "fs" not in first and "fs" not in second
    assert second["cache"] == {}
    assert first["cache"] is not second["cache"]
    assert first["cache"] is not defaults["cache"]
    assert second["batch_size"] == 1
    assert second["language"] == "中文"
    assert second["hotwords"] == ["测试"]
    assert second["beam_size"] == 2
    assert defaults == original


async def test_asr_fs_filter_does_not_change_vad_overrides(nano, monkeypatch):
    from ..funasr_realtime import _ExistingVAD

    defaults = {"fs": 16000, "cache": {"vad_state": []}}
    nano.owner._model.vad_kwargs = defaults
    vad = _ExistingVAD(nano.owner._model)
    original_inference = nano.owner._model.inference
    calls = []

    def inference(input, **kwargs):
        if "model" in kwargs:
            calls.append(kwargs)
            return [{"value": []}]
        return original_inference(input, **kwargs)

    monkeypatch.setattr(nano.owner._model, "inference", inference)
    assert vad.generate(torch.zeros(960), fs=8000, is_final=True) == [{"value": []}]
    assert calls[0]["model"] is vad.model
    assert calls[0]["kwargs"] is vad.kwargs
    assert calls[0]["fs"] == 8000
    assert calls[0]["is_final"] is True
    assert vad.kwargs == defaults
    assert vad.kwargs["cache"] is not defaults["cache"]
    nano.owner._model_spec.default_transcription_config = {"fs": 16000}
    nano.session._audio = np.full(123, 0.1, dtype=np.float32)
    nano.session._decode(0, 123)
    assert "fs" not in nano.calls[-1][1]
    assert defaults == {"fs": 16000, "cache": {"vad_state": []}}


def test_installed_nano_sdk_accepts_tensor_without_weights():
    try:
        sdk = import_module("funasr.models.fun_asr_nano.model")
    except ImportError:
        pytest.skip("Optional FunASR Nano SDK or its import dependencies are missing")
    waveform = torch.linspace(-1, 1, 16000, dtype=torch.float32)
    # Exercise the installed SDK, without constructing a model or using CUDA.
    messages = sdk.FunASRNano.generate_chatml(None, "语音转写：", waveform)
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert messages[1]["audio"] is waveform


async def test_half_sample_rejected_at_end(nano):
    await nano.session.receive_input(b"\x01")
    with pytest.raises(RealtimeError, match="Incomplete"):
        await nano.session.finish_input()
    assert not nano.calls


async def test_forced_segments_bounded_audio_and_stable_sentences(nano):
    session = nano.session
    session.max_segment_samples = 1920
    session.overlap_samples = 320
    session.lookback_samples = 960
    outputs = asyncio.create_task(collect(session))
    for _ in range(20):
        await session.receive_input(speech(960))
        assert len(session._audio) <= 2240
    await session.finish_input()
    data = await outputs
    final = data[-1]
    assert len(final["sentences"]) == 10
    assert all(call[0] <= 2240 for call in nano.calls)
    for before, after in zip(data, data[1:]):
        assert after["sentences"][: len(before["sentences"])] == before["sentences"]
    assert final["sentences"][-1]["end"] == 1200


async def test_cancel_has_no_final_and_rejects_more_input(nano):
    await nano.session.receive_input(speech(100))
    await nano.session.close("cancelled")
    assert not nano.calls
    assert not nano.session._pending
    with pytest.raises(RealtimeError):
        await nano.session.receive_input(speech(1))


@pytest.mark.parametrize(
    "change",
    [
        {"input": {"audio": {**PCM_FORMAT, "sample_rate": 8000}}},
        {"input": {"audio": {**PCM_FORMAT, "sample_rate": 16000.0}}},
        {"input": {"audio": {**PCM_FORMAT, "channels": True}}},
        {"parameters": {"unknown": True}},
        {"parameters": {"hotwords": "wrong"}},
        {"parameters": {"language": 1}},
    ],
)
def test_config_validation(change):
    with pytest.raises(RealtimeError):
        validate_config({**CONFIG, **change})


def test_disabled_instances_do_not_claim_streaming():
    model = FunASRModel("nano", "/unused", SimpleNamespace())
    assert model.get_realtime_capabilities() == {}
    with pytest.raises(ValueError):
        model.create_realtime_session(CONFIG, None)


def test_overlap_merge_is_conservative():
    from ..funasr_realtime import trim_overlap

    assert trim_overlap("今天开会", "开会讨论方案") == "讨论方案"
    assert trim_overlap("hello world", "world again") == "again"
    assert trim_overlap("go", "good") == "good"
    assert trim_overlap("重复", "重复") == "重复"


async def test_session_parameters_and_state_are_isolated(nano):
    other = NanoRealtimeSession(
        nano.owner,
        {**CONFIG, "parameters": {"language": "en", "hotwords": ["other"]}},
        nano.runtime,
    )
    first_outputs = asyncio.create_task(collect(nano.session))
    other_outputs = asyncio.create_task(collect(other))
    try:
        for _ in range(17):
            await nano.session.receive_input(speech(960))
            await other.receive_input(speech(960))
        await nano.session.finish_input()
        await other.finish_input()
        first, second = await asyncio.gather(first_outputs, other_outputs)
        assert first[-1]["is_final"] and second[-1]["is_final"]
        assert nano.session._sentences is not other._sentences
        assert nano.session.vad is not other.vad
        assert nano.session.vad.model.kwargs is not other.vad.model.kwargs
        assert {tuple(kwargs["hotwords"]) for _, kwargs in nano.calls} == {
            ("测试",),
            ("other",),
        }
        assert {kwargs["language"] for _, kwargs in nano.calls} == {"中文", "English"}
    finally:
        await other.close("test")
        for task in (first_outputs, other_outputs):
            if not task.done():
                task.cancel()
        await asyncio.gather(first_outputs, other_outputs, return_exceptions=True)


@pytest.mark.parametrize(
    "scenario",
    ["normal", "silence", "cancel", "disconnect", "invalid_config", "odd_sample"],
)
async def test_live_gateway_probe(nano, tmp_path, scenario):
    import importlib.util
    import wave
    from pathlib import Path

    from ....api.tests.test_realtime_api import serve_gateway

    path = Path(__file__).resolve().parents[4] / "benchmark/realtime_ws_probe.py"
    spec = importlib.util.spec_from_file_location("realtime_probe", path)
    probe_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe_module)
    audio_path = tmp_path / "speech.wav"
    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(speech(19200))
    async with serve_gateway(backend_model=nano.owner) as service:
        args = SimpleNamespace(
            audio=audio_path,
            url=service.url,
            scenario=scenario,
            chunk_ms=100,
            language="zh",
            hotword=[],
            timeout=5,
            cancel_after_ms=1000,
            speech_start_ms=0,
            require_interim=True,
        )
        report = await probe_module.probe(args, key="")
        assert report["passed"], report
        assert "识别" not in str(report)  # Reports do not expose transcription text.


async def test_probe_cli_writes_report_and_exit_code(nano, tmp_path):
    import json
    import os
    import wave
    from pathlib import Path

    from ....api.tests.test_realtime_api import serve_gateway

    audio_path = tmp_path / "speech.wav"
    report_path = tmp_path / "report.json"
    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(speech(19200))
    probe = Path(__file__).resolve().parents[4] / "benchmark/realtime_ws_probe.py"
    env = {**os.environ, "XINFERENCE_API_KEY": ""}
    async with serve_gateway(backend_model=nano.owner) as service:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(probe),
            "--url",
            service.url,
            "--audio",
            str(audio_path),
            "--report",
            str(report_path),
            "--require-interim",
            "--timeout",
            "5",
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
        assert process.returncode == 0, (stdout, stderr)
        report = json.loads(report_path.read_text())
        assert report["summary"]["passed"] == 1
        assert report["summary"]["failed"] == 0
        assert report["reports"][0]["architecture_checks"].startswith("not_verified")
