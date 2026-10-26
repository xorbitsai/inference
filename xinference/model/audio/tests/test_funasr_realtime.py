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
import json
import sys
import traceback
from dataclasses import replace
from importlib import import_module
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import pytest_asyncio
import torch

from ....core.realtime.protocol import RealtimeError
from ....core.realtime.runtime import RealtimeRuntime
from ..funasr import FunASRModel
from ..funasr_realtime import (
    PCM_FORMAT,
    NanoRealtimeOptions,
    NanoRealtimeSession,
    validate_config,
)

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


@pytest.mark.parametrize(
    "texts,expected",
    [
        (["hello", "world"], "hello world"),
        (["你好", "世界"], "你好世界"),
        (["hello,", "world!"], "hello, world!"),
        (["café", "déjà vu"], "café déjà vu"),
        (["こんにちは", "世界"], "こんにちは世界"),
        (["안녕", "세계"], "안녕세계"),
        (["hello", ", world"], "hello, world"),
    ],
)
async def test_vad_sentence_text_boundaries(nano, monkeypatch, texts, expected):
    hypotheses = iter(texts)
    monkeypatch.setattr(
        nano.owner._model, "inference", lambda **kwargs: [{"text": next(hypotheses)}]
    )
    for _ in texts:
        await nano.session.receive_input(speech(960))
        await nano.session.receive_input(bytes(1920))
    await nano.session.finish_input()
    final = (await collect(nano.session))[-1]
    assert [sentence["text"] for sentence in final["sentences"]] == texts
    assert final["text"] == expected


@pytest.mark.parametrize(
    "texts,expected",
    [
        (["hello world", "world again"], "hello world again"),
        (["今天开会", "开会讨论方案"], "今天开会讨论方案"),
    ],
)
async def test_forced_overlap_text_boundaries(nano, monkeypatch, texts, expected):
    hypotheses = iter(texts)
    monkeypatch.setattr(
        nano.owner._model, "inference", lambda **kwargs: [{"text": next(hypotheses)}]
    )
    nano.session.max_segment_samples = 1920
    nano.session.overlap_samples = 320
    await nano.session.receive_input(speech(2880))
    await nano.session.finish_input()
    final = (await collect(nano.session))[-1]
    assert len(final["sentences"]) == 2
    assert final["text"] == expected


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


async def test_end_hint_skips_previews_not_pcm_or_required_commits(nano):
    session = nano.session
    session.options = replace(session.options, optimize=True)
    session.max_segment_samples = 1920
    session.overlap_samples = 320
    session.partial_samples = session.first_partial_samples = 960
    session.request_input_end()  # Reader observes end before the processor drains.
    outputs = asyncio.create_task(collect(session))
    pcm = speech(4801)
    session.notify_input(len(pcm))
    await session.receive_input(pcm)
    await session.finish_input()
    data = await outputs
    assert session._total == 4801
    assert len(data[-1]["sentences"]) == 3
    assert session._decode_counts == {"forced_commit": 2, "final": 1}
    assert session._metrics["previews_skipped_end"] > 0
    assert nano.calls[-1][0] == 4801 - 3840 + 320
    assert data[-1]["is_final"] and sum(item["is_final"] for item in data) == 1
    assert session.observations()["backlog_audio_ms"] == 0


async def test_latest_prefix_coalesces_backlog_without_queueing_jobs(nano):
    session = nano.session
    session.options = replace(session.options, optimize=True)
    outputs = asyncio.create_task(collect(session))
    pcm = speech(19200)
    session.notify_input(len(pcm))
    await session.receive_input(pcm)
    # All obsolete eligible prefixes were skipped; only the newest was decoded.
    assert [length for length, _ in nano.calls] == [19200]
    assert session._metrics["previews_coalesced"] > 0
    session.request_input_end()
    await session.finish_input()
    final = (await outputs)[-1]
    assert len(nano.calls) == 1  # Exact same input safely reused by final commit.
    assert session._metrics["decode_reused"] == 1
    assert final["text"] == "识别19200"
    assert session._decode_cache is None


@pytest.mark.parametrize(
    "change", ["tail", "language", "hotword", "defaults", "generation", "context"]
)
async def test_decode_reuse_invalidated_by_input_or_context(nano, change):
    session = nano.session
    session.options = replace(session.options, optimize=True)
    session._audio = np.full(160, 0.1, dtype=np.float32)
    session._total = 160
    session._decode(0, 160)
    assert session._decode(0, 160) == "识别160"
    assert len(nano.calls) == 1
    end = 160
    if change == "tail":
        session._audio = np.full(161, 0.1, dtype=np.float32)
        session._total = end = 161
    elif change == "language":
        session.parameters["language"] = "en"
    elif change == "hotword":
        session.parameters["hotwords"].append("new")
    elif change == "defaults":
        nano.owner._model_spec.default_transcription_config = {"beam_size": 2}
    elif change == "generation":
        nano.runtime.generation = "reloaded"
    else:
        session._sentences.append({"start": 0, "end": 1, "text": "context"})
    session._decode(0, end)
    assert len(nano.calls) == 2


async def test_optimized_final_preserves_single_sample_tail(nano):
    session = nano.session
    session.options = replace(session.options, optimize=True)
    outputs = asyncio.create_task(collect(session))
    await session.receive_input(speech(15360))
    session.request_input_end()
    await session.receive_input(speech(1))
    await session.finish_input()
    data = await outputs
    assert [length for length, _ in nano.calls] == [15360, 15361]
    assert data[-1]["text"] == "识别15361"
    assert session._metrics["decode_reused"] == 0


async def test_cancel_fences_decode_and_outputs(nano):
    nano.session.request_cancel()
    await nano.session.emit({"must_not_escape": True})
    assert nano.session._outputs.empty()
    with pytest.raises(RealtimeError, match="cancelled"):
        nano.session._feed(speech(960))
    assert not nano.calls


@pytest.mark.parametrize(
    "options",
    [
        [],
        {"unknown": True},
        {"optimize": "true"},
        {"max_segment_ms": True},
        {"partial_interval_ms": 0},
        {"preview_backlog_ms": -1},
        {"max_segment_ms": 600},
        {"first_partial_ms": 6000, "max_segment_ms": 4000},
    ],
)
def test_realtime_options_reject_invalid(options):
    with pytest.raises(ValueError):
        NanoRealtimeOptions.from_dict(options)


def test_realtime_options_are_model_local_and_default_off():
    from ....deploy.utils import handle_click_args_type

    assert not NanoRealtimeOptions().optimize
    options = {"optimize": True, "max_segment_ms": 6000, "partial_interval_ms": 1500}
    model = FunASRModel(
        "nano", "/unused", SimpleNamespace(), realtime=True, realtime_options=options
    )
    options["max_segment_ms"] = 4000
    assert model._realtime_options.max_segment_ms == 6000
    assert "realtime_options" not in model._kwargs
    parsed = handle_click_args_type(
        '{"optimize":true,"max_segment_ms":6000,"partial_interval_ms":1500}'
    )
    assert NanoRealtimeOptions.from_dict(parsed) == model._realtime_options
    with pytest.raises(ValueError, match="requires"):
        FunASRModel("nano", "/unused", SimpleNamespace(), realtime_options={})


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {
            "optimize": True,
            "first_partial_ms": 960,
            "partial_interval_ms": 1500,
            "max_segment_ms": 6000,
        },
        {"end_priority": False, "coalesce_partials": False, "reuse_decode": False},
    ],
)
def test_realtime_options_object_and_legacy_json_are_equivalent(options, as_json):
    original = copy.deepcopy(options)
    value = " \n" + json.dumps(options) + "\t " if as_json else options
    expected = NanoRealtimeOptions.from_dict(options)
    model = FunASRModel(
        "nano", "/unused", SimpleNamespace(), realtime=True, realtime_options=value
    )
    assert model._realtime_options == expected
    assert "realtime_options" not in model._kwargs
    assert options == original
    options["optimize"] = not expected.optimize
    assert model._realtime_options == expected


@pytest.mark.parametrize("realtime", [False, True])
def test_realtime_options_explicit_none_retains_default_behavior(realtime):
    model = FunASRModel(
        "nano", "/unused", SimpleNamespace(), realtime=realtime, realtime_options=None
    )
    assert model._realtime_options is None
    assert "realtime_options" not in model._kwargs
    assert NanoRealtimeOptions.from_dict({}) == NanoRealtimeOptions()
    assert NanoRealtimeOptions.from_dict("{}") == NanoRealtimeOptions()
    assert not NanoRealtimeOptions().optimize
    assert NanoRealtimeOptions().partial_interval_ms == 960
    assert NanoRealtimeOptions().max_segment_ms == 15000


@pytest.mark.parametrize("options", ["{}", '{"optimize":true}'])
def test_realtime_options_legacy_json_still_requires_realtime(options):
    with pytest.raises(ValueError, match="realtime_options requires realtime=True"):
        FunASRModel("nano", "/unused", SimpleNamespace(), realtime_options=options)


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "{",
        "{'optimize': True}",
        "'{\"optimize\":true}'",
        '{"optimize":NaN}',
        '{"partial_interval_ms":Infinity}',
        '{"max_segment_ms":-Infinity}',
    ],
)
def test_realtime_options_json_syntax_errors_are_distinct(value):
    with pytest.raises(ValueError, match="realtime_options must contain valid JSON"):
        NanoRealtimeOptions.from_dict(value)


@pytest.mark.parametrize(
    "value",
    [None, [], True, 960, "null", "[]", "true", "960", '"text"', '"{}"'],
)
def test_realtime_options_requires_object_after_single_decode(value):
    with pytest.raises(ValueError, match="realtime_options must be a JSON object"):
        NanoRealtimeOptions.from_dict(value)


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize(
    "options,message",
    [
        ({"unknown": True}, "contains unsupported fields"),
        ({"optimize": "true"}, "optimize must be a boolean"),
        ({"end_priority": 1}, "end_priority must be a boolean"),
        ({"max_segment_ms": True}, "max_segment_ms must be an integer"),
        ({"partial_interval_ms": "1500"}, "partial_interval_ms must be an integer"),
        ({"first_partial_ms": 960.0}, "first_partial_ms must be an integer"),
        ({"partial_interval_ms": 10001}, "partial_interval_ms must be 120..10000"),
        ({"max_segment_ms": 999}, "max_segment_ms must be 1000..15000"),
        ({"preview_backlog_ms": -1}, "preview_backlog_ms must be 0..10000"),
        (
            {"first_partial_ms": 6000, "max_segment_ms": 4000},
            "realtime_options: Partial intervals must not exceed max_segment_ms",
        ),
    ],
)
def test_realtime_options_json_preserves_strict_validation(options, message, as_json):
    value = json.dumps(options) if as_json else options
    with pytest.raises(ValueError, match=message):
        NanoRealtimeOptions.from_dict(value)


@pytest.mark.parametrize(
    "value",
    [
        '{"PRIVATE_PAYLOAD":',
        '{"PRIVATE_PAYLOAD":true}',
        '{"optimize":"PRIVATE_PAYLOAD"}',
        '{"max_segment_ms":"PRIVATE_PAYLOAD"}',
        '{"PRIVATE_PAYLOAD":NaN}',
    ],
)
def test_realtime_options_errors_do_not_echo_input(value):
    with pytest.raises(ValueError) as error:
        NanoRealtimeOptions.from_dict(value)
    rendered = "".join(traceback.format_exception(error.value))
    assert "PRIVATE_PAYLOAD" not in rendered
    assert "realtime_options" in str(error.value)


@pytest.mark.parametrize("window_ms", [4000, 6000, 8000, 15000])
async def test_candidate_windows_bound_decode_and_audio(nano, window_ms):
    session = nano.session
    session.options = NanoRealtimeOptions.from_dict(
        {"optimize": True, "max_segment_ms": window_ms}
    )
    session.max_segment_samples = window_ms * 16
    session.request_input_end()
    outputs = asyncio.create_task(collect(session))
    for _ in range(300):
        await session.receive_input(speech(960))
        assert (
            len(session._audio) <= session.max_segment_samples + session.overlap_samples
        )
    await session.finish_input()
    final = (await outputs)[-1]
    assert session._total == 288000
    assert all(
        length <= session.max_segment_samples + session.overlap_samples
        for length, _ in nano.calls
    )
    assert final["duration_ms"] == 18000
    assert sum(s["end"] - s["start"] for s in final["sentences"]) == 18000


@pytest.mark.parametrize("kind", ["partial", "forced_commit", "final"])
@pytest.mark.parametrize("tied_trace_timestamps", [False, True])
async def test_nano_native_stage_cancel_has_trace_and_safe_cleanup(
    nano, monkeypatch, kind, tied_trace_timestamps
):
    import threading

    import aiohttp

    from ....api.tests.test_realtime_api import serve_gateway
    from ....core.tests.test_realtime_runtime import wait_empty

    if tied_trace_timestamps:
        from ....core.realtime import telemetry

        monkeypatch.setattr(telemetry, "time", SimpleNamespace(monotonic=lambda: 0.0))

    options = {
        "optimize": True,
        "coalesce_partials": False,
        "max_segment_ms": 1000,
        "first_partial_ms": 120 if kind == "partial" else 1000,
        "partial_interval_ms": 1000,
    }
    nano.owner._realtime_options = NanoRealtimeOptions.from_dict(options)
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    original = nano.owner._model.inference

    def inference(input, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        release.wait(5)
        return original(input, **kwargs)

    monkeypatch.setattr(nano.owner._model, "inference", inference)
    async with serve_gateway(backend_model=nano.owner) as service:
        try:
            async with aiohttp.ClientSession() as client:
                ws = await client.ws_connect(service.url)
                await ws.send_json(CONFIG)
                await ws.receive_json()
                await ws.send_bytes(
                    speech(
                        {"partial": 1920, "forced_commit": 17280, "final": 123}[kind]
                    )
                )
                if kind == "final":
                    await ws.send_json({"event": "input.end"})
                await asyncio.wait_for(entered.wait(), 2)
                await ws.send_json({"event": "session.cancel"})
                assert (await asyncio.wait_for(ws.receive_json(), 2))[
                    "reason"
                ] == "cancelled"
                assert service.runtime.active_count == 1
                assert service.runtime._inference_lock.locked()
                await ws.close()
        finally:
            release.set()
            await wait_empty(service.runtime)
        summary = service.runtime.recent_summaries[-1]
        starts = [e for e in summary["events"] if e["stage"] == "native_start"]
        assert [e["kind"] for e in starts] == [kind]
        stages = [event["stage"] for event in summary["events"]]
        # Ordering, not strictly increasing clock readings, proves cancellation
        # happened during native work, including on coarse Windows clocks.
        assert (
            stages.index("native_start")
            < stages.index("cancel_processed")
            < stages.index("native_end")
        )
        if tied_trace_timestamps:
            assert {event["at_ms"] for event in summary["events"]} == {0.0}
        assert "识别" not in str(summary)


async def test_optimized_odd_tail_remains_enforced(nano):
    session = nano.session
    session.options = replace(session.options, optimize=True)
    session.request_input_end()
    await session.receive_input(speech(960) + b"\x01")
    with pytest.raises(RealtimeError, match="Incomplete"):
        await session.finish_input()
    assert not nano.calls


async def test_optimized_duration_limit_remains_enforced(nano):
    nano.session.options = replace(nano.session.options, optimize=True)
    nano.runtime.limits = replace(nano.runtime.limits, max_duration=0.001)
    with pytest.raises(RealtimeError, match="Maximum audio duration"):
        await nano.session.receive_input(speech(960))
    assert not nano.calls


async def test_unknown_config_disables_reuse_and_invalidates_old_entry(nano):
    session = nano.session
    session.options = replace(session.options, optimize=True)
    session._audio = np.full(160, 0.1, dtype=np.float32)
    session._total = 160
    session._decode(0, 160)
    assert session._decode_cache is not None
    nano.owner._model_spec.default_transcription_config = {"custom": ("tuple",)}
    session._decode(0, 160)
    session._decode(0, 160)
    assert session._decode_cache is None and len(nano.calls) == 3
