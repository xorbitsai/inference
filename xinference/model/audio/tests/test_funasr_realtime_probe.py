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


"""Exercise the standalone probe over real local WebSockets, without ASR weights."""

import asyncio
import importlib.util
import json
import os
import sys
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web

SECRET = "probe-test-secret-not-for-reports"
SESSION_ID = "probe-test-session"


@pytest.fixture
def probe_module():
    path = Path(__file__).resolve().parents[4] / "benchmark/realtime_ws_probe.py"
    spec = importlib.util.spec_from_file_location("realtime_probe_fault_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe_args(tmp_path):
    path = tmp_path / "speech.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\xe8\x03" * 19200)
    return SimpleNamespace(
        audio=path,
        url="",
        scenario="normal",
        chunk_ms=100,
        language="zh",
        hotword=[],
        timeout=3,
        cancel_after_ms=300,
        speech_start_ms=None,
        require_interim=False,
    )


@asynccontextmanager
async def serve_probe(handler):
    async def endpoint(request):
        ws = web.WebSocketResponse(compress=False)
        await ws.prepare(request)
        try:
            await handler(ws)
        finally:
            await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/v1/models/nano/realtime", endpoint)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        yield f"ws://127.0.0.1:{port}/v1/models/nano/realtime"
    finally:
        await runner.cleanup()


async def start_and_receive_audio(ws):
    assert (await ws.receive_json())["event"] == "session.start"
    await ws.send_json({"event": "session.started", "session_id": SESSION_ID})
    frame = await ws.receive()
    assert frame.type == aiohttp.WSMsgType.BINARY


async def send_error(ws):
    await ws.send_json({"event": "error", "code": "inference_failed", "detail": SECRET})


def assert_no_probe_tasks():
    assert not [
        task
        for task in asyncio.all_tasks()
        if task.get_name().startswith("xinference-realtime-probe-")
    ]


@pytest.mark.parametrize("close_delay", [0, 0.25])
@pytest.mark.parametrize(
    "scenario", ["normal", "silence", "cancel", "disconnect", "odd_sample"]
)
async def test_probe_retains_midstream_server_error(
    probe_module, probe_args, close_delay, scenario
):
    late_frames = []

    async def handler(ws):
        await start_and_receive_audio(ws)
        await send_error(ws)
        if close_delay:
            try:
                frame = await asyncio.wait_for(ws.receive(), close_delay)
                if frame.type in (aiohttp.WSMsgType.BINARY, aiohttp.WSMsgType.TEXT):
                    late_frames.append(frame.type)
            except asyncio.TimeoutError:
                pass
        await ws.close(code=1013)

    async with serve_probe(handler) as url:
        probe_args.url = url
        probe_args.scenario = scenario
        report = await probe_module.probe(probe_args, key=SECRET)
    assert not report["passed"]
    assert report["error_code"] == "inference_failed"
    assert report["close_code"] == 1013
    assert report["final_count"] == 0
    assert report["last_seq"] == 0
    assert report["frames_sent"] == 1
    assert "server_error" in report["timings_ms"]
    assert "input_end" not in report["timings_ms"]
    assert not late_frames
    assert SECRET not in json.dumps(report)
    assert_no_probe_tasks()


async def test_probe_disconnect_finishes_close_handshake(probe_module, probe_args):
    async def handler(ws):
        await start_and_receive_audio(ws)
        async for frame in ws:
            assert frame.type == aiohttp.WSMsgType.BINARY

    probe_args.scenario = "disconnect"
    async with serve_probe(handler) as url:
        probe_args.url = url
        report = await probe_module.probe(probe_args, key=SECRET)
    assert report["passed"], report
    assert report["error_code"] is None
    assert report["close_code"] == 1000
    assert "disconnect_started" in report["timings_ms"]
    assert "disconnect_sent" in report["timings_ms"]
    assert_no_probe_tasks()


async def test_probe_sender_failure_does_not_discard_server_error(
    probe_module, probe_args, monkeypatch
):
    original_send = aiohttp.ClientWebSocketResponse.send_bytes

    async def fail_after_write(ws, data, *args, **kwargs):
        await original_send(ws, data, *args, **kwargs)
        raise ConnectionResetError(SECRET)

    monkeypatch.setattr(aiohttp.ClientWebSocketResponse, "send_bytes", fail_after_write)

    async def handler(ws):
        await start_and_receive_audio(ws)
        # The sender fails before the receiver can observe the error frame.
        await asyncio.sleep(0.05)
        await send_error(ws)
        await ws.close(code=1013)

    async with serve_probe(handler) as url:
        probe_args.url = url
        report = await probe_module.probe(probe_args, key=SECRET)
    assert not report["passed"]
    assert report["error_code"] == "inference_failed"
    assert report["sender_failure_type"] == "ConnectionResetError"
    assert report["failure_type"] == "ConnectionResetError"
    assert report["close_code"] == 1013
    assert "input_end" not in report["timings_ms"]
    assert SECRET not in json.dumps(report)
    assert_no_probe_tasks()


@pytest.mark.parametrize("server_error", [False, True])
async def test_probe_receiver_failure_preserves_observations(
    probe_module, probe_args, server_error
):
    async def handler(ws):
        await start_and_receive_audio(ws)
        if server_error:
            await send_error(ws)
        await ws.send_str(SECRET)  # Invalid JSON; must not be copied to reports.
        await ws.close(code=1013)

    async with serve_probe(handler) as url:
        probe_args.url = url
        report = await probe_module.probe(probe_args, key=SECRET)
    assert not report["passed"]
    assert report["error_code"] == ("inference_failed" if server_error else None)
    assert report["receiver_failure_type"] == "JSONDecodeError"
    assert report["failure_type"] == "JSONDecodeError"
    assert report["final_count"] == 0
    assert report["frames_sent"] == 1
    assert "input_end" not in report["timings_ms"]
    assert SECRET not in json.dumps(report)
    assert_no_probe_tasks()


@pytest.mark.parametrize("timeout_source", ["overall", "drain"])
async def test_probe_error_without_server_close_is_bounded(
    probe_module, probe_args, monkeypatch, timeout_source
):
    late_frames = []

    async def handler(ws):
        await start_and_receive_audio(ws)
        await send_error(ws)
        async for frame in ws:
            late_frames.append(frame.type)

    if timeout_source == "overall":
        probe_args.timeout = 0.5
    else:
        monkeypatch.setattr(probe_module, "CLOSE_TIMEOUT", 0.15)
    async with serve_probe(handler) as url:
        probe_args.url = url
        report = await asyncio.wait_for(probe_module.probe(probe_args, SECRET), 2)
    assert not report["passed"]
    assert report["error_code"] == "inference_failed"
    assert report["failure_type"] == "TimeoutError"
    assert report["exchange_failure_type"] == "TimeoutError"
    assert report["final_count"] == 0
    assert not late_frames
    assert SECRET not in json.dumps(report)
    assert_no_probe_tasks()


async def test_probe_sender_failure_without_server_error_has_bounded_drain(
    probe_module, probe_args, monkeypatch
):
    original_send = aiohttp.ClientWebSocketResponse.send_bytes

    async def fail_after_write(ws, data, *args, **kwargs):
        await original_send(ws, data, *args, **kwargs)
        raise ConnectionResetError(SECRET)

    monkeypatch.setattr(aiohttp.ClientWebSocketResponse, "send_bytes", fail_after_write)
    monkeypatch.setattr(probe_module, "CLOSE_TIMEOUT", 0.15)

    async def handler(ws):
        await start_and_receive_audio(ws)
        async for _ in ws:
            pass

    async with serve_probe(handler) as url:
        probe_args.url = url
        report = await asyncio.wait_for(probe_module.probe(probe_args, SECRET), 2)
    assert not report["passed"]
    assert report["error_code"] is None
    assert report["sender_failure_type"] == "ConnectionResetError"
    assert report["failure_type"] == "ConnectionResetError"
    assert report["exchange_failure_type"] == "TimeoutError"
    assert SECRET not in json.dumps(report)
    assert_no_probe_tasks()


async def test_probe_transport_close_does_not_invent_server_error(
    probe_module, probe_args
):
    async def handler(ws):
        await start_and_receive_audio(ws)
        await ws.close(code=1011)

    async with serve_probe(handler) as url:
        probe_args.url = url
        report = await probe_module.probe(probe_args, key=SECRET)
    assert not report["passed"]
    assert report["error_code"] is None
    assert report["close_code"] == 1011
    assert report["final_count"] == 0
    assert_no_probe_tasks()


async def test_probe_server_error_cli_report_and_exit_code(
    probe_module, probe_args, tmp_path
):
    async def handler(ws):
        await start_and_receive_audio(ws)
        await send_error(ws)
        await ws.close(code=1013)

    report_path = tmp_path / "failure.json"
    async with serve_probe(handler) as url:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            probe_module.__file__,
            "--url",
            url,
            "--audio",
            str(probe_args.audio),
            "--report",
            str(report_path),
            "--timeout",
            "3",
            env={**os.environ, "XINFERENCE_API_KEY": SECRET},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
    assert process.returncode == 1, (stdout, stderr)
    result = json.loads(report_path.read_text())
    assert result["summary"]["failed"] == 1
    report = result["reports"][0]
    assert report["error_code"] == "inference_failed"
    assert report["close_code"] == 1013
    assert report["final_count"] == 0
    assert SECRET not in report_path.read_text()
    assert SECRET.encode() not in stdout + stderr
    assert_no_probe_tasks()


@pytest.fixture
def trace_module():
    path = Path(__file__).resolve().parents[4] / "benchmark/realtime_trace_report.py"
    spec = importlib.util.spec_from_file_location("realtime_trace_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "start,end,verified",
    [(1, 3, True), (3, 4, False), (0, 1, False), (None, 3, False), (1, None, False)],
)
def test_trace_native_cancel_requires_bracketing_interval(
    trace_module, start, end, verified
):
    events = [{"stage": "cancel_processed", "at_ms": 2}]
    if start is not None:
        events.append({"stage": "native_start", "at_ms": start})
    if end is not None:
        events.append({"stage": "native_end", "at_ms": end})
    probe = {"trace_id": "a" * 16, "scenario": "cancel", "passed": True}
    summary = {"trace_id": "a" * 16, "reason": "cancelled", "events": events}
    result = trace_module.analyze([probe], [summary])
    assert result["reports"][0]["native_cancel_verified"] is verified
    # A failed client scenario or ambiguous server association cannot certify.
    assert not trace_module.analyze([{**probe, "passed": False}], [summary])["reports"][
        0
    ]["native_cancel_verified"]
    assert not trace_module.analyze([probe], [summary, summary])["reports"][0][
        "native_cancel_verified"
    ]


def test_trace_groups_sample_sizes_by_replica_and_duration(trace_module):
    probes, summaries = [], []
    for index in range(40):
        trace_id = f"{index:016x}"
        probes.append(
            {
                "trace_id": trace_id,
                "passed": index != 39,
                "end_to_final_ms": index,
                "audio_duration_ms": 60000,
                "scenario": "normal",
            }
        )
        summaries.append(
            {
                "trace_id": trace_id,
                "replica_model_uid": f"nano-{index // 20}",
                "events": [],
                "untrusted_text": SECRET,
            }
        )
    result = trace_module.analyze(probes, summaries)
    first, second = result["groups"]
    assert first["samples"] == 20 and first["p95_end_to_final_ms"] == 18
    assert second["samples"] == 19 and second["p95_end_to_final_ms"] is None
    assert second["failed"] == 1
    assert SECRET not in json.dumps(result)
    assert trace_module.analyze(probes, [])["groups"][0]["p95_end_to_final_ms"] is None


async def test_trace_report_cli_missing_phase_is_nonzero(trace_module, tmp_path):
    probe_path, log_path, report_path = (
        tmp_path / name for name in ("probe.json", "runtime.log", "report.json")
    )
    probe_path.write_text(
        json.dumps(
            {"reports": [{"trace_id": "a" * 16, "scenario": "cancel", "passed": True}]}
        )
    )
    log_path.write_text("credentials=" + SECRET + "\nmalformed diagnostics=not-json\n")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        trace_module.__file__,
        "--probe-report",
        str(probe_path),
        "--runtime-log",
        str(log_path),
        "--report",
        str(report_path),
        "--require-native-cancel",
    )
    assert await asyncio.wait_for(process.wait(), 5) == 1
    assert SECRET not in report_path.read_text()
