#!/usr/bin/env python3
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


"""Replay PCM WAV through Xinference realtime; secrets/media/text are not reported."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import time
import wave
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlsplit

import aiohttp

CLOSE_TIMEOUT = 5.0


async def probe(args, key: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "scenario": args.scenario,
        "passed": False,
        "timings_ms": {},
        "assertions": {},
    }
    report["protocol_version"] = 1
    report["task"] = "audio.transcription"
    path_parts = urlsplit(args.url).path.rstrip("/").split("/")
    report["model_uid"] = path_parts[-2] if len(path_parts) >= 2 else None
    clock = time.monotonic()
    timings = report["timings_ms"]

    def stamp(name):
        timings.setdefault(name, round((time.monotonic() - clock) * 1000, 3))

    try:
        with wave.open(str(args.audio), "rb") as audio:
            if (
                audio.getnchannels(),
                audio.getsampwidth(),
                audio.getframerate(),
                audio.getcomptype(),
            ) != (1, 2, 16000, "NONE"):
                raise ValueError("Expected uncompressed mono 16 kHz PCM16 WAV")
            duration_ms = audio.getnframes() * 1000 / 16000
            report.update(
                audio_duration_ms=duration_ms,
                chunk_ms=args.chunk_ms,
                frames_sent=0,
                pcm_bytes_sent=0,
            )
            config: Dict[str, Any] = {
                "event": "session.start",
                "protocol_version": 1,
                "task": "audio.transcription",
                "input": {
                    "audio": {
                        "encoding": "pcm_s16le",
                        "sample_rate": 16000,
                        "channels": 1,
                    }
                },
                "parameters": {"language": args.language, "hotwords": args.hotword},
            }
            if args.scenario == "invalid_config":
                config["input"]["audio"]["sample_rate"] = 8000
            headers = {"Authorization": "Bearer " + key} if key else {}
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
            async with aiohttp.ClientSession(
                timeout=timeout, trust_env=False
            ) as client:
                async with client.ws_connect(
                    args.url,
                    headers=headers,
                    heartbeat=25,
                    max_msg_size=2097152,
                    compress=0,
                ) as ws:
                    stamp("connected")
                    await ws.send_json(config)
                    first = await asyncio.wait_for(ws.receive_json(), 15)
                    if args.scenario == "invalid_config":
                        report["assertions"]["expected_rejection"] = first.get(
                            "event"
                        ) == "error" and first.get("code") in {
                            "unsupported_format",
                            "invalid_config",
                        }
                        report["error_code"] = first.get("code")
                        report["passed"] = all(report["assertions"].values())
                        return report
                    if first.get("event") != "session.started":
                        report["error_code"] = first.get(
                            "code", "missing_session_started"
                        )
                        return report
                    session_id = first["session_id"]
                    report["trace_id"] = hashlib.sha256(
                        session_id.encode()
                    ).hexdigest()[:16]
                    stamp("session_started")
                    stop = asyncio.Event()
                    final_count = 0
                    last_seq = 0
                    stable_sentences: list = []
                    closed_reason = None
                    protocol_error = None

                    async def send():
                        samples = args.chunk_ms * 16
                        audio_start = time.monotonic()
                        stamp("audio_start")
                        sent_samples = 0
                        while not stop.is_set():
                            pcm = audio.readframes(samples)
                            if not pcm:
                                break
                            if args.scenario == "silence":
                                pcm = bytes(len(pcm))
                            sent_samples += len(pcm) // 2
                            # Model microphone capture: a frame is available at
                            # its end, not 100 ms before those samples exist.
                            await asyncio.sleep(
                                max(
                                    0,
                                    audio_start
                                    + sent_samples / 16000
                                    - time.monotonic(),
                                )
                            )
                            if stop.is_set():
                                return
                            await ws.send_bytes(pcm)
                            report["frames_sent"] += 1
                            report["pcm_bytes_sent"] += len(pcm)
                            stamp("first_audio_sent")
                            timings["last_audio_sent"] = round(
                                (time.monotonic() - clock) * 1000, 3
                            )
                            if (
                                args.scenario in {"cancel", "disconnect"}
                                and sent_samples / 16 >= args.cancel_after_ms
                            ):
                                if args.scenario == "cancel":
                                    await ws.send_json({"event": "session.cancel"})
                                    stamp("cancel_sent")
                                else:
                                    stamp("disconnect_started")
                                    await ws.close()
                                    stamp("disconnect_sent")
                                return
                        if stop.is_set():
                            return
                        if args.scenario in {"cancel", "disconnect"}:
                            raise ValueError(
                                "Audio shorter than requested cancellation point"
                            )
                        if args.scenario == "odd_sample":
                            await ws.send_bytes(b"\x01")
                        await ws.send_json({"event": "input.end"})
                        stamp("input_end")

                    async def receive():
                        nonlocal final_count, last_seq, stable_sentences, closed_reason, protocol_error
                        async for frame in ws:
                            if frame.type == aiohttp.WSMsgType.ERROR:
                                raise ws.exception() or ConnectionError(
                                    "Realtime transport failed"
                                )
                            if frame.type != aiohttp.WSMsgType.TEXT:
                                raise ValueError("Unexpected non-text result")
                            event = json.loads(frame.data)
                            kind = event.get("event")
                            if kind == "error":
                                protocol_error = event.get("code", "unknown_error")
                                stamp("server_error")
                                stop.set()
                                continue
                            if event.get("session_id") != session_id:
                                raise ValueError("Mismatched session ID")
                            if kind == "output":
                                if final_count:
                                    raise ValueError(
                                        "Output received after final result"
                                    )
                                seq = event.get("seq")
                                if type(seq) is not int or seq <= last_seq:
                                    raise ValueError("Non-increasing result sequence")
                                last_seq = seq
                                data = event["data"]
                                sentences = data.get("sentences", [])
                                if (
                                    sentences[: len(stable_sentences)]
                                    != stable_sentences
                                ):
                                    raise ValueError("Stable sentence snapshot changed")
                                stable_sentences = sentences
                                nonempty = bool(
                                    data.get("partial")
                                    or data.get("text")
                                    or any(s.get("text") for s in sentences)
                                )
                                if nonempty:
                                    stamp("first_nonempty_output")
                                    if (
                                        data.get("partial")
                                        and data.get("is_final") is not True
                                    ):
                                        stamp("first_nonempty_partial")
                                if data.get("is_final") is True:
                                    final_count += 1
                                    stamp("final")
                            elif kind == "session.closed":
                                closed_reason = event.get("reason")
                                stamp("session_closed")
                                stop.set()
                            else:
                                raise ValueError("Unexpected server event")
                        stamp("connection_closed")

                    async def observed(direction, operation):
                        try:
                            await operation()
                        except Exception as exc:
                            # Keep only exception types: strings may contain
                            # credentials, media, or transcription text.
                            failure = type(exc).__name__
                            report[direction + "_failure_type"] = failure
                            report.setdefault("failure_type", failure)
                            stamp(direction + "_failed")
                            stop.set()
                        finally:
                            if direction == "receiver":
                                stop.set()

                    sender = asyncio.create_task(
                        observed("sender", send),
                        name="xinference-realtime-probe-sender",
                    )
                    receiver = asyncio.create_task(
                        observed("receiver", receive),
                        name="xinference-realtime-probe-receiver",
                    )
                    stopped = asyncio.create_task(
                        stop.wait(), name="xinference-realtime-probe-stop"
                    )

                    async def exchange():
                        await asyncio.wait(
                            (sender, receiver, stopped),
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if sender.done() and not stop.is_set():
                            # Normal upload completion still needs Final/close.
                            await asyncio.wait(
                                (receiver, stopped),
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                        if not sender.done():
                            if (
                                "disconnect_started" in timings
                                and protocol_error is None
                                and not report.get("receiver_failure_type")
                            ):
                                # aiohttp wakes its receiver while close() is
                                # still completing the requested handshake.
                                await asyncio.wait_for(
                                    asyncio.shield(sender), CLOSE_TIMEOUT
                                )
                            else:
                                sender.cancel()
                        await asyncio.gather(sender, return_exceptions=True)
                        # On upload failure, allow a bounded receive-only drain
                        # to retain a racing server error and its close code.
                        await asyncio.wait_for(asyncio.shield(receiver), CLOSE_TIMEOUT)

                    try:
                        await asyncio.wait_for(
                            exchange(),
                            args.timeout or duration_ms / 1000 + 60,
                        )
                    except Exception as exc:
                        report["exchange_failure_type"] = type(exc).__name__
                        raise
                    finally:
                        stop.set()
                        for task in (sender, receiver, stopped):
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(
                            sender, receiver, stopped, return_exceptions=True
                        )
                        # Persist observations even when upload, receive, or
                        # the bounded drain failed. Never lose a server error
                        # just because a later client write was reset.
                        report.update(
                            final_count=final_count,
                            last_seq=last_seq,
                            error_code=protocol_error,
                            close_code=ws.close_code,
                            closed_reason=closed_reason,
                        )
                    checks = report["assertions"]
                    checks["no_client_failure"] = not any(
                        report.get(name)
                        for name in ("sender_failure_type", "receiver_failure_type")
                    )
                    if args.scenario == "cancel":
                        report["native_cancel_stage"] = "not_verified_by_client"
                        checks.update(
                            no_error=protocol_error is None,
                            cancelled=closed_reason == "cancelled",
                            no_success_final=final_count == 0,
                        )
                    elif args.scenario == "disconnect":
                        checks["no_error"] = protocol_error is None
                        checks["disconnect_sent"] = "disconnect_sent" in timings
                        report["server_cleanup"] = "not_verified_by_client"
                    elif args.scenario == "odd_sample":
                        checks.update(
                            expected_rejection=protocol_error == "invalid_audio",
                            no_success_final=final_count == 0,
                        )
                    else:
                        checks.update(
                            no_error=protocol_error is None,
                            unique_final=final_count == 1,
                            completed=closed_reason == "completed",
                            normal_close=ws.close_code == 1000,
                            final_after_input_end=(
                                "input_end" in timings
                                and timings.get("final", -1) >= timings["input_end"]
                            ),
                        )
                        if args.scenario == "silence":
                            checks["no_text_for_silence"] = (
                                "first_nonempty_output" not in timings
                            )
                        else:
                            checks["nonempty_transcription"] = (
                                "first_nonempty_output" in timings
                            )
                        if args.scenario != "silence" and (
                            args.require_interim or duration_ms >= 30000
                        ):
                            first_output = timings.get(
                                "first_nonempty_output", float("inf")
                            )
                            checks["text_before_last_audio_and_end"] = (
                                first_output < timings.get("last_audio_sent", -1)
                                and first_output < timings.get("input_end", -1)
                            )
                        if "final" in timings and "input_end" in timings:
                            report["end_to_final_ms"] = (
                                timings["final"] - timings["input_end"]
                            )
                        if (
                            args.speech_start_ms is not None
                            and "first_nonempty_partial" in timings
                        ):
                            report["first_partial_latency_ms"] = (
                                timings["first_nonempty_partial"]
                                - timings["audio_start"]
                                - args.speech_start_ms
                            )
                        else:
                            report["first_partial_latency_ms"] = None
                            report["first_partial_latency_note"] = (
                                "Requires speech onset annotation and nonempty partial"
                            )
                    report["architecture_checks"] = (
                        "not_verified_by_client; use runtime integration tests and telemetry"
                    )
                    report["passed"] = bool(checks) and all(checks.values())
    except Exception as exc:
        # Exception strings may contain request headers or model output.
        report.setdefault("failure_type", type(exc).__name__)
    return report


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


async def run(args, key):
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(index):
        async with semaphore:
            return {"run": index, **await probe(args, key)}

    # Bound task creation as well as simultaneous connections for large repeats.
    reports = []
    for offset in range(0, args.repeat, args.concurrency):
        reports.extend(
            await asyncio.gather(
                *(
                    one(i)
                    for i in range(offset, min(args.repeat, offset + args.concurrency))
                )
            )
        )
    summary = {
        "runs": len(reports),
        "passed": sum(r["passed"] for r in reports),
        "failed": sum(not r["passed"] for r in reports),
    }
    for name in ("first_partial_latency_ms", "end_to_final_ms"):
        values = [
            r[name]
            for r in reports
            if r["passed"] and isinstance(r.get(name), (int, float))
        ]
        summary[name] = {
            "samples": len(values),
            "p50": percentile(values, 0.5) if values else None,
            "p95": percentile(values, 0.95) if len(values) >= 20 else None,
        }
    summary["p95_note"] = (
        "Reported only for >=20 valid samples; goals require a controlled, warmed GPU baseline"
    )
    return {"summary": summary, "reports": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument(
        "--scenario",
        choices=[
            "normal",
            "silence",
            "cancel",
            "disconnect",
            "invalid_config",
            "odd_sample",
        ],
        default="normal",
    )
    parser.add_argument("--language", default="zh")
    parser.add_argument("--hotword", action="append", default=[])
    parser.add_argument("--speech-start-ms", type=float)
    parser.add_argument("--require-interim", action="store_true")
    parser.add_argument("--cancel-after-ms", type=int, default=1000)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--repeat", type=int, default=1, help="Total number of sessions"
    )
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    url = urlsplit(args.url)
    if (
        url.scheme not in {"ws", "wss"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        parser.error(
            "Use a ws/wss endpoint without URL credentials or query parameters"
        )
    if (
        not 1 <= args.chunk_ms <= 1000
        or not 1 <= args.concurrency <= 128
        or args.repeat < 1
        or args.cancel_after_ms < 1
        or (args.timeout is not None and args.timeout <= 0)
        or (args.speech_start_ms is not None and args.speech_start_ms < 0)
    ):
        parser.error("Invalid frame, concurrency, repetition or timing limits")
    key = os.environ.get("XINFERENCE_API_KEY", "")
    if args.env_file and not key:
        try:
            from dotenv import dotenv_values
        except ImportError:
            parser.error("--env-file requires python-dotenv")
        key = dotenv_values(args.env_file).get("XINFERENCE_API_KEY") or ""
    result = asyncio.run(run(args, key))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0 if result["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
