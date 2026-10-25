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


"""Persistent, authenticated WebSocket endpoint in the loaded model process."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
import logging
import os
import secrets
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from contextvars import ContextVar, copy_context
from typing import Any, Callable, Dict, Optional

from aiohttp import WSMsgType, web

from .protocol import RealtimeError, RealtimeLimits, parse_event, validate_start
from .telemetry import RealtimeTrace

logger = logging.getLogger(__name__)


class SessionCancelled(Exception):
    pass


async def _settle(future: asyncio.Future) -> Any:
    """Repeated cancellation must not abandon an owned native/cleanup future."""
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    return future.result()


class RealtimeRuntime:
    def __init__(
        self,
        model: Any,
        model_uid: str,
        replica_model_uid: str,
        *,
        limits: Optional[RealtimeLimits] = None,
        inference_lock: Optional[asyncio.Lock] = None,
        admission: Optional[Callable[[], bool]] = None,
    ):
        self.model = model
        self.model_uid = model_uid
        self.replica_model_uid = replica_model_uid
        self.limits = limits or RealtimeLimits()
        self.generation = uuid.uuid4().hex
        self._secret = secrets.token_bytes(32)
        self._tickets: Dict[str, float] = {}
        self._handlers: Dict[asyncio.Task, web.WebSocketResponse] = {}
        self._inference_lock = inference_lock or asyncio.Lock()
        self._admission = admission or (lambda: True)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="realtime"
        )
        self._runner: Optional[web.AppRunner] = None
        self._stop_task: Optional[asyncio.Task] = None
        self.endpoint = ""
        self.draining = False
        self.active_count = 0
        self.completed_count = 0
        self.failed_count = 0
        self.input_queue_peak = 0
        self.input_bytes_peak = 0
        self.recent_summaries: deque = deque(maxlen=8)
        self._trace: ContextVar = ContextVar("realtime_trace", default=None)
        self.capabilities = model.get_realtime_capabilities()

    async def start(self, host: str, advertised_host: Optional[str] = None) -> None:
        app = web.Application(client_max_size=self.limits.max_frame_bytes)
        app.router.add_get("/realtime", self._handle)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, 0)
        try:
            await site.start()
            addresses = self._runner.addresses
            port = addresses[0][1]
            address = advertised_host or host
            if ":" in address:
                address = f"[{address}]"
            self.endpoint = f"ws://{address}:{port}/realtime"
        except BaseException:
            await self._runner.cleanup()
            self._runner = None
            self._executor.shutdown(wait=False, cancel_futures=True)
            raise

    def describe(self) -> Dict[str, Any]:
        return {
            "model_uid": self.model_uid,
            "replica_model_uid": self.replica_model_uid,
            "runtime_generation": self.generation,
            "endpoint": self.endpoint,
            "protocol_versions": [1],
            "capabilities": self.capabilities,
            "readiness": bool(self.endpoint) and not self.draining,
            "draining": self.draining,
            "pid": os.getpid(),
            "input_bytes_peak": self.input_bytes_peak,
        }

    def issue_ticket(self, subject: str, task: str) -> Dict[str, Any]:
        if self.draining or not self.endpoint:
            raise RealtimeError("endpoint_unavailable", "Realtime endpoint is draining")
        if task not in self.capabilities.get("tasks", {}):
            raise RealtimeError("unsupported_task", "Model does not support this task")
        now = time.time()
        self._tickets = {k: v for k, v in self._tickets.items() if v > now}
        if len(self._tickets) >= 4096:
            raise RealtimeError("capacity_exceeded", "Too many outstanding handshakes")
        claims: Dict[str, Any] = {
            "session_id": uuid.uuid4().hex,
            "subject": subject,
            "task": task,
            "model_uid": self.model_uid,
            "replica_model_uid": self.replica_model_uid,
            "runtime_generation": self.generation,
            "expires": now + self.limits.ticket_ttl,
            "max_duration": self.limits.max_duration,
        }
        payload = json.dumps(claims, separators=(",", ":"), sort_keys=True).encode()
        signature = hmac.new(self._secret, payload, hashlib.sha256).hexdigest()
        token = payload.hex() + "." + signature
        # A ticket can be consumed exactly once, including failed admission.
        self._tickets[claims["session_id"]] = claims["expires"]
        return {**self.describe(), "ticket": token}

    def _consume_ticket(self, authorization: str) -> Dict[str, Any]:
        try:
            if not authorization.startswith("Bearer ") or len(authorization) > 8192:
                raise ValueError
            payload_hex, signature = authorization[7:].split(".")
            payload = bytes.fromhex(payload_hex)
            expected = hmac.new(self._secret, payload, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError
            claims = json.loads(payload)
            if (
                claims["expires"] <= time.time()
                or claims["model_uid"] != self.model_uid
                or claims["replica_model_uid"] != self.replica_model_uid
                or claims["runtime_generation"] != self.generation
                or self._tickets.pop(claims["session_id"], None) is None
            ):
                raise ValueError
            return claims
        except (ValueError, TypeError, KeyError) as exc:
            raise web.HTTPForbidden(
                text="Invalid internal realtime credential"
            ) from exc

    async def invoke(self, fn: Callable, *args, **kwargs):
        """Serialize with file inference; never release a running GPU operation."""
        waiting = time.monotonic()
        trace = self._trace.get()
        async with self._inference_lock:
            if trace is not None:
                trace.duration("lock_wait", time.monotonic() - waiting)
            if inspect.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)

            submitted = time.monotonic()

            def work():
                entered = time.monotonic()
                if trace is not None:
                    trace.duration("executor_wait", entered - submitted)
                try:
                    return fn(*args, **kwargs)
                finally:
                    if trace is not None:
                        trace.duration("invoke", time.monotonic() - entered)

            future = asyncio.get_running_loop().run_in_executor(
                self._executor, copy_context().run, work
            )
            try:
                return await asyncio.shield(future)
            except asyncio.CancelledError:
                # Threads cannot be killed safely. Retain lock/capacity until done.
                with suppress(Exception):
                    await _settle(future)
                raise

    def record_event(self, stage: str, **fields: Any) -> None:
        trace = self._trace.get()
        if trace is not None:
            trace.event(stage, **fields)

    async def _send(self, ws, data: Dict[str, Any]) -> None:
        text = json.dumps(data, ensure_ascii=False)
        if len(text.encode("utf-8")) > self.limits.max_output_bytes:
            raise RealtimeError(
                "message_too_large", "Output message exceeds limit", 1009
            )
        started = time.monotonic()
        try:
            await asyncio.wait_for(ws.send_str(text), self.limits.send_timeout)
        finally:
            trace = self._trace.get()
            if trace is not None:
                trace.duration("send", time.monotonic() - started)

    async def _handle(self, request: web.Request) -> web.WebSocketResponse:
        claims = self._consume_ticket(request.headers.get("Authorization", ""))
        if self.draining:
            raise web.HTTPServiceUnavailable(
                text="Realtime endpoint is draining",
                headers={"X-Xinference-Realtime-Error": "endpoint_unavailable"},
            )
        if self.active_count >= self.limits.max_sessions or not self._admission():
            raise web.HTTPServiceUnavailable(
                text="Realtime capacity unavailable",
                headers={"X-Xinference-Realtime-Error": "capacity_exceeded"},
            )
        # No await between the authoritative local check and increment.
        self.active_count += 1
        ws = web.WebSocketResponse(
            heartbeat=25, max_msg_size=self.limits.max_frame_bytes, compress=False
        )
        handler = asyncio.current_task()
        assert handler is not None
        self._handlers[handler] = ws
        session = None
        reason = "error"
        started = time.monotonic()
        trace = RealtimeTrace(claims["session_id"])
        trace_token = self._trace.set(trace)
        try:
            await ws.prepare(request)
            first = await asyncio.wait_for(ws.receive(), self.limits.start_timeout)
            if first.type != WSMsgType.TEXT:
                raise RealtimeError(
                    "invalid_state", "First message must be session.start"
                )
            config = validate_start(parse_event(first.data))
            if config["task"] != claims["task"]:
                raise RealtimeError(
                    "unsupported_task", "Task differs from authorization"
                )
            # Factory validates task-specific input and parameters locally.
            session = self.model.create_realtime_session(config, self)
            await self._send(
                ws,
                {
                    "event": "session.started",
                    "session_id": claims["session_id"],
                    "protocol_version": 1,
                    "task": config["task"],
                },
            )
            await asyncio.wait_for(
                self._run(ws, session, claims["session_id"]),
                self.limits.max_duration + self.limits.final_timeout,
            )
            reason = "completed"
            self.completed_count += 1
        except SessionCancelled:
            reason = "cancelled"
            if not ws.closed:
                with suppress(Exception):
                    await self._send(
                        ws,
                        {
                            "event": "session.closed",
                            "session_id": claims["session_id"],
                            "reason": reason,
                        },
                    )
        except (ConnectionError, asyncio.CancelledError):
            reason = "disconnected"
        except Exception as exc:
            self.failed_count += 1
            if isinstance(exc, RealtimeError):
                error = exc
            elif isinstance(exc, asyncio.TimeoutError):
                error = RealtimeError("timeout", "Realtime operation timed out", 1013)
            else:
                # Do not expose backend exceptions, media, text or credentials.
                logger.error("Realtime session failed (%s)", type(exc).__name__)
                error = RealtimeError(
                    "inference_failed", "Realtime inference failed", 1011
                )
            with suppress(Exception):
                await self._send(ws, error.event())
                await ws.close(code=error.close_code)
        finally:

            async def cleanup():
                observations = {}
                try:
                    if session is not None:
                        observations = session.observations()
                        await session.close(reason)
                finally:
                    try:
                        with suppress(Exception):
                            await ws.close()
                        trace.event("closed", reason=reason)
                        summary = trace.snapshot()
                        summary.update(
                            reason=reason,
                            adapter=observations,
                            replica_model_uid=self.replica_model_uid,
                            runtime_generation=self.generation,
                        )
                        self.recent_summaries.append(summary)
                        logger.info(
                            "Realtime session summary replica=%s elapsed_ms=%.1f diagnostics=%s",
                            self.replica_model_uid,
                            (time.monotonic() - started) * 1000,
                            json.dumps(summary, separators=(",", ":")),
                        )
                    finally:
                        self.active_count -= 1
                        self._handlers.pop(handler, None)

            try:
                await _settle(asyncio.create_task(cleanup()))
            finally:
                self._trace.reset(trace_token)
        return ws

    async def _run(self, ws, session, session_id: str) -> None:
        inputs: asyncio.Queue = asyncio.Queue(maxsize=self.limits.input_queue_size)
        input_ended = asyncio.Event()
        byte_budget = asyncio.Condition()
        buffered_bytes = 0

        async def enqueue(frame):
            nonlocal buffered_bytes
            size = len(frame)
            if size > self.limits.max_input_queue_bytes:
                raise RealtimeError(
                    "message_too_large", "Input byte budget exceeded", 1009
                )
            async with byte_budget:
                await byte_budget.wait_for(
                    lambda: buffered_bytes + size <= self.limits.max_input_queue_bytes
                )
                buffered_bytes += size
                self.input_bytes_peak = max(self.input_bytes_peak, buffered_bytes)
            try:
                await inputs.put(frame)
            except BaseException:
                async with byte_budget:
                    buffered_bytes -= size
                    byte_budget.notify_all()
                raise

        async def read():
            ended = False
            while True:
                message = await asyncio.wait_for(
                    ws.receive(), None if ended else self.limits.idle_timeout
                )
                if message.type == WSMsgType.BINARY:
                    if ended:
                        raise RealtimeError("invalid_state", "Media after input.end")
                    if not message.data:
                        raise RealtimeError("invalid_audio", "Empty media frame")
                    session.notify_input(len(message.data))
                    await asyncio.wait_for(
                        enqueue(message.data), self.limits.queue_timeout
                    )
                    self.input_queue_peak = max(self.input_queue_peak, inputs.qsize())
                elif message.type == WSMsgType.TEXT:
                    event = parse_event(message.data)
                    if set(event) != {"event"}:
                        raise RealtimeError(
                            "invalid_event", "Unexpected control fields"
                        )
                    if event["event"] == "session.cancel":
                        raise SessionCancelled
                    if event["event"] != "input.end" or ended:
                        raise RealtimeError("invalid_state", "Unexpected control event")
                    ended = True
                    input_ended.set()
                    self.record_event(
                        "input_end",
                        buffered_bytes=buffered_bytes,
                        queued_frames=inputs.qsize(),
                    )
                    session.request_input_end()
                    await asyncio.wait_for(inputs.put(None), self.limits.queue_timeout)
                elif message.type in (
                    WSMsgType.CLOSE,
                    WSMsgType.CLOSED,
                    WSMsgType.ERROR,
                ):
                    raise ConnectionError("Realtime peer disconnected")

        async def process():
            nonlocal buffered_bytes
            while True:
                frame = await inputs.get()
                if frame is None:
                    self.record_event("input_drained")
                    self.record_event("finish_start")
                    await asyncio.wait_for(
                        session.finish_input(), self.limits.final_timeout
                    )
                    self.record_event("finish_end")
                    return
                try:
                    await session.receive_input(frame)
                finally:
                    async with byte_budget:
                        buffered_bytes -= len(frame)
                        byte_budget.notify_all()

        async def write():
            seq = 0
            async for data in session.iter_outputs():
                seq += 1
                await self._send(
                    ws,
                    {
                        "event": "output",
                        "session_id": session_id,
                        "seq": seq,
                        "data": data,
                    },
                )
            await self._send(
                ws,
                {
                    "event": "session.closed",
                    "session_id": session_id,
                    "reason": "completed",
                },
            )
            self.record_event("outputs_sent")

        async def final_deadline():
            await input_ended.wait()
            await asyncio.sleep(self.limits.final_timeout)
            raise asyncio.TimeoutError

        reader, processor, writer, deadline = [
            asyncio.create_task(fn()) for fn in (read, process, write, final_deadline)
        ]
        tasks = {reader, processor, writer, deadline}
        pending = set(tasks)
        watchdog = None

        async def quarantine_if_stuck():
            await asyncio.sleep(self.limits.drain_timeout)
            if not processor.done():
                # Do not wait for a peer's close handshake to supervise native
                # drain. Quarantine admission, never unload active weights.
                self.draining = True
                self.record_event("native_drain_timeout")
                logger.error(
                    "Realtime native drain timed out; endpoint quarantined until controlled recovery"
                )

        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    task.result()
                if writer in done:
                    # The adapter ends outputs only after completing accepted input.
                    await processor
                    return
        except BaseException as exc:
            session.request_cancel()
            self.record_event(
                "cancel_processed"
                if isinstance(exc, SessionCancelled)
                else "abort_processed"
            )
            # Fence/cancel output before sending a terminal control message.
            for task in tasks:
                if not task.done():
                    task.cancel()
            watchdog = asyncio.create_task(quarantine_if_stuck())
            # Signal failure/cancellation before waiting for a non-interruptible
            # native call. The socket can close promptly while capacity and the
            # inference lock remain held until the operation safely settles.
            with suppress(Exception):
                if not ws.closed:
                    if isinstance(exc, SessionCancelled):
                        await self._send(
                            ws,
                            {
                                "event": "session.closed",
                                "session_id": session_id,
                                "reason": "cancelled",
                            },
                        )
                        await ws.close()
                    elif isinstance(exc, ConnectionError):
                        await ws.close(code=1001)
                    elif isinstance(exc, asyncio.CancelledError) and self.draining:
                        await ws.close(code=1012)
                    else:
                        error = (
                            exc
                            if isinstance(exc, RealtimeError)
                            else RealtimeError(
                                (
                                    "timeout"
                                    if isinstance(
                                        exc,
                                        (asyncio.TimeoutError, asyncio.CancelledError),
                                    )
                                    else "inference_failed"
                                ),
                                "Realtime operation did not complete",
                                1013,
                            )
                        )
                        await self._send(ws, error.event())
                        await ws.close(code=error.close_code)
            raise
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            settlement = asyncio.gather(*tasks, return_exceptions=True)
            try:
                await _settle(settlement)
            finally:
                if watchdog is not None:
                    watchdog.cancel()
                    await _settle(asyncio.gather(watchdog, return_exceptions=True))

    async def stop(self) -> None:
        self.draining = True
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(self._stop())
        await _settle(self._stop_task)

    async def _stop(self) -> None:
        handlers = list(self._handlers)
        if handlers:
            _, pending = await asyncio.wait(handlers, timeout=self.limits.drain_timeout)
            if pending:
                logger.warning(
                    "Realtime draining %d sessions; native work must settle before unload",
                    len(pending),
                )
            for task in pending:
                task.cancel()
            # A non-cancellable native operation must settle before weights unload.
            await asyncio.gather(*pending, return_exceptions=True)
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self.endpoint = ""
        self._tickets.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
