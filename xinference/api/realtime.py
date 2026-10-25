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


"""Common realtime gateway: authenticate, route once, then proxy a socket."""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress

import aiohttp
from fastapi import HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.security import SecurityScopes

from ..core.realtime.protocol import (
    RealtimeError,
    RealtimeLimits,
    parse_event,
    validate_start,
)


async def realtime_session(api, websocket: WebSocket, model_uid: str) -> None:
    limits = RealtimeLimits()
    tasks = []
    accepted = False
    try:
        origin = websocket.headers.get("origin")
        allowed = {
            x.strip()
            for x in os.getenv("XINFERENCE_REALTIME_ALLOWED_ORIGINS", "").split(",")
            if x.strip()
        }
        if origin and origin not in allowed:
            raise HTTPException(403, "Realtime Origin is not allowed")
        # Reuse the complete HTTP credential policy (scope, live user state,
        # expiry, bans), not a separate API-key-only validator.
        scope = dict(websocket.scope, type="http", method="GET")
        request = Request(scope)
        subject = "anonymous"
        if api.is_authenticated():
            authorization = websocket.headers.get("authorization", "")
            token = (
                authorization[7:].strip()
                if authorization.lower().startswith("bearer ")
                else None
            )
            user = await api._auth_service(
                request=request,
                security_scopes=SecurityScopes(["models:read"]),
                token=token,
            )
            subject = str(user["id"])
        supervisor = await api._get_supervisor_ref()
        # Metadata lookup does not select a replica; model-type key permissions
        # must be checked as well as UID permissions before routing.
        description = await supervisor.describe_model(model_uid)
        api._check_model_access(request, model_uid, description.get("model_type"))
        await websocket.accept()
        accepted = True
        message = await asyncio.wait_for(websocket.receive(), limits.start_timeout)
        if message["type"] == "websocket.disconnect":
            return
        if message.get("text") is None:
            raise RealtimeError("invalid_state", "First message must be session.start")
        config = validate_start(parse_event(message["text"]))
        model = await supervisor.get_model(model_uid)
        # This is the last control-plane call for this session. No media or
        # output polling ever goes through a model Actor.
        descriptor = await model.get_realtime_endpoint(subject, config["task"])
        if not descriptor.get("readiness") or descriptor.get("model_uid") != model_uid:
            raise RealtimeError(
                "endpoint_unavailable", "Realtime endpoint is not ready", 1013
            )
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=limits.start_timeout)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as client:
            async with client.ws_connect(
                descriptor["endpoint"],
                headers={"Authorization": "Bearer " + descriptor["ticket"]},
                heartbeat=25,
                max_msg_size=limits.max_output_bytes,
                compress=0,
            ) as backend:
                await backend.send_json(config)

                async def upload():
                    while True:
                        frame = await websocket.receive()
                        if frame["type"] == "websocket.disconnect":
                            return
                        if frame.get("bytes") is not None:
                            data = frame["bytes"]
                            if len(data) > limits.max_frame_bytes:
                                raise RealtimeError(
                                    "message_too_large",
                                    "Media frame exceeds limit",
                                    1009,
                                )
                            await asyncio.wait_for(
                                backend.send_bytes(data), limits.send_timeout
                            )
                        elif frame.get("text") is not None:
                            data = frame["text"]
                            parse_event(data, limits.max_frame_bytes)
                            await asyncio.wait_for(
                                backend.send_str(data), limits.send_timeout
                            )

                async def download():
                    async for frame in backend:
                        if frame.type == aiohttp.WSMsgType.TEXT:
                            # The gateway forwards task results without looking
                            # at ASR fields such as partial/sentences/is_final.
                            await asyncio.wait_for(
                                websocket.send_text(frame.data), limits.send_timeout
                            )
                        elif frame.type == aiohttp.WSMsgType.BINARY:
                            await asyncio.wait_for(
                                websocket.send_bytes(frame.data), limits.send_timeout
                            )
                        elif frame.type == aiohttp.WSMsgType.ERROR:
                            raise RealtimeError(
                                "endpoint_unavailable",
                                "Internal realtime connection failed",
                                1011,
                            )
                    await websocket.close(code=backend.close_code or 1011)

                tasks = [asyncio.create_task(upload()), asyncio.create_task(download())]
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
    except HTTPException:
        # Before upgrade, ASGI close rejects the handshake. Never send an
        # HTTP-only Response via a WebSocket route.
        with suppress(Exception):
            await websocket.close(code=1008)
    except (WebSocketDisconnect, ConnectionError):
        pass
    except aiohttp.WSServerHandshakeError as exc:
        with suppress(Exception):
            code = "endpoint_unavailable"
            if (
                exc.headers
                and exc.headers.get("X-Xinference-Realtime-Error")
                == "capacity_exceeded"
            ):
                code = "capacity_exceeded"
            if accepted:
                await asyncio.wait_for(
                    websocket.send_json(
                        RealtimeError(code, "Realtime admission failed", 1013).event()
                    ),
                    limits.send_timeout,
                )
            await websocket.close(code=1013)
    except Exception as exc:
        error = (
            exc
            if isinstance(exc, RealtimeError)
            else RealtimeError(
                "endpoint_unavailable", "Realtime session could not be completed", 1011
            )
        )
        with suppress(Exception):
            if accepted:
                await asyncio.wait_for(
                    websocket.send_json(error.event()), limits.send_timeout
                )
            await websocket.close(code=error.close_code)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        with suppress(Exception):
            await websocket.close()
