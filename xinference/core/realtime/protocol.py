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


"""Versioned common framing, independent of ASR payloads."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict


class RealtimeError(ValueError):
    def __init__(self, code: str, detail: str, close_code: int = 1008):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.close_code = close_code

    def __reduce__(self):
        # Connect-time errors can cross xoscar; ValueError.args alone loses code.
        return type(self), (self.code, self.detail, self.close_code)

    def event(self) -> Dict[str, Any]:
        return {"event": "error", "code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class RealtimeLimits:
    max_sessions: int = 1
    max_frame_bytes: int = 65536
    max_output_bytes: int = 2097152
    input_queue_size: int = 8
    output_queue_size: int = 8
    start_timeout: float = 10
    idle_timeout: float = 60
    queue_timeout: float = 10
    send_timeout: float = 10
    final_timeout: float = 30
    max_duration: float = 1800
    ticket_ttl: float = 30
    drain_timeout: float = 30
    # Independent byte bound, retaining the previous eight maximum-size frames.
    max_input_queue_bytes: int = 524288


def parse_event(text: str, max_bytes: int = 65536) -> Dict[str, Any]:
    if len(text.encode("utf-8")) > max_bytes:
        raise RealtimeError("message_too_large", "Control message exceeds limit", 1009)
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise RealtimeError("invalid_event", "Expected a JSON control event") from exc
    if not isinstance(value, dict) or not isinstance(value.get("event"), str):
        raise RealtimeError("invalid_event", "Expected an event object")
    return value


def validate_start(config: Dict[str, Any]) -> Dict[str, Any]:
    if set(config) - {"event", "protocol_version", "task", "input", "parameters"}:
        raise RealtimeError("invalid_config", "Unknown session configuration field")
    if config.get("event") != "session.start":
        raise RealtimeError("invalid_state", "First message must be session.start")
    if (
        type(config.get("protocol_version")) is not int
        or config["protocol_version"] != 1
    ):
        raise RealtimeError("unsupported_version", "Supported protocol version is 1")
    if not isinstance(config.get("task"), str) or not config["task"]:
        raise RealtimeError("invalid_config", "A task is required")
    if not isinstance(config.get("input"), dict) or not isinstance(
        config.get("parameters", {}), dict
    ):
        raise RealtimeError("invalid_config", "input and parameters must be objects")
    return config
