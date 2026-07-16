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

import copy
from typing import Any, Dict, List

from .utils import is_valid_model_uid

DEFAULT_PRIORITY = 100
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_INTERVAL_SECONDS = 30
REDACTED_VALUE = "******"
SENSITIVE_KEYWORDS = (
    "token",
    "secret",
    "password",
    "passwd",
    "pwd",
    "api_key",
    "apikey",
    "access_key",
    "credential",
)


def normalize_launch_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Autostart launch config must be a JSON object.")

    launch = {key: value for key, value in payload.items() if value is not None}
    model_name = launch.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Autostart launch config requires a non-empty model_name.")

    model_uid = launch.get("model_uid") or model_name
    if not isinstance(model_uid, str) or not is_valid_model_uid(model_uid):
        raise ValueError(
            "Autostart launch config requires a non-empty model_uid up to 100 characters."
        )

    launch["model_name"] = model_name.strip()
    launch["model_uid"] = model_uid.strip()
    launch.setdefault("model_type", "LLM")
    launch.pop("wait_ready", None)
    return launch


def normalize_autostart_model_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("Autostart model entry must be a JSON object.")

    raw_launch = entry.get("launch")
    if raw_launch is None and "model_name" in entry:
        raw_launch = {
            key: value
            for key, value in entry.items()
            if key
            not in {
                "enabled",
                "priority",
                "max_retries",
                "retry_interval_seconds",
                "state",
            }
        }
    if raw_launch is None:
        raise ValueError("Autostart model entry requires a launch object.")

    try:
        priority = int(entry.get("priority", DEFAULT_PRIORITY))
    except (TypeError, ValueError):
        raise ValueError("Autostart model priority must be an integer.")

    try:
        max_retries = int(entry.get("max_retries", DEFAULT_MAX_RETRIES))
    except (TypeError, ValueError):
        raise ValueError("Autostart model max_retries must be an integer.")

    try:
        retry_interval_seconds = int(
            entry.get("retry_interval_seconds", DEFAULT_RETRY_INTERVAL_SECONDS)
        )
    except (TypeError, ValueError):
        raise ValueError("Autostart model retry_interval_seconds must be an integer.")

    if max_retries < 1:
        raise ValueError("Autostart model max_retries must be greater than 0.")
    if retry_interval_seconds < 1:
        raise ValueError(
            "Autostart model retry_interval_seconds must be greater than 0."
        )

    return {
        "enabled": bool(entry.get("enabled", True)),
        "priority": priority,
        "max_retries": max_retries,
        "retry_interval_seconds": retry_interval_seconds,
        "launch": normalize_launch_payload(raw_launch),
    }


def _is_sensitive_key(key: str) -> bool:
    normalized_key = key.lower().replace("-", "_")
    return any(keyword in normalized_key for keyword in SENSITIVE_KEYWORDS)


def _redact_sensitive_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                REDACTED_VALUE
                if _is_sensitive_key(str(key))
                else _redact_sensitive_values(val)
            )
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_values(item) for item in value]
    return value


def redact_launch_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    redacted = _redact_sensitive_values(copy.deepcopy(payload))
    envs = redacted.get("envs")
    if isinstance(envs, dict):
        redacted["envs"] = {key: REDACTED_VALUE for key in envs}
    return redacted


def redact_autostart_model_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    redacted = copy.deepcopy(entry)
    launch = redacted.get("launch")
    if isinstance(launch, dict):
        redacted["launch"] = redact_launch_payload(launch)
    return redacted


def redact_autostart_model_entries(
    entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [redact_autostart_model_entry(entry) for entry in entries]
