# Copyright 2022-2026 XProbe Inc.
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
import json
import os
from json import JSONDecodeError
from typing import Any, Dict, List, Optional

from ..constants import XINFERENCE_AUTOSTART_CONFIG_FILE
from .utils import is_valid_model_uid

AUTOSTART_CONFIG_VERSION = 1
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


def get_autostart_config_path(path: Optional[str] = None) -> str:
    return path or XINFERENCE_AUTOSTART_CONFIG_FILE


def empty_autostart_config() -> Dict[str, Any]:
    return {
        "version": AUTOSTART_CONFIG_VERSION,
        "concurrency": 1,
        "models": [],
    }


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


def normalize_autostart_config(config: Any) -> Dict[str, Any]:
    if config is None:
        return empty_autostart_config()
    if isinstance(config, list):
        config = {"models": config}
    if not isinstance(config, dict):
        raise ValueError("Autostart config must be a JSON object.")

    try:
        concurrency = int(config.get("concurrency", 1))
    except (TypeError, ValueError):
        raise ValueError("Autostart config concurrency must be an integer.")
    if concurrency < 1:
        raise ValueError("Autostart config concurrency must be greater than 0.")

    raw_models = config.get("models", [])
    if raw_models is None:
        raw_models = []
    if not isinstance(raw_models, list):
        raise ValueError("Autostart config models must be a list.")

    models: List[Dict[str, Any]] = []
    seen_model_uids = set()
    for raw_entry in raw_models:
        entry = normalize_autostart_model_entry(raw_entry)
        model_uid = entry["launch"]["model_uid"]
        if model_uid in seen_model_uids:
            raise ValueError(f"Duplicate autostart model_uid: {model_uid}")
        seen_model_uids.add(model_uid)
        models.append(entry)

    return {
        "version": AUTOSTART_CONFIG_VERSION,
        "concurrency": concurrency,
        "models": models,
    }


def load_autostart_config(path: Optional[str] = None) -> Dict[str, Any]:
    config_path = get_autostart_config_path(path)
    if not os.path.exists(config_path):
        return empty_autostart_config()

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except JSONDecodeError as e:
            raise ValueError(f"Invalid autostart config JSON: {e}") from e
        return normalize_autostart_config(data)


def save_autostart_config(
    config: Dict[str, Any], path: Optional[str] = None
) -> Dict[str, Any]:
    normalized = normalize_autostart_config(config)
    config_path = get_autostart_config_path(path)
    config_dir = os.path.dirname(config_path)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)
    tmp_path = f"{config_path}.tmp"
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    if hasattr(os, "fchmod"):
        os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, config_path)
    try:
        os.chmod(config_path, 0o600)
    except OSError:
        pass
    return normalized


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


def redact_autostart_config(config: Dict[str, Any]) -> Dict[str, Any]:
    redacted = copy.deepcopy(config)
    for entry in redacted.get("models", []):
        launch = entry.get("launch")
        if isinstance(launch, dict):
            entry["launch"] = redact_launch_payload(launch)
    return redacted


def upsert_autostart_model_entry(
    config: Dict[str, Any], entry: Dict[str, Any]
) -> Dict[str, Any]:
    normalized_config = normalize_autostart_config(config)
    normalized_entry = normalize_autostart_model_entry(entry)
    model_uid = normalized_entry["launch"]["model_uid"]

    models = []
    replaced = False
    for item in normalized_config["models"]:
        if item["launch"]["model_uid"] == model_uid:
            models.append(normalized_entry)
            replaced = True
        else:
            models.append(item)
    if not replaced:
        models.append(normalized_entry)

    normalized_config["models"] = models
    return normalize_autostart_config(normalized_config)


def remove_autostart_model_entry(
    config: Dict[str, Any], model_uid: str
) -> Dict[str, Any]:
    normalized_config = normalize_autostart_config(config)
    normalized_config["models"] = [
        item
        for item in normalized_config["models"]
        if item["launch"]["model_uid"] != model_uid
    ]
    return normalized_config
