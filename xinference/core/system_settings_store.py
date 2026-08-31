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

"""File-backed system settings used by model download processes."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

logger = logging.getLogger(__name__)

SYSTEM_SETTINGS_VERSION = 1
TOKEN_MASK = "********"
DOWNLOAD_SOURCES = {
    "auto",
    "huggingface",
    "modelscope",
    "openmind_hub",
    "csghub",
}

ENV_MODEL_SOURCE = "XINFERENCE_MODEL_SRC"
ENV_HF_ENDPOINT = "HF_ENDPOINT"
ENV_HF_TOKEN = "HUGGING_FACE_HUB_TOKEN"
ENV_HF_TOKEN_ALIAS = "HF_TOKEN"
ENV_PIP_INDEX_URL = "PIP_INDEX_URL"
ENV_DOWNLOAD_MAX_ATTEMPTS = "XINFERENCE_DOWNLOAD_MAX_ATTEMPTS"
ENV_HUB_DETECT_TIMEOUT = "XINFERENCE_HUB_DETECT_TIMEOUT"
ENV_MODEL_DOWNLOAD_WORKERS = "XINFERENCE_MODEL_DOWNLOAD_WORKERS"


@dataclass(frozen=True)
class SystemSettings:
    download_source: str
    hf_endpoint: str
    hf_token: str
    pip_index_url: str
    download_max_attempts: int
    hub_detect_timeout: float
    model_download_workers: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SystemSettings":
        fields = set(cls.__dataclass_fields__)
        missing = fields.difference(data)
        unknown = set(data).difference(fields)
        if missing:
            raise ValueError(f"Missing system setting fields: {sorted(missing)}")
        if unknown:
            raise ValueError(f"Unknown system setting fields: {sorted(unknown)}")

        download_source = data["download_source"]
        if not isinstance(download_source, str):
            raise ValueError("download_source must be a string")
        download_source = download_source.strip().lower()
        if download_source not in DOWNLOAD_SOURCES:
            raise ValueError(
                "download_source must be one of: auto, huggingface, modelscope, "
                "openmind_hub, csghub"
            )

        strings: Dict[str, str] = {}
        for field in ("hf_endpoint", "hf_token", "pip_index_url"):
            value = data[field]
            if not isinstance(value, str):
                raise ValueError(f"{field} must be a string")
            strings[field] = value.strip()

        download_max_attempts = data["download_max_attempts"]
        if (
            isinstance(download_max_attempts, bool)
            or not isinstance(download_max_attempts, int)
            or download_max_attempts < 1
        ):
            raise ValueError("download_max_attempts must be an integer greater than 0")

        hub_detect_timeout = data["hub_detect_timeout"]
        if isinstance(hub_detect_timeout, bool) or not isinstance(
            hub_detect_timeout, (int, float)
        ):
            raise ValueError("hub_detect_timeout must be a number greater than 0")
        hub_detect_timeout = float(hub_detect_timeout)
        if not math.isfinite(hub_detect_timeout) or hub_detect_timeout <= 0:
            raise ValueError(
                "hub_detect_timeout must be a finite number greater than 0"
            )

        model_download_workers = data["model_download_workers"]
        if (
            isinstance(model_download_workers, bool)
            or not isinstance(model_download_workers, int)
            or model_download_workers < 1
        ):
            raise ValueError("model_download_workers must be an integer greater than 0")

        return cls(
            download_source=download_source,
            hf_endpoint=strings["hf_endpoint"],
            hf_token=strings["hf_token"],
            pip_index_url=strings["pip_index_url"],
            download_max_attempts=download_max_attempts,
            hub_detect_timeout=hub_detect_timeout,
            model_download_workers=model_download_workers,
        )


def apply_system_settings(
    settings: SystemSettings,
    environ: Optional[MutableMapping[str, str]] = None,
) -> None:
    """Apply settings to future download work in the current process."""
    target_environ = environ if environ is not None else os.environ

    def set_optional(name: str, value: str) -> None:
        if value:
            target_environ[name] = value
        else:
            target_environ.pop(name, None)

    target_environ[ENV_MODEL_SOURCE] = settings.download_source
    set_optional(ENV_HF_ENDPOINT, settings.hf_endpoint)
    set_optional(ENV_HF_TOKEN, settings.hf_token)
    set_optional(ENV_HF_TOKEN_ALIAS, settings.hf_token)
    set_optional(ENV_PIP_INDEX_URL, settings.pip_index_url)
    target_environ[ENV_DOWNLOAD_MAX_ATTEMPTS] = str(settings.download_max_attempts)
    target_environ[ENV_HUB_DETECT_TIMEOUT] = str(settings.hub_detect_timeout)
    target_environ[ENV_MODEL_DOWNLOAD_WORKERS] = str(settings.model_download_workers)

    constants = sys.modules.get("xinference.constants")
    if constants is not None:
        setattr(
            constants,
            "XINFERENCE_DOWNLOAD_MAX_ATTEMPTS",
            settings.download_max_attempts,
        )
        setattr(
            constants,
            "XINFERENCE_HUB_DETECT_TIMEOUT",
            settings.hub_detect_timeout,
        )
        setattr(
            constants,
            "XINFERENCE_MODEL_DOWNLOAD_WORKERS",
            settings.model_download_workers,
        )

    model_utils = sys.modules.get("xinference.model.utils")
    if model_utils is not None:
        setattr(
            model_utils,
            "XINFERENCE_DOWNLOAD_MAX_ATTEMPTS",
            settings.download_max_attempts,
        )
        setattr(
            model_utils,
            "XINFERENCE_HUB_DETECT_TIMEOUT",
            settings.hub_detect_timeout,
        )
        setattr(
            model_utils,
            "XINFERENCE_MODEL_DOWNLOAD_WORKERS",
            settings.model_download_workers,
        )
        detect_lock = getattr(model_utils, "_auto_detect_hub_lock", None)
        if detect_lock is None:
            setattr(model_utils, "_auto_detected_hub", None)
        else:
            with detect_lock:
                setattr(model_utils, "_auto_detected_hub", None)

    worker = sys.modules.get("xinference.core.worker")
    if worker is not None:
        setattr(
            worker,
            "XINFERENCE_MODEL_DOWNLOAD_WORKERS",
            settings.model_download_workers,
        )

    hf_constants = sys.modules.get("huggingface_hub.constants")
    if hf_constants is not None:
        endpoint = settings.hf_endpoint.rstrip("/")
        if not endpoint:
            default_name = (
                "_HF_DEFAULT_STAGING_ENDPOINT"
                if getattr(hf_constants, "_staging_mode", False)
                else "_HF_DEFAULT_ENDPOINT"
            )
            endpoint = getattr(
                hf_constants,
                default_name,
                "https://huggingface.co",
            )
        url_template = endpoint + "/{repo_id}/resolve/{revision}/{filename}"
        for module_name, module in tuple(sys.modules.items()):
            if module is None or not (
                module_name == "huggingface_hub"
                or module_name.startswith("huggingface_hub.")
            ):
                continue
            if hasattr(module, "ENDPOINT"):
                setattr(module, "ENDPOINT", endpoint)
            if hasattr(module, "HUGGINGFACE_CO_URL_TEMPLATE"):
                setattr(module, "HUGGINGFACE_CO_URL_TEMPLATE", url_template)


class SystemSettingsStore:
    """Keep an immutable startup baseline and an optional saved full snapshot."""

    def __init__(
        self,
        path: str,
        environ: Optional[MutableMapping[str, str]] = None,
    ) -> None:
        self._path = Path(path)
        self._environ = environ if environ is not None else os.environ
        self._lock = threading.RLock()
        self._startup = self._settings_from_environment(dict(self._environ))
        self._current = self._load_saved() or self._startup

    @property
    def path(self) -> str:
        return str(self._path)

    @staticmethod
    def _parse_positive_int(environ: Mapping[str, str], name: str, default: int) -> int:
        raw = environ.get(name)
        if raw is None or raw == "":
            return default
        try:
            value = int(raw)
        except (TypeError, ValueError):
            logger.warning("Invalid %s value; using default %s", name, default)
            return default
        if value < 1:
            logger.warning("Invalid %s value; using default %s", name, default)
            return default
        return value

    @staticmethod
    def _parse_positive_float(
        environ: Mapping[str, str], name: str, default: float
    ) -> float:
        raw = environ.get(name)
        if raw is None or raw == "":
            return default
        try:
            value = float(raw)
        except (TypeError, ValueError):
            logger.warning("Invalid %s value; using default %s", name, default)
            return default
        if not math.isfinite(value) or value <= 0:
            logger.warning("Invalid %s value; using default %s", name, default)
            return default
        return value

    @classmethod
    def _settings_from_environment(cls, environ: Mapping[str, str]) -> SystemSettings:
        download_source = environ.get(ENV_MODEL_SOURCE, "auto").strip().lower()
        if download_source not in DOWNLOAD_SOURCES:
            logger.warning("Invalid %s value; using default auto", ENV_MODEL_SOURCE)
            download_source = "auto"

        return SystemSettings(
            download_source=download_source,
            hf_endpoint=environ.get(ENV_HF_ENDPOINT, "").strip(),
            hf_token=(
                environ.get(ENV_HF_TOKEN) or environ.get(ENV_HF_TOKEN_ALIAS) or ""
            ).strip(),
            pip_index_url=environ.get(ENV_PIP_INDEX_URL, "").strip(),
            download_max_attempts=cls._parse_positive_int(
                environ, ENV_DOWNLOAD_MAX_ATTEMPTS, 3
            ),
            hub_detect_timeout=cls._parse_positive_float(
                environ, ENV_HUB_DETECT_TIMEOUT, 3.0
            ),
            model_download_workers=cls._parse_positive_int(
                environ, ENV_MODEL_DOWNLOAD_WORKERS, 8
            ),
        )

    def _load_saved(self) -> Optional[SystemSettings]:
        if not self._path.is_file():
            return None
        try:
            with self._path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            if not isinstance(payload, dict):
                raise ValueError("root must be an object")
            if payload.get("version") != SYSTEM_SETTINGS_VERSION:
                raise ValueError("unsupported version")
            settings = payload.get("settings")
            if not isinstance(settings, dict):
                raise ValueError("settings must be an object")
            startup_settings = self._startup.to_dict()
            unknown = set(settings).difference(startup_settings)
            if unknown:
                raise ValueError(f"Unknown system setting fields: {sorted(unknown)}")
            startup_settings.update(settings)
            return SystemSettings.from_dict(startup_settings)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning(
                "Ignoring invalid system settings file %s: %s", self._path, exc
            )
            return None

    @staticmethod
    def _masked_token(token: str) -> str:
        if not token:
            return ""
        if len(token) <= 8:
            return TOKEN_MASK
        return f"{token[:4]}{TOKEN_MASK}{token[-4:]}"

    @classmethod
    def public_dict(cls, settings: SystemSettings) -> Dict[str, Any]:
        data = settings.to_dict()
        data["hf_token"] = cls._masked_token(settings.hf_token)
        return data

    def get(self) -> SystemSettings:
        with self._lock:
            return self._current

    def get_startup(self) -> SystemSettings:
        return self._startup

    def get_public(self) -> Dict[str, Any]:
        return self.public_dict(self.get())

    def _write(self, settings: SystemSettings) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = file.name
                json.dump(
                    {
                        "version": SYSTEM_SETTINGS_VERSION,
                        "settings": settings.to_dict(),
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, self._path)
            temporary_path = None
        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass

    def save_public(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        """Persist and immediately apply a full snapshot."""
        incoming_token = data.get("hf_token")
        if not isinstance(incoming_token, str):
            raise ValueError("hf_token must be a string")

        with self._lock:
            actual_token = (
                self._current.hf_token if "*" in incoming_token else incoming_token
            )
            full_data = dict(data)
            full_data["hf_token"] = actual_token
            settings = SystemSettings.from_dict(full_data)
            self._write(settings)
            self._current = settings
            self.apply_to_environment(settings)
            return self.public_dict(settings)

    def reset(self) -> Dict[str, Any]:
        """Remove the snapshot and restore the page state to the startup baseline."""
        with self._lock:
            try:
                self._path.unlink()
            except FileNotFoundError:
                pass
            self._current = self._startup
            self.apply_to_environment(self._startup)
            return self.public_dict(self._startup)

    def apply_to_environment(self, settings: Optional[SystemSettings] = None) -> None:
        apply_system_settings(settings or self.get(), self._environ)


def get_system_settings_from_environment(
    environ: Optional[Mapping[str, str]] = None,
) -> SystemSettings:
    source = environ if environ is not None else os.environ
    return SystemSettingsStore._settings_from_environment(dict(source))


_store: Optional[SystemSettingsStore] = None
_store_lock = threading.Lock()


def get_system_settings_store(path: str) -> SystemSettingsStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = SystemSettingsStore(path)
        elif _store.path != str(Path(path)):
            raise RuntimeError(
                f"System settings store already initialized at {_store.path}"
            )
        return _store


def initialize_system_settings(path: str) -> SystemSettingsStore:
    store = get_system_settings_store(path)
    store.apply_to_environment()
    return store
