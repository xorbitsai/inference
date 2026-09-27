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

"""File-backed OpenTelemetry settings."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

from .system_settings_store import (
    SYSTEM_SETTINGS_VERSION,
    get_system_settings_file_lock,
    load_system_settings_document,
    load_system_settings_document_for_update,
    write_or_remove_system_settings_document,
    write_system_settings_document,
)

logger = logging.getLogger(__name__)

SECRET_MASK = "********"
OTEL_PROTOCOLS = {"http/protobuf", "grpc"}

ENV_ENABLE_OTEL = "XINFERENCE_ENABLE_OTEL"
ENV_OTLP_BASE_ENDPOINT = "XINFERENCE_OTLP_BASE_ENDPOINT"
ENV_OTLP_TRACE_ENDPOINT = "XINFERENCE_OTLP_TRACE_ENDPOINT"
ENV_OTLP_METRIC_ENDPOINT = "XINFERENCE_OTLP_METRIC_ENDPOINT"
ENV_OTLP_API_KEY = "XINFERENCE_OTLP_API_KEY"
ENV_OTEL_EXPORTER_OTLP_PROTOCOL = "XINFERENCE_OTEL_EXPORTER_OTLP_PROTOCOL"
ENV_OTEL_SAMPLING_RATE = "XINFERENCE_OTEL_SAMPLING_RATE"
ENV_OTEL_BATCH_EXPORT_SCHEDULE_DELAY = "XINFERENCE_OTEL_BATCH_EXPORT_SCHEDULE_DELAY"
ENV_OTEL_MAX_QUEUE_SIZE = "XINFERENCE_OTEL_MAX_QUEUE_SIZE"
ENV_OTEL_MAX_EXPORT_BATCH_SIZE = "XINFERENCE_OTEL_MAX_EXPORT_BATCH_SIZE"
ENV_OTEL_METRIC_EXPORT_INTERVAL = "XINFERENCE_OTEL_METRIC_EXPORT_INTERVAL"
ENV_OTEL_BATCH_EXPORT_TIMEOUT = "XINFERENCE_OTEL_BATCH_EXPORT_TIMEOUT"
ENV_OTEL_METRIC_EXPORT_TIMEOUT = "XINFERENCE_OTEL_METRIC_EXPORT_TIMEOUT"


@dataclass(frozen=True)
class OpenTelemetrySettings:
    enable_otel: bool
    otlp_base_endpoint: str
    otlp_trace_endpoint: str
    otlp_metric_endpoint: str
    otlp_api_key: str
    otel_exporter_otlp_protocol: str
    otel_sampling_rate: float
    otel_batch_export_schedule_delay: int
    otel_max_queue_size: int
    otel_max_export_batch_size: int
    otel_metric_export_interval: int
    otel_batch_export_timeout: int
    otel_metric_export_timeout: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OpenTelemetrySettings":
        fields = set(cls.__dataclass_fields__)
        missing = fields.difference(data)
        unknown = set(data).difference(fields)
        if missing:
            raise ValueError(f"Missing OpenTelemetry setting fields: {sorted(missing)}")
        if unknown:
            raise ValueError(f"Unknown OpenTelemetry setting fields: {sorted(unknown)}")

        enable_otel = data["enable_otel"]
        if not isinstance(enable_otel, bool):
            raise ValueError("enable_otel must be a boolean")

        strings: Dict[str, str] = {}
        for field in (
            "otlp_base_endpoint",
            "otlp_trace_endpoint",
            "otlp_metric_endpoint",
            "otlp_api_key",
        ):
            value = data[field]
            if not isinstance(value, str):
                raise ValueError(f"{field} must be a string")
            strings[field] = value.strip()

        strings["otlp_base_endpoint"] = strings["otlp_base_endpoint"].rstrip("/")
        if not strings["otlp_base_endpoint"]:
            raise ValueError("otlp_base_endpoint must not be empty")

        protocol = data["otel_exporter_otlp_protocol"]
        if not isinstance(protocol, str):
            raise ValueError("otel_exporter_otlp_protocol must be a string")
        protocol = protocol.strip().lower()
        if protocol not in OTEL_PROTOCOLS:
            raise ValueError(
                "otel_exporter_otlp_protocol must be one of: http/protobuf, grpc"
            )

        sampling_rate = data["otel_sampling_rate"]
        if isinstance(sampling_rate, bool) or not isinstance(
            sampling_rate, (int, float)
        ):
            raise ValueError("otel_sampling_rate must be a number between 0 and 1")
        sampling_rate = float(sampling_rate)
        if not math.isfinite(sampling_rate) or not 0 <= sampling_rate <= 1:
            raise ValueError(
                "otel_sampling_rate must be a finite number between 0 and 1"
            )

        positive_ints: Dict[str, int] = {}
        for field in (
            "otel_batch_export_schedule_delay",
            "otel_max_queue_size",
            "otel_max_export_batch_size",
            "otel_metric_export_interval",
            "otel_batch_export_timeout",
            "otel_metric_export_timeout",
        ):
            value = data[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be an integer greater than 0")
            positive_ints[field] = value

        if (
            positive_ints["otel_max_export_batch_size"]
            > positive_ints["otel_max_queue_size"]
        ):
            raise ValueError(
                "otel_max_export_batch_size must not exceed otel_max_queue_size"
            )

        return cls(
            enable_otel=enable_otel,
            otlp_base_endpoint=strings["otlp_base_endpoint"],
            otlp_trace_endpoint=strings["otlp_trace_endpoint"],
            otlp_metric_endpoint=strings["otlp_metric_endpoint"],
            otlp_api_key=strings["otlp_api_key"],
            otel_exporter_otlp_protocol=protocol,
            otel_sampling_rate=sampling_rate,
            otel_batch_export_schedule_delay=positive_ints[
                "otel_batch_export_schedule_delay"
            ],
            otel_max_queue_size=positive_ints["otel_max_queue_size"],
            otel_max_export_batch_size=positive_ints["otel_max_export_batch_size"],
            otel_metric_export_interval=positive_ints["otel_metric_export_interval"],
            otel_batch_export_timeout=positive_ints["otel_batch_export_timeout"],
            otel_metric_export_timeout=positive_ints["otel_metric_export_timeout"],
        )


@dataclass(frozen=True)
class ObservabilitySettings:
    opentelemetry: OpenTelemetrySettings

    def to_dict(self) -> Dict[str, Any]:
        return {"opentelemetry": self.opentelemetry.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObservabilitySettings":
        unknown = set(data).difference({"opentelemetry"})
        if unknown:
            raise ValueError(f"Unknown observability providers: {sorted(unknown)}")
        opentelemetry = data.get("opentelemetry")
        if not isinstance(opentelemetry, Mapping):
            raise ValueError("opentelemetry must be an object")
        return cls(opentelemetry=OpenTelemetrySettings.from_dict(opentelemetry))


def apply_observability_settings(
    settings: ObservabilitySettings,
    environ: Optional[MutableMapping[str, str]] = None,
) -> None:
    """Apply settings for the next OpenTelemetry initialization."""
    target_environ = environ if environ is not None else os.environ
    opentelemetry = settings.opentelemetry

    def set_optional(name: str, value: str) -> None:
        if value:
            target_environ[name] = value
        else:
            target_environ.pop(name, None)

    target_environ[ENV_ENABLE_OTEL] = str(opentelemetry.enable_otel).lower()
    target_environ[ENV_OTLP_BASE_ENDPOINT] = opentelemetry.otlp_base_endpoint
    set_optional(ENV_OTLP_TRACE_ENDPOINT, opentelemetry.otlp_trace_endpoint)
    set_optional(ENV_OTLP_METRIC_ENDPOINT, opentelemetry.otlp_metric_endpoint)
    set_optional(ENV_OTLP_API_KEY, opentelemetry.otlp_api_key)
    target_environ[ENV_OTEL_EXPORTER_OTLP_PROTOCOL] = (
        opentelemetry.otel_exporter_otlp_protocol
    )
    target_environ[ENV_OTEL_SAMPLING_RATE] = str(opentelemetry.otel_sampling_rate)
    target_environ[ENV_OTEL_BATCH_EXPORT_SCHEDULE_DELAY] = str(
        opentelemetry.otel_batch_export_schedule_delay
    )
    target_environ[ENV_OTEL_MAX_QUEUE_SIZE] = str(opentelemetry.otel_max_queue_size)
    target_environ[ENV_OTEL_MAX_EXPORT_BATCH_SIZE] = str(
        opentelemetry.otel_max_export_batch_size
    )
    target_environ[ENV_OTEL_METRIC_EXPORT_INTERVAL] = str(
        opentelemetry.otel_metric_export_interval
    )
    target_environ[ENV_OTEL_BATCH_EXPORT_TIMEOUT] = str(
        opentelemetry.otel_batch_export_timeout
    )
    target_environ[ENV_OTEL_METRIC_EXPORT_TIMEOUT] = str(
        opentelemetry.otel_metric_export_timeout
    )

    constants = sys.modules.get("xinference.constants")
    if constants is None:
        return

    default_trace_endpoint = f"{opentelemetry.otlp_base_endpoint}/v1/traces"
    default_metric_endpoint = f"{opentelemetry.otlp_base_endpoint}/v1/metrics"
    setattr(constants, "XINFERENCE_ENABLE_OTEL", opentelemetry.enable_otel)
    setattr(
        constants,
        "_XINFERENCE_OTLP_BASE_ENDPOINT",
        opentelemetry.otlp_base_endpoint,
    )
    setattr(
        constants,
        "XINFERENCE_OTLP_TRACE_ENDPOINT",
        opentelemetry.otlp_trace_endpoint or default_trace_endpoint,
    )
    setattr(
        constants,
        "XINFERENCE_OTLP_METRIC_ENDPOINT",
        opentelemetry.otlp_metric_endpoint or default_metric_endpoint,
    )
    setattr(constants, "XINFERENCE_OTLP_API_KEY", opentelemetry.otlp_api_key)
    setattr(
        constants,
        "XINFERENCE_OTEL_EXPORTER_OTLP_PROTOCOL",
        opentelemetry.otel_exporter_otlp_protocol,
    )
    setattr(
        constants,
        "XINFERENCE_OTEL_SAMPLING_RATE",
        opentelemetry.otel_sampling_rate,
    )
    setattr(
        constants,
        "XINFERENCE_OTEL_BATCH_EXPORT_SCHEDULE_DELAY",
        opentelemetry.otel_batch_export_schedule_delay,
    )
    setattr(
        constants,
        "XINFERENCE_OTEL_MAX_QUEUE_SIZE",
        opentelemetry.otel_max_queue_size,
    )
    setattr(
        constants,
        "XINFERENCE_OTEL_MAX_EXPORT_BATCH_SIZE",
        opentelemetry.otel_max_export_batch_size,
    )
    setattr(
        constants,
        "XINFERENCE_OTEL_METRIC_EXPORT_INTERVAL",
        opentelemetry.otel_metric_export_interval,
    )
    setattr(
        constants,
        "XINFERENCE_OTEL_BATCH_EXPORT_TIMEOUT",
        opentelemetry.otel_batch_export_timeout,
    )
    setattr(
        constants,
        "XINFERENCE_OTEL_METRIC_EXPORT_TIMEOUT",
        opentelemetry.otel_metric_export_timeout,
    )


class ObservabilitySettingsStore:
    """Keep an immutable startup baseline and an optional saved snapshot."""

    def __init__(
        self,
        path: str,
        environ: Optional[MutableMapping[str, str]] = None,
    ) -> None:
        self._path = Path(path)
        self._environ = environ if environ is not None else os.environ
        self._lock = get_system_settings_file_lock(self._path)
        self._startup = self._settings_from_environment(dict(self._environ))
        self._current = self._load_saved() or self._startup

    @property
    def path(self) -> str:
        return str(self._path)

    @staticmethod
    def _parse_bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
        raw = environ.get(name)
        if raw is None or raw == "":
            return default
        normalized = raw.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        logger.warning("Invalid %s value; using default %s", name, default)
        return default

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
    def _parse_sampling_rate(
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
        if not math.isfinite(value) or not 0 <= value <= 1:
            logger.warning("Invalid %s value; using default %s", name, default)
            return default
        return value

    @classmethod
    def _settings_from_environment(
        cls, environ: Mapping[str, str]
    ) -> ObservabilitySettings:
        protocol = (
            environ.get(ENV_OTEL_EXPORTER_OTLP_PROTOCOL, "http/protobuf")
            .strip()
            .lower()
        )
        if protocol not in OTEL_PROTOCOLS:
            logger.warning(
                "Invalid %s value; using default http/protobuf",
                ENV_OTEL_EXPORTER_OTLP_PROTOCOL,
            )
            protocol = "http/protobuf"

        max_queue_size = cls._parse_positive_int(environ, ENV_OTEL_MAX_QUEUE_SIZE, 2048)
        max_export_batch_size = cls._parse_positive_int(
            environ, ENV_OTEL_MAX_EXPORT_BATCH_SIZE, 512
        )
        if max_export_batch_size > max_queue_size:
            max_export_batch_size = min(512, max_queue_size)
            logger.warning(
                "%s must not exceed %s; using %s",
                ENV_OTEL_MAX_EXPORT_BATCH_SIZE,
                ENV_OTEL_MAX_QUEUE_SIZE,
                max_export_batch_size,
            )

        base_endpoint = (environ.get(ENV_OTLP_BASE_ENDPOINT) or "").strip().rstrip(
            "/"
        ) or "http://localhost:4318"
        return ObservabilitySettings(
            opentelemetry=OpenTelemetrySettings(
                enable_otel=cls._parse_bool(environ, ENV_ENABLE_OTEL, False),
                otlp_base_endpoint=base_endpoint,
                otlp_trace_endpoint=environ.get(ENV_OTLP_TRACE_ENDPOINT, "").strip(),
                otlp_metric_endpoint=environ.get(ENV_OTLP_METRIC_ENDPOINT, "").strip(),
                otlp_api_key=environ.get(ENV_OTLP_API_KEY, "").strip(),
                otel_exporter_otlp_protocol=protocol,
                otel_sampling_rate=cls._parse_sampling_rate(
                    environ, ENV_OTEL_SAMPLING_RATE, 0.1
                ),
                otel_batch_export_schedule_delay=cls._parse_positive_int(
                    environ, ENV_OTEL_BATCH_EXPORT_SCHEDULE_DELAY, 5000
                ),
                otel_max_queue_size=max_queue_size,
                otel_max_export_batch_size=max_export_batch_size,
                otel_metric_export_interval=cls._parse_positive_int(
                    environ, ENV_OTEL_METRIC_EXPORT_INTERVAL, 60000
                ),
                otel_batch_export_timeout=cls._parse_positive_int(
                    environ, ENV_OTEL_BATCH_EXPORT_TIMEOUT, 10000
                ),
                otel_metric_export_timeout=cls._parse_positive_int(
                    environ, ENV_OTEL_METRIC_EXPORT_TIMEOUT, 30000
                ),
            ),
        )

    def _load_saved(self) -> Optional[ObservabilitySettings]:
        if not self._path.is_file():
            return None
        try:
            payload = load_system_settings_document(self._path)
            observability = payload.get("observability")
            if observability is None:
                return None
            if not isinstance(observability, dict):
                raise ValueError("observability must be an object")
            saved = observability.get("opentelemetry")
            startup_settings = self._startup.opentelemetry.to_dict()
            if saved is None and set(observability).intersection(startup_settings):
                # Compatibility with the short-lived flat observability draft.
                saved = observability
            if saved is None:
                return None
            if not isinstance(saved, dict):
                raise ValueError("observability.opentelemetry must be an object")
            unknown = set(saved).difference(startup_settings)
            if unknown:
                raise ValueError(
                    f"Unknown OpenTelemetry setting fields: {sorted(unknown)}"
                )
            startup_settings.update(saved)
            return ObservabilitySettings(
                opentelemetry=OpenTelemetrySettings.from_dict(startup_settings)
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning(
                "Ignoring invalid observability settings file %s: %s",
                self._path,
                exc,
            )
            return None

    @staticmethod
    def _masked_secret(secret: str) -> str:
        if not secret:
            return ""
        if len(secret) <= 8:
            return SECRET_MASK
        return f"{secret[:4]}{SECRET_MASK}{secret[-4:]}"

    @classmethod
    def public_dict(cls, settings: ObservabilitySettings) -> Dict[str, Any]:
        data = settings.to_dict()
        data["opentelemetry"]["otlp_api_key"] = cls._masked_secret(
            settings.opentelemetry.otlp_api_key
        )
        return data

    def get(self) -> ObservabilitySettings:
        with self._lock:
            return self._current

    def get_startup(self) -> ObservabilitySettings:
        return self._startup

    def get_public(self) -> Dict[str, Any]:
        return self.public_dict(self.get())

    def _write(self, settings: ObservabilitySettings) -> None:
        payload = load_system_settings_document_for_update(self._path)
        payload.setdefault("settings", {})
        observability = payload.get("observability")
        observability = dict(observability) if isinstance(observability, dict) else {}
        observability["opentelemetry"] = settings.opentelemetry.to_dict()
        payload["observability"] = observability
        write_system_settings_document(self._path, payload)

    def save_public(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        incoming_opentelemetry = data.get("opentelemetry")
        if not isinstance(incoming_opentelemetry, Mapping):
            raise ValueError("opentelemetry must be an object")
        incoming_key = incoming_opentelemetry.get("otlp_api_key")
        if not isinstance(incoming_key, str):
            raise ValueError("otlp_api_key must be a string")

        with self._lock:
            full_opentelemetry = dict(incoming_opentelemetry)
            full_opentelemetry["otlp_api_key"] = (
                self._current.opentelemetry.otlp_api_key
                if "*" in incoming_key
                else incoming_key
            )
            settings = ObservabilitySettings.from_dict(
                {**data, "opentelemetry": full_opentelemetry}
            )
            self._write(settings)
            self._current = settings
            self.apply_to_environment(settings)
            return self.public_dict(settings)

    def reset(self) -> Dict[str, Any]:
        with self._lock:
            try:
                payload = load_system_settings_document(self._path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                payload = {"version": SYSTEM_SETTINGS_VERSION, "settings": {}}
            observability = payload.get("observability")
            if isinstance(observability, dict):
                observability = dict(observability)
                observability.pop("opentelemetry", None)
                if set(observability).intersection(
                    OpenTelemetrySettings.__dataclass_fields__
                ):
                    observability = {}
                if observability:
                    payload["observability"] = observability
                else:
                    payload.pop("observability", None)
            payload.setdefault("settings", {})
            write_or_remove_system_settings_document(self._path, payload)
            self._current = self._startup
            self.apply_to_environment(self._startup)
            return self.public_dict(self._startup)

    def apply_to_environment(
        self, settings: Optional[ObservabilitySettings] = None
    ) -> None:
        apply_observability_settings(settings or self.get(), self._environ)


_store: Optional[ObservabilitySettingsStore] = None
_store_lock = threading.Lock()


def get_observability_settings_store(path: str) -> ObservabilitySettingsStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = ObservabilitySettingsStore(path)
        elif _store.path != str(Path(path)):
            raise RuntimeError(
                "Observability settings store already initialized at " f"{_store.path}"
            )
        return _store


def initialize_observability_settings(path: str) -> ObservabilitySettingsStore:
    store = get_observability_settings_store(path)
    store.apply_to_environment()
    return store
