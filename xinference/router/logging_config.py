# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Logging helpers for the Xinference Token-aware Router."""

from __future__ import annotations

import logging
import logging.config
import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ..constants import (
    XINFERENCE_LOG_BACKUP_COUNT,
    XINFERENCE_LOG_MAX_BYTES,
    XINFERENCE_LOG_RETENTION_DAYS,
)
from ..deploy.utils import get_config_dict, get_log_file, update_all_formatter_addresses

_ROUTER_LOG_FIELDS = frozenset(
    {
        "event",
        "request_id",
        "router_uid",
        "node_id",
        "assignment_id",
        "assignment_generation",
        "instance_id",
        "listen_port",
        "config_revision",
        "requested_model",
        "logical_model",
        "route_profile",
        "rule_id",
        "route_reason",
        "backend_id",
        "backend_model_uid",
        "prompt_tokens",
        "output_tokens",
        "total_budget",
        "stream",
        "revision",
        "previous_revision",
        "target_revision",
        "current_revision",
        "status_code",
        "outcome",
        "elapsed_seconds",
        "tokenizer_asset_id",
        "listen_address",
        "backend_url",
        "backend_mapping",
        "enabled",
    }
)
_VALID_LOG_LEVELS = frozenset(
    {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
)
_ROUTER_LOG_IDENTITY: dict[str, Any] = {}


def normalize_log_level(log_level: str) -> str:
    value = str(log_level or "").upper()
    return value if value in _VALID_LOG_LEVELS else "INFO"


def router_access_log_enabled() -> bool:
    return (
        os.getenv("XINFERENCE_TOKEN_ROUTER_ACCESS_LOG", "false").strip().lower()
        == "true"
    )


def router_log_extra(**fields: Any) -> dict[str, dict[str, Any]]:
    """Build a whitelisted ``logging`` extra mapping for Router events."""

    values = {**_ROUTER_LOG_IDENTITY, **fields}
    return {
        "xinference_fields": {
            key: value
            for key, value in values.items()
            if key in _ROUTER_LOG_FIELDS and value is not None
        }
    }


def set_router_log_identity(**fields: Any) -> None:
    """Set process-wide Runtime identity fields added to structured events.

    A Router Runtime serves exactly one Assignment in one process, so a small
    process-local context is sufficient and avoids threading identity through
    every data-plane logging call.
    """

    for key, value in fields.items():
        if key in _ROUTER_LOG_FIELDS:
            if value is None:
                _ROUTER_LOG_IDENTITY.pop(key, None)
            else:
                _ROUTER_LOG_IDENTITY[key] = value


def sanitize_log_url(value: str) -> str:
    """Remove credentials, query parameters, and fragments from a log URL."""

    try:
        parsed = urlsplit(value)
        if not parsed.scheme or not parsed.hostname:
            return "<invalid-url>"
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = host
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except (TypeError, ValueError):
        return "<invalid-url>"


def _address(host: str, port: int | None) -> str:
    return f"{host}:{port}" if port is not None else host


def configure_router_logging(
    log_level: str, host: str = "", port: int | None = None
) -> dict:
    """Configure Xinference-standard Router logging and return its dict config."""

    normalized_level = normalize_log_level(log_level)
    logging_conf = get_config_dict(
        normalized_level,
        get_log_file("router"),
        XINFERENCE_LOG_BACKUP_COUNT,
        XINFERENCE_LOG_MAX_BYTES,
        role="router",
        address=_address(host, port),
        log_retention_days=XINFERENCE_LOG_RETENTION_DAYS,
    )
    logging.config.dictConfig(logging_conf)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return logging_conf


def update_router_logging_address(logging_conf: dict, host: str, port: int) -> dict:
    """Update live and reusable logging config with the resolved listen address."""

    address = _address(host, port)
    for formatter in logging_conf.get("formatters", {}).values():
        if isinstance(formatter, dict) and "address" in formatter:
            formatter["address"] = address
    update_all_formatter_addresses("router", address)
    return logging_conf
