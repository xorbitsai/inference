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
"""Audit logging for API Key model calls."""

import json
import logging
import os
import socket
from datetime import datetime, timezone
from typing import Optional

from ....constants import (
    XINFERENCE_AUDIT_LOG_RETENTION_DAYS,
    XINFERENCE_LOG_BACKUP_COUNT,
    XINFERENCE_LOG_DIR,
    XINFERENCE_LOG_MAX_BYTES,
)
from ....core.log import create_rotating_handler

_audit_logger: Optional[logging.Logger] = None


def _get_audit_logger() -> logging.Logger:
    global _audit_logger
    if _audit_logger is not None:
        return _audit_logger

    _audit_logger = logging.getLogger("xinference.audit")
    _audit_logger.setLevel(logging.INFO)
    _audit_logger.propagate = False

    handler = create_rotating_handler(
        filename=os.path.join(XINFERENCE_LOG_DIR, "audit.log"),
        retention_days=XINFERENCE_AUDIT_LOG_RETENTION_DAYS,
        rotation="daily+size",
        max_bytes=XINFERENCE_LOG_MAX_BYTES,
        backup_count=XINFERENCE_LOG_BACKUP_COUNT,
        formatter=logging.Formatter("%(message)s"),
    )
    _audit_logger.addHandler(handler)
    return _audit_logger


_NODE_NAME = socket.gethostname()
_SERVICE_ADDRESS = ""

# Endpoints that should NOT be audited
_AUDIT_SKIP_ENDPOINTS = (
    "/v1/audit/",
    "/v1/cluster/auth",
    "/v1/cluster/ui_config",
    "/status",
    "/v1/address",
)

# Endpoint prefixes for category classification
_INFERENCE_PREFIXES = (
    "/v1/chat/",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/rerank",
    "/v1/images/",
    "/v1/audio/",
    "/v1/video/",
    "/v1/sdapi/",
)

_AUTH_PREFIXES = (
    "/token",
    "/v1/auth/",
)


def should_skip_audit(endpoint: str) -> bool:
    for prefix in _AUDIT_SKIP_ENDPOINTS:
        if endpoint == prefix or endpoint.startswith(prefix):
            return True
    return False


def classify_endpoint(endpoint: str) -> str:
    for prefix in _INFERENCE_PREFIXES:
        if endpoint.startswith(prefix):
            return "inference"
    for prefix in _AUTH_PREFIXES:
        if endpoint == prefix or endpoint.startswith(prefix):
            return "auth"
    return "admin"


# Shared model UID → (model_name, model_type) cache, populated by RESTfulAPI
_uid_model_cache: dict[str, tuple[str, str]] = {}


def update_model_cache(uid: str, model_name: str, model_type: str = "") -> None:
    _uid_model_cache[uid] = (model_name, model_type)


def evict_model_cache(uid: str) -> None:
    _uid_model_cache.pop(uid, None)


def resolve_model_info(model_id: str) -> tuple[str, str]:
    """Return (model_name, model_type) for a given model_id/uid."""
    if not model_id:
        return ("", "")
    entry = _uid_model_cache.get(model_id)
    if entry:
        return entry
    return ("", "")


def _get_service_address() -> str:
    global _SERVICE_ADDRESS
    if _SERVICE_ADDRESS:
        return _SERVICE_ADDRESS
    from ....deploy import utils as _deploy_utils

    _JsonFileFormatter = getattr(_deploy_utils, "JsonFileFormatter", None)
    if _JsonFileFormatter is None:
        return ""
    for inst in _JsonFileFormatter._instances:
        if inst.address:
            _SERVICE_ADDRESS = inst.address
            return _SERVICE_ADDRESS
    return ""


def record_audit_event(
    *,
    user: str,
    api_key_name: str,
    api_key_prefix: str,
    model_id: str,
    model_name: str = "",
    model_type: str,
    endpoint: str,
    status: str,
    latency_ms: float = 0.0,
    client_ip: str = "",
    node: str = "",
    address: str = "",
    category: str = "",
    auth_type: str = "",
) -> None:
    if should_skip_audit(endpoint):
        return
    if not address:
        address = _get_service_address()
    if not category:
        category = classify_endpoint(endpoint)
    entry = {
        "@timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        + "Z",
        "event_type": "api_call",
        "category": category,
        "auth_type": auth_type,
        "user": user,
        "api_key_name": api_key_name,
        "api_key_prefix": api_key_prefix,
        "model_id": model_id,
        "model_name": model_name,
        "model_type": model_type,
        "endpoint": endpoint,
        "status": status,
        "latency_ms": latency_ms,
        "client_ip": client_ip,
        "node": node or _NODE_NAME,
        "address": address,
    }
    _get_audit_logger().info(json.dumps(entry, ensure_ascii=False))
