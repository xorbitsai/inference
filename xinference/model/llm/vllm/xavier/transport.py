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

from typing import Any, Dict, Optional
from urllib.parse import urlsplit

XAVIER_TRANSPORT_BACKEND_KEY = "vllm_transfer_backend_type"
XAVIER_TRANSPORT_BACKEND_ALIAS_KEY = "transfer_backend_type"

XAVIER_TRANSPORT_XAVIER = "xavier"

XAVIER_CONNECTOR = "XavierConnector"
XAVIER_CONNECTOR_MODULE = "xinference.model.llm.vllm.xavier.v1_connector"


def normalize_xavier_transport_backend(backend: Optional[str]) -> str:
    if backend is None or backend == "":
        return XAVIER_TRANSPORT_XAVIER
    backend = str(backend).strip().lower()
    if backend not in ("xavier", "nixl"):
        raise ValueError(
            f"Unknown vLLM transfer backend: {backend!r}; use xavier or nixl"
        )
    return backend


def get_xavier_transport_backend(xavier_config: Optional[Dict[str, Any]]) -> str:
    if not xavier_config:
        return XAVIER_TRANSPORT_XAVIER
    return normalize_xavier_transport_backend(
        xavier_config.get(
            XAVIER_TRANSPORT_BACKEND_KEY,
            xavier_config.get(XAVIER_TRANSPORT_BACKEND_ALIAS_KEY),
        )
    )


def set_xavier_transport_backend(
    xavier_config: Dict[str, Any], backend: Optional[str]
) -> Dict[str, Any]:
    normalized = normalize_xavier_transport_backend(backend)
    xavier_config[XAVIER_TRANSPORT_BACKEND_KEY] = normalized
    xavier_config[XAVIER_TRANSPORT_BACKEND_ALIAS_KEY] = normalized
    return xavier_config


def validate_gpu_cache_budget(value, enabled: bool, replicas: int):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("xavier_gpu_cache_bytes must be a non-negative integer")
    if not enabled or replicas <= 1:
        raise ValueError(
            "xavier_gpu_cache_bytes requires Xavier with multiple replicas"
        )
    return value


def get_transport_host(address: str) -> Optional[str]:
    return urlsplit(address if "://" in address else "tcp://" + address).hostname


def gpu_pool_options(address: str, env: Dict[str, str]) -> Dict[str, str]:
    import os
    from importlib.metadata import version
    from importlib.util import find_spec

    from packaging.version import Version

    from ..pd import resolve_nixl_host

    if Version(version("xoscar")) < Version("0.11.1") or find_spec("nixl") is None:
        raise RuntimeError(
            "Xavier GPU transfer requires xoscar[nixl]>=0.11.1 on the worker and in the model environment"
        )
    host = get_transport_host(address)
    if not host:
        raise ValueError("Cannot determine Xavier NIXL worker host")
    host = resolve_nixl_host(host, "a reachable worker address")
    if not host:
        raise ValueError("Cannot determine Xavier NIXL worker host")
    if ":" in host:
        host = f"[{host}]"
    env["UCX_MEMTYPE_CACHE"] = "n"
    env.setdefault("UCX_TLS", os.environ.get("UCX_TLS", "tcp,cuda_copy,cuda_ipc"))
    return {"external_address": f"nixl://{host}:0"}


def uses_direct_handoff(config: Optional[Dict[str, Any]]) -> bool:
    """GPU P/D uses request-scoped handoff; CPU and hybrid caches stay unchanged."""
    return bool(
        config
        and config.get("gpu_cache_bytes") is not None
        and config.get("role") in {"prefill", "decode"}
    )
