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

XAVIER_TRANSPORT_BACKEND_KEY = "vllm_transfer_backend_type"
XAVIER_TRANSPORT_BACKEND_ALIAS_KEY = "transfer_backend_type"

XAVIER_TRANSPORT_XAVIER = "xavier"

XAVIER_CONNECTOR = "XavierConnector"
XAVIER_CONNECTOR_MODULE = "xinference.model.llm.vllm.xavier.v1_connector"


def normalize_xavier_transport_backend(backend: Optional[str]) -> str:
    if backend in (None, ""):
        return XAVIER_TRANSPORT_XAVIER
    return XAVIER_TRANSPORT_XAVIER


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
