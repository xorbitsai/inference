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

from __future__ import annotations

import os


def token_router_data_plane_token() -> str:
    """Return the credential selected by the Supervisor for Runtime requests."""

    return (
        os.getenv("XINFERENCE_TOKEN_ROUTER_DATA_PLANE_TOKEN")
        or os.getenv("XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN")
        or ""
    )
