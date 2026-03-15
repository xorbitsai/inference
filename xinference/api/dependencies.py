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

"""FastAPI dependency injection for REST API handlers.

The API instance is attached to the app in RESTfulAPI.serve() via app.state.api.
Handlers inject it via Depends(get_api); supervisor ref is obtained in-handler
with await api._get_supervisor_ref() when needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from .restful_api import RESTfulAPI


def get_api(request: Request) -> "RESTfulAPI":
    """Return the RESTfulAPI instance for the current app (set in serve())."""
    return request.app.state.api
