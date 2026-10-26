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


"""Generic realtime route registration; no model-family dependencies."""

from fastapi import WebSocket

from ..realtime import realtime_session


def register_routes(api) -> None:
    async def connect(websocket: WebSocket, model_uid: str) -> None:
        await realtime_session(api, websocket, model_uid)

    api._router.add_api_websocket_route("/v1/models/{model_uid}/realtime", connect)
