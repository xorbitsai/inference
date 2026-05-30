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
"""Launch history route registration and handlers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from fastapi import Depends, HTTPException, Query, Request, Security

from ..dependencies import get_api
from ..responses import JSONResponse

if TYPE_CHECKING:
    from ..restful_api import RESTfulAPI

logger = logging.getLogger(__name__)


async def list_launch_history(
    model_name: Optional[str] = Query(None),
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    try:
        data = api._launch_history_store.list(model_name=model_name)
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def create_launch_history(
    request: Request,
    api: "RESTfulAPI" = Depends(get_api),
    user: Optional[dict] = None,
) -> JSONResponse:
    try:
        body = await request.json()
        model_name = body.get("model_name")
        model_uid = body.get("model_uid", "")
        data = body.get("data", {})
        if not model_name:
            raise HTTPException(status_code=400, detail="model_name is required")
        username = user.get("username", "") if user else ""
        api._launch_history_store.upsert(
            model_name=model_name,
            model_uid=model_uid,
            data=data,
            username=username,
        )
        return JSONResponse(content={"status": "ok"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def delete_launch_history(
    model_name: str,
    model_uid: str = "",
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    try:
        deleted = api._launch_history_store.delete(model_name, model_uid)
        if not deleted:
            raise HTTPException(status_code=404, detail="Record not found")
        return JSONResponse(content={"status": "ok"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def register_routes(api: "RESTfulAPI") -> None:
    router = api._router
    auth = api._auth_service
    is_auth = api.is_authenticated()

    if is_auth:

        async def create_handler(
            request: Request,
            user: dict = Security(auth, scopes=["models:write"]),
            api_: "RESTfulAPI" = Depends(get_api),
        ) -> JSONResponse:
            return await create_launch_history(request, api_, user)

    else:
        create_handler = create_launch_history  # type: ignore[assignment]

    router.add_api_route(
        "/v1/launch_history",
        list_launch_history,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:list"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/launch_history",
        create_handler,
        methods=["POST"],
    )
    router.add_api_route(
        "/v1/launch_history/{model_name}/{model_uid}",
        delete_launch_history,
        methods=["DELETE"],
        dependencies=([Security(auth, scopes=["models:write"])] if is_auth else None),
    )
    # Variant for the common case of an empty model_uid, which cannot be
    # expressed as a non-empty path segment.
    router.add_api_route(
        "/v1/launch_history/{model_name}",
        delete_launch_history,
        methods=["DELETE"],
        dependencies=([Security(auth, scopes=["models:write"])] if is_auth else None),
    )
