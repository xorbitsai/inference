# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Private, account-scoped model favorites. Synchronous handlers use a thread pool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from fastapi import Depends, HTTPException, Request, Security
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from ..dependencies import get_api
from ..responses import JSONResponse

if TYPE_CHECKING:
    from ..restful_api import RESTfulAPI


class FavoriteModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: StrictStr = Field(min_length=1, max_length=512)

    @field_validator("model_name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        if value != value.strip() or any(ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError("Invalid model name")
        return value


def _owner(user: Optional[dict], authenticated: bool) -> Optional[int]:
    if user is None:
        if authenticated:
            raise HTTPException(
                status_code=403, detail="Account identity is unavailable"
            )
        return None
    user_id = user.get("id")
    if type(user_id) is not int or user_id <= 0:
        raise HTTPException(status_code=403, detail="Account identity is unavailable")
    return user_id


def _require_user_session(request: Request) -> None:
    if getattr(request.state, "authenticated_api_key", None) is not None:
        raise HTTPException(
            status_code=403, detail="User session required to change favorites"
        )


def register_routes(api: "RESTfulAPI") -> None:
    # This dependency is also visible to API composition hooks such as role
    # admission, and the existing auth service remains the identity authority.
    auth = api._auth_service
    authenticated = api.is_authenticated()

    current_user = (
        Security(auth, scopes=["models:list"])
        if authenticated
        else Depends(lambda: None)
    )

    def list_favorites(
        user: Optional[dict] = current_user,
        api_: "RESTfulAPI" = Depends(get_api),
    ) -> JSONResponse:
        result = api_._model_favorites_store.list(_owner(user, authenticated))
        if not isinstance(result, list) or any(not isinstance(n, str) for n in result):
            raise TypeError("Model favorites store must return a list of names")
        return JSONResponse(content=result)

    def add_favorite(
        body: FavoriteModel,
        request: Request,
        user: Optional[dict] = current_user,
        api_: "RESTfulAPI" = Depends(get_api),
    ) -> JSONResponse:
        _require_user_session(request)
        api_._model_favorites_store.add(_owner(user, authenticated), body.model_name)
        return JSONResponse(content={"status": "ok"})

    def remove_favorite(
        body: FavoriteModel,
        request: Request,
        user: Optional[dict] = current_user,
        api_: "RESTfulAPI" = Depends(get_api),
    ) -> JSONResponse:
        _require_user_session(request)
        api_._model_favorites_store.remove(_owner(user, authenticated), body.model_name)
        return JSONResponse(content={"status": "ok"})

    for method, handler in (
        ("GET", list_favorites),
        ("PUT", add_favorite),
        ("DELETE", remove_favorite),
    ):
        api._router.add_api_route("/v1/model_favorites", handler, methods=[method])
