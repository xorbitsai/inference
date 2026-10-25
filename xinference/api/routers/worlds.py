"""World generation route registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Security

from ...types import VideoList

if TYPE_CHECKING:
    from ..restful_api import RESTfulAPI


def register_routes(api: "RESTfulAPI") -> None:
    api._router.add_api_route(
        "/v1/worlds/generations",
        api.create_world,
        methods=["POST"],
        response_model=VideoList,
        dependencies=(
            [Security(api._auth_service, scopes=["models:read"])]
            if api.is_authenticated()
            else None
        ),
    )
