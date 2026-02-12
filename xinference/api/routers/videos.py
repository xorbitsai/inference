"""Video route registration (text-to-video, image-to-video, flf-to-video)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Security

from ...types import VideoList

if TYPE_CHECKING:
    from ..restful_api import RESTfulAPI


def register_routes(api: "RESTfulAPI") -> None:
    router = api._router
    auth = api._auth_service
    is_auth = api.is_authenticated()

    router.add_api_route(
        "/v1/video/generations",
        api.create_videos,
        methods=["POST"],
        response_model=VideoList,
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/video/generations/image",
        api.create_videos_from_images,
        methods=["POST"],
        response_model=VideoList,
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/video/generations/flf",
        api.create_videos_from_first_last_frame,
        methods=["POST"],
        response_model=VideoList,
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
