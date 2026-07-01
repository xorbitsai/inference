"""Image route registration (generations, variations, inpainting, ocr, edits, SD API)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Security

from ...types import ImageList, SDAPIResult

if TYPE_CHECKING:
    from ..restful_api import RESTfulAPI


def register_routes(api: "RESTfulAPI") -> None:
    router = api._router
    auth = api._auth_service
    is_auth = api.is_authenticated()

    router.add_api_route(
        "/v1/images/generations",
        api.create_images,
        methods=["POST"],
        response_model=ImageList,
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/images/variations",
        api.create_variations,
        methods=["POST"],
        response_model=ImageList,
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/images/inpainting",
        api.create_inpainting,
        methods=["POST"],
        response_model=ImageList,
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/images/ocr",
        api.create_ocr,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/images/edits",
        api.create_image_edits,
        methods=["POST"],
        response_model=ImageList,
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )

    # SD WebUI API
    router.add_api_route(
        "/sdapi/v1/options",
        api.sdapi_options,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/sdapi/v1/sd-models",
        api.sdapi_sd_models,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/sdapi/v1/samplers",
        api.sdapi_samplers,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/sdapi/v1/txt2img",
        api.sdapi_txt2img,
        methods=["POST"],
        response_model=SDAPIResult,
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/sdapi/v1/img2img",
        api.sdapi_img2img,
        methods=["POST"],
        response_model=SDAPIResult,
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
