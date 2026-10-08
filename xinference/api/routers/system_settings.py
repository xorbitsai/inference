"""System settings API routes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from fastapi import Body, HTTPException, Request, Security
from pydantic import BaseModel, Field

from ..responses import JSONResponse

if TYPE_CHECKING:
    from ..restful_api import RESTfulAPI


class SystemSettingsPayload(BaseModel):
    class Config:
        extra = "forbid"

    download_source: Literal[
        "auto", "huggingface", "modelscope", "openmind_hub", "csghub"
    ]
    hf_endpoint: str
    hf_token: str
    pip_index_url: str
    download_max_attempts: int = Field(ge=1)
    hub_detect_timeout: float = Field(gt=0)
    model_download_workers: int = Field(ge=1)


async def get_system_settings(request: Request) -> JSONResponse:
    store = request.app.state.system_settings_store
    return JSONResponse(content=store.get_public())


async def _apply_system_settings_to_cluster(request: Request) -> None:
    store = request.app.state.system_settings_store
    api = request.app.state.api
    supervisor_ref = await api._get_supervisor_ref()
    await supervisor_ref.update_system_settings(store.get().to_dict())


async def update_system_settings(
    request: Request,
    body: SystemSettingsPayload = Body(...),
) -> JSONResponse:
    store = request.app.state.system_settings_store
    try:
        settings = store.save_public(body.dict())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    await _apply_system_settings_to_cluster(request)
    return JSONResponse(content=settings)


async def reset_system_settings(request: Request) -> JSONResponse:
    store = request.app.state.system_settings_store
    settings = store.reset()
    await _apply_system_settings_to_cluster(request)
    return JSONResponse(content=settings)


def register_routes(api: "RESTfulAPI") -> None:
    router = api._router
    auth = api._auth_service
    read_dependencies = (
        [Security(auth, scopes=["settings:read"])] if api.is_authenticated() else None
    )
    write_dependencies = (
        [Security(auth, scopes=["settings:write"])] if api.is_authenticated() else None
    )

    router.add_api_route(
        "/v1/cluster/system_settings",
        get_system_settings,
        methods=["GET"],
        dependencies=read_dependencies,
    )
    router.add_api_route(
        "/v1/cluster/system_settings",
        update_system_settings,
        methods=["PUT"],
        dependencies=write_dependencies,
    )
    router.add_api_route(
        "/v1/cluster/system_settings/reset",
        reset_system_settings,
        methods=["POST"],
        dependencies=write_dependencies,
    )
