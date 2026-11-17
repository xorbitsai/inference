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

"""Observability settings API routes."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from fastapi import Body, HTTPException, Request, Security
from pydantic import BaseModel, Field

from ..responses import JSONResponse

if TYPE_CHECKING:
    from ...core.observability_settings_store import ObservabilitySettingsStore
    from ..restful_api import RESTfulAPI


class OpenTelemetrySettingsPayload(BaseModel):
    class Config:
        extra = "forbid"

    enable_otel: bool
    otlp_base_endpoint: str
    otlp_trace_endpoint: str
    otlp_metric_endpoint: str
    otlp_api_key: str
    otel_exporter_otlp_protocol: Literal["http/protobuf", "grpc"]
    otel_sampling_rate: float = Field(ge=0, le=1)
    otel_batch_export_schedule_delay: int = Field(ge=1)
    otel_max_queue_size: int = Field(ge=1)
    otel_max_export_batch_size: int = Field(ge=1)
    otel_metric_export_interval: int = Field(ge=1)
    otel_batch_export_timeout: int = Field(ge=1)
    otel_metric_export_timeout: int = Field(ge=1)


class ObservabilitySettingsPayload(BaseModel):
    class Config:
        extra = "forbid"

    opentelemetry: OpenTelemetrySettingsPayload
    runtime_enable_otel: Optional[bool] = None
    restart_required: Optional[bool] = None
    restart_supported: Optional[bool] = None
    restart_components: Optional[List[str]] = None
    restart_error: Optional[str] = None


def _with_runtime_state(
    store: "ObservabilitySettingsStore", settings: Dict[str, Any]
) -> Dict[str, Any]:
    from ...core.observability_settings_store import observability_runtime_state

    runtime = observability_runtime_state(store.get_runtime())
    pending = store.get() != store.get_runtime() or bool(runtime["error"])
    return {
        **settings,
        "runtime_enable_otel": store.get_runtime().opentelemetry.enable_otel,
        "restart_required": pending,
        "restart_supported": False,
        "restart_components": ["api"] if pending else [],
        "restart_error": runtime["error"],
    }


async def _settings_operation(request: Request, operation: str, data=None):
    from ...deploy.api_host import API_HOST_ENV, host_request

    if os.environ.get(API_HOST_ENV):
        try:
            return await host_request(
                "observability-settings", operation=operation, data=data
            )
        except ValueError as error:
            if str(error) != "Unknown API host command":
                raise
    store = request.app.state.observability_settings_store
    if operation == "get":
        settings = store.get_public()
    elif operation == "save":
        settings = store.save_public(data)
    else:
        settings = store.reset()
    return _with_runtime_state(store, settings)


async def get_observability_settings(request: Request) -> JSONResponse:
    return JSONResponse(content=await _settings_operation(request, "get"))


async def update_observability_settings(
    request: Request,
    body: ObservabilitySettingsPayload = Body(...),
) -> JSONResponse:
    try:
        settings = await _settings_operation(
            request, "save", body.dict(include={"opentelemetry"})
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return JSONResponse(content=settings)


async def reset_observability_settings(request: Request) -> JSONResponse:
    return JSONResponse(content=await _settings_operation(request, "reset"))


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
        "/v1/cluster/observability_settings",
        get_observability_settings,
        methods=["GET"],
        dependencies=read_dependencies,
    )
    router.add_api_route(
        "/v1/cluster/observability_settings",
        update_observability_settings,
        methods=["PUT"],
        dependencies=write_dependencies,
    )
    router.add_api_route(
        "/v1/cluster/observability_settings/reset",
        reset_observability_settings,
        methods=["POST"],
        dependencies=write_dependencies,
    )
