# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Token-aware Router control-plane REST endpoints."""

from __future__ import annotations

import hmac
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from fastapi import Depends, Header, HTTPException, Query, Request, Security
from fastapi.responses import Response

from ..._compat import ValidationError
from ..dependencies import get_api
from ..responses import JSONResponse
from ..schemas.router import (
    RouterConfigAck,
    RouterRuntimeHeartbeat,
    RouterRuntimeRegister,
    TokenRouterCreate,
    TokenRouterUpdate,
)

if TYPE_CHECKING:
    from ..restful_api import RESTfulAPI


def _username(user: Optional[dict]) -> str:
    return user.get("username", "") if user else ""


def _internal_authorized(authorization: str) -> bool:
    expected = os.environ.get("XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN", "")
    if not expected or not authorization.startswith("Bearer "):
        return False
    return hmac.compare_digest(authorization[7:], expected)


def _require_internal(authorization: str = Header(default="")) -> None:
    if not _internal_authorized(authorization):
        raise HTTPException(status_code=401, detail="Invalid Token Router credential")


async def _parse_payload(request: Request, payload_model: Any):
    try:
        body = await request.json()
        return payload_model.parse_obj(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail="Request body must be valid JSON"
        ) from exc


async def _supervisor(api: "RESTfulAPI"):
    return await api._get_supervisor_ref()


async def list_routers(api: "RESTfulAPI") -> JSONResponse:
    return JSONResponse(content=await (await _supervisor(api)).list_token_routers())


async def list_backend_candidates(api: "RESTfulAPI") -> JSONResponse:
    return JSONResponse(
        content=await (await _supervisor(api)).list_token_router_backend_candidates()
    )


async def list_tokenizer_assets(api: "RESTfulAPI") -> JSONResponse:
    return JSONResponse(content=await (await _supervisor(api)).list_tokenizer_assets())


async def get_tokenizer_asset(asset_id: str, api: "RESTfulAPI") -> JSONResponse:
    try:
        result = await (await _supervisor(api)).get_tokenizer_asset(asset_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="Tokenizer asset not found"
        ) from exc
    return JSONResponse(content=result)


async def validate_tokenizer_asset(asset_id: str, api: "RESTfulAPI") -> JSONResponse:
    try:
        result = await (await _supervisor(api)).validate_tokenizer_asset(asset_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="Tokenizer asset not found"
        ) from exc
    return JSONResponse(content=result)


async def create_router(
    payload: TokenRouterCreate, api: "RESTfulAPI", user: Optional[dict]
) -> JSONResponse:
    data = payload.dict()
    router_uid = data.pop("router_uid")
    try:
        result = await (await _supervisor(api)).create_token_router(
            router_uid, data, _username(user)
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="Tokenizer asset not found"
        ) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(status_code=201, content=result)


async def get_router(router_uid: str, api: "RESTfulAPI") -> JSONResponse:
    result = await (await _supervisor(api)).get_token_router(router_uid)
    if result is None:
        raise HTTPException(status_code=404, detail="Token Router not found")
    return JSONResponse(content=result)


async def update_router(
    router_uid: str,
    payload: TokenRouterUpdate,
    api: "RESTfulAPI",
    user: Optional[dict],
) -> JSONResponse:
    try:
        result = await (await _supervisor(api)).update_token_router(
            router_uid, payload.dict(), _username(user)
        )
    except KeyError as exc:
        detail = (
            "Token Router not found"
            if str(exc).strip("'") == router_uid
            else "Tokenizer asset not found"
        )
        raise HTTPException(status_code=404, detail=detail) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(content=result)


async def delete_router(router_uid: str, api: "RESTfulAPI") -> JSONResponse:
    try:
        deleted = await (await _supervisor(api)).delete_token_router(router_uid)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Token Router not found")
    return JSONResponse(content={"status": "ok"})


async def set_router_enabled(
    router_uid: str, enabled: bool, api: "RESTfulAPI", user: Optional[dict]
) -> JSONResponse:
    try:
        result = await (await _supervisor(api)).set_token_router_enabled(
            router_uid, enabled, _username(user)
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Token Router not found") from exc
    return JSONResponse(content=result)


async def validate_router(router_uid: str, api: "RESTfulAPI") -> JSONResponse:
    result = await (await _supervisor(api)).validate_token_router(router_uid)
    if result is None:
        raise HTTPException(status_code=404, detail="Token Router not found")
    return JSONResponse(content=result)


async def router_status(router_uid: str, api: "RESTfulAPI") -> JSONResponse:
    result = await (await _supervisor(api)).get_token_router_status(router_uid)
    if result is None:
        raise HTTPException(status_code=404, detail="Token Router not found")
    return JSONResponse(content=result)


async def router_instances(router_uid: str, api: "RESTfulAPI") -> JSONResponse:
    if await (await _supervisor(api)).get_token_router(router_uid) is None:
        raise HTTPException(status_code=404, detail="Token Router not found")
    return JSONResponse(
        content=await (await _supervisor(api)).list_token_router_instances(router_uid)
    )


async def router_metrics(router_uid: str, api: "RESTfulAPI") -> JSONResponse:
    result = await (await _supervisor(api)).get_token_router_metrics(router_uid)
    if result is None:
        raise HTTPException(status_code=404, detail="Token Router not found")
    return JSONResponse(content=result)


async def runtime_register(
    payload: RouterRuntimeRegister, api: "RESTfulAPI"
) -> JSONResponse:
    if await (await _supervisor(api)).get_token_router(payload.router_uid) is None:
        raise HTTPException(status_code=404, detail="Token Router not found")
    data = payload.dict()
    router_uid = data.pop("router_uid")
    instance_id = data.pop("instance_id")
    try:
        result = await (await _supervisor(api)).register_token_router_instance(
            router_uid, instance_id, data
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Token Router not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(content=result)


async def runtime_heartbeat(
    instance_id: str, payload: RouterRuntimeHeartbeat, api: "RESTfulAPI"
) -> JSONResponse:
    try:
        result = await (await _supervisor(api)).heartbeat_token_router_instance(
            instance_id, payload.dict()
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="Router instance not found"
        ) from exc
    return JSONResponse(content=result)


async def runtime_config(
    router_uid: str, after_revision: int, api: "RESTfulAPI"
) -> Response:
    result = await (await _supervisor(api)).get_token_router_config_after(
        router_uid, after_revision
    )
    if result is None:
        return Response(status_code=204)
    return JSONResponse(content=result)


async def runtime_ack(
    instance_id: str, payload: RouterConfigAck, api: "RESTfulAPI"
) -> JSONResponse:
    try:
        result = await (await _supervisor(api)).ack_token_router_config(
            instance_id, payload.router_uid, payload.revision, payload.error
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="Router instance not found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(content=result)


async def runtime_unregister(instance_id: str, api: "RESTfulAPI") -> JSONResponse:
    deleted = await (await _supervisor(api)).unregister_token_router_instance(
        instance_id
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Router instance not found")
    return JSONResponse(content={"status": "ok"})


def register_routes(api: "RESTfulAPI") -> None:
    router = api._router
    auth = api._auth_service
    is_auth = api.is_authenticated()

    def route(
        path: str,
        endpoint: Callable[..., Any],
        methods: list[str],
        scope: str,
        *,
        payload_model: Optional[type] = None,
        include_user: bool = False,
        enabled: Optional[bool] = None,
        query_params: tuple[str, ...] = (),
    ) -> None:
        async def invoke(
            request: Request,
            api_: "RESTfulAPI",
            user: Optional[dict],
        ) -> JSONResponse:
            kwargs: Dict[str, Any] = dict(request.path_params)
            for name in query_params:
                kwargs[name] = request.query_params.get(name, "")
            if payload_model is not None:
                kwargs["payload"] = await _parse_payload(request, payload_model)
            if enabled is not None:
                kwargs["enabled"] = enabled
            kwargs["api"] = api_
            if include_user:
                kwargs["user"] = user
            return await endpoint(**kwargs)

        if is_auth:

            async def authed(
                request: Request,
                user: dict = Security(auth, scopes=[scope]),
                api_: "RESTfulAPI" = Depends(get_api),
            ) -> JSONResponse:
                return await invoke(request, api_, user)

            handler: Callable[..., Any] = authed
        else:

            async def anonymous(
                request: Request,
                api_: "RESTfulAPI" = Depends(get_api),
            ) -> JSONResponse:
                return await invoke(request, api_, None)

            handler = anonymous
        router.add_api_route(path, handler, methods=methods)

    route("/v1/tokenizer_assets", list_tokenizer_assets, ["GET"], "routers:list")
    route(
        "/v1/tokenizer_assets/{asset_id}",
        get_tokenizer_asset,
        ["GET"],
        "routers:read",
    )
    route(
        "/v1/tokenizer_assets/{asset_id}/validate",
        validate_tokenizer_asset,
        ["POST"],
        "routers:operate",
    )

    route("/v1/token_routers", list_routers, ["GET"], "routers:list")
    route(
        "/v1/token_routers/backend-candidates",
        list_backend_candidates,
        ["GET"],
        "routers:list",
    )
    route(
        "/v1/token_routers",
        create_router,
        ["POST"],
        "routers:write",
        payload_model=TokenRouterCreate,
        include_user=True,
    )
    route("/v1/token_routers/{router_uid}", get_router, ["GET"], "routers:read")
    route(
        "/v1/token_routers/{router_uid}",
        update_router,
        ["PUT"],
        "routers:write",
        payload_model=TokenRouterUpdate,
        include_user=True,
    )
    route(
        "/v1/token_routers/{router_uid}",
        delete_router,
        ["DELETE"],
        "routers:write",
    )
    route(
        "/v1/token_routers/{router_uid}/validate",
        validate_router,
        ["POST"],
        "routers:operate",
    )
    route(
        "/v1/token_routers/{router_uid}/enable",
        set_router_enabled,
        ["POST"],
        "routers:operate",
        include_user=True,
        enabled=True,
    )
    route(
        "/v1/token_routers/{router_uid}/disable",
        set_router_enabled,
        ["POST"],
        "routers:operate",
        include_user=True,
        enabled=False,
    )
    route(
        "/v1/token_routers/{router_uid}/status",
        router_status,
        ["GET"],
        "routers:read",
    )
    route(
        "/v1/token_routers/{router_uid}/instances",
        router_instances,
        ["GET"],
        "routers:read",
    )
    route(
        "/v1/token_routers/{router_uid}/metrics-summary",
        router_metrics,
        ["GET"],
        "routers:read",
    )

    async def register_runtime_handler(
        request: Request,
        _: None = Depends(_require_internal),
        api_: "RESTfulAPI" = Depends(get_api),
    ) -> JSONResponse:
        payload = await _parse_payload(request, RouterRuntimeRegister)
        return await runtime_register(payload, api_)

    async def heartbeat_runtime_handler(
        instance_id: str,
        request: Request,
        _: None = Depends(_require_internal),
        api_: "RESTfulAPI" = Depends(get_api),
    ) -> JSONResponse:
        payload = await _parse_payload(request, RouterRuntimeHeartbeat)
        return await runtime_heartbeat(instance_id, payload, api_)

    async def get_runtime_config_handler(
        router_uid: str,
        after_revision: int = Query(default=0, ge=0),
        _: None = Depends(_require_internal),
        api_: "RESTfulAPI" = Depends(get_api),
    ) -> Response:
        try:
            return await runtime_config(router_uid, after_revision, api_)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="Token Router not found"
            ) from exc

    async def ack_runtime_handler(
        instance_id: str,
        request: Request,
        _: None = Depends(_require_internal),
        api_: "RESTfulAPI" = Depends(get_api),
    ) -> JSONResponse:
        payload = await _parse_payload(request, RouterConfigAck)
        return await runtime_ack(instance_id, payload, api_)

    async def unregister_runtime_handler(
        instance_id: str,
        _: None = Depends(_require_internal),
        api_: "RESTfulAPI" = Depends(get_api),
    ) -> JSONResponse:
        return await runtime_unregister(instance_id, api_)

    router.add_api_route(
        "/v1/internal/token-router/instances/register",
        register_runtime_handler,
        methods=["POST"],
    )
    router.add_api_route(
        "/v1/internal/token-router/instances/{instance_id}/heartbeat",
        heartbeat_runtime_handler,
        methods=["POST"],
    )
    router.add_api_route(
        "/v1/internal/token-router/configs/{router_uid}",
        get_runtime_config_handler,
        methods=["GET"],
    )
    router.add_api_route(
        "/v1/internal/token-router/instances/{instance_id}/config-ack",
        ack_runtime_handler,
        methods=["POST"],
    )
    router.add_api_route(
        "/v1/internal/token-router/instances/{instance_id}/unregister",
        unregister_runtime_handler,
        methods=["POST"],
    )
