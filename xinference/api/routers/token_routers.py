# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Token-aware Router control-plane REST endpoints."""

from __future__ import annotations

import hmac
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type

from fastapi import Depends, Header, HTTPException, Query, Request, Security
from fastapi.responses import Response

from ..._compat import BaseModel, ValidationError
from ...constants import XINFERENCE_TOKEN_ROUTER_ENABLED
from ..dependencies import get_api
from ..responses import JSONResponse
from ..schemas.router import (
    RouterAssignmentStatus,
    RouterConfigAck,
    RouterNodeHeartbeat,
    RouterNodeLabelsUpdate,
    RouterNodeRegister,
    RouterNodeStateUpdate,
    RouterRuntimeHeartbeat,
    RouterRuntimeRegister,
    TokenizerAssetBindingCreate,
    TokenizerAssetBindingStatus,
    TokenizerAssetBindingUpdate,
    TokenizerAssetCreate,
    TokenizerAssetUpdate,
    TokenRouterCreate,
    TokenRouterDeploymentUpdate,
    TokenRouterUpdate,
    normalize_token_router_backend_url,
)

if TYPE_CHECKING:
    from ..restful_api import RESTfulAPI


_DEFAULT_BACKEND_URL_ENV = "XINFERENCE_TOKEN_ROUTER_DEFAULT_BACKEND_URL"
_UNUSABLE_REST_HOSTS = {"", "0.0.0.0", "::", "127.0.0.1", "localhost"}


def _unavailable_backend_defaults(source: str, error: str) -> Dict[str, Any]:
    return {
        "backend": {
            "mode": "current_supervisor",
            "display_name": "Current Supervisor",
            "backend_url": None,
            "source": source,
            "available": False,
            "error": error,
        }
    }


def _rest_host_url(host: Any, port: Any) -> Optional[str]:
    normalized_host = str(host or "").strip()
    if normalized_host.lower() in _UNUSABLE_REST_HOSTS or port is None:
        return None
    try:
        normalized_port = int(port)
    except (TypeError, ValueError):
        return None
    if ":" in normalized_host and not normalized_host.startswith("["):
        normalized_host = f"[{normalized_host}]"
    return f"http://{normalized_host}:{normalized_port}"


def _token_router_defaults(api: "RESTfulAPI") -> Dict[str, Any]:
    configured = os.environ.get(_DEFAULT_BACKEND_URL_ENV, "").strip()
    if configured:
        try:
            backend_url = normalize_token_router_backend_url(configured)
        except ValueError:
            return _unavailable_backend_defaults(
                "server_config",
                f"{_DEFAULT_BACKEND_URL_ENV} is not a valid base HTTP(S) endpoint",
            )
        source = "server_config"
    else:
        candidate = _rest_host_url(
            getattr(api, "_host", ""), getattr(api, "_port", None)
        )
        if candidate is None:
            return _unavailable_backend_defaults(
                "unavailable",
                f"Configure {_DEFAULT_BACKEND_URL_ENV} with the internal Supervisor REST endpoint",
            )
        try:
            backend_url = normalize_token_router_backend_url(candidate)
        except ValueError:
            return _unavailable_backend_defaults(
                "unavailable",
                f"Configure {_DEFAULT_BACKEND_URL_ENV} with the internal Supervisor REST endpoint",
            )
        source = "rest_endpoint"

    return {
        "backend": {
            "mode": "current_supervisor",
            "display_name": "Current Supervisor",
            "backend_url": backend_url,
            "source": source,
            "available": True,
        }
    }


def _username(user: Optional[dict]) -> str:
    return user.get("username", "") if user else ""


def _internal_authorized(authorization: str) -> bool:
    expected = os.environ.get("XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN", "")
    if not expected or not authorization.startswith("Bearer "):
        return False
    return hmac.compare_digest(authorization[7:], expected)


def _require_token_router_enabled() -> None:
    if not XINFERENCE_TOKEN_ROUTER_ENABLED:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "TOKEN_ROUTER_DISABLED",
                "message": "Token Router feature is disabled",
            },
        )


def _require_internal(authorization: str = Header(default="")) -> None:
    _require_token_router_enabled()
    if not _internal_authorized(authorization):
        raise HTTPException(status_code=401, detail="Invalid Token Router credential")


async def _parse_payload(request: Request, payload_model: Type[BaseModel]):
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


async def get_router_defaults(api: "RESTfulAPI") -> JSONResponse:
    return JSONResponse(content=_token_router_defaults(api))


async def list_tokenizer_assets(api: "RESTfulAPI") -> JSONResponse:
    return JSONResponse(content=await (await _supervisor(api)).list_tokenizer_assets())


async def create_tokenizer_asset(
    payload: TokenizerAssetCreate, api: "RESTfulAPI", user: Optional[dict]
) -> JSONResponse:
    try:
        result = await (await _supervisor(api)).create_tokenizer_asset(
            payload.dict(), _username(user)
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(status_code=201, content=result)


async def update_tokenizer_asset(
    asset_id: str,
    payload: TokenizerAssetUpdate,
    api: "RESTfulAPI",
    user: Optional[dict],
) -> JSONResponse:
    try:
        result = await (await _supervisor(api)).update_tokenizer_asset(
            asset_id, payload.dict(exclude_none=True), _username(user)
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="Tokenizer asset not found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(content=result)


async def delete_tokenizer_asset(asset_id: str, api: "RESTfulAPI") -> JSONResponse:
    try:
        deleted = await (await _supervisor(api)).delete_tokenizer_asset(asset_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Tokenizer asset not found")
    return JSONResponse(content={"status": "ok"})


async def list_tokenizer_asset_bindings(
    asset_id: str, api: "RESTfulAPI"
) -> JSONResponse:
    return JSONResponse(
        content=await (await _supervisor(api)).list_tokenizer_asset_bindings(
            asset_id=asset_id
        )
    )


async def get_tokenizer_asset_binding(
    asset_id: str, node_id: str, api: "RESTfulAPI"
) -> JSONResponse:
    items = await (await _supervisor(api)).list_tokenizer_asset_bindings(
        asset_id=asset_id, node_id=node_id
    )
    if not items:
        raise HTTPException(status_code=404, detail="Tokenizer Asset Binding not found")
    return JSONResponse(content=items[0])


async def create_tokenizer_asset_bindings(
    asset_id: str,
    payload: TokenizerAssetBindingCreate,
    api: "RESTfulAPI",
    user: Optional[dict],
) -> JSONResponse:
    try:
        result = await (await _supervisor(api)).create_tokenizer_asset_bindings(
            asset_id, payload.dict(), _username(user)
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="Asset or Router node not found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(status_code=201, content=result)


async def update_tokenizer_asset_binding(
    asset_id: str,
    node_id: str,
    payload: TokenizerAssetBindingUpdate,
    api: "RESTfulAPI",
    user: Optional[dict],
) -> JSONResponse:
    try:
        result = await (await _supervisor(api)).update_tokenizer_asset_binding(
            asset_id, node_id, payload.dict(exclude_none=True), _username(user)
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="Tokenizer Asset Binding not found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(content=result)


async def delete_tokenizer_asset_binding(
    asset_id: str, node_id: str, api: "RESTfulAPI"
) -> JSONResponse:
    try:
        deleted = await (await _supervisor(api)).delete_tokenizer_asset_binding(
            asset_id, node_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Tokenizer Asset Binding not found")
    return JSONResponse(content={"status": "ok"})


async def revalidate_tokenizer_asset_binding(
    asset_id: str, node_id: str, api: "RESTfulAPI", user: Optional[dict]
) -> JSONResponse:
    try:
        result = await (await _supervisor(api)).revalidate_tokenizer_asset_binding(
            asset_id, node_id, _username(user)
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="Tokenizer Asset Binding not found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(content=result)


async def list_backend_candidates(
    api: "RESTfulAPI", tokenizer_asset_id: str = ""
) -> JSONResponse:
    return JSONResponse(
        content=await (await _supervisor(api)).list_token_router_backend_candidates(
            tokenizer_asset_id
        )
    )


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
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
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


async def get_router_deployment(router_uid: str, api: "RESTfulAPI") -> JSONResponse:
    result = await (await _supervisor(api)).get_token_router_deployment(router_uid)
    if result is None:
        raise HTTPException(status_code=404, detail="Token Router not found")
    return JSONResponse(content=result)


async def update_router_deployment(
    router_uid: str, payload: TokenRouterDeploymentUpdate, api: "RESTfulAPI"
) -> JSONResponse:
    try:
        result = await (await _supervisor(api)).update_token_router_deployment(
            router_uid, payload.dict(exclude_none=True)
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Token Router not found") from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(content=result)


async def router_assignments(router_uid: str, api: "RESTfulAPI") -> JSONResponse:
    if await (await _supervisor(api)).get_token_router(router_uid) is None:
        raise HTTPException(status_code=404, detail="Token Router not found")
    return JSONResponse(
        content=await (await _supervisor(api)).list_token_router_assignments(
            router_uid=router_uid
        )
    )


async def list_router_nodes(
    api: "RESTfulAPI", include_offline: str = ""
) -> JSONResponse:
    normalized = include_offline.strip().lower()
    if normalized not in {"", "0", "1", "false", "true", "no", "yes"}:
        raise HTTPException(
            status_code=422,
            detail="include_offline must be a boolean value",
        )
    # Preserve the historical API default for compatibility.  The Web UI uses
    # include_offline=false explicitly for its active-node view.
    include = normalized in {"", "1", "true", "yes"}
    return JSONResponse(
        content=await (await _supervisor(api)).list_token_router_nodes(
            include_offline=include
        )
    )


async def get_router_node(node_id: str, api: "RESTfulAPI") -> JSONResponse:
    result = await (await _supervisor(api)).get_token_router_node(node_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Router node not found")
    return JSONResponse(content=result)


async def update_router_node_state(
    node_id: str, payload: RouterNodeStateUpdate, api: "RESTfulAPI"
) -> JSONResponse:
    try:
        result = await (await _supervisor(api)).set_token_router_node_state(
            node_id, payload.desired_state
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Router node not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(content=result)


async def update_router_node_labels(
    node_id: str, payload: RouterNodeLabelsUpdate, api: "RESTfulAPI"
) -> JSONResponse:
    try:
        result = await (await _supervisor(api)).set_token_router_node_labels(
            node_id, payload.labels
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Router node not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(content=result)


async def list_router_node_tokenizer_assets(
    node_id: str, api: "RESTfulAPI"
) -> JSONResponse:
    if await (await _supervisor(api)).get_token_router_node(node_id) is None:
        raise HTTPException(status_code=404, detail="Router node not found")
    return JSONResponse(
        content=await (await _supervisor(api)).list_tokenizer_asset_bindings(
            node_id=node_id
        )
    )


async def register_router_node(
    payload: RouterNodeRegister, api: "RESTfulAPI"
) -> JSONResponse:
    try:
        result = await (await _supervisor(api)).register_token_router_node(
            payload.dict()
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(content=result)


async def heartbeat_router_node(
    node_id: str, payload: RouterNodeHeartbeat, api: "RESTfulAPI"
) -> JSONResponse:
    try:
        result = await (await _supervisor(api)).heartbeat_token_router_node(
            node_id, payload.dict()
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Router node not found") from exc
    return JSONResponse(content=result)


async def report_assignment_status(
    assignment_id: str, payload: RouterAssignmentStatus, api: "RESTfulAPI"
) -> JSONResponse:
    try:
        result = await (await _supervisor(api)).report_token_router_assignment_status(
            assignment_id, payload.dict()
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="Router Assignment not found"
        ) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(content=result)


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
        payload_model: Optional[Type[BaseModel]] = None,
        include_user: bool = False,
        enabled: Optional[bool] = None,
        query_params: tuple[str, ...] = (),
    ) -> None:
        async def invoke(
            request: Request,
            api_: "RESTfulAPI",
            user: Optional[dict],
        ) -> JSONResponse:
            _require_token_router_enabled()
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

        handler: Callable[..., Any]
        if is_auth:

            async def authed(
                request: Request,
                user: dict = Security(auth, scopes=[scope]),
                api_: "RESTfulAPI" = Depends(get_api),
            ) -> JSONResponse:
                return await invoke(request, api_, user)

            handler = authed
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
        "/v1/tokenizer_assets",
        create_tokenizer_asset,
        ["POST"],
        "routers:write",
        payload_model=TokenizerAssetCreate,
        include_user=True,
    )
    route(
        "/v1/tokenizer_assets/{asset_id}",
        get_tokenizer_asset,
        ["GET"],
        "routers:read",
    )
    route(
        "/v1/tokenizer_assets/{asset_id}",
        update_tokenizer_asset,
        ["PATCH"],
        "routers:write",
        payload_model=TokenizerAssetUpdate,
        include_user=True,
    )
    route(
        "/v1/tokenizer_assets/{asset_id}",
        delete_tokenizer_asset,
        ["DELETE"],
        "routers:write",
    )
    route(
        "/v1/tokenizer_assets/{asset_id}/bindings",
        list_tokenizer_asset_bindings,
        ["GET"],
        "routers:read",
    )
    route(
        "/v1/tokenizer_assets/{asset_id}/bindings",
        create_tokenizer_asset_bindings,
        ["POST"],
        "routers:write",
        payload_model=TokenizerAssetBindingCreate,
        include_user=True,
    )
    route(
        "/v1/tokenizer_assets/{asset_id}/bindings/{node_id}",
        get_tokenizer_asset_binding,
        ["GET"],
        "routers:read",
    )
    route(
        "/v1/tokenizer_assets/{asset_id}/bindings/{node_id}",
        update_tokenizer_asset_binding,
        ["PATCH"],
        "routers:write",
        payload_model=TokenizerAssetBindingUpdate,
        include_user=True,
    )
    route(
        "/v1/tokenizer_assets/{asset_id}/bindings/{node_id}",
        delete_tokenizer_asset_binding,
        ["DELETE"],
        "routers:write",
    )
    route(
        "/v1/tokenizer_assets/{asset_id}/bindings/{node_id}/revalidate",
        revalidate_tokenizer_asset_binding,
        ["POST"],
        "routers:operate",
        include_user=True,
    )
    route(
        "/v1/tokenizer_assets/{asset_id}/validate",
        validate_tokenizer_asset,
        ["POST"],
        "routers:operate",
    )

    route("/v1/token_routers", list_routers, ["GET"], "routers:list")
    route(
        "/v1/token_routers/defaults",
        get_router_defaults,
        ["GET"],
        "routers:list",
    )
    route(
        "/v1/token_routers/backend-candidates",
        list_backend_candidates,
        ["GET"],
        "routers:list",
        query_params=("tokenizer_asset_id",),
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
        "/v1/token_routers/{router_uid}/deployment",
        get_router_deployment,
        ["GET"],
        "routers:read",
    )
    route(
        "/v1/token_routers/{router_uid}/deployment",
        update_router_deployment,
        ["PUT"],
        "routers:write",
        payload_model=TokenRouterDeploymentUpdate,
    )
    route(
        "/v1/token_routers/{router_uid}/assignments",
        router_assignments,
        ["GET"],
        "routers:read",
    )
    route(
        "/v1/token_router_nodes",
        list_router_nodes,
        ["GET"],
        "routers:list",
        query_params=("include_offline",),
    )
    route(
        "/v1/token_router_nodes/{node_id}",
        get_router_node,
        ["GET"],
        "routers:read",
    )
    route(
        "/v1/token_router_nodes/{node_id}/tokenizer_assets",
        list_router_node_tokenizer_assets,
        ["GET"],
        "routers:read",
    )
    route(
        "/v1/token_router_nodes/{node_id}/labels",
        update_router_node_labels,
        ["PATCH"],
        "routers:write",
        payload_model=RouterNodeLabelsUpdate,
    )
    route(
        "/v1/token_router_nodes/{node_id}/state",
        update_router_node_state,
        ["PUT"],
        "routers:operate",
        payload_model=RouterNodeStateUpdate,
    )
    route(
        "/v1/token_routers/{router_uid}/metrics-summary",
        router_metrics,
        ["GET"],
        "routers:read",
    )

    async def register_node_handler(
        request: Request,
        _: None = Depends(_require_internal),
        api_: "RESTfulAPI" = Depends(get_api),
    ) -> JSONResponse:
        payload = await _parse_payload(request, RouterNodeRegister)
        return await register_router_node(payload, api_)

    async def heartbeat_node_handler(
        node_id: str,
        request: Request,
        _: None = Depends(_require_internal),
        api_: "RESTfulAPI" = Depends(get_api),
    ) -> JSONResponse:
        payload = await _parse_payload(request, RouterNodeHeartbeat)
        return await heartbeat_router_node(node_id, payload, api_)

    async def watch_assignments_handler(
        node_id: str,
        after_cursor: str = Query(default=""),
        wait_seconds: float = Query(default=30.0, ge=0.0, le=60.0),
        _: None = Depends(_require_internal),
        api_: "RESTfulAPI" = Depends(get_api),
    ) -> Response:
        try:
            result = await (await _supervisor(api_)).watch_token_router_assignments(
                node_id, after_cursor, wait_seconds
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="Router node not found"
            ) from exc
        if result is None:
            return Response(status_code=204)
        return JSONResponse(content=result)

    async def watch_asset_bindings_handler(
        node_id: str,
        after_cursor: str = Query(default=""),
        wait_seconds: float = Query(default=30.0, ge=0.0, le=60.0),
        _: None = Depends(_require_internal),
        api_: "RESTfulAPI" = Depends(get_api),
    ) -> Response:
        try:
            result = await (await _supervisor(api_)).watch_tokenizer_asset_bindings(
                node_id, after_cursor, wait_seconds
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="Router node not found"
            ) from exc
        if result is None:
            return Response(status_code=204)
        return JSONResponse(content=result)

    async def asset_binding_status_handler(
        node_id: str,
        request: Request,
        _: None = Depends(_require_internal),
        api_: "RESTfulAPI" = Depends(get_api),
    ) -> JSONResponse:
        payload = await _parse_payload(request, TokenizerAssetBindingStatus)
        data = payload.dict()
        asset_id = data.pop("asset_id")
        try:
            result = await (
                await _supervisor(api_)
            ).report_tokenizer_asset_binding_status(asset_id, node_id, data)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="Tokenizer Asset Binding not found"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(content=result)

    async def assignment_status_handler(
        assignment_id: str,
        request: Request,
        _: None = Depends(_require_internal),
        api_: "RESTfulAPI" = Depends(get_api),
    ) -> JSONResponse:
        payload = await _parse_payload(request, RouterAssignmentStatus)
        return await report_assignment_status(assignment_id, payload, api_)

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
        "/v1/internal/token-router/nodes/register",
        register_node_handler,
        methods=["POST"],
    )
    router.add_api_route(
        "/v1/internal/token-router/nodes/{node_id}/heartbeat",
        heartbeat_node_handler,
        methods=["POST"],
    )
    router.add_api_route(
        "/v1/internal/token-router/nodes/{node_id}/assignments",
        watch_assignments_handler,
        methods=["GET"],
    )
    router.add_api_route(
        "/v1/internal/token-router/nodes/{node_id}/asset-bindings",
        watch_asset_bindings_handler,
        methods=["GET"],
    )
    router.add_api_route(
        "/v1/internal/token-router/nodes/{node_id}/asset-bindings/status",
        asset_binding_status_handler,
        methods=["POST"],
    )
    router.add_api_route(
        "/v1/internal/token-router/assignments/{assignment_id}/status",
        assignment_status_handler,
        methods=["PUT"],
    )
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
