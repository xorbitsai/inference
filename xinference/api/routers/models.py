"""Model management route registration (list / launch / terminate / register / etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable

from fastapi import Request, Response, Security

if TYPE_CHECKING:
    from ..restful_api import RESTfulAPI


def register_routes(api: "RESTfulAPI") -> None:
    router = api._router
    auth = api._auth_service
    is_auth = api.is_authenticated()

    # --- must be registered before /v1/models/{model_uid} to avoid conflicts ---
    router.add_api_route(
        "/v1/models/prompts", api._get_builtin_prompts, methods=["GET"]
    )
    router.add_api_route(
        "/v1/models/families", api._get_builtin_families, methods=["GET"]
    )
    router.add_api_route(
        "/v1/models/llm/auto-register",
        api.build_llm_registration_from_config,
        methods=["POST"],
        dependencies=(
            [Security(auth, scopes=["models:register"])] if is_auth else None
        ),
    )
    router.add_api_route(
        "/v1/models/vllm-supported",
        api.list_vllm_supported_model_families,
        methods=["GET"],
    )
    router.add_api_route(
        "/v1/models/update_type",
        api.update_model_type,
        methods=["POST"],
        dependencies=(
            [Security(auth, scopes=["models:register"])] if is_auth else None
        ),
    )
    router.add_api_route(
        "/v1/models/instances",
        api.get_instance_info,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:list"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/models/instance",
        api.launch_model_by_version,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:write"])] if is_auth else None),
    )

    # --- engines ---
    router.add_api_route(
        "/v1/engines/{model_name}",
        api.query_engines_by_model_name,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:list"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/engines/{model_type}/{model_name}",
        api.query_engines_by_model_name,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:list"])] if is_auth else None),
    )

    # --- model versions ---
    router.add_api_route(
        "/v1/models/{model_type}/{model_name}/versions",
        api.get_model_versions,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:list"])] if is_auth else None),
    )

    get_autostart_config_handler: Callable[..., Awaitable[Any]]
    upsert_autostart_model_handler: Callable[..., Awaitable[Any]]
    if is_auth:

        async def get_autostart_config_handler_authed(
            user: Any = Security(auth, scopes=["models:write"]),
        ) -> Response:
            return await api.get_autostart_config(user)

        async def upsert_autostart_model_handler_authed(
            request: Request,
            user: Any = Security(auth, scopes=["models:write"]),
        ) -> Response:
            return await api.upsert_autostart_model(request, user)

        get_autostart_config_handler = get_autostart_config_handler_authed
        upsert_autostart_model_handler = upsert_autostart_model_handler_authed
    else:

        async def get_autostart_config_handler_anon() -> Response:
            return await api.get_autostart_config(None)

        async def upsert_autostart_model_handler_anon(request: Request) -> Response:
            return await api.upsert_autostart_model(request, None)

        get_autostart_config_handler = get_autostart_config_handler_anon
        upsert_autostart_model_handler = upsert_autostart_model_handler_anon

    # --- model autostart ---
    router.add_api_route(
        "/v1/autostart/models",
        get_autostart_config_handler,
        methods=["GET"],
    )
    router.add_api_route(
        "/v1/autostart/models/summary",
        api.get_autostart_model_summary,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:list"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/autostart/models",
        upsert_autostart_model_handler,
        methods=["POST"],
    )
    router.add_api_route(
        "/v1/autostart/models/{model_uid}",
        api.remove_autostart_model,
        methods=["DELETE"],
        dependencies=([Security(auth, scopes=["models:write"])] if is_auth else None),
    )

    # --- CRUD on running models ---
    router.add_api_route(
        "/v1/models",
        api.list_models,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:list"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/models",
        api.launch_model,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:write"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/models/{model_uid}",
        api.describe_model,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:list"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/models/{model_uid}",
        api.terminate_model,
        methods=["DELETE"],
        dependencies=([Security(auth, scopes=["models:write"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/models/{model_uid}/events",
        api.get_model_events,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/models/{model_uid}/replicas",
        api.get_model_replicas,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:list"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/models/{model_uid}/replicas/{replica_id}",
        api.terminate_model_replica,
        methods=["DELETE"],
        dependencies=([Security(auth, scopes=["models:write"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/models/{model_uid}/requests/{request_id}/abort",
        api.abort_request,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/models/{model_uid}/progress",
        api.get_launch_model_progress,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/models/{model_uid}/cancel",
        api.cancel_launch_model,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:write"])] if is_auth else None),
    )

    # --- model registrations ---
    router.add_api_route(
        "/v1/model_registrations/{model_type}",
        api.register_model,
        methods=["POST"],
        dependencies=(
            [Security(auth, scopes=["models:register"])] if is_auth else None
        ),
    )
    router.add_api_route(
        "/v1/model_registrations/{model_type}/{model_name}",
        api.unregister_model,
        methods=["DELETE"],
        dependencies=(
            [Security(auth, scopes=["models:register"])] if is_auth else None
        ),
    )
    router.add_api_route(
        "/v1/model_registrations/{model_type}",
        api.list_model_registrations,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:list"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/model_registrations/{model_type}/{model_name}",
        api.get_model_registrations,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:list"])] if is_auth else None),
    )
