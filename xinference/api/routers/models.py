"""Model management route registration (list / launch / terminate / register / etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Security

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
        dependencies=([Security(auth, scopes=["models:add"])] if is_auth else None),
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
        dependencies=([Security(auth, scopes=["models:start"])] if is_auth else None),
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
        dependencies=([Security(auth, scopes=["models:start"])] if is_auth else None),
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
        dependencies=([Security(auth, scopes=["models:stop"])] if is_auth else None),
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
        dependencies=([Security(auth, scopes=["models:stop"])] if is_auth else None),
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
            [Security(auth, scopes=["models:unregister"])] if is_auth else None
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

    # --- Gradio UI ---
    router.add_api_route(
        "/v1/ui/{model_uid}",
        api.build_gradio_interface,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/ui/images/{model_uid}",
        api.build_gradio_media_interface,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/ui/audios/{model_uid}",
        api.build_gradio_media_interface,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/ui/videos/{model_uid}",
        api.build_gradio_media_interface,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
