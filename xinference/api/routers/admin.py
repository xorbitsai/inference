"""Admin / cluster / infrastructure route registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Security

if TYPE_CHECKING:
    from ..restful_api import RESTfulAPI


def register_routes(api: "RESTfulAPI") -> None:
    router = api._router
    auth = api._auth_service
    is_auth = api.is_authenticated()

    # internal
    router.add_api_route("/status", api.get_status, methods=["GET"])
    router.add_api_route("/v1/address", api.get_address, methods=["GET"])

    # auth
    router.add_api_route("/token", api.login_for_access_token, methods=["POST"])
    router.add_api_route(
        "/v1/cluster/auth", api.is_cluster_authenticated, methods=["GET"]
    )

    # cluster
    router.add_api_route(
        "/v1/cluster/info",
        api.get_cluster_device_info,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cluster/version",
        api.get_cluster_version,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cluster/devices",
        api._get_devices_count,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:list"])] if is_auth else None),
    )

    # workers / supervisor
    router.add_api_route(
        "/v1/workers",
        api.get_workers_info,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/supervisor",
        api.get_supervisor_info,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/clusters",
        api.abort_cluster,
        methods=["DELETE"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )

    # cache
    router.add_api_route(
        "/v1/cache/models",
        api.list_cached_models,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["cache:list"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cache/models/files",
        api.list_model_files,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["cache:list"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cache/models",
        api.confirm_and_remove_model,
        methods=["DELETE"],
        dependencies=([Security(auth, scopes=["cache:delete"])] if is_auth else None),
    )

    # virtualenvs
    router.add_api_route(
        "/v1/virtualenvs",
        api.list_virtual_envs,
        methods=["GET"],
        dependencies=(
            [Security(auth, scopes=["virtualenv:list"])] if is_auth else None
        ),
    )
    router.add_api_route(
        "/v1/virtualenvs",
        api.remove_virtual_env,
        methods=["DELETE"],
        dependencies=(
            [Security(auth, scopes=["virtualenv:delete"])] if is_auth else None
        ),
    )

    # progress
    router.add_api_route(
        "/v1/requests/{request_id}/progress",
        api.get_progress,
        methods=["get"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
