"""Embeddings route registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Security

if TYPE_CHECKING:
    from ..restful_api import RESTfulAPI


def register_routes(api: "RESTfulAPI") -> None:
    router = api._router
    auth = api._auth_service
    is_auth = api.is_authenticated()

    router.add_api_route(
        "/v1/embeddings",
        api.create_embedding,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/convert_ids_to_tokens",
        api.convert_ids_to_tokens,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
