"""Audio route registration (transcriptions, translations, speech)."""

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
        "/v1/audio/transcriptions",
        api.create_transcriptions,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/audio/translations",
        api.create_translations,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/audio/speech",
        api.create_speech,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
