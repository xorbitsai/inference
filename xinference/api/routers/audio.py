"""Audio route registration (embeddings, transcriptions, translations, speech)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request, Security
from fastapi.responses import JSONResponse

from ..utils import require_model

if TYPE_CHECKING:
    from ..restful_api import RESTfulAPI


def register_routes(api: "RESTfulAPI") -> None:
    router = api._router
    auth = api._auth_service
    is_auth = api.is_authenticated()

    async def capabilities(request: Request, model_uid: str) -> JSONResponse:
        api._check_model_access(request, model_uid, "audio")
        model = await require_model(
            api._get_supervisor_ref, model_uid, api._report_error_event
        )
        return JSONResponse(content=await model.get_audio_capabilities())

    router.add_api_route(
        "/v1/models/{model_uid}/audio/capabilities",
        capabilities,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/audio/embeddings",
        api.create_audio_embedding,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
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
