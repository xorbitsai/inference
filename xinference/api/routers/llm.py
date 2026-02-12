"""LLM route registration (completions, chat/completions, Anthropic messages, flexible infer)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Security

from ...types import ANTHROPIC_AVAILABLE, AnthropicMessage, ChatCompletion, Completion

if TYPE_CHECKING:
    from ..restful_api import RESTfulAPI


def register_routes(api: "RESTfulAPI") -> None:
    router = api._router
    auth = api._auth_service
    is_auth = api.is_authenticated()

    router.add_api_route(
        "/v1/completions",
        api.create_completion,
        methods=["POST"],
        response_model=Completion,
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )

    if ANTHROPIC_AVAILABLE:
        router.add_api_route(
            "/anthropic/v1/messages",
            api.create_message,
            methods=["POST"],
            response_model=AnthropicMessage,
            dependencies=(
                [Security(auth, scopes=["models:read"])] if is_auth else None
            ),
        )
        router.add_api_route(
            "/anthropic/v1/models",
            api.anthropic_list_models,
            methods=["GET"],
            dependencies=(
                [Security(auth, scopes=["models:list"])] if is_auth else None
            ),
        )
        router.add_api_route(
            "/anthropic/v1/models/{model_id}",
            api.anthropic_get_model,
            methods=["GET"],
            dependencies=(
                [Security(auth, scopes=["models:list"])] if is_auth else None
            ),
        )

    router.add_api_route(
        "/v1/chat/completions",
        api.create_chat_completion,
        methods=["POST"],
        response_model=ChatCompletion,
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/flexible/infers",
        api.create_flexible_infer,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )
