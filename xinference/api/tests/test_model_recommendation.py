from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from xinference.api.restful_api import RESTfulAPI
from xinference.api.routers import models
from xinference.core.model_recommendation import RecommendationModelNotFound


@pytest.mark.asyncio
async def test_recommendation_api_validation_errors_and_contract():
    supervisor = AsyncMock()
    supervisor.recommend_model.return_value = {
        "status": "no_recommendation",
        "config": None,
        "reasons": [],
        "warnings": [],
    }
    api = SimpleNamespace(_get_supervisor_ref=AsyncMock(return_value=supervisor))
    app = FastAPI()
    app.add_api_route(
        "/v1/models/recommend",
        RESTfulAPI.recommend_model.__get__(api),
        methods=["POST"],
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for body in [
            {},
            {"model_name": "test", "model_type": "image"},
            {"model_name": "test", "constraints": {"n_gpu": 0}},
            {"model_name": "test", "constraints": {"typo": True}},
        ]:
            assert (
                await client.post("/v1/models/recommend", json=body)
            ).status_code == 422
        assert (
            await client.post("/v1/models/recommend", content="{")
        ).status_code == 422
        supervisor.recommend_model.assert_not_awaited()
        response = await client.post(
            "/v1/models/recommend",
            json={"model_name": "test", "constraints": {"n_gpu": None}},
        )
        assert response.status_code == 200 and response.json()["config"] is None
        supervisor.recommend_model.assert_awaited_once_with(
            {"model_name": "test", "constraints": {"n_gpu": None}}
        )
        for model_type in ("embedding", "rerank", "audio"):
            body = {"model_name": "test", "model_type": model_type}
            response = await client.post("/v1/models/recommend", json=body)
            assert response.status_code == 200
            supervisor.recommend_model.assert_awaited_with(body)
            response = await client.post(
                "/v1/models/recommend",
                json={**body, "constraints": {"model_size_in_billions": 7}},
            )
            assert response.status_code == 422
        for error, status in [
            (RecommendationModelNotFound("not found"), 404),
            (KeyError("private failure"), 500),
            (RuntimeError("private failure"), 500),
        ]:
            supervisor.recommend_model.side_effect = error
            response = await client.post(
                "/v1/models/recommend", json={"model_name": "test"}
            )
            assert response.status_code == status
            assert "private failure" not in response.text


@pytest.mark.parametrize("authenticated", [True, False])
def test_recommendation_route_uses_model_list_permission(authenticated):
    api = MagicMock()
    api.is_authenticated.return_value = authenticated
    models.register_routes(api)
    routes = api._router.add_api_route.call_args_list
    recommendation = next(c for c in routes if c.args[0] == "/v1/models/recommend")
    assert recommendation.kwargs["methods"] == ["POST"]
    deps = recommendation.kwargs["dependencies"]
    assert deps[0].scopes == ["models:list"] if authenticated else deps is None
    assert routes.index(recommendation) < next(
        i for i, c in enumerate(routes) if c.args[0] == "/v1/models/{model_uid}"
    )


@pytest.mark.asyncio
async def test_launch_api_forwards_unified_context_and_rejects_invalid_values():
    from xinference.core.supervisor import SupervisorActor

    supervisor = SimpleNamespace(
        launch_builtin_model=AsyncMock(return_value="test-model")
    )
    api = SimpleNamespace(
        _get_supervisor_ref=AsyncMock(return_value=supervisor), _uid_to_model_name={}
    )
    app = FastAPI()
    app.add_api_route(
        "/v1/models", RESTfulAPI.launch_model.__get__(api), methods=["POST"]
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        body = {"model_name": "test", "model_engine": "vLLM", "context_length": 32768}
        response = await client.post("/v1/models", json=body)
        assert response.status_code == 200
        assert (
            supervisor.launch_builtin_model.call_args.kwargs["context_length"] == 32768
        )
        supervisor.launch_builtin_model = SupervisorActor.launch_builtin_model.__get__(
            SimpleNamespace()
        )
        for value in [0, -1, True, 32768.5, "32768"]:
            response = await client.post(
                "/v1/models", json={**body, "context_length": value}
            )
            assert response.status_code == 400
            assert "context_length" in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("context_error", [False, True])
async def test_context_overflow_is_a_client_error(
    chat, stream, context_error, monkeypatch
):
    from xinference.api import restful_api
    from xinference.model.llm.context_length import ContextLengthExceeded

    error_cls = ContextLengthExceeded if context_error else ValueError
    error = error_cls("Prompt has 5000 tokens, but context_length is 4096")
    model = SimpleNamespace(
        uid="test",
        is_vllm_backend=AsyncMock(return_value=False),
        is_transformers_batching_enabled=AsyncMock(return_value=False),
        generate=AsyncMock(side_effect=error),
        chat=AsyncMock(side_effect=error),
        abort_request=AsyncMock(return_value="DONE"),
        decrease_serve_count=AsyncMock(),
    )
    monkeypatch.setattr(restful_api, "require_model", AsyncMock(return_value=model))
    supervisor = SimpleNamespace(
        describe_model=AsyncMock(return_value={}),
        resolve_token_router_runtime=AsyncMock(return_value=None),
    )
    api = SimpleNamespace(
        extract_guided_params=lambda **kwargs: {},
        _set_trace_model=lambda *args: None,
        _set_trace_model_type=lambda *args: None,
        _check_model_access=lambda *args: None,
        _get_supervisor_ref=AsyncMock(return_value=supervisor),
        _report_error_event=AsyncMock(),
        _get_model_last_error=AsyncMock(side_effect=lambda uid, error: error),
        _is_qwen38_description=lambda desc: False,
        handle_request_limit_error=RESTfulAPI.handle_request_limit_error,
        _chat_completion_error_detail=RESTfulAPI._chat_completion_error_detail,
    )
    app = FastAPI()
    method = RESTfulAPI.create_chat_completion if chat else RESTfulAPI.create_completion
    path = "/v1/chat/completions" if chat else "/v1/completions"
    app.add_api_route(path, method.__get__(api), methods=["POST"])
    body = {"model": "test", "stream": stream}
    body.update(
        {"messages": [{"role": "user", "content": "hello"}]}
        if chat
        else {"prompt": "hello"}
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(path, json=body)
    if stream:
        assert response.status_code == 200
        detail = RESTfulAPI._chat_completion_error_detail() if chat else str(error)
        assert '"error":' in response.text and detail in response.text
    else:
        assert response.status_code == (400 if context_error else 500)
        detail = str(error)
        if chat:
            detail = (
                "Request exceeds the model's configured context length."
                if context_error
                else RESTfulAPI._chat_completion_error_detail()
            )
        assert response.json()["detail"] == detail
