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
            {"model_name": "test", "model_type": "embedding"},
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
