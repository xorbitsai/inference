import asyncio
import builtins
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from ..docanalyze import mineru25


@pytest.mark.parametrize(
    "allowed_model,status", [("allowed-model", 403), ("mineru", 200)]
)
def test_docanalyze_enforces_model_authorization(
    tmp_path, monkeypatch, allowed_model, status
):
    from xinference.api import restful_api
    from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService
    from xinference.api.oauth2.advanced.crypto import get_password_hash
    from xinference.api.routers.images import register_routes
    from xinference.core.utils import CancelMixin

    service = AdvancedAuthService(
        db_path=str(tmp_path / "auth.db"),
        jwt_secret_key="test-secret",
        encryption_key="test-encryption-key",
    )
    user_id = service.db.create_user(
        username="docanalyze-user",
        password_hash=get_password_hash("pass"),
        source="local",
        enabled=1,
        must_change_password=0,
        permissions=["models:read"],
    )
    key = service.create_api_key_for_user(
        user_id=user_id,
        model_permissions=[
            {"permission_type": "model_id", "permission_value": allowed_model}
        ],
    )["key"]
    assert service.validate_model_access(key, "mineru", "image") == (status == 200)
    api = restful_api.RESTfulAPI.__new__(restful_api.RESTfulAPI)
    CancelMixin.__init__(api)
    api._router = APIRouter()
    api._auth_service = service
    api._advanced_auth_service = service
    api._get_supervisor_ref = AsyncMock()
    api._report_error_event = AsyncMock()
    api._set_trace_model = Mock()
    api._set_trace_model_type = Mock()
    api._record_audit = Mock()
    actor = SimpleNamespace(docanalyze=AsyncMock(return_value=b"[]"))
    lookup = AsyncMock(return_value=actor)
    monkeypatch.setattr(restful_api, "require_model", lookup)
    register_routes(api)
    app = FastAPI()
    app.include_router(api._router)
    with TestClient(app) as client:
        response = client.post(
            "/v1/images/docanalyze",
            headers={"Authorization": f"Bearer {key}"},
            data={"model": "mineru"},
            files={"file": ("document.pdf", b"pdf", "application/pdf")},
        )
    assert response.status_code == status
    if status == 403:
        lookup.assert_not_awaited()
        actor.docanalyze.assert_not_awaited()
    else:
        assert response.json() == []
        lookup.assert_awaited_once()
        actor.docanalyze.assert_awaited_once_with(
            file_bytes=b"pdf", file_name="document.pdf"
        )


def test_transformers_loads_without_vllm(monkeypatch):
    # Keep the real vendor package and lazy loader. This startup path does not
    # perform async file I/O, so provide that optional dependency for minimal CI.
    aiofiles = ModuleType("aiofiles")
    aiofiles.open = Mock(
        side_effect=AssertionError("No file I/O expected during startup")
    )
    monkeypatch.setitem(sys.modules, "aiofiles", aiofiles)

    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm."):
            monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "vllm", None)
    attempted = []
    original_import = builtins.__import__

    def import_without_vllm(name, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm."):
            attempted.append(name)
            raise ModuleNotFoundError("No module named 'vllm'", name="vllm")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_vllm)
    weights = SimpleNamespace(
        generate=Mock(), config=SimpleNamespace(max_position_embeddings=4096)
    )
    processor = SimpleNamespace(apply_chat_template=Mock(), tokenizer=SimpleNamespace())
    transformers = ModuleType("transformers")
    transformers.__version__ = "4.57.6"
    transformers.Qwen2VLForConditionalGeneration = SimpleNamespace(
        from_pretrained=Mock(return_value=weights)
    )
    transformers.AutoProcessor = SimpleNamespace(
        from_pretrained=Mock(return_value=processor)
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    model = mineru25.Mineru2_5Model(
        "mineru",
        model_path="model-path",
        model_spec=SimpleNamespace(model_ability=["docanalyze"]),
        backend="transformers",
        batch_size=2,
    )
    model.load()
    from xinference.thirdparty.mineru_vl_utils import MinerUClient
    from xinference.thirdparty.mineru_vl_utils.vlm_client.transformers_client import (
        TransformersVlmClient,
    )

    assert isinstance(model._model, MinerUClient)
    assert isinstance(model._model.client, TransformersVlmClient)
    assert model._model.client.model is weights
    assert model._model.client.batch_size == 2
    assert attempted == []


def actor_stub(count=0):
    return SimpleNamespace(
        _serve_count=count,
        _request_limits=1,
        _metrics_labels={},
        model_uid=lambda: "mineru",
        record_metrics=AsyncMock(),
        _require_ready=Mock(),
        _model=SimpleNamespace(docanalyze=AsyncMock()),
        _call_wrapper_json=AsyncMock(return_value=b"[]"),
    )


def test_docanalyze_rejects_at_capacity():
    from xinference.core.model import ModelActor

    actor = actor_stub(count=1)
    with pytest.raises(RuntimeError, match="Rate limit reached"):
        asyncio.run(ModelActor.docanalyze(actor, b"pdf", "document.pdf"))
    actor._call_wrapper_json.assert_not_awaited()
    actor._require_ready.assert_not_called()
    assert actor._serve_count == 1


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
def test_docanalyze_restores_serve_count(outcome):
    from xinference.core.model import ModelActor

    actor = actor_stub()

    async def call(*args, **kwargs):
        assert actor._serve_count == 1
        if outcome == "error":
            raise ValueError("inference failed")
        if outcome == "cancel":
            raise asyncio.CancelledError()
        return b"[]"

    actor._call_wrapper_json.side_effect = call
    if outcome == "success":
        assert (
            asyncio.run(ModelActor.docanalyze(actor, b"pdf", "document.pdf")) == b"[]"
        )
    else:
        with pytest.raises(
            ValueError if outcome == "error" else asyncio.CancelledError
        ):
            asyncio.run(ModelActor.docanalyze(actor, b"pdf", "document.pdf"))
    assert actor._serve_count == 0
    actor._call_wrapper_json.assert_awaited_once()
