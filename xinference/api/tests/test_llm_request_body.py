# CI: linux
# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import xoscar as xo
from fastapi import FastAPI

from ...core.pd_model import PDModelActor
from .. import restful_api
from .test_llm_request_abort import _new_api
from .test_model_request_logging import _app

ROUTES = {
    "/v1/completions": ("create_completion", '{"model": "m", "prompt": "hi"'),
    "/v1/chat/completions": (
        "create_chat_completion",
        '{"model": "m", "messages": [{"role": "user", "content": "hi"}]',
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ROUTES)
@pytest.mark.parametrize(
    "body,detail",
    [
        ("truncated", "Request body must be valid JSON"),
        ("truncated-stream", "Request body must be valid JSON"),
        ("[]", "Request body must be a JSON object"),
        ("null", "Request body must be a JSON object"),
    ],
)
async def test_llm_routes_reject_a_body_that_is_not_a_json_object(
    monkeypatch, path, body, detail
):
    handler, prefix = ROUTES[path]
    if body == "truncated":
        body = prefix
    elif body == "truncated-stream":
        body = prefix + ', "stream": true'
    require_model = AsyncMock()
    monkeypatch.setattr(restful_api, "require_model", require_model)
    api = _new_api()
    api._get_supervisor_ref = AsyncMock()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=_app(path, getattr(api, handler)), raise_app_exceptions=False
        ),
        base_url="http://test",
    ) as client:
        response = await client.post(
            path, content=body, headers={"content-type": "application/json"}
        )

    assert response.status_code == 400
    assert response.json() == {"detail": detail}
    require_model.assert_not_awaited()
    api._get_supervisor_ref.assert_not_awaited()


class _PDStage(xo.Actor):
    def __init__(self, prefill=False):
        super().__init__()
        self.prefill = prefill
        self.error = None

    async def set_error(self, error):
        self.error = error

    async def prepare_anthropic_request(self, messages, config):
        return dict(messages=messages, generate_config=config, input_tokens=1)

    async def abort_request(self, *args, **kwargs):
        return "DONE"

    async def decrease_serve_count(self, **kwargs):
        pass

    @xo.generator
    async def chat(self, messages, config, **kwargs):
        if self.prefill:
            return {"_pd_kv_transfer_params": {"do_remote_prefill": True}}
        if self.error is not None:
            raise self.error
        return json.dumps(
            dict(
                id="completion-1",
                model="pd",
                created=1,
                object="chat.completion",
                choices=[
                    dict(
                        index=0,
                        message=dict(role="assistant", content="ok"),
                        finish_reason="stop",
                    )
                ],
                usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        ).encode()

    generate = chat


@pytest.fixture
async def pd_api(monkeypatch):
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        prefill = await xo.create_actor(
            _PDStage, True, address=pool.external_address, uid=b"prefill"
        )
        decode = await xo.create_actor(
            _PDStage, address=pool.external_address, uid=b"decode"
        )
        model = await xo.create_actor(
            PDModelActor,
            "pd",
            model_engine="heterogeneous",
            address=pool.external_address,
            uid=b"pd",
        )
        await model.add_prefill_actor("p", prefill)
        await model.add_decode_actor("d", decode)
        api = object.__new__(restful_api.RESTfulAPI)
        api._advanced_auth_service = None
        api._uid_to_model_name = {}
        api._check_model_access = lambda *args: None
        api._set_trace_model = api._set_trace_model_type = lambda *args: None
        api._report_error_event = AsyncMock()
        api._get_supervisor_ref = AsyncMock(
            return_value=SimpleNamespace(
                describe_model=AsyncMock(
                    return_value={"model_name": "pd", "model_engine": "vLLM"}
                ),
                resolve_token_router_runtime=AsyncMock(return_value=None),
            )
        )
        monkeypatch.setattr(restful_api, "require_model", AsyncMock(return_value=model))
        monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
        app = FastAPI()
        for endpoint, handler in (
            ("completions", api.create_completion),
            ("chat/completions", api.create_chat_completion),
            ("responses", api.create_response),
            ("messages", api.create_message),
        ):
            app.add_api_route("/v1/" + endpoint, handler, methods=["POST"])
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
                headers={"anthropic-version": "2023-06-01"},
            ) as client:
                yield client, decode, model.address
        finally:
            await xo.destroy_actor(model)
            await xo.destroy_actor(prefill)
            await xo.destroy_actor(decode)


def _pd_body(endpoint, **kwargs):
    body = dict(model="pd", max_tokens=8, **kwargs)
    body.update(
        {"prompt": "hi"}
        if endpoint == "completions"
        else (
            {"input": "hi"}
            if endpoint == "responses"
            else {"messages": [{"role": "user", "content": "hi"}]}
        )
    )
    return body


_PD_N_ERROR = "PD KV handoff currently requires n=1"
_PD_SAMPLING_ERROR = (
    "Cross-engine PD does not yet transfer structured sampling or logprobs"
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint,options,detail",
    [
        ("chat/completions", {"n": 2}, _PD_N_ERROR),
        ("completions", {"n": 2}, _PD_N_ERROR),
        (
            "responses",
            {"text": {"format": {"type": "json_object"}}},
            _PD_SAMPLING_ERROR,
        ),
        (
            "messages",
            {
                "output_config": {
                    "format": {"type": "json_schema", "schema": {"type": "object"}}
                }
            },
            _PD_SAMPLING_ERROR,
        ),
    ],
)
async def test_pd_request_limits_are_client_errors(pd_api, endpoint, options, detail):
    client, _, address = pd_api
    response = await client.post("/v1/" + endpoint, json=_pd_body(endpoint, **options))
    assert response.status_code == 400
    data = response.json()
    if endpoint in ("responses", "messages"):
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["message"] == detail
    else:
        assert data["detail"] == detail
    assert address not in response.text and "[address=" not in response.text


@pytest.mark.asyncio
async def test_pd_valid_request(pd_api):
    client, _, _ = pd_api
    response = await client.post(
        "/v1/chat/completions", json=_pd_body("chat/completions", n=1)
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_pd_unrelated_actor_value_error_stays_500(pd_api):
    client, decode, _ = pd_api
    # An engine's ValueError with the same message is not a router validation.
    await decode.set_error(ValueError(_PD_N_ERROR))
    response = await client.post(
        "/v1/chat/completions", json=_pd_body("chat/completions")
    )
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_pd_streaming_rejection_before_headers(pd_api):
    client, _, address = pd_api
    response = await client.post(
        "/v1/responses",
        json=_pd_body(
            "responses", stream=True, text={"format": {"type": "json_object"}}
        ),
    )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == _PD_SAMPLING_ERROR
    assert address not in response.text and "[address=" not in response.text


@pytest.mark.asyncio
async def test_pd_streaming_rejection_after_headers_keeps_status(pd_api):
    client, _, address = pd_api
    response = await client.post(
        "/v1/chat/completions",
        json=_pd_body("chat/completions", stream=True, guided_json={"type": "object"}),
    )
    assert response.status_code == 200
    events = [
        json.loads(line[5:])
        for line in response.text.splitlines()
        if line.startswith("data:")
    ]
    assert events == [{"error": _PD_SAMPLING_ERROR}]
    assert address not in response.text and "[address=" not in response.text
