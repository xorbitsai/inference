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
from unittest.mock import AsyncMock, Mock

import pytest

from ..model import ModelActor


@pytest.fixture(autouse=True)
def inline_threads(monkeypatch):
    monkeypatch.setattr(
        "asyncio.to_thread", AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw))
    )


def _actor(method, response):
    model = SimpleNamespace(
        model_family=SimpleNamespace(to_description=lambda: {}), allow_batch=True
    )
    impl = (Mock if method == "generate" else AsyncMock)(return_value=response)
    setattr(model, method, impl)
    actor = ModelActor("test:123", "test:345", model, "test-model-0")  # type: ignore
    actor._model_state = "ready"
    actor.record_metrics = AsyncMock()
    return actor


def _assert_metrics(actor, tokens, stream):
    calls = actor.record_metrics.await_args_list
    for name, value in (
        ("input_tokens_total_counter", 7),
        ("output_tokens_total_counter", 3),
        ("generate_tokens_total", 3),
    ):
        recorded = [c.args[1:] for c in calls if c.args[0] == name]
        assert recorded == (
            [("add", {"labels": actor._metrics_labels, "value": value})]
            if tokens
            else []
        )
    ttft = [c.args[1:] for c in calls if c.args[0] == "time_to_first_token_seconds"]
    assert len(ttft) == 1
    op, data = ttft[0]
    assert op == "observe"
    assert data["labels"] == {**actor._metrics_labels, "stream": stream}
    assert data["value"] >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "async_generate", "chat"])
@pytest.mark.parametrize(
    "response,tokens",
    [
        pytest.param(
            {"usage": {"prompt_tokens": 7, "completion_tokens": 3}}, True, id="tokens"
        ),
        ({"choices": []}, False),
        ({"usage": None}, False),
        ({"usage": "invalid"}, False),
        ({"usage": {"prompt_tokens": None, "completion_tokens": 3}}, False),
        ({"usage": {"prompt_tokens": 7, "completion_tokens": None}}, False),
    ],
)
async def test_non_stream_completion_metrics(method, response, tokens):
    actor = _actor(method, response)
    result = await (actor.chat([]) if method == "chat" else actor.generate("hi"))
    assert json.loads(result) == response
    _assert_metrics(actor, tokens, "false")


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "async_generate"])
async def test_stream_generate_records_metrics_once(method):
    async def chunks():
        yield {"choices": [{"text": "hi"}]}
        yield {"usage": {"prompt_tokens": 7, "completion_tokens": 3}}

    actor = _actor(method, chunks())
    result = await actor.generate("hi")
    result._actor_ref = actor
    result._gc_destroy = False
    assert len([chunk async for chunk in result]) == 2
    _assert_metrics(actor, True, "true")
