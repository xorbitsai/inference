# Copyright 2022-2026 Xinference Holdings Pte. Ltd
import copy
from types import SimpleNamespace

import pytest
from anthropic.types import ThinkingBlock, ToolUseBlock

from . import test_anthropic_engine_integration as native


@pytest.mark.parametrize(
    "engine,failure",
    [
        ("MLX", None),
        ("MLX", "lookup"),
        ("MLX", "result"),
        ("MLX", "parallel"),
        ("MLX", "second_parallel"),
        ("vLLM", None),
        ("vLLM", "parallel"),
        ("vLLM", "second_parallel"),
    ],
)
def test_tool_loop_probe_keeps_result_validation(monkeypatch, engine, failure):
    lookup = ToolUseBlock(
        id="lookup", type="tool_use", name="lookup", input={"value": "alpha"}
    )
    premature = ToolUseBlock(
        id="early-finish", type="tool_use", name="finish", input={"value": "alpha"}
    )
    finish = ToolUseBlock(
        id="finish",
        type="tool_use",
        name="finish",
        input={"value": "alpha" if failure == "result" else "NATIVE_RESULT_42"},
    )
    first = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=42),
        content=[
            ThinkingBlock(type="thinking", thinking="Use lookup.", signature="sig"),
            *([] if failure == "lookup" else [lookup]),
            *([premature] if failure == "parallel" else []),
        ],
    )
    second = SimpleNamespace(
        content=([lookup, finish] if failure == "second_parallel" else [finish]),
        stop_reason="tool_use",
    )
    requests = []

    def create(model, messages, **kwargs):
        requests.append(copy.deepcopy(messages))
        return first if len(requests) == 1 else second

    monkeypatch.setattr(native, "create", create)
    counts = []

    def count(url, **kwargs):
        counts.append(kwargs["json"])
        assert url == "http://probe/v1/messages/count_tokens"
        assert kwargs["json"]["messages"] == requests[0]
        return SimpleNamespace(status_code=200, json=lambda: {"input_tokens": 42})

    monkeypatch.setattr(native.requests, "post", count)
    model = SimpleNamespace(
        engine=engine,
        abilities=["reasoning"],
        endpoint="http://probe",
        key="test",
        model="probe",
    )
    if failure:
        with pytest.raises(AssertionError):
            native.test_thinking_replay_and_tool_loop(model, stream=False)
        assert len(requests) == (2 if failure in {"result", "second_parallel"} else 1)
        return

    native.test_thinking_replay_and_tool_loop(model, stream=False)
    assert len(requests) == 2
    assert len(counts) == 1
    results = [
        part for part in requests[1][-1]["content"] if part["type"] == "tool_result"
    ]
    assert [result["tool_use_id"] for result in results] == ["lookup"]
    assert results[0]["content"] == "NATIVE_RESULT_42"
