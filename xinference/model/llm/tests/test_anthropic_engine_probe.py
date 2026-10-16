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
        content=[
            ThinkingBlock(type="thinking", thinking="Use lookup.", signature="sig"),
            *([] if failure == "lookup" else [lookup]),
            *([premature] if engine == "MLX" or failure == "parallel" else []),
        ]
    )
    second = SimpleNamespace(
        content=(
            [lookup, finish]
            if engine == "MLX" or failure == "second_parallel"
            else [finish]
        ),
        stop_reason="tool_use",
    )
    requests = []

    def create(model, messages, **kwargs):
        requests.append(copy.deepcopy(messages))
        return first if len(requests) == 1 else second

    monkeypatch.setattr(native, "create", create)
    model = SimpleNamespace(engine=engine, abilities=["reasoning"])
    if failure:
        with pytest.raises(AssertionError):
            native.test_thinking_replay_and_tool_loop(model, stream=False)
        assert len(requests) == (2 if failure in {"result", "second_parallel"} else 1)
        return

    native.test_thinking_replay_and_tool_loop(model, stream=False)
    assert len(requests) == 2
    results = requests[1][-1]["content"]
    assert [result["tool_use_id"] for result in results] == (
        ["lookup", "early-finish"] if engine == "MLX" else ["lookup"]
    )
    assert results[0]["content"] == "NATIVE_RESULT_42"
    if engine == "MLX":
        assert results[1]["is_error"] is True
