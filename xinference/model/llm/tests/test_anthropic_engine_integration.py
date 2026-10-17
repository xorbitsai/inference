# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Native regressions, enabled by XINFERENCE_TEST_ANTHROPIC_ENGINE.

Set XINFERENCE_TEST_ANTHROPIC_ENDPOINT and XINFERENCE_TEST_ANTHROPIC_MODEL to
check an existing test deployment. Otherwise launch a dedicated test model.
XINFERENCE_TEST_ANTHROPIC_LAUNCH_CONFIG overrides launch parameters as JSON.
XINFERENCE_TEST_ANTHROPIC_API_KEY sets the server key.
XINFERENCE_TEST_ANTHROPIC_ENGINE_VERSION is required for an existing deployment.
XINFERENCE_TEST_CLAUDE_CODE=1 also exercises the installed Claude Code CLI.
"""

import base64
import io
import json
import os
import shutil
import subprocess
import uuid
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace

import pytest
import requests


@pytest.fixture(scope="module")
def native_model(record_testsuite_property):
    engine = os.environ.get("XINFERENCE_TEST_ANTHROPIC_ENGINE")
    if not engine:
        pytest.skip("Set XINFERENCE_TEST_ANTHROPIC_ENGINE for native model tests")
    assert engine in {"MLX", "vLLM", "SGLang"}
    from anthropic import Anthropic
    from openai import OpenAI

    from ....client import Client
    from ....conftest import _setup_cluster

    endpoint = os.environ.get("XINFERENCE_TEST_ANTHROPIC_ENDPOINT")
    cluster = None
    client = None
    launched = False
    image_limit = None
    if not endpoint:
        cluster = contextmanager(_setup_cluster)(use_test_pool=False)
        endpoint, _ = cluster.__enter__()
    key = os.environ.get("XINFERENCE_TEST_ANTHROPIC_API_KEY", "native-test")
    model = os.environ.get("XINFERENCE_TEST_ANTHROPIC_MODEL")
    try:
        client = Client(endpoint, api_key=key)
        if not model:
            assert not os.environ.get(
                "XINFERENCE_TEST_ANTHROPIC_ENDPOINT"
            ), "An existing endpoint requires XINFERENCE_TEST_ANTHROPIC_MODEL"
            config = {
                "model_name": "qwen3.5",
                "model_size_in_billions": 4,
                "model_engine": engine,
                "model_format": "mlx" if engine == "MLX" else "awq",
                "quantization": "4bit" if engine == "MLX" else "Int4",
                "enable_virtual_env": False,
                "enable_thinking": False,
                "reasoning_content": True,
            }
            if engine == "vLLM":
                config.update(
                    max_model_len=16384,
                    max_num_seqs=1,
                    gpu_memory_utilization=0.6,
                    enforce_eager=True,
                    # Keep repeated greedy probes on the same full-prefill path.
                    enable_prefix_caching=False,
                    limit_mm_per_prompt={"image": 2, "video": 0},
                )
            elif engine == "SGLang":
                config.update(
                    context_length=16384,
                    mem_fraction_static=0.65,
                    max_running_requests=1,
                    disable_cuda_graph=True,
                )
            else:
                config["context_length"] = 16384
            config.update(
                json.loads(
                    os.environ.get("XINFERENCE_TEST_ANTHROPIC_LAUNCH_CONFIG", "{}")
                )
            )
            config["model_uid"] = "anthropic-native-" + uuid.uuid4().hex[:12]
            raw_limit = config.get("limit_mm_per_prompt")
            if engine == "vLLM" and isinstance(raw_limit, dict):
                image_limit = raw_limit.get("image")
            model = client.launch_model(**config)
            launched = True
        description = client.list_models()[model]
        assert description["model_engine"] == engine, description
        abilities = set(description["model_ability"])
        assert "tools" in abilities, description
        for name in ["xinference", "anthropic", "openai"]:
            try:
                record_testsuite_property(name, version(name))
            except PackageNotFoundError:
                pass
        record_testsuite_property("model", json.dumps(description, sort_keys=True))
        record_testsuite_property("engine", engine)
        engine_version = os.environ.get("XINFERENCE_TEST_ANTHROPIC_ENGINE_VERSION")
        if os.environ.get("XINFERENCE_TEST_ANTHROPIC_ENDPOINT"):
            assert engine_version, "Record the external server's engine version"
        else:
            engine_version = version(
                {"MLX": "mlx-vlm", "vLLM": "vllm", "SGLang": "sglang"}[engine]
            )
        record_testsuite_property("engine_version", engine_version)
        with (
            Anthropic(
                base_url=endpoint, api_key=key, timeout=180, max_retries=0
            ) as anthropic,
            OpenAI(
                base_url=endpoint + "/v1", api_key=key, timeout=180, max_retries=0
            ) as openai,
        ):
            yield SimpleNamespace(
                engine=engine,
                abilities=abilities,
                image_limit=image_limit,
                model=model,
                endpoint=endpoint,
                key=key,
                anthropic=anthropic,
                openai=openai,
                description=description,
            )
    finally:
        try:
            if launched and client is not None:
                client.terminate_model(model)
        finally:
            if cluster is not None:
                cluster.__exit__(None, None, None)


def image(color):
    from PIL import Image

    data = io.BytesIO()
    Image.new("RGB", (128, 128), color).save(data, "PNG")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(data.getvalue()).decode(),
        },
    }


def text(message):
    return "".join(block.text for block in message.content if block.type == "text")


def create(native, messages, stream=False, **kwargs):
    request = dict(model=native.model, max_tokens=768, temperature=0, messages=messages)
    request.update(kwargs)
    # Newer Anthropic SDKs expose sampling controls only through extra_body.
    request["extra_body"] = {
        **{
            name: request.pop(name)
            for name in ("temperature", "top_p")
            if name in request
        },
        **request.get("extra_body", {}),
    }
    if not stream:
        return native.anthropic.messages.create(**request)
    with native.anthropic.messages.stream(**request) as response:
        events = list(response)
        result = response.get_final_message()
    assert events[0].type == "message_start"
    assert events[-1].type == "message_stop"
    assert any(event.type == "content_block_delta" for event in events)
    assert (
        result.usage.input_tokens
        + (getattr(result.usage, "cache_read_input_tokens", 0) or 0)
        > 0
    )
    assert result.usage.output_tokens > 0
    return result


def test_create_preserves_protocol_extensions():
    from ....api.protocols import parse_anthropic_request

    def send(**request):
        extra_body = request.pop("extra_body")
        return parse_anthropic_request({**request, **extra_body})

    native = SimpleNamespace(
        model="test", anthropic=SimpleNamespace(messages=SimpleNamespace(create=send))
    )
    request = create(
        native,
        [{"role": "user", "content": "hello"}],
        temperature=0.6,
        extra_body={
            "context_management": {"edits": [{"type": "clear_thinking_20251015"}]}
        },
    )
    assert "context_management" in request.requested_features
    assert request.temperature == 0.6


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "colors,in_tool_result",
    [(("red",), False), (("red", "blue"), False), (("red", "blue"), True)],
    ids=["single", "multiple", "tool-result"],
)
def test_images(native_model, colors, in_tool_result, stream):
    assert "vision" in native_model.abilities
    content = [image(color) for color in colors]
    messages = []
    kwargs = {}
    if in_tool_result:
        messages = [
            {"role": "user", "content": "Inspect the images returned by lookup."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_images",
                        "name": "lookup",
                        "input": {"value": "images"},
                    },
                ],
            },
        ]
        content = [
            {"type": "tool_result", "tool_use_id": "toolu_images", "content": content}
        ]
        kwargs = {"tools": tools(), "tool_choice": {"type": "none"}}
    content.append(
        {
            "type": "text",
            "text": "Name the dominant color of each image in order. Answer briefly in English.",
        }
    )
    result = create(
        native_model,
        messages + [{"role": "user", "content": content}],
        stream=stream,
        thinking={"type": "disabled"},
        **kwargs,
    )
    assert result.stop_reason == "end_turn", result
    counted = requests.post(
        native_model.endpoint + "/v1/messages/count_tokens",
        headers={"anthropic-version": "2023-06-01", "x-api-key": native_model.key},
        json={
            "model": native_model.model,
            "messages": messages + [{"role": "user", "content": content}],
            "thinking": {"type": "disabled"},
            **kwargs,
        },
        timeout=180,
    )
    assert counted.status_code == 200, counted.text
    assert counted.json()["input_tokens"] == result.usage.input_tokens + (
        getattr(result.usage, "cache_read_input_tokens", 0) or 0
    )
    answer = text(result).lower()
    assert all(color in answer for color in colors), result


def tools():
    return [
        {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        }
        for name, description in [
            ("lookup", "Look up a value."),
            (
                "finish",
                "Copy the latest lookup tool result verbatim into value.",
            ),
        ]
    ]


@pytest.mark.parametrize("stream", [False, True])
def test_thinking_replay_and_tool_loop(native_model, stream):
    assert "reasoning" in native_model.abilities
    messages = [
        {
            "role": "user",
            "content": "Call lookup with value alpha.",
        }
    ]
    kwargs = dict(
        max_tokens=4096,
        temperature=0.6,
        top_p=0.95,
        tools=tools(),
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
    )
    kwargs["tool_choice"] = {
        "type": "tool",
        "name": "lookup",
        **({"disable_parallel_tool_use": True} if native_model.engine == "MLX" else {}),
    }
    first = create(native_model, messages, stream=stream, **kwargs)
    thought = [b for b in first.content if b.type == "thinking"]
    assert thought and all(b.thinking and b.signature for b in thought), first
    calls = [b for b in first.content if b.type == "tool_use"]
    assert len(calls) == 1 and calls[0].name == "lookup", first
    messages += [
        {"role": "assistant", "content": [b.model_dump() for b in first.content]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": calls[0].id,
                    "content": "NATIVE_RESULT_42",
                },
                {
                    "type": "text",
                    "text": "Now call finish, copying the latest lookup tool result verbatim into finish.value.",
                },
            ],
        },
    ]
    kwargs["tool_choice"] = {
        "type": "tool",
        "name": "finish",
        **({"disable_parallel_tool_use": True} if native_model.engine == "MLX" else {}),
    }
    messages[:0] = [
        {"role": "user", "content": "An earlier task."},
        {
            "role": "assistant",
            "content": [
                *(block.model_dump() for block in thought),
                {
                    "type": "text",
                    "text": "The earlier task is complete.",
                },
            ],
        },
    ]
    # Qwen3 already hides thinking before the latest user query. This native
    # probe checks edit acceptance and active tool-loop replay; semantic clearing
    # is asserted in api/tests/test_anthropic_protocol.py.
    context_management = {
        "edits": [
            {
                "type": "clear_thinking_20251015",
                "keep": {"type": "thinking_turns", "value": 1},
            }
        ]
    }
    second = create(
        native_model,
        messages,
        stream=stream,
        extra_body={"context_management": context_management},
        **kwargs,
    )
    calls = [b for b in second.content if b.type == "tool_use"]
    assert len(calls) == 1 and calls[0].name == "finish", second
    assert all(call.input == {"value": "NATIVE_RESULT_42"} for call in calls), second
    assert second.stop_reason == "tool_use", second


def test_counting_and_unsupported_controls(native_model):
    native = native_model
    request = {
        "model": native.model,
        "messages": [{"role": "user", "content": "Reply READY."}],
        "thinking": {"type": "disabled"},
    }
    counted = native.anthropic.messages.count_tokens(**request)
    result = create(native, request["messages"], thinking=request["thinking"])
    assert counted.input_tokens == result.usage.input_tokens + (
        getattr(result.usage, "cache_read_input_tokens", 0) or 0
    )


@pytest.mark.parametrize("stream", [False, True])
def test_structured_schema_is_enforced(native_model, stream):
    result = create(
        native_model,
        [{"role": "user", "content": "Reply with plain text HELLO, not JSON."}],
        stream=stream,
        thinking={"type": "disabled"},
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"status": {"const": "SCHEMA_ACCEPTED"}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
            }
        },
    )
    assert json.loads(text(result)) == {"status": "SCHEMA_ACCEPTED"}
    assert result.stop_reason == "end_turn"


@pytest.mark.parametrize("stream", [False, True])
def test_sglang_thinking_budget(native_model, stream, record_property):
    if native_model.engine != "SGLang":
        pytest.skip("SGLang request-level reasoning budget")
    from transformers import AutoTokenizer

    from ..llm_family import match_llm

    description = native_model.description
    family = match_llm(
        description["model_name"],
        description["model_format"],
        description["model_size_in_billions"],
        description["quantization"],
    )
    assert family is not None
    launch = json.loads(os.environ.get("XINFERENCE_TEST_ANTHROPIC_LAUNCH_CONFIG", "{}"))
    tokenizer = AutoTokenizer.from_pretrained(
        launch.get("model_path") or family.model_specs[0].model_id
    )
    result = create(
        native_model,
        [
            {
                "role": "user",
                "content": (
                    "Consider every integer from 1 through 300, checking each for primality "
                    "one by one in your reasoning. Then state how many are prime."
                ),
            }
        ],
        stream=stream,
        max_tokens=1536,
        thinking={"type": "enabled", "budget_tokens": 1024},
    )
    thought = "".join(
        block.thinking for block in result.content if block.type == "thinking"
    )
    # The native closer may add a newline and its closing marker at the boundary.
    count = len(tokenizer.encode(thought, add_special_tokens=False))
    record_property("thinking_tokens", count)
    assert 0 < count <= 1026, result
    assert text(result).strip(), result


@pytest.mark.parametrize("stream", [False, True])
def test_mlx_cache_breakpoints(native_model, stream):
    if native_model.engine != "MLX":
        pytest.skip("MLX explicit cache checkpoints")
    system = [
        {
            "type": "text",
            "text": uuid.uuid4().hex + " Stable context. " * 64,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Earlier facts. " * 32,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "Reply READY."},
            ],
        }
    ]
    cold = create(
        native_model,
        messages,
        stream=stream,
        system=system,
        thinking={"type": "disabled"},
    )
    hot = create(
        native_model,
        messages,
        stream=stream,
        system=system,
        thinking={"type": "disabled"},
    )
    assert cold.usage.cache_creation_input_tokens > 0
    assert hot.usage.cache_read_input_tokens == cold.usage.cache_creation_input_tokens
    assert hot.usage.cache_creation_input_tokens == 0
    assert text(cold).strip() and text(hot).strip()


@pytest.mark.parametrize("stream", [False, True])
def test_text_model_rejects_images(native_model, stream):
    if "vision" in native_model.abilities:
        pytest.skip("Requires a text-only test model")
    response = requests.post(
        native_model.endpoint + "/v1/messages",
        headers={"anthropic-version": "2023-06-01", "x-api-key": native_model.key},
        json={
            "model": native_model.model,
            "max_tokens": 32,
            "stream": stream,
            "messages": [{"role": "user", "content": [image("red")]}],
        },
        timeout=30,
    )
    assert response.status_code == 400, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert "vision support" in response.json()["error"]["message"]


@pytest.mark.parametrize("limit_kind", ["protocol", "deployment"])
def test_image_limits(native_model, limit_kind):
    if limit_kind == "deployment" and (
        native_model.image_limit is None or native_model.image_limit >= 64
    ):
        pytest.skip("Requires a known deployment image limit below 64")
    limit = 64 if limit_kind == "protocol" else native_model.image_limit
    response = requests.post(
        native_model.endpoint + "/v1/messages",
        headers={"anthropic-version": "2023-06-01", "x-api-key": native_model.key},
        json={
            "model": native_model.model,
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": [image("red")] * (limit + 1)}],
        },
        timeout=30,
    )
    assert response.status_code == 400, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert f"at most {limit}" in response.json()["error"]["message"]


@pytest.mark.parametrize("stream", [False, True])
def test_stop_sequences(native_model, stream):
    if native_model.engine != "vLLM" or "reasoning" not in native_model.abilities:
        pytest.skip("Regression for vLLM reasoning families")
    messages = [
        {"role": "user", "content": "Reply exactly: ALPHA BRAVO CHARLIE DELTA."}
    ]
    kwargs = {"max_tokens": 128, "thinking": {"type": "disabled"}}
    baseline = create(native_model, messages, stream=stream, **kwargs)
    output = text(baseline)
    assert baseline.stop_reason == "end_turn" and baseline.stop_sequence is None
    assert len(output) > 8, baseline
    # Derive a stop from actual greedy output instead of testing spelling or
    # counting ability. Keep a nonempty prefix to exercise terminal SSE usage.
    offset = len(output) // 2
    stop = output[offset : offset + 8]
    offset = output.index(stop)
    assert offset > 0, baseline
    stopped = create(
        native_model, messages, stream=stream, stop_sequences=[stop], **kwargs
    )
    assert text(stopped) == output[:offset], (baseline, stopped)
    assert stop not in text(stopped)
    assert stopped.stop_reason == "stop_sequence", stopped
    assert stopped.stop_sequence == stop, stopped


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("json_output", [False, True])
def test_openai_path(native_model, stream, json_output):
    kwargs = {"response_format": {"type": "json_object"}} if json_output else {}
    prompt = (
        "Return JSON with key answer and value 42."
        if json_output
        else "Reply exactly READY."
    )
    result = native_model.openai.chat.completions.create(
        model=native_model.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        stream=stream,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        **kwargs,
    )
    if stream:
        chunks = list(result)
        output = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
        assert chunks[-1].choices[0].finish_reason == "stop"
    else:
        output = result.choices[0].message.content
        assert result.choices[0].finish_reason == "stop"
    assert "<think>" not in output, output
    if json_output:
        parsed = json.loads(output)
        assert isinstance(parsed, dict) and parsed.get("answer") in (42, "42"), output
    else:
        assert "ready" in output.lower(), output


@pytest.mark.parametrize("image_count", [0, 1, 2])
def test_claude_code(native_model, tmp_path, image_count, record_testsuite_property):
    if os.environ.get("XINFERENCE_TEST_CLAUDE_CODE") != "1":
        pytest.skip("Set XINFERENCE_TEST_CLAUDE_CODE=1 to run Claude Code")
    executable = shutil.which("claude")
    assert executable, "Claude Code is required when explicitly enabled"
    native = native_model
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("NATIVE_RESULT_42\n", encoding="utf-8")
    colors = ["red", "blue"][:image_count]
    content = [
        *(image(color) for color in colors),
        {
            "type": "text",
            "text": (
                "Name the dominant color of each image briefly in English."
                if colors
                else f"Use Read to read {fixture}, then reply with its contents."
            ),
        },
    ]
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("ANTHROPIC_", "CLAUDE_")) and k != "CLAUDECODE"
    }
    env.update(
        ANTHROPIC_BASE_URL=native.endpoint,
        ANTHROPIC_API_KEY=native.key,
        ANTHROPIC_DEFAULT_HAIKU_MODEL=native.model,
        ANTHROPIC_DEFAULT_SONNET_MODEL=native.model,
        ANTHROPIC_DEFAULT_OPUS_MODEL=native.model,
        ANTHROPIC_SMALL_FAST_MODEL=native.model,
        CLAUDE_CONFIG_DIR=str(tmp_path / "config"),
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
        DISABLE_TELEMETRY="1",
        DISABLE_AUTOUPDATER="1",
        CLAUDE_CODE_MAX_OUTPUT_TOKENS="1024",
        MAX_THINKING_TOKENS="0",
    )
    args = [
        executable,
        "--bare",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--model",
        native.model,
        "--tools",
        "" if colors else "Read",
        "--allowedTools",
        "Read",
        "--permission-mode",
        "dontAsk",
        "--no-session-persistence",
        "-p",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
    ]
    trace_path = tmp_path / "claude.jsonl"
    try:
        result = subprocess.run(
            args,
            input=json.dumps(
                {"type": "user", "message": {"role": "user", "content": content}}
            )
            + "\n",
            text=True,
            capture_output=True,
            cwd=tmp_path,
            env=env,
            timeout=240,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or b""
        trace_path.write_bytes(
            output.encode("utf-8") if isinstance(output, str) else output
        )
        raise
    trace_path.write_text(result.stdout, encoding="utf-8")
    record_testsuite_property(
        "claude_version",
        subprocess.check_output(
            [executable, "--version"], text=True, timeout=10
        ).strip(),
    )
    assert result.returncode == 0, result.stderr + result.stdout[-2000:]
    events = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    ]
    final = next(event for event in reversed(events) if event.get("type") == "result")
    assert not final.get("is_error"), final
    assert any(event.get("type") == "stream_event" for event in events)
    if colors:
        assert all(color in final["result"].lower() for color in colors), final
    else:
        read_ids = {
            b["id"]
            for event in events
            if event.get("type") == "assistant"
            for b in event["message"]["content"]
            if b.get("type") == "tool_use" and b.get("name") == "Read"
        }
        assert read_ids, events
        assert any(
            b.get("type") == "tool_result"
            and b.get("tool_use_id") in read_ids
            and not b.get("is_error")
            and "NATIVE_RESULT_42" in b.get("content", "")
            for event in events
            if event.get("type") == "user"
            for b in event["message"]["content"]
        )
        assert final["num_turns"] >= 2 and final["result"], final
