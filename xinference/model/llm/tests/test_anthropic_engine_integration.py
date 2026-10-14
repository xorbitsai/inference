# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Native regressions, enabled by XINFERENCE_TEST_ANTHROPIC_ENGINE.

Set XINFERENCE_TEST_ANTHROPIC_ENDPOINT and XINFERENCE_TEST_ANTHROPIC_MODEL to
check an existing test deployment. Otherwise launch a dedicated test model.
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
                    enable_prefix_caching=True,
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
                model=model,
                endpoint=endpoint,
                key=key,
                anthropic=anthropic,
                openai=openai,
            )
    finally:
        if launched and client is not None:
            client.terminate_model(model)
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


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("colors", [("red",), ("red", "blue")])
def test_images(native_model, colors, stream):
    assert "vision" in native_model.abilities
    result = create(
        native_model,
        [
            {
                "role": "user",
                "content": [
                    *(image(color) for color in colors),
                    {
                        "type": "text",
                        "text": "Name the dominant color of each image in order. Answer briefly in English.",
                    },
                ],
            }
        ],
        stream=stream,
        thinking={"type": "disabled"},
    )
    assert result.stop_reason == "end_turn", result
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
            ("finish", "Record the value returned by lookup."),
        ]
    ]


@pytest.mark.parametrize("stream", [False, True])
def test_thinking_replay_and_tool_loop(native_model, stream):
    assert "reasoning" in native_model.abilities
    messages = [
        {
            "role": "user",
            "content": "Look up alpha, then record the value that lookup returns.",
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
    if native_model.engine != "MLX":
        kwargs["tool_choice"] = {"type": "tool", "name": "lookup"}
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
                }
            ],
        },
    ]
    if native_model.engine != "MLX":
        kwargs["tool_choice"] = {"type": "tool", "name": "finish"}
    second = create(native_model, messages, stream=stream, **kwargs)
    calls = [b for b in second.content if b.type == "tool_use"]
    assert len(calls) == 1 and calls[0].name == "finish", second
    assert calls[0].input == {"value": "NATIVE_RESULT_42"}, second
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
    if native.engine == "MLX":
        from anthropic import APIStatusError

        with pytest.raises(APIStatusError) as error:
            create(
                native,
                request["messages"],
                tools=tools(),
                tool_choice={"type": "any"},
                thinking=request["thinking"],
            )
        assert error.value.status_code == 501
    if native.engine == "SGLang":
        response = requests.post(
            native.endpoint + "/v1/messages/count_tokens",
            headers={"anthropic-version": "2023-06-01", "x-api-key": native.key},
            json={
                "model": native.model,
                "messages": [{"role": "user", "content": [image("red")]}],
            },
            timeout=30,
        )
        assert response.status_code == 501, response.text


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("json_output", [False, True])
def test_openai_path(native_model, stream, json_output):
    if json_output and native_model.engine == "MLX":
        pytest.skip("MLX has no constrained JSON support")
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
    assert (
        json.loads(output) == {"answer": 42}
        if json_output
        else "ready" in output.lower()
    ), output


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
    record_testsuite_property(
        "claude_version",
        subprocess.check_output([executable, "--version"], text=True).strip(),
    )
    (tmp_path / "claude.jsonl").write_text(result.stdout, encoding="utf-8")
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
        assert "NATIVE_RESULT_42" in final["result"], final
        assert any(
            b.get("type") == "tool_use" and b.get("name") == "Read"
            for event in events
            if event.get("type") == "assistant"
            for b in event["message"]["content"]
        )
