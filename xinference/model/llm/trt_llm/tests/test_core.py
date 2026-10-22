# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import copy
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ...core import chat_context_var
from ..core import TRTLLMModel


class Request:
    def __init__(self, fragments=("ab", "c")):
        self.prompt_token_ids = [10, 20]
        self.outputs = []
        self.finished = False
        self.aborted = False
        self.position = 0
        self.fragments = fragments

    def abort(self):
        self.aborted = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.position == len(self.fragments):
            raise StopAsyncIteration
        await asyncio.sleep(0)
        self.position += 1
        self.finished = self.position == len(self.fragments)
        text = "".join(self.fragments[: self.position])
        self.outputs = [
            SimpleNamespace(
                index=0,
                text=text,
                text_diff=self.fragments[self.position - 1],
                token_ids=list(range(len(text))),
                finish_reason="length" if self.finished else None,
            )
        ]
        return self

    def __await__(self):
        async def collect():
            async for _ in self:
                pass
            return self

        return collect().__await__()


class Engine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.requests = []
        self.tokenizer = SimpleNamespace(
            encode=lambda prompt, **kwargs: list(range(len(prompt))),
            chat_template="template",
            apply_chat_template=lambda messages, **kwargs: "hi",
        )

    def generate_async(self, tokens, params, streaming):
        self.tokens, self.params = tokens, params
        request = Request(getattr(self, "fragments", ("ab", "c")))
        self.requests.append(request)
        return request

    def shutdown(self):
        self.stopped = True


@pytest.fixture
def model(monkeypatch):
    def speculative_config(data):
        decoding_type = data.pop("decoding_type")
        return SimpleNamespace(decoding_type=decoding_type, **data)

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.llmapi.llm_args",
        SimpleNamespace(
            DecodingBaseConfig=SimpleNamespace(from_dict=speculative_config)
        ),
    )
    monkeypatch.setattr(
        "xinference.model.llm.trt_llm.compat.prepare_runtime",
        lambda config: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm",
        SimpleNamespace(LLM=Engine, SamplingParams=SimpleNamespace),
    )
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.2.1")
    family = SimpleNamespace(
        model_specs=[SimpleNamespace(quantization="none")],
        model_name="qwen3",
        model_family="qwen3",
        context_length=32,
        stop=["DEFAULT"],
        stop_token_ids=[999],
        model_ability=["chat", "hybrid"],
        reasoning_start_tag="<think>",
        reasoning_end_tag="</think>",
        chat_template=None,
    )
    result = TRTLLMModel(
        "tiny", family, "/model", {"max_seq_len": 32, "enable_thinking": False}
    )
    result.load()
    yield result
    result.stop()


@pytest.mark.asyncio
async def test_stream_is_incremental_and_usage_keeps_last_token(model):
    stream = await model.async_generate(
        "hi",
        {
            "max_tokens": 3,
            "stream": True,
            "stop": ["CUSTOM"],
            "stream_options": {"include_usage": True},
        },
        "request",
    )
    assert not model._engine.requests  # An unstarted/closed stream allocates nothing.
    first = await anext(stream)
    request = model._engine.requests[0]
    assert not request.finished
    assert first["choices"][0]["text"] == "ab"
    rest = [chunk async for chunk in stream]
    assert rest[0]["choices"][0]["text"] == "c"
    assert rest[0]["choices"][0]["finish_reason"] is None
    assert rest[1]["choices"][0]["text"] == ""
    assert rest[1]["choices"][0]["finish_reason"] == "length"
    assert rest[-1]["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    assert model._engine.params.max_tokens == 3
    assert model._engine.params.stop == ["CUSTOM"]
    assert not model._requests and not request.aborted


@pytest.mark.asyncio
async def test_close_stream_aborts_and_releases_request(model):
    stream = await model.async_generate("hi", {"stream": True}, "close")
    await anext(stream)
    request = model._engine.requests[-1]
    await stream.aclose()
    assert request.aborted and not model._requests
    assert await model.abort_request("close") == "NO_OP"


@pytest.mark.asyncio
async def test_explicit_abort_and_shutdown(model):
    stream = await model.async_generate("hi", {"stream": True}, "cancel")
    await anext(stream)
    assert await model.abort_request("cancel") == "DONE"
    assert model._engine.requests[-1].aborted
    engine = model._engine
    model.stop()
    assert engine.stopped
    await stream.aclose()


@pytest.mark.asyncio
async def test_cancel_nonstream_aborts_engine_request(model, monkeypatch):
    class WaitingRequest(Request):
        def __await__(self):
            return asyncio.Event().wait().__await__()

    request = WaitingRequest()
    monkeypatch.setattr(
        model._engine, "generate_async", lambda *args, **kwargs: request
    )
    task = asyncio.create_task(model.async_generate("hi", request_id="waiting"))
    await asyncio.sleep(0)
    assert model._requests["waiting"] is request
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert request.aborted and not model._requests


@pytest.mark.asyncio
async def test_duplicate_id_does_not_remove_original_request(model):
    first = await model.async_generate("hi", {"stream": True}, "same")
    await anext(first)
    request = model._requests["same"]
    duplicate = await model.async_generate("hi", {"stream": True}, "same")
    with pytest.raises(ValueError, match="already active"):
        await anext(duplicate)
    assert model._requests["same"] is request
    await first.aclose()


@pytest.mark.asyncio
async def test_late_stream_close_does_not_abort_a_shutdown_executor(model, monkeypatch):
    stream = await model.async_generate("hi", {"stream": True})
    await anext(stream)
    request = model._engine.requests[-1]
    model.stop()
    calls = []

    def unavailable():
        calls.append("abort")
        raise RuntimeError("executor was already shut down")

    monkeypatch.setattr(request, "abort", unavailable)
    await stream.aclose()
    assert not calls and not model._requests


@pytest.mark.asyncio
async def test_nonstream_clamps_to_context_without_mutating_input(model):
    config = {"max_tokens": 100, "temperature": 0.5}
    result = await model.async_generate("hi", config)
    assert result["choices"][0]["text"] == "abc"
    assert result["usage"]["completion_tokens"] == 3
    assert model._engine.params.max_tokens == 30
    assert config["max_tokens"] == 100
    with pytest.raises(ValueError, match="no room"):
        await model.async_generate("x" * 32)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        {"max_tokens": 0},
        {"max_tokens": True},
        {"n": 2},
        {"logprobs": 2},
        {"cache_salt": "tenant"},
    ],
)
async def test_unsupported_request_fails_before_submission(model, config):
    with pytest.raises(ValueError):
        await model.async_generate("hi", config)
    assert not model._engine.requests


@pytest.mark.asyncio
async def test_chat_and_chat_stream_use_shared_conversion(model):
    messages = [{"role": "user", "content": "hello"}]
    result = await model.async_chat(messages)
    assert result["choices"][0]["message"]["content"] == "abc"
    stream = await model.async_chat(messages, {"stream": True})
    chunks = [chunk async for chunk in stream]
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "abc"
    with pytest.raises(ValueError, match="plain text"):
        await model.async_chat([{"role": "user", "content": [{"type": "image_url"}]}])


def test_platform_format_and_model_guards(monkeypatch, model):
    from ...llm_family import match_llm

    family = match_llm("qwen3", "pytorch", "0_6", "none")
    spec = SimpleNamespace(model_format="pytorch")
    monkeypatch.setattr(sys, "platform", "linux")
    assert TRTLLMModel.match_json(family, spec, "none") is True
    assert TRTLLMModel.match_json(family, spec, "fp8")[0] is False
    family.architectures = ["Qwen3_5ForConditionalGeneration"]
    assert TRTLLMModel.match_json(family, spec, "none")[0] is False


def test_engine_construction_failure_clears_state(monkeypatch, model):
    monkeypatch.setattr(
        "xinference.model.llm.trt_llm.compat.prepare_runtime", lambda _: None
    )
    monkeypatch.setattr(
        sys.modules["tensorrt_llm"],
        "LLM",
        Mock(side_effect=RuntimeError("load failed")),
    )
    failed = TRTLLMModel("failed", model.model_family, "/model")
    with pytest.raises(RuntimeError, match="load failed"):
        failed.load()
    assert failed._engine is None and not failed._requests


@pytest.mark.parametrize("depth,expected", [(None, 3), ("2", 2), (5, 5)])
def test_draft_target_launch_options(model, depth, expected):
    model._model_config.update(draft_model_path="/draft", num_speculative_tokens=depth)
    model.stop()
    model.load()
    options = model._engine.kwargs
    assert vars(options["speculative_config"]) == {
        "decoding_type": "DraftTarget",
        "speculative_model": "/draft",
        "max_draft_len": expected,
    }
    assert options["disable_overlap_scheduler"] is True
    assert "draft_model_path" not in options
    assert "num_speculative_tokens" not in options


@pytest.mark.parametrize("depth", [0, -1, "invalid", 1.5])
def test_invalid_draft_depth_fails_before_engine_start(model, monkeypatch, depth):
    model.stop()
    engine = Mock()
    monkeypatch.setattr(sys.modules["tensorrt_llm"], "LLM", engine)
    model._model_config.update(draft_model_path="/draft", num_speculative_tokens=depth)
    with pytest.raises(ValueError, match="positive integer"):
        model.load()
    engine.assert_not_called()


def test_explicit_speculation_wins_and_engine_cannot_mutate_launch_config(
    model, monkeypatch, caplog
):
    model.stop()
    native = {
        "decoding_type": "NGram",
        "max_draft_len": 3,
        "max_matching_ngram_size": 4,
        "is_public_pool": False,
    }
    model._model_config.update(
        speculative_config=native,
        draft_model_path="/ignored",
        num_speculative_tokens="ignored",
        disable_overlap_scheduler=False,
    )

    def create(**kwargs):
        assert vars(kwargs["speculative_config"]) == native
        assert kwargs["disable_overlap_scheduler"] is False
        assert "draft_model_path" not in kwargs
        assert "num_speculative_tokens" not in kwargs
        return Engine(**kwargs)

    monkeypatch.setattr(sys.modules["tensorrt_llm"], "LLM", create)
    model.load()
    assert "Ignoring draft_model_path and num_speculative_tokens" in caplog.text
    assert native["decoding_type"] == "NGram"
    model.stop()
    model.load()  # Reusing launch state must still work.


@pytest.mark.parametrize("transport", ["_xavier_cache_config", "_nixl_config"])
@pytest.mark.parametrize(
    "speculation",
    [
        {"draft_model_path": "/draft"},
        {"speculative_config": {"decoding_type": "NGram", "max_draft_len": 3}},
    ],
)
def test_speculation_cannot_bypass_cache_and_pd_guards(
    model, monkeypatch, transport, speculation
):
    model.stop()
    engine = Mock()
    monkeypatch.setattr(sys.modules["tensorrt_llm"], "LLM", engine)
    model._model_config.update(speculation)
    model._model_config[transport] = {"role": "decode"}
    with pytest.raises(ValueError, match="requires ordinary replicas"):
        model.load()
    engine.assert_not_called()


def test_depth_without_drafter_is_rejected(model):
    model.stop()
    model._model_config["num_speculative_tokens"] = 3
    with pytest.raises(ValueError, match="requires draft_model_path"):
        model.load()


def test_empty_depth_without_drafter_is_unset(model):
    model.stop()
    model._model_config["num_speculative_tokens"] = ""
    model.load()
    assert "speculative_config" not in model._engine.kwargs
    assert "num_speculative_tokens" not in model._engine.kwargs


def test_speculative_config_json_string(model):
    import json

    model.stop()
    value = json.dumps({"decoding_type": "NGram", "max_draft_len": 3})
    model._model_config["speculative_config"] = value
    model.load()
    assert model._engine.kwargs["speculative_config"].decoding_type == "NGram"
    assert model._model_config["speculative_config"] == value


@pytest.mark.parametrize("value", ["invalid", "[]", "null", "3"])
def test_invalid_speculative_json_fails_before_engine_start(model, monkeypatch, value):
    model.stop()
    engine = Mock()
    monkeypatch.setattr(sys.modules["tensorrt_llm"], "LLM", engine)
    model._model_config["speculative_config"] = value
    with pytest.raises(ValueError, match="speculative_config must be a JSON object"):
        model.load()
    engine.assert_not_called()


@pytest.mark.parametrize("enable_mtp", [False, True])
def test_factory_accepts_local_or_declared_trt_drafter(model, monkeypatch, enable_mtp):
    from ... import llm_family
    from ...core import create_llm_model_instance

    match = llm_family.match_llm

    def declared_family(*args):
        family = match(*args).copy(deep=True)
        family.model_specs[0].draft_model_id = "Qwen/Qwen3-0.6B"
        return family

    monkeypatch.setattr(llm_family, "match_llm", declared_family)

    monkeypatch.setattr(
        "xinference.model.llm.llm_family.check_engine_by_spec_parameters",
        lambda *args: TRTLLMModel,
    )
    monkeypatch.setattr(
        "xinference.model.llm.cache_manager.LLMCacheManager.cache",
        lambda self: "/draft",
    )
    launched = create_llm_model_instance(
        "draft-target",
        "qwen3",
        "TRT-LLM",
        model_size_in_billions="0_6",
        model_format="pytorch",
        quantization="none",
        model_path="/target",
        enable_virtual_env=False,
        context_length=1024,
        **({"enable_mtp": True} if enable_mtp else {"draft_model_path": "/draft"}),
        num_speculative_tokens="2",
    )
    try:
        launched.load()
        config = launched._engine.kwargs
        assert config["speculative_config"].speculative_model == "/draft"
        assert config["speculative_config"].max_draft_len == 2
        assert config["max_seq_len"] == 1024
        assert "enable_mtp" not in config
    finally:
        launched.stop()


async def collect_chat(model, config):
    result = await model.async_chat([{"role": "user", "content": "hello"}], config)
    if not config.get("stream"):
        return result["choices"][0]["message"]
    chunks = [chunk async for chunk in result]
    return {
        field: "".join(
            choice["delta"].get(field) or ""
            for chunk in chunks
            for choice in chunk["choices"]
        )
        for field in ("content", "reasoning_content")
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("launch_thinking", [False, True])
async def test_request_thinking_overrides_do_not_mutate_parser(
    model, monkeypatch, stream, launch_thinking
):
    model._enable_thinking = launch_thinking
    model.prepare_parse_reasoning_content(True, launch_thinking)
    render = Mock(return_value="hi")
    monkeypatch.setattr(model._tokenizer, "apply_chat_template", render)
    context = {"enable_thinking": not launch_thinking}
    config = {"stream": stream, "chat_template_kwargs": context}
    token = chat_context_var.set({"enable_thinking": launch_thinking})
    try:
        message = await collect_chat(model, config)
        assert message["content"] == ("abc" if launch_thinking else "")
        assert (message.get("reasoning_content") or "") == (
            "" if launch_thinking else "abc"
        )
        assert render.call_args.kwargs["enable_thinking"] is not launch_thinking
        assert model.reasoning_parser.enable_thinking is launch_thinking
        assert config["chat_template_kwargs"] == {
            "enable_thinking": not launch_thinking
        }
        if not stream:
            assert chat_context_var.get()["enable_thinking"] is launch_thinking
    finally:
        chat_context_var.reset(token)


@pytest.mark.asyncio
async def test_interleaved_chat_streams_keep_their_thinking_context(model):
    model._enable_thinking = True
    model.prepare_parse_reasoning_content(True, True)
    token = chat_context_var.set({})
    try:
        streams = [
            await model.async_chat(
                [{"role": "user", "content": "hello"}],
                {"stream": True, "chat_template_kwargs": {"enable_thinking": thinking}},
            )
            for thinking in (False, True)
        ]
        chunks = [[], []]
        active = set(range(2))
        while active:
            for index in list(active):
                try:
                    chunks[index].append(await anext(streams[index]))
                except StopAsyncIteration:
                    active.remove(index)
        for index, field in enumerate(("content", "reasoning_content")):
            assert (
                "".join(
                    choice["delta"].get(field) or ""
                    for chunk in chunks[index]
                    for choice in chunk["choices"]
                )
                == "abc"
            )
        assert model.reasoning_parser.enable_thinking is True
    finally:
        chat_context_var.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "last,reasoning,content", [("c", "abc", ""), ("</think>answer", "ab", "answer")]
)
async def test_terminal_chat_text_is_parsed_before_finish(
    model, last, reasoning, content
):
    model.prepare_parse_reasoning_content(True, True)
    model._engine.fragments = ("ab", last)
    token = chat_context_var.set({"enable_thinking": True})
    try:
        stream = await model.async_chat(
            [{"role": "user", "content": "hello"}],
            {"stream": True, "stream_options": {"include_usage": True}},
        )
        chunks = [chunk async for chunk in stream]
        choices = [choice for chunk in chunks for choice in chunk["choices"]]
        assert (
            "".join(c["delta"].get("reasoning_content") or "" for c in choices)
            == reasoning
        )
        assert "".join(c["delta"].get("content") or "" for c in choices) == content
        terminal = [c for c in choices if c["finish_reason"] is not None]
        assert len(terminal) == 1 and terminal[0]["finish_reason"] == "length"
        assert not terminal[0]["delta"].get("content")
        assert chunks[-1]["usage"]["completion_tokens"] == len("ab" + last)
    finally:
        chat_context_var.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_chat_accepts_json_string_template_kwargs(model, stream):
    model._enable_thinking = True
    model.prepare_parse_reasoning_content(True, True)
    token = chat_context_var.set({})
    try:
        message = await collect_chat(
            model,
            {"stream": stream, "chat_template_kwargs": '{"enable_thinking": false}'},
        )
        assert message["content"] == "abc"
        assert not message.get("reasoning_content")
    finally:
        chat_context_var.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_chat_normalizes_text_parts_without_changing_input(
    model, monkeypatch, stream
):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ],
        }
    ]
    original = copy.deepcopy(messages)
    render = Mock(return_value="hi")
    monkeypatch.setattr(model._tokenizer, "apply_chat_template", render)
    result = await model.async_chat(messages, {"stream": stream})
    if stream:
        assert [chunk async for chunk in result]
    assert render.call_args.args[0] == [{"role": "user", "content": "hello\nworld"}]
    assert messages == original


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["image_url", "input_audio", "video_url"])
async def test_chat_still_rejects_media_parts_before_submission(model, kind):
    with pytest.raises(ValueError, match="plain text"):
        await model.async_chat(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}, {"type": kind}],
                }
            ]
        )
    assert not model._engine.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("options", [{"echo": True}, {"logprobs": 0}])
async def test_unsupported_completion_options_fail_before_submission(
    model, stream, options
):
    with pytest.raises(ValueError, match="does not support"):
        await model.async_generate("hi", {**options, "stream": stream})
    assert not model._engine.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_chat_logprobs_false_remains_supported(model, stream):
    message = await collect_chat(model, {"logprobs": False, "stream": stream})
    assert message["content"] == "abc"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "chat"])
@pytest.mark.parametrize("stream", [False, True])
async def test_rest_admission_preserves_id_for_explicit_actor_abort(
    model, monkeypatch, kind, stream
):
    import json
    from unittest.mock import AsyncMock

    from xinference.api import restful_api
    from xinference.api.tests.test_llm_request_abort import _new_api, _Request
    from xinference.core.model import ModelActor
    from xinference.core.utils import CancelMixin

    entered = asyncio.Event()

    class PendingRequest(Request):
        async def __anext__(self):
            entered.set()
            await asyncio.Event().wait()

    native = PendingRequest()
    monkeypatch.setattr(model._engine, "generate_async", lambda *a, **kw: native)
    monkeypatch.setattr(model.model_family, "to_description", lambda: {}, raising=False)
    actor = ModelActor("test:1", "test:2", model, "tiny-rep0")
    actor._model_state = "ready"
    actor.record_metrics = AsyncMock()

    async def invoke(name, *args, **kwargs):
        result = await getattr(actor, name)(*args, **kwargs)
        if hasattr(result, "_actor_ref"):
            result._actor_ref = actor
            result._gc_destroy = False
        return result

    async def abort(request_id):
        return await actor.abort_request(request_id, block_duration=0)

    proxy = SimpleNamespace(
        uid=b"tiny",
        generate=lambda *a, **kw: invoke("generate", *a, **kw),
        chat=lambda *a, **kw: invoke("chat", *a, **kw),
        is_vllm_backend=AsyncMock(return_value=False),
        is_transformers_batching_enabled=AsyncMock(return_value=False),
        abort_request=abort,
        decrease_serve_count=actor.decrease_serve_count,
    )

    async def require_model(*a, **kw):
        return proxy

    async def supervisor_abort(uid, request_id, block_duration, **kwargs):
        return {"msg": await actor.abort_request(request_id, block_duration)}

    supervisor = SimpleNamespace(
        describe_model=AsyncMock(return_value={"model_family": "qwen3"}),
        abort_request=supervisor_abort,
    )
    monkeypatch.setattr(restful_api, "require_model", require_model)
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    api = _new_api()
    CancelMixin.__init__(api)
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    request_id = f"public-{kind}-{stream}"
    body = {"model": "tiny", "request_id": request_id, "stream": stream}
    if kind == "completion":
        body["prompt"] = "hi"
        create = api.create_completion
    else:
        body["messages"] = [{"role": "user", "content": "hi"}]
        create = api.create_chat_completion
    response = None
    if stream:
        response = await create(_Request(body))
        task = asyncio.create_task(anext(response.body_iterator))
    else:
        task = asyncio.create_task(create(_Request(body)))
    try:
        await asyncio.wait_for(entered.wait(), timeout=3)
        admitted = list(actor._running_tasks)
        native_ids = list(model._requests)
        aborted = await api.abort_request(
            _Request({"block_duration": 0}), "tiny", request_id
        )
        assert json.loads(aborted.body) == {"msg": "DONE"}
        assert admitted == native_ids == [request_id]
        await asyncio.wait({task}, timeout=1)
        assert task.done() and native.aborted
        assert not model._requests
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if response is not None:
            await response.body_iterator.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "chat"])
@pytest.mark.parametrize("keyword_config", [False, True])
@pytest.mark.parametrize("explicit_id", [None, "keyword-id"])
async def test_actor_config_request_id_respects_explicit_keyword(
    model, monkeypatch, method, keyword_config, explicit_id
):
    import json
    from unittest.mock import AsyncMock

    from xinference.core.model import ModelActor

    monkeypatch.setattr(model.model_family, "to_description", lambda: {}, raising=False)
    actor = ModelActor("test:1", "test:2", model, "tiny-rep0")
    actor._model_state = "ready"
    actor.record_metrics = AsyncMock()
    config = {"request_id": "config-id"}
    args, kwargs = (
        ((), {"generate_config": config}) if keyword_config else ((config,), {})
    )
    if explicit_id is not None:
        kwargs["request_id"] = explicit_id
    prompt = "hi" if method == "generate" else [{"role": "user", "content": "hi"}]
    result = json.loads(await getattr(actor, method)(prompt, *args, **kwargs))
    expected = explicit_id or "config-id"
    assert result["id"] == ("chat" + expected if method == "chat" else expected)
    assert list(actor._running_tasks) == [expected]
    assert config == {"request_id": "config-id"}


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "chat"])
@pytest.mark.parametrize("stop_ids", [151645, 0, None, [], [151645]])
async def test_scalar_stop_tokens_are_normalized_without_mutating_config(
    model, method, stop_ids
):
    config = {"stop_token_ids": stop_ids}
    original = copy.deepcopy(config)
    if method == "generate":
        await model.async_generate("hi", config)
    else:
        await model.async_chat([{"role": "user", "content": "hi"}], config)
    assert model._engine.params.stop_token_ids == (
        [stop_ids] if type(stop_ids) is int else stop_ids
    )
    assert config == original


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "chat"])
async def test_other_async_engines_keep_existing_request_id_admission(
    monkeypatch, method
):
    from unittest.mock import AsyncMock

    from xinference.core.model import ModelActor

    legacy = SimpleNamespace(
        model_family=SimpleNamespace(to_description=lambda: {}),
        allow_batch=True,
        async_generate=AsyncMock(return_value={"choices": []}),
        async_chat=AsyncMock(return_value={"choices": []}),
    )
    actor = ModelActor("test:1", "test:2", legacy, "legacy-rep0")
    actor._model_state = "ready"
    actor.record_metrics = AsyncMock()
    monkeypatch.setattr("xinference.core.model.uuid.uuid1", lambda: "actor-generated")
    prompt = "hi" if method == "generate" else []
    await getattr(actor, method)(prompt, {"request_id": "config-id"})
    call = getattr(legacy, "async_" + method).await_args
    assert call.kwargs["request_id"] == "actor-generated"
    assert call.args[1] == {"request_id": "config-id"}
