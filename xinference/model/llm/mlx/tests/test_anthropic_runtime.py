import hashlib
import sys
from types import SimpleNamespace

import pytest

from .. import core


@pytest.mark.parametrize("start,end", [("analysis", "final"), ("<think>", None)])
def test_manual_budget_rejects_missing_reasoning_markers(start, end):
    from ...reasoning_parser import ReasoningParser

    model = object.__new__(core.MLXChatModel)
    model.reasoning_parser = ReasoningParser(
        reasoning_start_tag=start, reasoning_end_tag=end
    )
    model._tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text.encode()))
    with pytest.raises(NotImplementedError, match="thinking budgets require"):
        model._apply_thinking_budget(
            {
                "chat_template_kwargs": {
                    "_anthropic_request": True,
                    "enable_thinking": True,
                    "thinking_budget": 1024,
                }
            }
        )


@pytest.mark.parametrize("engine", ["mlx", "vllm", "sglang"])
@pytest.mark.parametrize("markers", [None, (None, None), ("", "")])
def test_non_reasoning_models_ignore_anthropic_thinking_budget(engine, markers):
    from ...reasoning_parser import ReasoningParser
    from ...sglang.core import SGLANGChatModel
    from ...vllm.core import VLLMChatModel

    cls = {"mlx": core.MLXChatModel, "vllm": VLLMChatModel, "sglang": SGLANGChatModel}[
        engine
    ]
    model = object.__new__(cls)
    model.reasoning_parser = (
        ReasoningParser(reasoning_start_tag=markers[0], reasoning_end_tag=markers[1])
        if markers is not None
        else None
    )
    config = {
        "_anthropic_request": True,
        "chat_template_kwargs": {
            "_anthropic_request": True,
            "enable_thinking": True,
            "thinking_budget": 1024,
        },
    }
    model._apply_thinking_budget(config)
    assert (
        not {"_mlx_thinking_budget", "thinking_token_budget", "max_thinking_tokens"}
        & config.keys()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("control", ["json", "tool"])
async def test_xavier_rejects_openai_constraints_before_handoff(role, stream, control):
    from unittest.mock import AsyncMock

    model = control_model()
    model._xavier = SimpleNamespace(role=role, fetch=AsyncMock(), prefill=AsyncMock())
    config = {"stream": stream}
    if control == "json":
        config["response_format"] = {"type": "json_object"}
    else:
        config["tool_choice"] = "required"
    with pytest.raises(NotImplementedError, match="P/D models do not support"):
        await model.async_generate("prompt", config)
    model._xavier.fetch.assert_not_called()
    model._xavier.prefill.assert_not_called()


@pytest.mark.parametrize("start", ["<think>", "<|channel>thought\n"])
def test_reasoning_opener_preserves_existing_newline(start):
    model = control_model()
    model.reasoning_parser.reasoning_start_tag = start
    config = {
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert model._prepare_reasoning_prompt("assistant\n", config) == (
        "assistant\n" + start.rstrip("\n") + "\n"
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "terminal,request_stop", [(106, None), (106, [77]), (77, [77]), (0, 0), (99, None)]
)
def test_vlm_family_stop_preserves_final_detokenizer_fragment(
    monkeypatch, stream, terminal, request_stop
):
    fake_core = SimpleNamespace(metal=SimpleNamespace(get_peak_memory=lambda: 0))
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake_core))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.utils",
        SimpleNamespace(GenerationResponse=lambda **kw: SimpleNamespace(**kw)),
    )
    monkeypatch.setattr(
        core,
        "_ensure_mlx_vlm_thread_local_stream",
        lambda: SimpleNamespace(
            generate_step=lambda *args, **kw: iter([(10, None), (terminal, None)])
        ),
    )

    class Detokenizer:
        last_segment = ""

        def reset(self):
            self.last_segment = ""

        def add_token(self, token):
            pass

        def finalize(self):
            self.last_segment = "tail"

    model = control_model(core.MLXVisionModel)
    model.model_family.stop_token_ids = [106]
    model._tokenizer = SimpleNamespace(eos_token_id=99)
    model._processor = SimpleNamespace(
        tokenizer=model._tokenizer, detokenizer=Detokenizer()
    )
    model._model = object()
    model._draft_model = None
    model._prompt_cache = None
    model._cleanup_memory = lambda: None
    model._structured_processor = lambda config: []
    model._prepare_inputs = lambda *args: (([1, 2], None, None, {}), 2, 0)
    chunks = list(
        model._generate_stream(
            {},
            model._sanitize_generate_config(
                {"max_tokens": 4, "stream": stream, "stop_token_ids": request_stop}
            ),
        )
    )
    assert "".join(chunk["choices"][0]["text"] for chunk, _ in chunks) == "tail"
    assert chunks[-1][1]["completion_tokens"] == 2
    assert chunks[-1][0]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("cls", [core.MLXChatModel, core.MLXVisionModel])
@pytest.mark.parametrize("request_stop", [0, [77]])
def test_constraints_mask_model_and_request_stop_ids(monkeypatch, cls, request_stop):
    from .. import structured

    model = control_model(cls)
    model.model_family.stop_token_ids = [106]
    model._tokenizer.eos_token_ids = 99
    seen = []

    def processor(*args):
        seen.append(args[-1])
        return object()

    monkeypatch.setattr(structured, "JSONLogitsProcessor", processor)
    monkeypatch.setattr(structured, "ReasoningBudgetLogitsProcessor", processor)
    assert (
        len(
            model._structured_processor(
                {
                    "_mlx_thinking_budget": 4,
                    "response_format": {"type": "json_object"},
                    "stop_token_ids": request_stop,
                }
            )
        )
        == 2
    )
    expected = [0, 99, 106] if request_stop == 0 else [77, 99, 106]
    assert seen == [expected, expected]


class Array:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, item):
        return self


def test_cached_text_embeddings_restore_rotary_offsets(monkeypatch):
    fake = SimpleNamespace(array=Array)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake))
    monkeypatch.setitem(sys.modules, "mlx.core", fake)
    language = SimpleNamespace(_position_ids="old", _rope_deltas=None)

    class Model:
        language_model = language

        def get_input_embeddings(self, *args, **kwargs):
            language._position_ids = None
            language._rope_deltas = None
            return "embeddings"

    wrapped = core._CachedTextVisionModel(Model())
    assert wrapped.get_input_embeddings([1]) == "embeddings"
    assert language._rope_deltas.value == 0
    assert language._position_ids is None


def test_older_mlx_vlm_prefills_only_missing_prefix_and_copies_cache(monkeypatch):
    fake = SimpleNamespace(array=Array, eval=lambda arrays: None)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake))
    monkeypatch.setitem(sys.modules, "mlx.core", fake)
    monkeypatch.setattr(core.time, "monotonic", lambda: 100)
    entry = SimpleNamespace(state=["initial"])
    calls = []
    stored = []

    class Language:
        _rope_deltas = None
        _position_ids = None

        def __call__(self, inputs, cache):
            calls.append((inputs.value, self._rope_deltas.value))
            cache[0].state.append("prefilled")

    class Cache:
        def insert_cache(self, *args, **kwargs):
            stored.append(args)

        def __len__(self):
            return len(stored)

    model = object.__new__(core.MLXVisionModel)
    model._model_config = {}
    model._model = SimpleNamespace(language_model=Language())
    model._reusable_prompt_cache = Cache()
    model._reusable_prompt_cache_model_key = "model"
    model._anthropic_cache_expiry = {}
    model._cache_has_checkpoint_api = False
    model._fetch_reusable_prompt_cache = lambda tokens: ([entry], tokens[3:], 3)
    remaining, kwargs, cached = model._prepare_reusable_prompt_cache(
        [1, 2, 3, 4, 5], 4, 300
    )
    assert remaining == [5]
    assert calls == [([4], 0)]
    assert cached == 3
    assert "prompt_cache_checkpoint" not in kwargs
    assert stored[0][1] == [1, 2, 3, 4]
    entry.state.append("generated")
    assert stored[0][2][0].state == ["initial", "prefilled"]
    assert next(iter(model._anthropic_cache_expiry.values())) == 400


@pytest.mark.parametrize("stored_tokens", [[1, 2, 3], [1, 2, 3, 9, 10]])
def test_expired_cache_is_not_reused(monkeypatch, stored_tokens):
    model = object.__new__(core.MLXVisionModel)
    model._reusable_prompt_cache_model_key = "model"

    class Cache:
        expired = False

        def trim_to(self, *, n_sequences):
            assert n_sequences == 0
            self.expired = True

        def fetch_nearest_cache(self, key, tokens):
            assert self.expired
            return None, tokens

    model._reusable_prompt_cache = Cache()
    key = hashlib.sha256(str(stored_tokens).encode()).hexdigest()
    model._anthropic_cache_expiry = {key: 50}
    monkeypatch.setattr(core.time, "monotonic", lambda: 100)
    assert model._fetch_reusable_prompt_cache([1, 2, 3, 4, 5]) == (
        None,
        [1, 2, 3, 4, 5],
        0,
    )
    assert not model._anthropic_cache_expiry


def test_invalid_schema_fails_and_default_effort_preserves_prompt():
    model = object.__new__(core.MLXChatModel)
    model.model_family = SimpleNamespace(model_ability=["chat"])
    model.reasoning_parser = None
    model._tokenizer = None
    messages = [{"role": "user", "content": "hello"}]
    prepared = model.prepare_anthropic_request(messages, {"reasoning_effort": "medium"})
    assert prepared["messages"] == messages
    assert messages == [{"role": "user", "content": "hello"}]
    with pytest.raises(ValueError, match="JSON schema object"):
        model.prepare_anthropic_request(
            messages, {"response_format": {"type": "json_schema"}}
        )
    prepared = model.prepare_anthropic_request(messages, {"parallel_tool_calls": False})
    assert prepared["generate_config"]["parallel_tool_calls"] is False


def test_explicit_checkpoints_refresh_and_expire_independently(monkeypatch):
    from .. import prompt_cache

    clock = [100.0]
    monkeypatch.setattr(prompt_cache.time, "monotonic", lambda: clock[0])
    cache = prompt_cache.TimedPromptCache()
    state = [SimpleNamespace(state=["original"])]
    assert cache.store([1, 2], state, 3600)
    assert cache.store([1, 2, 3, 4], state, 300)
    state[0].state.append("generation")
    clock[0] = 300
    hit, length = cache.fetch([1, 2, 3, 4, 5])
    assert length == 4 and hit[0].state == ["original"]
    hit[0].state.append("request")
    assert cache.entries[(1, 2)].expires == 3900
    assert cache.entries[(1, 2, 3, 4)].expires == 600
    clock[0] = 600
    hit, length = cache.fetch([1, 2, 3, 4, 5])
    assert length == 2 and hit[0].state == ["original"]
    assert (1, 2, 3, 4) not in cache.entries
    clock[0] = 4200
    assert cache.fetch([1, 2, 3, 4, 5]) == (None, 0)


def test_explicit_checkpoint_memory_limit_and_unrelated_expiry(monkeypatch):
    from .. import prompt_cache

    clock = [0.0]
    monkeypatch.setattr(prompt_cache.time, "monotonic", lambda: clock[0])
    state = [SimpleNamespace(state=SimpleNamespace(nbytes=10))]
    cache = prompt_cache.TimedPromptCache(max_bytes=15)
    cache.store([1], state, 300)
    cache.store([2], state, 3600)
    assert list(cache.entries) == [(2,)]
    clock[0] = 300
    assert cache.fetch([2, 3])[1] == 1
    assert cache.entries[(2,)].expires == 3900
    cache = prompt_cache.TimedPromptCache(max_bytes=5)
    assert not cache.store([1], state, 300)
    assert cache.fetch([1, 2]) == (None, 0)


def test_added_checkpoint_is_prefilled_but_eviction_does_not_force_a_miss():
    from ..prompt_cache import TimedPromptCache

    state = [SimpleNamespace(state=[])]
    cache = TimedPromptCache(max_size=1)
    cache.store([1, 2, 3, 4], state, 300)
    assert cache.fetch([1, 2, 3, 4, 5], [2, 4]) == (None, 0)
    cache.store([1, 2], state, 3600)
    cache.store([1, 2, 3, 4], state, 300)
    assert cache.fetch([1, 2, 3, 4, 5], [2, 4])[1] == 4


@pytest.mark.parametrize("cls", [core.MLXModel, core.MLXVisionModel])
def test_final_detokenizer_fragment_preserves_length_finish(cls):
    model = object.__new__(cls)
    model.model_uid = "test"
    model._tokenizer = SimpleNamespace(eos_token_id=99)
    model._prompt_cache = SimpleNamespace(tokens=[])
    model._prepare_inputs = lambda prompt, config: ([], 1, 0)
    model._get_max_tokens = lambda length, requested: 2
    model._structured_processor = lambda config: []
    model._generate_stream_inner = lambda **kwargs: iter(
        [
            SimpleNamespace(token=1, text="a", generation_tokens=1),
            SimpleNamespace(token=2, text="b", generation_tokens=2),
            SimpleNamespace(token=2, text="", generation_tokens=2),
        ]
    )
    chunks = list(
        model._generate_stream(
            "prompt",
            {
                "max_tokens": 2,
                "temperature": 0,
                "top_p": 1,
                "top_k": 0,
                "repetition_penalty": None,
                "repetition_context_size": 20,
            },
        )
    )
    assert chunks[-1][0]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1][1]["completion_tokens"] == 2
    assert model._prompt_cache.tokens == [1, 2]


def test_structured_text_requests_have_independent_kv_caches(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models.cache",
        SimpleNamespace(
            make_prompt_cache=lambda *args: [object()],
        ),
    )
    model = object.__new__(core.MLXChatModel)
    model._tokenizer = SimpleNamespace(encode=lambda prompt: [1, 2, 3])
    model._model = object()
    model._context_length_limit = None
    model._max_kv_size = None
    first = {"response_format": {"type": "json_object"}}
    second = {"response_format": {"type": "json_object"}}
    assert model._prepare_inputs("prompt", first) == ([1, 2, 3], 3, 0)
    assert model._prepare_inputs("prompt", second) == ([1, 2, 3], 3, 0)
    assert first["_mlx_prompt_cache"] is not second["_mlx_prompt_cache"]


def test_cache_hit_can_extend_lifetime_without_revoking_an_existing_lease(monkeypatch):
    from .. import prompt_cache

    clock = [0.0]
    monkeypatch.setattr(prompt_cache.time, "monotonic", lambda: clock[0])
    cache = prompt_cache.TimedPromptCache()
    cache.store([1, 2], [SimpleNamespace(state=[])], 300)
    clock[0] = 100
    assert cache.fetch([1, 2, 3], [2], {2: 3600})[1] == 2
    assert cache.entries[(1, 2)].expires == 3700
    clock[0] = 200
    cache.fetch([1, 2, 3], [2], {2: 300})
    assert cache.entries[(1, 2)].expires == 3700


def test_cache_write_lifetime_includes_prefill_time(monkeypatch):
    from .. import prompt_cache

    monkeypatch.setattr(prompt_cache.time, "monotonic", lambda: 200)
    cache = prompt_cache.TimedPromptCache()
    cache.store([1, 2], [SimpleNamespace(state=[])], 300, started=100)
    assert cache.entries[(1, 2)].expires == 400


def control_model(cls=core.MLXChatModel):
    from ...reasoning_parser import ReasoningParser

    model = object.__new__(cls)
    model.model_uid = "controls"
    model.model_family = SimpleNamespace(
        model_name="qwen3",
        model_family="qwen3",
        chat_template="template",
        model_ability=["chat", "vision"],
        stop=None,
        stop_token_ids=None,
    )
    model.reasoning_parser = ReasoningParser(
        reasoning_start_tag="<think>",
        reasoning_end_tag="</think>",
        enable_thinking=True,
        allow_disable_thinking=True,
    )
    model.tool_parser = None
    model._tokenizer = SimpleNamespace(
        encode=lambda text, **kw: list(text.encode()), eos_token_id=0
    )
    model._model = object()
    model._model_generation_config = {}
    model._context_length_limit = None
    model._xavier = None
    return model


@pytest.mark.parametrize("cls", [core.MLXChatModel, core.MLXVisionModel])
@pytest.mark.parametrize("format_type", ["text", "json_object"])
def test_openai_controls_normalize_format_and_string_kwargs(cls, format_type):
    model = control_model(cls)
    original = {
        "response_format": {"type": format_type},
        "chat_template_kwargs": '{"enable_thinking":true}',
        "tool_choice": "none",
        "tools": [{"type": "function"}],
        "parallel_tool_calls": False,
    }
    parallel, config = model._prepare_chat_controls(original)
    assert parallel is False and "tools" not in config
    assert config["chat_template_kwargs"]["enable_thinking"] is (format_type == "text")
    assert ("response_format" in config) is (format_type != "text")
    assert original["tools"] and isinstance(original["chat_template_kwargs"], str)
    assert "response_format" not in model._sanitize_generate_config(
        {"response_format": {"type": "text"}}
    )


def test_openai_thinking_budget_is_ignored():
    model = control_model()
    _, config = model._prepare_chat_controls(
        {"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": "unused"}}
    )
    assert "_mlx_thinking_budget" not in config
    with pytest.raises(ValueError, match="positive integer"):
        model._prepare_chat_controls(config, anthropic=True)


@pytest.mark.parametrize(
    "prompt", ["assistant\n", "<think>old</think>assistant\n", "assistant<think>\n"]
)
@pytest.mark.parametrize("control", ["json", "budget"])
def test_reasoning_prompt_opens_actual_block_and_count_matches(prompt, control):
    model = control_model()
    config = {
        "chat_template_kwargs": {"_anthropic_request": True, "enable_thinking": True}
    }
    if control == "json":
        config["response_format"] = {"type": "json_object"}
    else:
        config["chat_template_kwargs"]["thinking_budget"] = 4
    _, config = model._prepare_chat_controls(config)
    model.get_full_context = lambda *args, **kwargs: prompt
    expected = prompt if prompt.endswith("<think>\n") else prompt + "<think>\n"
    assert model._prepare_reasoning_prompt(prompt, config) == expected
    assert model.count_tokens([{"role": "user", "content": "hello"}], config) == len(
        expected
    )
    config["chat_template_kwargs"]["enable_thinking"] = False
    assert model._prepare_reasoning_prompt(prompt, config) == prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_fallback_decode_and_close_do_not_block_actor_loop(stream):
    import asyncio
    import threading
    import time
    from unittest.mock import AsyncMock

    model = control_model()
    model._batch_model = SimpleNamespace(generate=AsyncMock(return_value=("batch", {})))
    owner = threading.get_ident()
    workers = []
    ticks = []
    closed = []

    def generate(*args):
        try:
            for _ in range(3):
                workers.append(threading.get_ident())
                time.sleep(0.02)
                yield {"choices": [{"text": "x", "finish_reason": "stop"}]}, {}
        finally:
            closed.append(threading.get_ident())

    model._generate_stream = generate

    async def heartbeat():
        for _ in range(5):
            await asyncio.sleep(0.005)
            ticks.append(len(workers))

    async def run():
        result = await model.async_generate(
            "prompt", {"stream": stream, "response_format": {"type": "json_object"}}
        )
        if stream:
            assert len([chunk async for chunk in result]) == 3
        else:
            assert result["choices"][0]["text"] == "xxx"

    try:
        await asyncio.gather(run(), heartbeat())
        assert ticks and min(ticks) < 3
        assert len(set(workers)) == 1 and workers[0] != owner
        assert closed == [workers[0]]
        model._batch_model.generate.assert_not_called()
        result = await model.async_generate(
            "prompt", {"response_format": {"type": "text"}, "max_tokens": 1}
        )
        assert result["choices"][0]["text"] == "batch"
    finally:
        model.stop()


def test_store_purges_expired_entries_before_live_eviction(monkeypatch):
    from .. import prompt_cache

    clock = [0]
    monkeypatch.setattr(prompt_cache.time, "monotonic", lambda: clock[0])
    cache = prompt_cache.TimedPromptCache(max_size=2)
    state = [SimpleNamespace(state=[])]
    cache.store([1], state, 3600)
    cache.store([2], state, 300)
    clock[0] = 301
    cache.store([3], state, 300)
    assert list(cache.entries) == [(1,), (3,)]
    assert not cache.store([4], state, 300, started=0)
    assert list(cache.entries) == [(1,), (3,)]


@pytest.mark.parametrize("max_bytes,created", [(None, 4), (0, 0)])
def test_cache_creation_counts_writes_even_when_later_prefix_evicts_earlier(
    monkeypatch, max_bytes, created
):
    fake = SimpleNamespace(array=Array, eval=lambda arrays: None)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake))
    monkeypatch.setitem(sys.modules, "mlx.core", fake)
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models.cache",
        SimpleNamespace(
            make_prompt_cache=lambda *args: [
                SimpleNamespace(state=SimpleNamespace(nbytes=10))
            ]
        ),
    )
    model = control_model()
    model._model = lambda *args, **kwargs: None
    model._model_config = {"prompt_cache_size": 1, "prompt_cache_bytes": max_bytes}
    model._max_kv_size = None
    tokens = [1, 2, 3, 4, 5]
    config = {}
    assert model._prepare_explicit_prompt_cache(tokens, [(2, 3600), (4, 300)], config)[
        0
    ] == [5]
    assert config["_anthropic_cache_created_tokens"] == created
    assert config["_anthropic_cache_creation"] == {
        "ephemeral_5m_input_tokens": created // 2,
        "ephemeral_1h_input_tokens": created // 2,
    }
    if created:
        assert list(model._explicit_cache.entries) == [(1, 2, 3, 4)]
        assert (
            model._prepare_explicit_prompt_cache(tokens, [(2, 3600), (4, 300)], config)[
                2
            ]
            == 4
        )
        assert config["_anthropic_cache_created_tokens"] == 0
    assert model._explicit_cache.max_size == 1
    model._model_config = {}
    model._explicit_cache_key = None
    model._prepare_explicit_prompt_cache(tokens, [(2, 300)], {})
    assert model._explicit_cache.max_size == core._DEFAULT_MLX_PROMPT_CACHE_SIZE


def test_coincident_cache_boundaries_use_longer_lifetime():
    model = control_model()
    model._get_reusable_prompt_prefix_len = lambda *args: 10
    config = {}
    model._set_explicit_cache_boundaries(
        [],
        "prompt",
        "template",
        None,
        {},
        {"breakpoints": [{"ttl": 3600}, {"ttl": 300}]},
        config,
    )
    assert config["_mlx_cache_boundaries"] == [(10, 3600)]


def test_vlm_without_prompt_cache_api_does_not_apply_explicit_markers(monkeypatch):
    model = control_model(core.MLXVisionModel)
    model._model_config = {}
    monkeypatch.setattr(
        core,
        "_ensure_mlx_vlm_thread_local_stream",
        lambda: SimpleNamespace(generate_step=lambda model, input_ids: None),
    )
    model._init_reusable_prompt_cache()
    assert model._cache_has_prompt_cache_api is False
    model.count_tokens = lambda *args: 3
    prepared = model.prepare_anthropic_request(
        [{"role": "user", "content": "hello"}],
        {
            "_anthropic_cache_control": {
                "messages": [{"role": "user", "content": "hello"}],
                "ttl": 300,
            }
        },
    )
    assert prepared["cache_control_applied"] is False
    assert "_anthropic_cache_control" not in prepared["generate_config"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [core.MLXChatModel, core.MLXVisionModel])
@pytest.mark.parametrize("choice", ["required", "named", "none", "auto"])
async def test_openai_tool_controls_reach_text_and_vision_generation(cls, choice):
    from unittest.mock import AsyncMock, Mock

    model = control_model(cls)
    declared = [
        {
            "type": "function",
            "function": {"name": "Read", "parameters": {"type": "object"}},
        }
    ]
    generated = {
        "id": "test",
        "model": "test",
        "created": 1,
        "choices": [
            {
                "index": 0,
                "text": '[{"name":"Read","arguments":{}}]',
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }
    seen = []

    def render(*args, **kwargs):
        seen.append(kwargs.get("tools", []))
        return "prompt"

    model.get_full_context = render
    model._get_reusable_prompt_prefix_len = lambda *args: None
    model._set_explicit_cache_boundaries = lambda *args: None

    def prepare(messages, config):
        seen.append(config.pop("tools", []))
        return {"prompt": "prompt"}, seen[-1], config["chat_template_kwargs"], config

    model._prepare_chat_inputs = prepare
    model._to_chat_completion = lambda c, *args: {
        "id": "test",
        "choices": [
            {"message": {"content": c["choices"][0]["text"]}, "finish_reason": "stop"}
        ],
        "usage": {},
    }
    model._post_process_completion = lambda *args: {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "one", "function": {"name": "Read", "arguments": "{}"}},
                        {"id": "two", "function": {"name": "Read", "arguments": "{}"}},
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    generate = (
        AsyncMock(return_value=generated)
        if cls is core.MLXChatModel
        else Mock(return_value=generated)
    )
    if cls is core.MLXChatModel:
        model.async_generate = generate
    else:
        model.generate = generate
    config = {
        "tools": declared,
        "tool_choice": (
            {"type": "function", "function": {"name": "Read"}}
            if choice == "named"
            else choice
        ),
        "parallel_tool_calls": False,
        "chat_template_kwargs": '{"enable_thinking":true}',
    }
    messages = [{"role": "user", "content": "read"}]
    if cls is core.MLXChatModel:
        result = await model.async_chat(messages, config)
    else:
        result = model.chat(messages, config)
    used = generate.call_args.args[1]
    if choice in {"named", "required"}:
        assert used["response_format"]["json_schema"]["schema"]["maxItems"] == 1
        assert used["chat_template_kwargs"]["enable_thinking"] is False
        assert (
            result["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
            == "Read"
        )
    elif choice == "none":
        assert seen == [[]]
        assert "tool_calls" not in result["choices"][0]["message"]
    else:
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1


@pytest.mark.asyncio
async def test_closing_fallback_stream_closes_decode_on_its_worker():
    import threading

    model = control_model()
    model._batch_model = None
    closed = []

    def generate(*args):
        try:
            yield {"choices": [{"text": "x"}]}, {}
            raise AssertionError("closed stream must not decode again")
        finally:
            closed.append(threading.get_ident())

    model._generate_stream = generate
    chunks = await model.async_generate("prompt", {"stream": True})
    try:
        assert await anext(chunks)
        await chunks.aclose()
        assert len(closed) == 1 and closed[0] != threading.get_ident()
    finally:
        model.stop()


@pytest.mark.asyncio
async def test_cancelling_inflight_fallback_waits_for_worker_cleanup():
    import asyncio
    import threading
    import time

    model = control_model()
    model._batch_model = None
    started = threading.Event()
    closed = []

    def generate(*args):
        try:
            started.set()
            time.sleep(0.05)
            yield {"choices": [{"text": "x"}]}, {}
        finally:
            closed.append(threading.get_ident())

    model._generate_stream = generate
    task = asyncio.create_task(model.async_generate("prompt", {"max_tokens": 1}))
    try:

        async def wait_for_decode():
            while not started.is_set():
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait_for_decode(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(closed) == 1 and closed[0] != threading.get_ident()
        assert not model._fallback_lock.locked()
    finally:
        model.stop()


def test_active_system_checkpoint_survives_known_history_overflow():
    from ..prompt_cache import TimedPromptCache

    cache = TimedPromptCache(max_size=2)
    state = [SimpleNamespace(state=[])]
    previous = 0
    for turn in range(25):
        tokens = list(range(6 + 2 * turn))
        boundaries = [2, len(tokens) - 1]
        _, cached = cache.fetch(tokens, boundaries)
        assert cached == previous, (turn, cached, previous)
        for end in boundaries:
            if end > cached:
                assert cache.store(tokens[:end], state, 300)
        previous = boundaries[-1]
        assert len(cache.entries) <= 2
        assert len(cache._known) <= 8
        assert (0, 1) in cache._known
    assert len(cache._known) == 8


@pytest.mark.parametrize("candidate", [[1], [9]])
def test_oversized_checkpoint_does_not_evict_live_cache(candidate):
    from ..prompt_cache import TimedPromptCache

    cache = TimedPromptCache(max_bytes=10)
    assert cache.store([1], [SimpleNamespace(state=SimpleNamespace(nbytes=5))], 300)
    assert not cache.store(
        candidate, [SimpleNamespace(state=SimpleNamespace(nbytes=11))], 300
    )
    assert list(cache.entries) == [(1,)]
    assert cache.fetch([1, 2], [1])[1] == 1
    assert cache.entries[(1,)].nbytes == 5


@pytest.mark.parametrize("role", [None, "prefill", "decode"])
def test_xavier_keeps_automatic_cache_without_explicit_checkpoints(role):
    model = control_model()
    model._model_config = {}
    model._xavier = SimpleNamespace(role=role)
    model.count_tokens = lambda *args: 5
    messages = [{"role": "user", "content": "hello"}]
    prepared = model.prepare_anthropic_request(
        messages, {"_anthropic_cache_control": {"messages": messages, "ttl": 300}}
    )
    assert prepared["cache_control_applied"] is False
    assert "_anthropic_cache_control" not in prepared["generate_config"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("fail", [False, True])
async def test_stop_during_decode_keeps_cleanup_on_worker_and_preserves_error(
    stream, fail
):
    import asyncio
    import threading
    import time

    model = control_model()
    model._batch_model = None
    started = threading.Event()
    closed = []

    def generate(*args):
        try:
            started.set()
            time.sleep(0.05)
            if fail:
                raise ValueError("original decode error")
            yield {"choices": [{"text": "x"}]}, {}
        finally:
            closed.append(threading.get_ident())

    model._generate_stream = generate

    async def run():
        result = await model.async_generate(
            "prompt", {"stream": stream, "max_tokens": 1}
        )
        if stream:
            return [chunk async for chunk in result]
        return result

    task = asyncio.create_task(run())
    try:

        async def wait_for_decode():
            while not started.is_set():
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait_for_decode(), timeout=5)
        await asyncio.to_thread(model.stop)
        with pytest.raises(
            ValueError if fail else RuntimeError,
            match="original decode error" if fail else "model is stopping",
        ):
            await task
        assert len(closed) == 1 and closed[0] != threading.get_ident()
        assert model._fallback_executor is None
        with pytest.raises(RuntimeError, match="model is stopping"):
            await model.async_generate("prompt", {"max_tokens": 1})
    finally:
        model.stop()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode", ["standard", "json", "budget", "checkpoint"])
@pytest.mark.parametrize(
    "ending", ["eos", "second_eos", "custom_stop", "length", "first_eos"]
)
def test_serial_stop_usage_and_kv_tokens_stay_aligned(
    monkeypatch, stream, mode, ending
):
    from itertools import count
    from unittest.mock import Mock

    clock = Mock(wraps=core.time)
    clock.time.side_effect = count(100.0, 0.01)
    monkeypatch.setattr(core, "time", clock)
    prompt = [1, 2, 3]
    terminal = {
        "eos": 99,
        "second_eos": 98,
        "custom_stop": 77,
        "length": 11,
        "first_eos": 99,
    }[ending]
    if ending == "custom_stop" and not stream:
        terminal = 0
    generated = (
        [99]
        if ending == "first_eos"
        else [10, 11] if ending == "length" else [10, 11, terminal]
    )
    maximum = len(generated) if ending != "first_eos" else 5

    class Cache:
        def __init__(self, tokens=()):
            self.tokens = list(tokens)

    caches = []

    def make_cache(*args):
        cache = [Cache()]
        caches.append(cache)
        return cache

    def native_generate(model, tokenizer, tokens, **kwargs):
        cache = kwargs["prompt_cache"][0]
        cache.tokens.extend(tokens)
        for index, token in enumerate(generated):
            cache.tokens.append(
                token
            )  # mlx-lm feeds the sampled token before yielding it.
            text = (
                "<STOP>" if token in {0, 77} else "tail" if token in {98, 99} else "x"
            )
            yield SimpleNamespace(token=token, text=text, generation_tokens=index + 1)
        if ending != "length":
            raise AssertionError("outer stop check must stop at the terminal token")

    # Use the real _generate_stream_inner adapter, with an mlx-lm-shaped source.
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.utils",
        SimpleNamespace(
            make_logits_processors=lambda **kwargs: [],
            make_sampler=lambda **kwargs: None,
            stream_generate=native_generate,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models.cache",
        SimpleNamespace(make_prompt_cache=make_cache),
    )
    model = control_model()
    model.model_path = "serial-model"
    model._max_kv_size = None
    model._prompt_cache = core.PromptCache() if mode == "standard" else None
    model._tokenizer = SimpleNamespace(
        encode=lambda text: prompt if text == "prompt" else prompt + generated + [42],
        eos_token_id=99,
        eos_token_ids={98, 99},
    )
    model._structured_processor = lambda config: []
    model._cleanup_memory = Mock()
    config = {
        "stream": stream,
        "stream_options": {"include_usage": True},
        "max_tokens": maximum,
        "stop_token_ids": (
            ([terminal] if stream else terminal) if ending == "custom_stop" else None
        ),
    }
    if mode == "json":
        config["response_format"] = {"type": "json_object"}
    elif mode == "budget":
        config["_mlx_thinking_budget"] = 10
    elif mode == "checkpoint":
        config["_mlx_cache_boundaries"] = [(2, 300)]

        def prepare(tokens, boundaries, config):
            cache = make_cache()
            cache[0].tokens.extend(tokens[:2])
            return tokens[2:], cache, 2

        model._prepare_explicit_prompt_cache = prepare
    config = model._sanitize_generate_config(config)
    chunks = list(model._generate_stream("prompt", config))
    usage = chunks[-1][1]
    assert usage["completion_tokens"] == len(generated)
    assert usage["total_tokens"] == len(prompt) + len(generated)
    assert usage["prompt_tokens_details"]["cached_tokens"] == (
        2 if mode == "checkpoint" else 0
    )
    terminal_chunks = [
        chunk
        for chunk, _ in chunks
        if chunk["choices"] and chunk["choices"][0]["finish_reason"]
    ]
    assert terminal_chunks[-1]["choices"][0]["finish_reason"] == (
        "length" if ending == "length" else "stop"
    )
    text = "".join(
        chunk["choices"][0]["text"] for chunk, _ in chunks if chunk["choices"]
    )
    assert "<STOP>" not in text
    if ending in {"eos", "second_eos", "first_eos"}:
        assert text.endswith("tail")
    assert caches[-1][0].tokens == prompt + generated
    if mode == "standard":
        assert model._prompt_cache.tokens == caches[-1][0].tokens
        assert model._prepare_inputs("followup", {}) == (
            [42],
            len(prompt + generated) + 1,
            len(prompt + generated),
        )
    else:
        assert model._prompt_cache is None
    model._cleanup_memory.assert_called_once()


def test_explicit_boundaries_encode_full_prompt_only_once():
    from unittest.mock import Mock

    model = control_model()
    model._tokenizer = SimpleNamespace(
        encode=Mock(side_effect=lambda text: list(text.encode()))
    )
    model.get_full_context = lambda messages, *args, **kwargs: messages[0]["content"]
    config = {}
    checkpoints = [
        {"messages": [{"role": "user", "content": "ab"}], "ttl": 3600},
        {"messages": [{"role": "user", "content": "abcd"}], "ttl": 300},
    ]
    full_tokens = model._tokenizer.encode("abcde")
    model._set_explicit_cache_boundaries(
        [{"role": "user", "content": "abcde"}],
        "abcde",
        "template",
        None,
        {},
        {"breakpoints": checkpoints},
        config,
        full_tokens,
    )
    model._get_reusable_prompt_prefix_len(
        [{"role": "user", "content": "abcde"}],
        "abcde",
        "template",
        None,
        {},
        None,
        full_tokens,
    )
    assert config["_mlx_cache_boundaries"] == [(2, 3600), (4, 300)]
    encoded = [call.args[0] for call in model._tokenizer.encode.call_args_list]
    assert encoded.count("abcde") == 1
    assert set(encoded) == {"abcde", "ab", "abcd", ""}


def test_native_mlx_lm_eos_usage_matches_kv_and_followup(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm import sample_utils
    from mlx_lm.models.qwen2 import Model, ModelArgs
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {
        value: index
        for index, value in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
    }
    vocab["<eos>"] = len(vocab)
    raw = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="<eos>")
    language = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=len(vocab),
            num_key_value_heads=2,
        )
    )
    language.set_dtype(mx.float16)
    mx.eval(language.parameters())
    generated = tokenizer.encode("ab") + [tokenizer.eos_token_id]
    samples = iter(generated + [tokenizer.eos_token_id])
    monkeypatch.setattr(
        sample_utils,
        "make_sampler",
        lambda **kwargs: lambda logits: mx.array([next(samples)]),
    )
    model = object.__new__(core.MLXModel)
    model.model_uid = "native-serial"
    model.model_path = "/unused"
    model._model = language
    model._tokenizer = tokenizer
    model._prompt_cache = core.PromptCache()
    model._max_kv_size = None
    model._context_length_limit = None
    model._model_generation_config = {}
    config = model._sanitize_generate_config(
        {"max_tokens": 5, "stop_token_ids": [tokenizer.eos_token_id]}
    )
    chunks = list(model._generate_stream("Q", config))
    expected = tokenizer.encode("Q") + generated
    assert chunks[-1][1]["completion_tokens"] == len(generated)
    assert model._prompt_cache.tokens == expected
    assert model._prompt_cache.cache[0].offset == len(expected)
    assert "".join(chunk["choices"][0]["text"] for chunk, _ in chunks) == "ab"
    remaining, count, cached = model._prepare_inputs(
        tokenizer.decode(expected) + "Z", {}
    )
    assert remaining == tokenizer.encode("Z")
    assert count == len(expected) + 1 and cached == len(expected)
