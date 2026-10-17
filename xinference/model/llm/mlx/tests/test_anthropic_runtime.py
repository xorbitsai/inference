import hashlib
import sys
from types import SimpleNamespace

import pytest

from .. import core


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


def test_final_detokenizer_fragment_preserves_length_finish():
    model = object.__new__(core.MLXVisionModel)
    model.model_uid = "test"
    model._tokenizer = SimpleNamespace(eos_token_id=99)
    model._prompt_cache = None
    model._prepare_inputs = lambda prompt, config: ([], 1, 0)
    model._get_max_tokens = lambda length, requested: 2
    model._structured_processor = lambda config: []
    model._generate_stream_inner = lambda **kwargs: iter(
        [
            SimpleNamespace(token=1, text="a"),
            SimpleNamespace(token=2, text="b"),
            SimpleNamespace(token=2, text=""),
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
