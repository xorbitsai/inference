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


def test_unsupported_controls_fail_and_default_effort_preserves_prompt():
    model = object.__new__(core.MLXChatModel)
    model.model_family = SimpleNamespace(model_ability=["chat"])
    model.reasoning_parser = None
    messages = [{"role": "user", "content": "hello"}]
    prepared = model.prepare_anthropic_request(messages, {"reasoning_effort": "medium"})
    assert prepared["messages"] == messages
    assert messages == [{"role": "user", "content": "hello"}]
    with pytest.raises(NotImplementedError, match="Structured"):
        model.prepare_anthropic_request(
            messages, {"response_format": {"type": "json_schema"}}
        )
    with pytest.raises(NotImplementedError, match="disable_parallel"):
        model.prepare_anthropic_request(messages, {"parallel_tool_calls": False})
