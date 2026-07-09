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

"""
Tests for ``EmbeddingModel._truncate_sentences`` (B1).

Verifies that the ``truncate_prompt_tokens`` parameter correctly bounds
input length before encoding, using a mock tokenizer so these tests
require no GPU, no model download, and no network.
"""

from unittest.mock import MagicMock

from ...core import _EMBEDDING_TRUNCATE_CHAR_PER_TOKEN, EmbeddingModel


class _StrTokenizer:
    """A minimal tokenizer that models 1 token ≈ 4 characters (English)."""

    def __call__(self, text, **kwargs):
        max_length = kwargs.get("max_length", None)
        tokens = [ord(c) for c in text[: max_length * 4]] if max_length else text
        return MagicMock(input_ids=tokens)

    def decode(self, ids, **kwargs):
        return "".join(chr(i) if isinstance(i, int) else i for i in ids)


class _NoneTokenizer:
    """A tokenizer that raises on __call__ (simulating a broken/unsupported
    tokenizer, or an engine that returns a non-callable _tokenizer)."""

    def __call__(self, text, **kwargs):
        raise RuntimeError("tokenizer unavailable")


class _DummyEmbeddingModel(EmbeddingModel):
    """Minimal EmbeddingModel subclass for testing _truncate_sentences.

    We only need ``_model_uid``, ``model_family`` with a ``max_tokens``
    attribute, and optionally ``_tokenizer``. Abstract methods are stubbed
    since the test never exercises them.
    """

    def __init__(self, *, tokenizer=None, max_tokens=None):
        self._model_uid = "test-model-0"
        self._tokenizer = tokenizer
        self.model_family = MagicMock(max_tokens=max_tokens)

    # Stub abstract methods — never called in these tests.
    def _create_embedding(self, sentences, **kwargs):
        raise NotImplementedError

    def check_lib(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    @classmethod
    def match_json(cls, *args, **kwargs):
        raise NotImplementedError


LONG_TEXT = "Hello world! " * 500  # ~7000 chars, ~1750 tokens (4 char/token)

# --- _truncate_sentences: None -------------------------------------------------


def test_truncate_none_unchanged():
    """truncate_prompt_tokens=None → return sentences intact (no truncation)."""
    model = _DummyEmbeddingModel(tokenizer=_StrTokenizer())
    result = model._truncate_sentences(LONG_TEXT, None)
    assert result is LONG_TEXT  # same object identity


# --- _truncate_sentences: >0 (explicit token limit) ----------------------------


def test_truncate_positive_with_tokenizer():
    """truncate_prompt_tokens=8 → input is truncated to ~8 tokens via
    tokenizer."""
    model = _DummyEmbeddingModel(tokenizer=_StrTokenizer())
    result = model._truncate_sentences(LONG_TEXT, 8)
    assert isinstance(result, str)
    # With our mock, 8 tokens × 4 char/token = ~32 chars returned
    assert len(result) <= 8 * 4


def test_truncate_positive_fallback_to_char():
    """truncate_prompt_tokens=8 + no tokenizer → fallback to char-based cut."""
    model = _DummyEmbeddingModel(tokenizer=None)  # no tokenizer (llama_cpp)
    result = model._truncate_sentences(LONG_TEXT, 8)
    assert isinstance(result, str)
    assert len(result) == 8 * _EMBEDDING_TRUNCATE_CHAR_PER_TOKEN


def test_truncate_positive_tokenizer_raises_fallback():
    """truncate_prompt_tokens=8 + tokenizer raises → fallback to char-based cut
    without propagating the exception."""
    model = _DummyEmbeddingModel(tokenizer=_NoneTokenizer())
    result = model._truncate_sentences(LONG_TEXT, 8)
    assert isinstance(result, str)
    assert len(result) == 8 * _EMBEDDING_TRUNCATE_CHAR_PER_TOKEN


# --- _truncate_sentences: <0 (model max_tokens) --------------------------------


def test_truncate_negative_with_max_tokens():
    """truncate_prompt_tokens=-1 + model has max_tokens=4 → truncate to 4."""
    model = _DummyEmbeddingModel(tokenizer=_StrTokenizer(), max_tokens=4)
    result = model._truncate_sentences(LONG_TEXT, -1)
    assert isinstance(result, str)
    assert len(result) <= 4 * 4


def test_truncate_negative_max_tokens_unknown_skips():
    """truncate_prompt_tokens=-1 + model max_tokens=None → skip truncation
    (defensive: unknown limit should not cause errors)."""
    model = _DummyEmbeddingModel(tokenizer=_StrTokenizer(), max_tokens=None)
    result = model._truncate_sentences(LONG_TEXT, -1)
    # Must return the original string unchanged (not truncate).
    assert result is LONG_TEXT


# --- _truncate_sentences: list[str] input ---------------------------------------


def test_truncate_list_input():
    """truncate_prompt_tokens=4 + list[str] input → each item truncated."""
    model = _DummyEmbeddingModel(tokenizer=_StrTokenizer())
    texts = ["This is a long first document.", "And another one, also long."]
    result = model._truncate_sentences(texts, 4)
    assert isinstance(result, list)
    assert len(result) == 2
    for r in result:
        assert isinstance(r, str)
        assert len(r) <= 4 * 4


# --- _truncate_sentences: non-string items in list ------------------------------


def test_truncate_non_string_items():
    """truncate_prompt_tokens=4 + list with non-string items → str() cast works."""
    model = _DummyEmbeddingModel(tokenizer=_StrTokenizer())
    texts = ["hello", 123]
    result = model._truncate_sentences(texts, 4)
    assert isinstance(result, list)
    assert len(result) == 2
