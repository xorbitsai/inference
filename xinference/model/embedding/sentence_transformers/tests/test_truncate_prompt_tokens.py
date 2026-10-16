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
        if max_length is not None:
            # Truncation requested: keep the first max_length tokens (~4 chars
            # each). max_length=0 models HF "max_length=0, truncation=True" ->
            # empty input_ids.
            tokens = [ord(c) for c in text[: max_length * 4]]
        else:
            tokens = [ord(c) for c in text]
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


# --- _truncate_sentences: non-text primitives are preserved ---------------------


def test_truncate_non_text_primitive_preserved():
    """A non-text primitive inside a list is preserved as-is (not str()'d),
    since only strings are truncatable text."""
    model = _DummyEmbeddingModel(tokenizer=_StrTokenizer())
    texts = ["hello", 123]
    result = model._truncate_sentences(texts, 4)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[1] == 123  # int preserved, not stringified to "123"


# --- _truncate_sentences: token arrays (List[int] / List[List[int]]) -----------
# Regression for the corruption bug: token arrays were str()'d, so [[1,2,3]]
# was embedded as the literal text "[[1,". They must be sliced to n_tokens ids
# and passed through to _fix_langchain_openai_inputs intact.


def test_truncate_flat_token_array_sliced():
    """List[int] (one doc as token ids) must be sliced to n_tokens ids, NOT
    stringified to ['1','2','3']."""
    model = _DummyEmbeddingModel(tokenizer=_StrTokenizer())
    result = model._truncate_sentences([1, 2, 3, 4, 5], 3)
    assert result == [1, 2, 3]


def test_truncate_nested_token_array_sliced():
    """List[List[int]] (several docs as token ids) must slice each inner list,
    preserving the nested structure so _fix_langchain_openai_inputs can still
    decode each doc."""
    model = _DummyEmbeddingModel(tokenizer=_StrTokenizer())
    result = model._truncate_sentences([[1, 2, 3, 4], [5, 6, 7, 8, 9]], 2)
    assert result == [[1, 2], [5, 6]]


def test_truncate_token_array_not_stringified():
    """Regression: the sliced result must remain int lists, never str."""
    model = _DummyEmbeddingModel(tokenizer=_StrTokenizer())
    result = model._truncate_sentences([[1, 2, 3]], 8)
    assert isinstance(result, list)
    assert isinstance(result[0], list)
    assert all(isinstance(i, int) for i in result[0])


# --- _truncate_sentences: multimodal / structured dicts ------------------------
# Regression: list(dict) iterated keys ({"text":"hello"} -> ["text"]) and
# str(dict) embedded the literal repr. Only string values may be truncated.


def test_truncate_single_dict_preserves_keys():
    """Dict[str,str] (e.g. {"text": ...}) must truncate only the value,
    preserving the key."""
    model = _DummyEmbeddingModel(tokenizer=_StrTokenizer())
    result = model._truncate_sentences({"text": LONG_TEXT}, 8)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"text"}
    assert len(result["text"]) <= 8 * 4


def test_truncate_list_of_dicts_preserves_structure():
    """List[Dict[str,str]] (multimodal): only string values truncated; keys and
    non-text values preserved verbatim; structure intact."""
    model = _DummyEmbeddingModel(tokenizer=_StrTokenizer())
    result = model._truncate_sentences(
        [{"text": LONG_TEXT}, {"text": "short", "image_url": "http://x/y.png"}],
        8,
    )
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], dict)
    assert set(result[0].keys()) == {"text"}
    assert len(result[0]["text"]) <= 8 * 4
    # non-string value preserved verbatim, key set unchanged
    assert set(result[1].keys()) == {"text", "image_url"}
    assert result[1]["image_url"] == "http://x/y.png"


# --- _truncate_sentences: multimodal media fields preserved --------------------
# Regression (PR #5151 review): only the ``text`` field may be truncated in a
# multimodal dict. ``image`` / ``video`` / ``audio`` carry URLs, file paths or
# base64 payloads; truncating them corrupts the media (a 222-char base64 was
# previously reduced to 32 chars at truncate_prompt_tokens=8).


def test_truncate_multimodal_media_fields_preserved():
    """Dict with all media keys: only ``text`` is bounded, media verbatim."""
    model = _DummyEmbeddingModel(tokenizer=_StrTokenizer())
    long_b64 = "data:image/png;base64," + "A" * 200  # 222 chars >> 8*4
    long_url = "https://example.com/path/" + "x" * 200
    result = model._truncate_sentences(
        {
            "text": LONG_TEXT,
            "image": long_b64,
            "video": long_url,
            "audio": long_url,
        },
        8,
    )
    assert isinstance(result, dict)
    assert set(result.keys()) == {"text", "image", "video", "audio"}
    assert len(result["text"]) <= 8 * 4  # text IS truncated
    assert result["image"] == long_b64  # media preserved verbatim
    assert result["video"] == long_url
    assert result["audio"] == long_url


def test_truncate_list_of_multimodal_dicts_preserves_media():
    """List[Dict[str,str]]: each item's media fields survive; only ``text``
    is bounded. Covers the char-fallback path (no tokenizer)."""
    model = _DummyEmbeddingModel(tokenizer=None)
    long_b64 = "data:image/png;base64," + "B" * 200  # >> 8*CHAR_PER_TOKEN
    result = model._truncate_sentences([{"text": LONG_TEXT, "image": long_b64}], 8)
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["image"] == long_b64
    assert len(result[0]["text"]) <= 8 * _EMBEDDING_TRUNCATE_CHAR_PER_TOKEN


def test_truncate_text_field_in_mixed_dict_is_bounded():
    """A multimodal dict still has its long ``text`` bounded, so the token
    budget still applies to the text payload alongside the preserved media."""
    model = _DummyEmbeddingModel(tokenizer=_StrTokenizer())
    result = model._truncate_sentences(
        {"image": "data:image/png;base64," + "A" * 200, "text": LONG_TEXT},
        4,
    )
    assert result["image"] == "data:image/png;base64," + "A" * 200
    assert len(result["text"]) <= 4 * 4


# --- _truncate_sentences: ==0 (vLLM parity: explicit empty) --------------------


def test_truncate_zero_yields_empty_char_fallback():
    """truncate_prompt_tokens=0 + no tokenizer -> char fallback s[:0] == ''.
    Must NOT silently expand to the model's full max_tokens (the bug)."""
    model = _DummyEmbeddingModel(tokenizer=None, max_tokens=8192)
    assert model._truncate_sentences(LONG_TEXT, 0) == ""


def test_truncate_zero_yields_empty_with_tokenizer():
    """truncate_prompt_tokens=0 + tokenizer -> max_length=0 -> empty input_ids
    -> ''. Matches vLLM (xinference/model/llm/vllm/core.py::_tokenize)."""
    model = _DummyEmbeddingModel(tokenizer=_StrTokenizer(), max_tokens=8192)
    assert model._truncate_sentences(LONG_TEXT, 0) == ""
