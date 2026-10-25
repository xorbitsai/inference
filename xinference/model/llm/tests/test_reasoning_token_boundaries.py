# Copyright 2022-2026 Xinference Holdings Pte. Ltd
from unittest.mock import Mock

import pytest

from ..reasoning_parser import KimiK3ReasoningParser, ReasoningParser


class CharacterTokenizer:
    """Keep marker boundaries deliberately split across several tokens."""

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


@pytest.mark.parametrize(
    "parser",
    [
        ReasoningParser(reasoning_start_tag="<think>", reasoning_end_tag="</think>"),
        KimiK3ReasoningParser(),
        ReasoningParser(
            reasoning_start_tag="<|channel>thought\n", reasoning_end_tag="<channel|>"
        ),
    ],
)
def test_latest_prompt_boundary_and_split_decode_marker(parser):
    tokenizer = CharacterTokenizer()
    parser.bind_tokenizer(tokenizer)
    opening, closing = parser.reasoning_start_tag, parser.reasoning_end_tag
    history = opening + "old reasoning" + closing + "old answer"
    disabled = history + opening + closing
    enabled = history + opening
    assert parser.is_reasoning_end(tokenizer.encode(disabled))
    assert not parser.is_reasoning_end(tokenizer.encode(enabled))
    assert not parser.is_reasoning_end([])
    output = enabled + "new reasoning"
    for index, char in enumerate(closing):
        output += char
        assert parser.is_reasoning_end_streaming(
            tokenizer.encode(output), iter(tokenizer.encode(char))
        ) is (index == len(closing) - 1)
    assert not parser.is_reasoning_end_streaming(
        tokenizer.encode(output + "x"), [ord("x")]
    )
    assert not parser.is_reasoning_end_streaming(tokenizer.encode(output), iter([]))
    assert parser.extract_content_ids(
        tokenizer.encode(output + "{}")
    ) == tokenizer.encode("{}")


def test_tokenizer_binding_is_not_repeated_during_decode():
    tokenizer = Mock()
    tokenizer.encode.side_effect = [[1], [2]]
    parser = ReasoningParser(
        reasoning_start_tag="<think>", reasoning_end_tag="</think>"
    )
    parser.bind_tokenizer(tokenizer)
    for _ in range(3):
        assert parser.is_reasoning_end([1, 3, 2])
        assert parser.is_reasoning_end_streaming([1, 3, 2], [2])
    assert tokenizer.encode.call_count == 2
    assert parser.extract_content_ids([1, 3]) == []


def test_empty_token_markers_are_rejected():
    parser = ReasoningParser()
    with pytest.raises(ValueError, match="non-empty"):
        parser.bind_tokenizer(CharacterTokenizer())


@pytest.mark.parametrize("unknown", [None, 7])
def test_ambiguous_or_unknown_token_markers_are_rejected(unknown):
    tokenizer = Mock(unk_token_id=unknown)
    tokenizer.encode.side_effect = [[7], [7 if unknown is None else 8]]
    parser = ReasoningParser(
        reasoning_start_tag="<think>", reasoning_end_tag="</think>"
    )
    with pytest.raises(ValueError, match="distinct known"):
        parser.bind_tokenizer(tokenizer)
