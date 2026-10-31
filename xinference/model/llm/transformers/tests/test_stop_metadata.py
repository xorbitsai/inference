# CI: linux

from xinference.model.llm.transformers.utils import _find_stop_sequence


def test_find_stop_sequence_returns_earliest_match():
    assert _find_stop_sequence("answer<END>tail", ["tail", "<END>"]) == "<END>"


def test_find_stop_sequence_returns_none_without_match():
    assert _find_stop_sequence("answer", ["<END>"]) is None
