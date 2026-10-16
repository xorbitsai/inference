# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import pytest

from ..utils import hash_block_tokens


@pytest.mark.parametrize(
    "first,previous,tokens,expected",
    [
        (True, None, [], 16765256144593249530),
        (True, -1, [1, 2, 3], 3926270711534698589),
        (False, 2**64 - 1, [0, 2147483647, -2147483648], 682864629166071906),
        (False, 123, [1] * 16, 9100343863099402539),
    ],
)
def test_token_hash_preserves_v1_wire_format(first, previous, tokens, expected):
    # Golden hashes produced before vectorizing token encoding. Existing
    # snapshots and ranks running older releases must keep matching.
    assert hash_block_tokens(first, previous, tokens) == expected
