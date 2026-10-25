# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import pytest

from ..contract import KVCacheContract


@pytest.fixture
def kv_contract():
    return KVCacheContract(
        weights_fingerprint="a" * 64,
        tokenizer_fingerprint="b" * 64,
        attention_fingerprint="c" * 64,
        position_fingerprint="d" * 64,
        num_layers=3,
        num_kv_heads=2,
        head_dim=4,
        block_size=4,
        logical_dtype="float16",
    )
