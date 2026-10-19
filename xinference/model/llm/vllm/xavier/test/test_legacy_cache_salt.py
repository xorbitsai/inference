# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import copy
import pickle
from collections import deque
from types import SimpleNamespace

import pytest
from packaging.version import Version


def test_legacy_sampling_salt_survives_clone_and_serialization():
    pytest.importorskip("vllm")
    from ..legacy_salt import XavierSamplingParams

    params = XavierSamplingParams(cache_salt="private-scope")
    assert params.clone().cache_salt == "private-scope"
    assert pickle.loads(pickle.dumps(params)).cache_salt == "private-scope"
    assert "private-scope" not in repr(params)


def test_v0_scheduler_scopes_native_and_remote_block_identity():
    vllm = pytest.importorskip("vllm")
    if Version(vllm.__version__) >= Version("0.11.0"):
        pytest.skip("V0 scheduler requires vLLM < 0.11")
    from ..scheduler import XavierScheduler

    scheduler = object.__new__(XavierScheduler)
    scheduler.waiting = deque()

    def submit(salt):
        seq = SimpleNamespace(extra_hash=lambda: None)
        group = SimpleNamespace(
            sampling_params=SimpleNamespace(cache_salt=salt), get_seqs=lambda: [seq]
        )
        scheduler.add_seq_group(group)
        return seq

    a, b, repeated, public = [submit(salt) for salt in ("a", "b", "a", None)]
    assert a.extra_hash() == repeated.extra_hash()
    assert a.extra_hash() != b.extra_hash()
    assert public.extra_hash() is None
    assert copy.deepcopy(a).extra_hash() == a.extra_hash()
