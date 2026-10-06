# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import pytest

from ..backends.bytes.storage import XavierBytesCacheActor
from ..contract import KVCacheContract


def contract():
    return KVCacheContract(
        *["0" * 64] * 4,
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        block_size=64,
        logical_dtype="float16",
    )


def store(capacity=2):
    c = contract()
    actor = XavierBytesCacheActor(capacity * c.layer_nbytes * c.num_layers)
    namespace = actor.configure(c.to_dict())
    page = bytes(c.layer_nbytes * c.num_layers)
    return actor, namespace, page


def test_capacity_geometry_and_contract():
    actor, namespace, page = store()
    keys = [str(i) * 64 for i in range(1, 4)]
    assert actor.put(namespace, keys, [page] * 3) == [False] * 3
    assert actor.get(namespace, keys) == []
    assert actor.put(namespace, keys[:2], [page] * 2) == [True] * 2
    assert actor.get_stats()["pages"] == 2
    with pytest.raises(ValueError, match="geometry"):
        actor.put(namespace, [keys[0]], [page[:-1]])
    with pytest.raises(ValueError, match="namespace"):
        actor.get("wrong", keys)
    metadata = contract().to_dict()
    metadata["weights_fingerprint"] = "1" * 64
    with pytest.raises(ValueError, match="weights_fingerprint"):
        actor.configure(metadata)
    with pytest.raises(ValueError, match="budget"):
        XavierBytesCacheActor(1).configure(contract().to_dict())


@pytest.mark.parametrize("touch", [False, True])
def test_eviction_preserves_chain_head_and_oversized_put_preserves_cache(touch):
    actor, namespace, page = store(3)
    keys = [str(i) * 64 for i in range(1, 5)]
    actor.put(namespace, keys[:3], [page] * 3)
    if touch:
        assert actor.get(namespace, keys[:3]) == [page] * 3
    actor.put(namespace, keys[3:], [page])
    assert actor.get(namespace, keys[:3]) == [page] * 2
    assert actor.get(namespace, keys[3:]) == [page]
    assert actor.put(namespace, keys, [page] * 4) == [False] * 4
    assert actor.get(namespace, keys[:3]) == [page] * 2


def test_put_reports_pages_that_survive_pinned_capacity_pressure():
    actor, namespace, page = store(3)
    keys = [str(i) * 64 for i in range(1, 5)]
    ticket = actor.prepare_handoff(namespace, keys[3:])
    actor.put(namespace, keys[3:], [page])
    assert actor.put(namespace, keys[:3], [page] * 3) == [True, True, False]
    assert actor.get(namespace, keys[:3]) == [page] * 2
    assert actor.get_handoff(namespace, keys[3:], ticket) == [page]


def test_suffix_publication_keeps_existing_chain_heads_newer_than_tails():
    actor, namespace, page = store(4)
    keys = [str(i) * 64 for i in range(1, 6)]
    actor.put(namespace, keys[:2], [page] * 2)
    assert actor.put(namespace, keys[:4], [page] * 2, start=2) == [True] * 4
    actor.put(namespace, keys[4:], [page])
    assert actor.get(namespace, keys[:4]) == [page] * 3
    with pytest.raises(ValueError, match="geometry"):
        actor.put(namespace, keys[:4], [page], start=2)


def test_leases_pin_absent_pages_and_handoff_is_consumed_once():
    actor, namespace, page = store()
    keys = ["1" * 64, "2" * 64]
    ticket = actor.prepare_handoff(namespace, keys)
    assert not actor.handoff_ready(namespace, keys, ticket)
    with pytest.raises(ValueError, match="incomplete"):
        actor.get_handoff(namespace, keys, ticket)
    assert actor.put(namespace, keys, [page, page]) == [True, True]
    assert actor.handoff_ready(namespace, keys, ticket)
    assert actor.put(namespace, ["3" * 64], [page]) == [False]
    with pytest.raises(RuntimeError, match="budget"):
        actor.prepare_handoff(namespace, ["3" * 64])
    with pytest.raises(ValueError, match="incomplete"):
        actor.get_handoff(namespace, [keys[0]], ticket)
    assert actor.get_handoff(namespace, keys, ticket) == [page, page]
    with pytest.raises(ValueError, match="consumed"):
        actor.handoff_ready(namespace, keys, ticket)
    with pytest.raises(ValueError, match="consumed"):
        actor.get_handoff(namespace, keys, ticket)
    actor.release_handoff(ticket)
    actor.release_handoff(ticket)
    assert actor.put(namespace, ["3" * 64], [page]) == [True]
    assert actor.get_stats()["active_handoffs"] == 0


def test_immutable_pages_expired_lease_and_empty_prefix():
    actor, namespace, page = store(1)
    key = "1" * 64
    actor.put(namespace, [key], [page])
    actor.put(namespace, [key], [b"x" * len(page)])
    assert actor.get(namespace, [key]) == [page]
    ticket = actor.prepare_handoff(namespace, [key])
    actor._handoffs[ticket]["deadline"] = 0
    assert actor.get_stats()["active_handoffs"] == 0
    ticket = actor.prepare_handoff(namespace, [])
    assert actor.get_handoff(namespace, [], ticket) == []
    actor.release_handoff(ticket)
    assert actor.get_stats()["handoff_reads"] == 1
