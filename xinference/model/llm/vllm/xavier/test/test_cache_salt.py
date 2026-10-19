# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xinference.cache_isolation import cache_scope


def test_shared_hashes_are_isolated_and_memoized_by_salt(connector):
    tokens = list(range(65))
    a = connector._build_xavier_hashes(tokens, "a")
    b = connector._build_xavier_hashes(tokens, "b")
    public = connector._build_xavier_hashes(tokens)
    assert not ({key for key, _ in a} & {key for key, _ in b + public})
    assert connector._build_xavier_hashes(tokens, "a") == a
    assert len(connector._hash_cache) == 3


def test_remote_probe_cannot_find_other_salt_and_store_uses_same_scope(
    connector, connector_module
):
    tokens = list(range(33))
    stored = connector._build_xavier_hashes(tokens[:-1], "a")
    available = {key: (key, i, i) for i, (key, _) in enumerate(stored)}
    connector._query_remote_blocks = AsyncMock(
        side_effect=lambda request_id, keys: {
            2: {available[key] for key, _ in keys if key in available}
        }
    )
    connector._reserve_load_request = AsyncMock(return_value=True)
    request = SimpleNamespace(request_id="r", prompt_token_ids=tokens, cache_salt="b")
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    request.cache_salt = "a"
    assert connector.get_num_new_matched_tokens(request, 0) == (32, False)
    connector.update_state_after_alloc(
        SimpleNamespace(request_id="p", cache_salt="a"), None, 0
    )
    metadata = connector_module.XavierConnectorMetadata()
    connector._add_store_request(metadata, "p", tokens, [[0, 1]])
    assert metadata.store_requests[0].block_hashes == [key for key, _ in stored]


@pytest.mark.parametrize("salt", [None, "b"])
def test_direct_handoff_rejects_scope_mismatch_before_loading(connector, salt):
    connector._direct_handoff = True
    request = SimpleNamespace(
        request_id="r",
        prompt_token_ids=list(range(33)),
        cache_salt=salt,
        kv_transfer_params={"xavier_direct": {"cache_scope": cache_scope("a")}},
    )
    with pytest.raises(ValueError, match="scope mismatch"):
        connector.get_num_new_matched_tokens(request, 0)
    assert not connector._requests_need_load


def test_salt_survives_chunked_prefill_and_is_released_on_finish(
    connector, connector_module
):
    connector.update_state_after_alloc(
        SimpleNamespace(request_id="r", cache_salt="a"), None, 0
    )
    tokens = list(range(65))
    first = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="r",
                prompt_token_ids=tokens,
                block_ids=([0, 1, 2, 3],),
                num_computed_tokens=0,
            )
        ],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"r": 32},
    )
    meta = connector_module.XavierConnectorMetadata()
    connector._build_store_meta(first, meta)
    assert meta.store_requests[0].block_hashes == [
        key for key, _ in connector._build_xavier_hashes(tokens[:32], "a")
    ]
    second = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r"], num_computed_tokens=[32], new_block_ids=[None]
        ),
        num_scheduled_tokens={"r": 33},
    )
    meta = connector_module.XavierConnectorMetadata()
    connector._build_store_meta(second, meta)
    assert meta.store_requests[0].block_hashes == [
        key for key, _ in connector._build_xavier_hashes(tokens[:64], "a")
    ]
    connector.request_finished(SimpleNamespace(request_id="r"), [])
    assert "r" not in connector._request_cache_salts
