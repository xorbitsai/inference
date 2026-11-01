# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
import sys
from types import SimpleNamespace

import pytest
import torch

from ....xavier.backends.bytes.storage import XavierBytesCacheActor
from ....xavier.prefix import prefix_keys
from ....xavier.tests.test_bytes_storage import contract
from .test_cross_engine import cross_module as cross_module
from .test_prefix import prefix_module as prefix_module


def test_prefill_history_imports_before_forward_and_preserves_live_pd(
    monkeypatch, connector_config, cross_module, prefix_module
):
    monkeypatch.setitem(
        sys.modules,
        "xinference.model.llm.vllm.xavier.prefix",
        prefix_module,
    )
    c = contract()
    config = dict(
        role="prefill",
        rank=1,
        contract=c.to_dict(),
        heterogeneous=True,
        gpu_cache_bytes=0,
        prefix_cache=dict(
            capacity_bytes=2 * c.num_layers * c.layer_nbytes, write_behind=True
        ),
    )
    connector_config.cache_config.block_size = 64
    connector_config.model_config.dtype = torch.float16
    connector_config.kv_transfer_config.get_from_extra_config = lambda *a: config
    caches = SimpleNamespace(
        num_blocks=4,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=[
                    "model.layers.0.self_attn.attn",
                    "model.layers.1.self_attn.attn",
                ],
                kv_cache_spec=SimpleNamespace(),
            )
        ],
    )
    p = cross_module.CrossEngineConnector(connector_config, None, caches)
    history = p._prefix_history
    store = XavierBytesCacheActor(config["prefix_cache"]["capacity_bytes"])
    ns = store.configure(c.to_dict())["namespace"]
    calls = []

    def rpc(method, *args):
        calls.append(method)
        return getattr(store, "exists" if method == "pending_prefix" else method)(*args)

    history._rpc = rpc
    native = {
        f"model.layers.{i}.self_attn.attn": torch.zeros(
            4, 2, 64, 2, 8, dtype=torch.float16
        )
        for i in range(2)
    }
    history.register_kv_caches(native)  # Stub only the live CUDA transport boundary.
    tokens = list(range(129))
    page = bytes([1]) * c.num_layers * c.layer_nbytes
    keys = prefix_keys(tokens[:-1], cache_salt="customer-a")
    store.put(ns, keys, [page, page])
    request = SimpleNamespace(
        request_id="p",
        prompt_token_ids=tokens,
        cache_salt="customer-b",
        kv_transfer_params={},
    )
    assert p.get_num_new_matched_tokens(request, 0) == (0, False)
    request.cache_salt = "customer-a"
    assert p.get_num_new_matched_tokens(request, 64) == (64, False)
    p.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda: ([2, 0, 3],)), 64
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="p",
                prompt_token_ids=tokens,
                block_ids=([2, 0, 3],),
                num_computed_tokens=128,
            )
        ],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"p": 1},
    )
    meta = p.build_connector_meta(output)
    assert meta.prefix_meta.prefix_loads == [([0], [page])]
    assert meta.loads == []  # P's history never enters D's live receive path.
    p._get_connector_metadata = lambda: meta
    p.start_load_kv(None)
    assert store.get_stats()["imported_pages"] == 1
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    p.wait_for_save()
    assert "pending_prefix" in calls and "exists" not in calls
    assert p.request_finished(request, [2, 0, 3]) == (False, None)
    assert not history._prefix_pending and not history._prefix_allocated
    p.shutdown()


@pytest.mark.asyncio
async def test_history_lookup_does_not_wait_for_storage_deadline(
    prefix_module, monkeypatch
):
    import asyncio
    import time
    from unittest.mock import AsyncMock

    c = contract()
    client = object.__new__(prefix_module.PrefixConnector)
    client.contract = c
    client._xavier_config = dict(write_behind=True, read_timeout=30)
    client._prefix_retry_at = 0
    client._prefix_read_retry_at = 0
    cancelled = asyncio.Event()

    async def stalled(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    client._prefix_ref = SimpleNamespace(get=AsyncMock(side_effect=stalled))
    monkeypatch.setattr(prefix_module, "PREFIX_RPC_TIMEOUT", 0.01)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            client._prefix_rpc("get", c.fingerprint, prefix_keys(list(range(129)))),
            0.5,
        )
    assert cancelled.is_set()
    assert client._prefix_read_retry_at > time.monotonic()
    client._prefix_ref = SimpleNamespace(pending_prefix=AsyncMock(return_value=0))
    assert await client._prefix_rpc("pending_prefix", c.fingerprint, []) == 0
