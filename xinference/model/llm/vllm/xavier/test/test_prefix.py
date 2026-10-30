# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ....xavier.backends.bytes.storage import XavierBytesCacheActor
from ....xavier.prefix import prefix_keys
from ....xavier.tests.test_bytes_storage import contract


@pytest.fixture
def prefix_module(connector_module, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "xinference.model.llm.vllm.xavier.v1_connector", connector_module
    )
    name = "xinference.model.llm.vllm.xavier._test_prefix"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "prefix.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_prefix_admission_owns_pages_across_eviction_and_tracks_salt(
    prefix_module, connector_config
):
    c = contract()
    config = dict(
        contract=c.to_dict(),
        shared_prefix=True,
        capacity_bytes=2 * c.num_layers * c.layer_nbytes,
    )
    connector_config.cache_config.block_size = 64
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
    client = prefix_module.PrefixConnector(connector_config, None, caches)
    store = XavierBytesCacheActor(config["capacity_bytes"])
    ns = store.configure(c.to_dict())["namespace"]
    client._rpc = lambda method, *args: getattr(store, method)(*args)
    native = {
        f"model.layers.{i}.self_attn.attn": torch.zeros(
            4, 2, 64, 2, 8, dtype=torch.float16
        )
        for i in range(2)
    }
    client.register_kv_caches(native)
    keys = prefix_keys(list(range(129))[:-1], cache_salt="a")
    page = bytes([1] * c.num_layers * c.layer_nbytes)
    store.put(ns, keys, [page, page])
    request = SimpleNamespace(
        request_id="r", prompt_token_ids=list(range(129)), cache_salt="b"
    )
    assert client.get_num_new_matched_tokens(request, 0) == (0, False)
    request.cache_salt = "a"
    assert client.get_num_new_matched_tokens(request, 0) == (128, False)
    store.put(ns, prefix_keys([5] * 128), [page, page])
    assert store.get(ns, keys) == []
    client.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda: ([2, 0, 3],)), 128
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="r",
                prompt_token_ids=request.prompt_token_ids,
                block_ids=([2, 0, 3],),
                num_computed_tokens=128,
            )
        ],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"r": 1},
    )
    metadata = client.build_connector_meta(output)
    client._get_connector_metadata = lambda: metadata
    client.start_load_kv(None)
    assert store.get_stats()["imported_pages"] == 2
    client.wait_for_save()
    assert store.get(ns, keys) == [page, page]
    assert client.request_finished(request, [2, 0, 3]) == (False, None)
    assert not client._prefix_pending and not client._prefix_allocated
    assert not client._request_cache_salts and not client._chunked_prefill
    client.shutdown()


def test_unavailable_storage_is_a_miss_and_request_state_is_released(
    prefix_module, connector_config
):
    c = contract()
    config = dict(
        contract=c.to_dict(),
        shared_prefix=True,
        capacity_bytes=c.num_layers * c.layer_nbytes,
    )
    connector_config.cache_config.block_size = 64
    connector_config.kv_transfer_config.get_from_extra_config = lambda *a: config
    caches = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[
            SimpleNamespace(layer_names=["layer"], kv_cache_spec=SimpleNamespace())
        ],
    )
    client = prefix_module.PrefixConnector(connector_config, None, caches)

    def unavailable(*args):
        raise ConnectionError("storage unavailable")

    client._rpc = unavailable
    request = SimpleNamespace(
        request_id="failed", prompt_token_ids=list(range(65)), cache_salt="a"
    )
    assert client.get_num_new_matched_tokens(request, 0) == (0, False)
    assert client.get_num_new_matched_tokens(request, 1) == (0, False)
    client.request_finished(request, [])
    assert not client._prefix_pending and not client._request_cache_salts
    client.shutdown()
