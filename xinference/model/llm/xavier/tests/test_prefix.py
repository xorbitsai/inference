# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
from dataclasses import replace

import pytest
import torch

from ..backends.bytes.storage import XavierBytesCacheActor
from ..backends.torch.pd import canonical_vllm_views
from ..backends.torch.prefix import pack_pages, unpack_pages
from ..prefix import prefix_keys
from .test_bytes_storage import contract


@pytest.mark.parametrize("salt", [None, "customer-a", "客户"])
def test_prefix_chain_matches_storage_protocol_and_is_causal(salt):
    tokens = list(range(130))
    keys = prefix_keys(tokens, cache_salt=salt)
    # Golden vectors from SGLang 0.5.21's native HiCache hash implementation.
    expected = {
        None: [
            "fea7b32778ecbdd7adee1941e98c89cf96bbc762f5f1beb0be24e36a456fbbc5",
            "1617a7384eff5e9135098c24794739af884859cdb17c1a61a834e8d6ac997351",
        ],
        "customer-a": [
            "0c48a0e6be2d2bc378ca598d9c9565af1916cfc658e7eccab16fb7fc4aae684a",
            "3344b0e53e7b6bab65929b8ef71c39aaa19388680d49746ea3ed020813925ed2",
        ],
        "客户": [
            "27dce5ba5c930c94b98ae692a0820035c101487282f6a5f9931a3102b33db9a5",
            "75c131ef2b54165ea603451b3ce54bd1b83c5f3166091a3b281e3cdef32c06c8",
        ],
    }
    assert keys == expected[salt]
    assert keys == prefix_keys(tokens[:128], cache_salt=salt)
    changed = list(tokens)
    changed[0] += 1
    assert not set(keys).intersection(prefix_keys(changed, cache_salt=salt))
    assert not set(keys).intersection(prefix_keys(tokens, cache_salt="other"))
    assert not set(keys).intersection(
        prefix_keys(tokens, cache_salt=salt, extra_key="adapter")
    )


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_native_vllm_and_trt_pages_share_bit_exact_canonical_bytes(dtype):
    from ...trt_llm.cross_engine import canonical_trt_views

    c = replace(contract(), logical_dtype=dtype)
    # Every physical coordinate, including the layer, has distinct raw bits.
    bits = torch.arange(4 * 2 * 2 * 64 * 2 * 8, dtype=torch.int32).to(torch.int16)
    bits = bits.reshape(4, 2, 2, 64, 2, 8)
    source = {
        f"model.layers.{i}.self_attn.attn": bits[:, i]
        .contiguous()
        .view(getattr(torch, dtype))
        for i in range(2)
    }
    views = canonical_vllm_views(source, 4, c)
    pages = pack_pages(views, [2, 0], c)
    assert pages == [bits[slot].numpy().tobytes() for slot in (2, 0)]
    # TRT's adapter supports BF16; the wire bytes themselves preserve both types.
    trt_contract = replace(c, logical_dtype="bfloat16")
    target = torch.zeros(4, 2, 2, 2 * 64 * 8, dtype=torch.bfloat16)
    trt_views = canonical_trt_views(target, trt_contract)
    unpack_pages(trt_views, [1, 3], pages, trt_contract)
    # Check native HND memory directly, independently of the TRT adapter views.
    for src, dst in ((2, 1), (0, 3)):
        for layer in range(c.num_layers):
            for kv in (0, 1):
                expected = (
                    bits[src, layer, kv].permute(1, 0, 2).contiguous().reshape(-1)
                )
                assert torch.equal(target[dst, layer, kv].view(torch.int16), expected)
    assert pack_pages(trt_views, [1, 3], trt_contract) == pages
    restored = {name: torch.zeros_like(cache) for name, cache in source.items()}
    output = canonical_vllm_views(restored, 4, c)
    unpack_pages(output, [0, 2], pages, c)
    for name, cache in restored.items():
        for src, dst in ((2, 0), (0, 2)):
            assert torch.equal(
                cache[dst].view(torch.int16), source[name][src].view(torch.int16)
            )
    assert pack_pages(output, [0, 2], c) == pages
    for cache in output.values():
        assert torch.count_nonzero(cache[1]) == 0


def test_owned_prefix_survives_eviction_and_incompatible_models_reject():
    c = contract()
    store = XavierBytesCacheActor(2 * c.num_layers * c.layer_nbytes)
    ns = store.configure(c.to_dict())["namespace"]
    page = bytes(c.num_layers * c.layer_nbytes)
    keys = prefix_keys(list(range(129)), cache_salt="a")
    store.put(ns, keys, [page, page])
    owned = store.get(ns, keys)
    store.put(ns, prefix_keys([7] * 129), [page, page])
    assert store.exists(ns, keys) == 0
    assert owned == [page, page]
    assert store.get(ns, prefix_keys(list(range(129)), cache_salt="b")) == []
    with pytest.raises(ValueError, match="weights_fingerprint"):
        store.configure(replace(c, weights_fingerprint="1" * 64).to_dict())
