# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from ..backends.torch.canonical import decode_layer, encode_layer, export_token_slots
from ..backends.torch.snapshot import KVSnapshotStore
from ..contract import KVLayerMetadata, build_prefix_keys


def _metadata(contract, num_blocks, layer=0):
    return KVLayerMetadata(
        contract,
        layer,
        tuple(
            build_prefix_keys(contract, list(range(num_blocks * contract.block_size)))
        ),
    )


@pytest.mark.parametrize("source_layout", ["N2THD", "N2HTD"])
@pytest.mark.parametrize("target_layout", ["N2THD", "N2HTD"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_all_fp16_bit_patterns_survive_layout_conversion(
    kv_contract, source_layout, target_layout, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    contract = replace(kv_contract, head_dim=16)
    metadata = _metadata(contract, 256)
    # Includes both zeros, subnormals, infinities and every NaN representation.
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    canonical = bits.view(torch.float16).reshape(256, 2, 4, 2, 16)
    source = canonical.to(device)
    if source_layout == "N2HTD":
        source = source.permute(0, 1, 3, 2, 4)
    payload = encode_layer(source, metadata, source_layout)
    source_metadata = KVLayerMetadata.from_dict(
        json.loads(json.dumps(metadata.to_dict()))
    )
    restored = decode_layer(
        payload, source_metadata, metadata, target_layout, device=device
    )
    expected = canonical
    if target_layout == "N2HTD":
        expected = expected.permute(0, 1, 3, 2, 4)
    assert restored.is_contiguous()
    assert restored.device.type == device
    assert torch.equal(
        restored.cpu().view(torch.int16), expected.contiguous().view(torch.int16)
    )
    assert payload.numel() == 256 * contract.layer_nbytes


def test_odd_byte_offset_and_ambiguous_axes_use_explicit_layout(kv_contract):
    contract = replace(kv_contract, block_size=2)
    metadata = _metadata(contract, 2)
    canonical = torch.arange(64, dtype=torch.float16).reshape(2, 2, 2, 2, 4)
    payload = encode_layer(canonical, metadata, "N2THD")
    sliced = torch.cat([torch.tensor([0], dtype=torch.uint8), payload])[1:]
    assert sliced.storage_offset() == 1 and sliced.is_contiguous()
    restored = decode_layer(sliced, metadata, metadata, "N2HTD")
    assert torch.equal(restored, canonical.permute(0, 1, 3, 2, 4).contiguous())


def test_export_and_import_do_not_alias_mutable_slots(kv_contract):
    metadata = _metadata(kv_contract, 2)
    canonical = torch.arange(128, dtype=torch.float16).reshape(2, 2, 4, 2, 4)
    payload = encode_layer(canonical, metadata, "N2THD")
    exported = payload.clone()
    restored = decode_layer(payload, metadata, metadata, "N2THD")
    canonical.zero_()
    assert torch.equal(payload, exported)
    payload.zero_()
    assert torch.equal(restored.flatten(), torch.arange(128, dtype=torch.float16))


def test_token_slot_export_uses_declared_order_and_k_before_v(kv_contract):
    metadata = _metadata(kv_contract, 2)
    keys = torch.arange(160, dtype=torch.float16).reshape(20, 2, 4)
    values = keys + 1000
    slots = torch.tensor([18, 3, 7, 1, 9, 2, 16, 5], dtype=torch.int32)
    payload = export_token_slots(keys, values, slots, metadata)
    restored = decode_layer(payload, metadata, metadata, "N2THD")
    assert torch.equal(restored[:, 0], keys[slots.long()].reshape(2, 4, 2, 4))
    assert torch.equal(restored[:, 1], values[slots.long()].reshape(2, 4, 2, 4))
    keys.zero_()
    values.zero_()
    assert restored.count_nonzero() > 0


@pytest.mark.parametrize(
    "change", ["dtype", "truncated", "oversized", "rank", "noncontiguous"]
)
def test_malformed_payload_is_rejected(kv_contract, change):
    metadata = _metadata(kv_contract, 1)
    payload = torch.zeros(kv_contract.layer_nbytes, dtype=torch.uint8)
    if change == "dtype":
        payload = payload.to(torch.int8)
    elif change == "truncated":
        payload = payload[:-1]
    elif change == "oversized":
        payload = torch.cat([payload, payload[:1]])
    elif change == "rank":
        payload = payload.reshape(2, -1)
    else:
        payload = torch.zeros(2 * payload.numel(), dtype=torch.uint8)[::2]
    with pytest.raises(ValueError, match="Invalid canonical KV payload"):
        decode_layer(payload, metadata, metadata, "N2THD")


@pytest.mark.parametrize("change", ["model", "tokenizer", "positions", "layer", "keys"])
def test_peer_mismatch_is_rejected_before_payload_access(kv_contract, change):
    metadata = _metadata(kv_contract, 1)
    expected = metadata
    if change in ("model", "tokenizer", "positions"):
        field = {
            "model": "weights_fingerprint",
            "tokenizer": "tokenizer_fingerprint",
            "positions": "position_fingerprint",
        }[change]
        expected = _metadata(replace(kv_contract, **{field: "e" * 64}), 1)
    elif change == "layer":
        expected = _metadata(kv_contract, 1, layer=1)
    else:
        expected = KVLayerMetadata(
            kv_contract, 0, tuple(build_prefix_keys(kv_contract, [9] * 4))
        )
    with pytest.raises(ValueError):
        decode_layer(object(), metadata, expected, "N2THD")


@pytest.mark.parametrize("change", ["dtype", "shape", "layout", "count"])
def test_physical_cache_must_match_declared_geometry(kv_contract, change):
    metadata = _metadata(kv_contract, 1)
    tensor = torch.zeros(1, 2, 4, 2, 4, dtype=torch.float16)
    layout = "N2THD"
    if change == "dtype":
        tensor = tensor.to(torch.bfloat16)
    elif change == "shape":
        tensor = tensor.reshape(1, 2, 2, 4, 4)
    elif change == "layout":
        layout = "auto"
    else:
        tensor = tensor.repeat(2, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        encode_layer(tensor, metadata, layout)


@pytest.mark.parametrize("slots", [[], [0], [0, 1, 2], [0, 1, 2, 3, 4]])
def test_partial_slot_groups_are_rejected(kv_contract, slots):
    pool = torch.zeros(8, 2, 4, dtype=torch.float16)
    with pytest.raises(ValueError, match="complete blocks"):
        export_token_slots(
            pool, pool, torch.tensor(slots, dtype=torch.long), _metadata(kv_contract, 1)
        )


def test_new_keys_use_existing_atomic_snapshots_and_leases(kv_contract):
    metadata = _metadata(kv_contract, 2)
    storage_keys = [key.storage_key for key in metadata.keys]
    store = KVSnapshotStore(capacity=2)
    layers = {str(layer) for layer in range(kv_contract.num_layers)}
    canonical = torch.arange(128, dtype=torch.float16).reshape(2, 2, 4, 2, 4)
    payload = encode_layer(canonical, metadata, "N2THD")
    for layer in range(kv_contract.num_layers):
        declared = replace(metadata, layer_index=layer)
        peer = KVLayerMetadata.from_dict(json.loads(json.dumps(declared.to_dict())))
        imported = decode_layer(payload, peer, declared, "N2THD")
        store.stage(str(layer), storage_keys, imported)
        published = store.publish(storage_keys, layers)
        assert published == (
            storage_keys if layer == kv_contract.num_layers - 1 else []
        )
    assert store.reserve("consumer:request", [storage_keys[0]])
    expected = canonical.clone()
    canonical.zero_()
    assert torch.equal(store.read("0", storage_keys), expected)
    # A leased block survives pressure, while the other complete block is evicted.
    next_key = build_prefix_keys(kv_contract, list(range(12)))[-1].storage_key
    store.stage("0", [next_key], expected[:1])
    assert storage_keys[0] in store.ready
    assert storage_keys[1] in store.evicted
    assert torch.equal(store.read("0", [storage_keys[0]]), expected[:1])
    store.release("consumer:request")
    assert not store.leases


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_cuda_export_and_import_preserve_bits(kv_contract):
    metadata = _metadata(kv_contract, 1)
    source = torch.arange(64, device="cuda", dtype=torch.int16).view(torch.float16)
    source = source.reshape(1, 2, 4, 2, 4)
    payload = encode_layer(source, metadata, "N2THD")
    restored = decode_layer(payload, metadata, metadata, "N2HTD", device="cuda")
    assert torch.equal(
        restored.permute(0, 1, 3, 2, 4).contiguous().view(torch.int16),
        source.view(torch.int16),
    )
