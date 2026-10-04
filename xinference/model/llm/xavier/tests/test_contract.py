# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import json
import subprocess
import sys
from dataclasses import replace

import pytest

from ..contract import (
    KVBlockKey,
    KVCacheContract,
    KVLayerMetadata,
    build_block_key,
    build_prefix_keys,
    fingerprint_files,
    fingerprint_metadata,
)


def test_contract_and_transfer_metadata_survive_json(kv_contract):
    keys = tuple(build_prefix_keys(kv_contract, list(range(8))))
    metadata = KVLayerMetadata(kv_contract, 2, keys)
    restored = KVLayerMetadata.from_dict(json.loads(json.dumps(metadata.to_dict())))
    assert restored == metadata
    restored.require_match(metadata)
    assert restored.contract.fingerprint == kv_contract.fingerprint
    assert restored.contract.layer_nbytes == 128
    assert all(key.storage_key.bit_length() > 64 for key in restored.keys)


def test_protocol_v1_golden_keys(kv_contract):
    # Changing this wire identity requires an explicit protocol-version decision.
    assert kv_contract.fingerprint == (
        "80508d1d6d3079602517a9d11e38640ae8c28752a4e5f9d49e9e4ffffe3c6dfe"
    )
    assert [key.digest for key in build_prefix_keys(kv_contract, list(range(8)))] == [
        "ca200c64b1cca99cb07a21118f581dca9af101e2689b200a16cf306189e1dab0",
        "6d54ad461d9d15d70e03af4558811fa7426833081a13c117774de4383b74aee9",
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("weights_fingerprint", "e" * 64),
        ("tokenizer_fingerprint", "e" * 64),
        ("attention_fingerprint", "e" * 64),
        ("position_fingerprint", "e" * 64),
        ("num_layers", 4),
        ("num_kv_heads", 4),
        ("head_dim", 8),
        ("block_size", 8),
    ],
)
def test_incompatible_semantics_cannot_share_keys(kv_contract, field, value):
    peer = replace(kv_contract, **{field: value})
    with pytest.raises(ValueError, match=field):
        kv_contract.require_match(peer)
    assert peer.fingerprint != kv_contract.fingerprint
    tokens = list(range(16))
    local_key = build_prefix_keys(kv_contract, tokens)[0]
    peer_key = build_prefix_keys(peer, tokens)[0]
    assert local_key.storage_key != peer_key.storage_key
    with pytest.raises(ValueError, match="incompatible prefix"):
        build_block_key(peer, [1] * peer.block_size, [0] * peer.block_size, local_key)


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol_version", 2),
        ("protocol_version", True),
        ("layout_version", 2),
        ("layout", "N2HTD"),
        ("transport_dtype", "float16"),
        ("logical_dtype", "auto"),
        ("logical_dtype", "bfloat16"),
        ("logical_dtype", "fp8"),
        ("tensor_parallel_size", 2),
        ("pipeline_parallel_size", 2),
        ("attention_kind", "sliding_window"),
        ("attention_kind", "recurrent"),
        ("weight_quantization", "int4"),
        ("is_multimodal", True),
        ("has_lora", True),
        ("has_lora", 0),
        ("num_layers", 0),
        ("num_kv_heads", -1),
        ("head_dim", 4.0),
        ("block_size", True),
        ("weights_fingerprint", "model-name"),
        ("tokenizer_fingerprint", "B" * 64),
        ("attention_fingerprint", ""),
        ("position_fingerprint", None),
    ],
)
def test_unsupported_or_incomplete_contract_is_rejected(kv_contract, field, value):
    metadata = kv_contract.to_dict()
    metadata[field] = value
    with pytest.raises(ValueError):
        KVCacheContract.from_dict(metadata)


@pytest.mark.parametrize("kind", ["contract", "key", "layer"])
@pytest.mark.parametrize("change", ["missing", "unknown", "not-mapping"])
def test_wire_metadata_never_ignores_fields(kv_contract, kind, change):
    key = build_prefix_keys(kv_contract, list(range(4)))[0]
    item = {
        "contract": kv_contract,
        "key": key,
        "layer": KVLayerMetadata(kv_contract, 0, (key,)),
    }[kind]
    metadata = item.to_dict()
    if change == "missing":
        metadata.pop(next(iter(metadata)))
    elif change == "unknown":
        metadata["engine"] = "sglang"
    else:
        metadata = []
    with pytest.raises(ValueError):
        type(item).from_dict(metadata)


def test_content_keys_follow_causal_prefix_and_request_salt(kv_contract):
    original = build_prefix_keys(kv_contract, [1] * 4 + [2] * 4)
    changed = build_prefix_keys(kv_contract, [9] * 4 + [2] * 4)
    assert original[0] != changed[0]
    assert original[1] != changed[1]
    assert original[0].prefix_tokens == 4
    assert original[1].prefix_tokens == 8
    assert original == build_prefix_keys(kv_contract, [1] * 4 + [2] * 4, list(range(8)))
    salted = build_prefix_keys(kv_contract, [1] * 4 + [2] * 4, cache_salt="tenant-a")
    assert all(local.digest != remote.digest for local, remote in zip(original, salted))


@pytest.mark.parametrize("length", [0, 1, 3, 4, 5, 7, 8, 9])
def test_only_complete_blocks_get_content_keys(kv_contract, length):
    keys = build_prefix_keys(kv_contract, list(range(length)))
    assert [key.prefix_tokens for key in keys] == list(range(4, length + 1, 4))


@pytest.mark.parametrize(
    "tokens,positions",
    [
        ([1, 2, 3, 4], [0, 1, 2]),
        ([1, 2, 3, 4], [1, 2, 3, 4]),
        ([1, 2, 3, 4], [0, 1, 3, 4]),
        ([1, 2, 3, 4], [False, 1, 2, 3]),
        ([1, 2, 3, 4], [0.0, 1, 2, 3]),
        ([1, -1, 3, 4], [0, 1, 2, 3]),
        ([1, 2**32, 3, 4], [0, 1, 2, 3]),
        ([1, True, 3, 4], [0, 1, 2, 3]),
        ([1, 2.0, 3, 4], [0, 1, 2, 3]),
    ],
)
def test_invalid_tokens_and_positions_are_rejected(kv_contract, tokens, positions):
    with pytest.raises(ValueError):
        build_block_key(kv_contract, tokens, positions)
    with pytest.raises(ValueError):
        build_prefix_keys(kv_contract, tokens, positions)


def test_partial_tail_is_validated_and_cannot_be_published(kv_contract):
    with pytest.raises(ValueError, match="Invalid token"):
        build_prefix_keys(kv_contract, [0, 1, 2, 3, -1])
    with pytest.raises(ValueError, match="complete block"):
        build_block_key(kv_contract, [0], [0])
    with pytest.raises(ValueError, match="cache_salt"):
        build_prefix_keys(kv_contract, [], cache_salt=None)
    with pytest.raises(ValueError, match="incompatible prefix"):
        build_block_key(
            kv_contract,
            [0] * 4,
            [0, 1, 2, 3],
            KVBlockKey(kv_contract.fingerprint, 3, "e" * 64),
        )


@pytest.mark.parametrize("layer", [-1, 3, True])
def test_logical_layer_indices_are_explicit(kv_contract, layer):
    keys = tuple(build_prefix_keys(kv_contract, list(range(4))))
    with pytest.raises(ValueError):
        KVLayerMetadata(kv_contract, layer, keys)


def test_layer_metadata_binds_contract_keys_and_order(kv_contract):
    keys = tuple(build_prefix_keys(kv_contract, list(range(8))))
    metadata = KVLayerMetadata(kv_contract, 0, keys)
    with pytest.raises(ValueError, match="order"):
        metadata.require_match(KVLayerMetadata(kv_contract, 0, keys[::-1]))
    with pytest.raises(ValueError, match="logical layer"):
        metadata.require_match(KVLayerMetadata(kv_contract, 1, keys))
    with pytest.raises(ValueError, match="Duplicate"):
        KVLayerMetadata(kv_contract, 0, (keys[0], keys[0]))
    with pytest.raises(ValueError, match="immutable sequence"):
        KVLayerMetadata(kv_contract, 0, list(keys))
    with pytest.raises(ValueError, match="incompatible contract"):
        KVLayerMetadata(replace(kv_contract, weights_fingerprint="e" * 64), 0, keys)


def test_metadata_fingerprint_is_order_independent_and_finite():
    a = {"rope": {"theta": 10000.0, "scale": 1.0}, "head_dim": 64}
    b = {"head_dim": 64, "rope": {"scale": 1.0, "theta": 10000.0}}
    assert fingerprint_metadata(a) == fingerprint_metadata(b)
    b["rope"]["theta"] = 100000.0
    assert fingerprint_metadata(a) != fingerprint_metadata(b)
    with pytest.raises(ValueError):
        fingerprint_metadata({"scale": float("nan")})


def test_asset_identity_survives_relocation_and_detects_changed_bytes(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "relocated"
    first.mkdir()
    second.mkdir()
    for root in (first, second):
        (root / "weights").write_bytes(b"weights")
        (root / "tokenizer").write_bytes(b"tokenizer")
    a = {name: first / name for name in ("weights", "tokenizer")}
    b = {name: second / name for name in ("tokenizer", "weights")}
    fingerprint = fingerprint_files(a)
    assert fingerprint == fingerprint_files(b)
    (second / "weights").write_bytes(b"changed")
    assert fingerprint != fingerprint_files(b)
    with pytest.raises(ValueError, match="nonempty"):
        fingerprint_files({})


def test_keys_are_stable_in_a_fresh_interpreter(kv_contract):
    tokens = list(range(9))
    expected = [key.to_dict() for key in build_prefix_keys(kv_contract, tokens)]
    script = """
import json, sys
from xinference.model.llm.xavier.contract import KVCacheContract, build_prefix_keys
contract = KVCacheContract.from_dict(json.loads(sys.argv[1]))
print(json.dumps([key.to_dict() for key in build_prefix_keys(contract, list(range(9)))]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(kv_contract.to_dict())],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected
