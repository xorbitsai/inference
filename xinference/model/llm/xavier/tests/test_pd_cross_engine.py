# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import json
import struct
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
import torch

from ...sglang.xavier.directory import XavierPDDirectory
from ..backends.torch.pd import (
    CrossEngineGPUActor,
    canonical_vllm_views,
    sglang_first_token_payload,
)
from ..pd_contract import build_pd_contract, prompt_digest


@pytest.mark.parametrize("layout", ["2NTHD", "N2THD", "BHN2D", "BHN2D_NHD"])
def test_vllm_canonical_views_preserve_engine_storage(kv_contract, layout):
    contract = replace(kv_contract, num_layers=2)
    tensors = {
        f"model.layers.{i}.self_attn.attn": torch.arange(
            7 * 2 * 4 * 2 * 4, dtype=torch.float16
        ).reshape(7, 2, 4, 2, 4)
        + i * 1000
        for i in range(2)
    }
    original = {name: tensor.clone() for name, tensor in tensors.items()}
    if layout == "2NTHD":
        tensors = {name: tensor.movedim(1, 0) for name, tensor in tensors.items()}
    if layout.startswith("BHN2D"):
        tensors = {
            name: tensor.permute(0, 3, 2, 1, 4).reshape(7, 2, 4, 8)
            for name, tensor in tensors.items()
        }
        if layout == "BHN2D_NHD":
            tensors = {
                name: tensor.transpose(1, 2).contiguous().transpose(1, 2)
                for name, tensor in tensors.items()
            }
    views = canonical_vllm_views(tensors, 7, contract)
    for kind in range(2):
        for index in range(2):
            name = f"model.layers.{index}.self_attn.attn"
            view = views[str(index + kind * 2)]
            assert torch.equal(view, original[name][:, kind])
            view[3, 2] = -1
            native = tensors[name] if layout == "N2THD" else tensors[name].movedim(1, 0)
            if layout.startswith("BHN2D"):
                assert (tensors[name][3, :, 2, kind * 4 : (kind + 1) * 4] == -1).all()
            else:
                assert (native[3, kind, 2] == -1).all()


def test_canonical_views_reject_missing_layer_and_dtype(kv_contract):
    contract = replace(kv_contract, num_layers=1)
    with pytest.raises(ValueError, match="Incomplete"):
        canonical_vllm_views({}, 7, contract)
    with pytest.raises(ValueError, match="layout"):
        canonical_vllm_views(
            {"model.layers.0.self_attn.attn": torch.zeros(7, 2, 4, 2, 4)}, 7, contract
        )


def test_first_token_metadata_and_room_marker():
    sizes = [64, 64, 64, 64, 512, 512, 64, 128, 1024, 64]
    payload = sglang_first_token_payload(sizes, 123, 42, 193)
    assert list(map(len, payload)) == sizes
    assert struct.unpack_from("<i", payload[0])[0] == 42
    assert struct.unpack_from("<i", payload[1])[0] == 193
    assert struct.unpack_from("<Q", payload[-1])[0] == 123
    with pytest.raises(ValueError, match="layout"):
        sglang_first_token_payload(sizes + [64], 123, 42, 193)


def test_directory_proves_imported_tokens_and_single_completion(kv_contract):
    directory = XavierPDDirectory()
    directory.configure(kv_contract.fingerprint)
    for role in ("prefill", "decode"):
        directory.prepare(1, kv_contract.fingerprint, prompt_digest([1, 2]), role, prompt_tokens=2)
    directory.complete(1, 4096, 1)
    directory.complete(1, 4096, 1)
    stats = directory.get_stats()
    assert stats["imported_tokens"] == 1
    assert stats["gpu_bytes"] == 4096
    assert stats["completed_requests"] == 1
    directory.release(1, "decode")
    assert directory.check(1)
    directory.release(1, "prefill")
    assert directory.get_stats()["active_handoffs"] == 0


def test_contract_fingerprints_actual_assets_and_context(tmp_path):
    config = dict(
        model_type="qwen2",
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_size=8,
        max_position_embeddings=4096,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "tokenizer.json").write_bytes(b"tokenizer")
    initial = build_pd_contract(str(tmp_path))
    initial.require_match(build_pd_contract(str(tmp_path), 4096))
    with pytest.raises(ValueError, match="position_fingerprint"):
        initial.require_match(build_pd_contract(str(tmp_path), 8192))
    (tmp_path / "model.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="weights_fingerprint"):
        initial.require_match(build_pd_contract(str(tmp_path)))


@pytest.mark.asyncio
async def test_late_source_publication_does_not_leak_a_room():
    directory = AsyncMock()
    directory.publish_source.side_effect = ValueError("cancelled room")
    actor = CrossEngineGPUActor(None, None, directory, "namespace", 1)
    with pytest.raises(ValueError, match="cancelled room"):
        await actor.open(123)
    assert not actor.rooms


@pytest.mark.asyncio
async def test_aborted_source_cannot_register_late_gpu_chunks():
    actor = CrossEngineGPUActor(None, None, None, "namespace", 1)
    actor.rooms[123] = {"aborted": True}
    with pytest.raises(RuntimeError, match="cancelled"):
        actor.init(123, 3, 0)
    with pytest.raises(RuntimeError, match="cancelled"):
        await actor.add_chunk(123, [1, 2, 3])
    assert actor.rooms[123] == {"aborted": True}
