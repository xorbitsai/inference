# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Opt-in real SGLang -> Xavier -> fresh SGLang prefix-cache regression."""

import asyncio
import json
import multiprocessing
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("XINFERENCE_TEST_SGLANG_XAVIER") != "1",
    reason="requires two CUDA GPUs, SGLang >= 0.5.21 and local model weights",
)


def test_salted_sglang_storage_hashes_cannot_reuse_other_salts(monkeypatch):
    from array import array
    from types import SimpleNamespace

    import torch
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.utils import get_storage_hash_str

    from ...xavier.backends.torch.storage import XavierCacheActor
    from ...xavier.contract import KVCacheContract
    from ..xavier.storage import XavierHiCacheStorage

    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 2, 2, 4, 4, "float16"
    )
    actor = XavierCacheActor(4 * contract.num_layers * contract.layer_nbytes)
    namespace = actor.configure(
        contract.to_dict(), {"engine": "sglang", "layout": "layer_first"}
    )
    storage = XavierHiCacheStorage(
        SimpleNamespace(
            tp_size=1,
            pp_size=1,
            attn_cp_size=1,
            is_mla_model=False,
            extra_config={"contract": contract.to_dict()},
        )
    )
    storage._namespace = namespace
    monkeypatch.setattr(
        storage, "_rpc", lambda method, *args: getattr(actor, method)(*args)
    )
    keys = [
        get_storage_hash_str(
            RadixKey(array("q", range(8)), cache_salt=salt), page_size=4
        )
        for salt in (None, "tenant-a", "tenant-b")
    ]
    page = torch.zeros(
        contract.num_layers * contract.layer_nbytes // 2, dtype=torch.float16
    )
    assert storage.batch_set(keys[1], [page, page])
    assert storage.batch_exists(keys[1]) == 2
    assert storage.batch_exists(keys[0]) == storage.batch_exists(keys[2]) == 0
    assert storage.batch_get(keys[2]) == [None, None]
    assert storage.batch_set(keys[2], [page + 1, page + 1])
    assert all(torch.equal(value, page) for value in storage.batch_get(keys[1]))
    assert all(torch.equal(value, page + 1) for value in storage.batch_get(keys[2]))
    storage.close()


@pytest.mark.asyncio
async def test_real_sglang_remote_prefix_restore(tmp_path):
    import requests
    import sglang as sgl
    import torch
    import xoscar as xo
    from transformers import AutoTokenizer

    from ...xavier.backends.torch.storage import XavierCacheActor
    from ..xavier.config import configure_xavier

    assert torch.cuda.device_count() >= 2
    multiprocessing.set_start_method("spawn", force=True)
    model_path = os.environ["XINFERENCE_TEST_PD_MODEL_PATH"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 120)[
        :1024
    ]
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        actor = await xo.create_actor(
            XavierCacheActor,
            64 * 1024 * 1024,
            address=pool.external_address,
            uid="xavier-test",
        )
        cache = {"address": actor.address, "uid": "xavier-test"}
        config = {
            "tp_size": 1,
            "mem_fraction_static": 0.3,
            "max_total_tokens": 4096,
            "context_length": 4096,
            "disable_cuda_graph": True,
            "hicache_ratio": 1.1,
            "enable_cache_report": True,
            "log_level": "info",
        }
        await asyncio.to_thread(configure_xavier, model_path, config, cache)
        engines = []
        results = {}

        async def generate(engine):
            response = await asyncio.to_thread(
                requests.post,
                engine.url + "/generate",
                json={
                    "input_ids": tokens,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 16,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()

        try:
            producer = await asyncio.to_thread(
                sgl.Runtime, model_path=model_path, base_gpu_id=0, **config
            )
            engines.append(producer)
            results["cold"] = await generate(producer)
            for _ in range(100):
                stats = await actor.get_stats()
                if stats["stored_pages"]:
                    break
                await asyncio.sleep(0.1)
            assert stats["stored_pages"] > 0, stats
            results["after_publish"] = stats
            consumer = await asyncio.to_thread(
                sgl.Runtime, model_path=model_path, base_gpu_id=1, **config
            )
            engines.append(consumer)
            results["warm"] = await generate(consumer)
            results["after_restore"] = await actor.get_stats()
            assert (
                results["after_restore"]["read_pages"]
                > results["after_publish"]["read_pages"]
            )
            assert results["warm"]["meta_info"]["cached_tokens"] > 0
            assert (
                results["warm"]["meta_info"]["cached_tokens_details"]["storage"] == 960
            )
            assert results["cold"]["text"] == results["warm"]["text"]
            assert [
                item[1]
                for item in results["cold"]["meta_info"]["output_token_logprobs"]
            ] == [
                item[1]
                for item in results["warm"]["meta_info"]["output_token_logprobs"]
            ]
            flushed = await asyncio.to_thread(
                requests.post, consumer.url + "/flush_cache", timeout=30
            )
            flushed.raise_for_status()
            flushed = await asyncio.to_thread(
                requests.post, consumer.url + "/flush_cache", timeout=30
            )
            flushed.raise_for_status()
            await xo.destroy_actor(actor)
            results["store_unavailable"] = await generate(consumer)
            assert results["store_unavailable"]["text"] == results["cold"]["text"]
            assert results["store_unavailable"]["meta_info"]["cached_tokens"] == 0
        finally:
            for engine in engines:
                await asyncio.to_thread(engine.shutdown)
            destination = Path(
                os.environ.get(
                    "XINFERENCE_SGLANG_XAVIER_RESULT", str(tmp_path / "result.json")
                )
            )
            destination.write_text(json.dumps(results, indent=2))
