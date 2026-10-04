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


@pytest.mark.asyncio
async def test_real_sglang_remote_prefix_restore(tmp_path):
    import requests
    import sglang as sgl
    import torch
    import xoscar as xo
    from transformers import AutoTokenizer

    from ...xavier.backends.torch.storage import XavierCacheActor
    from ..core import SGLANGModel
    from ..xavier.config import configure_xavier
    from ..xavier.pd import SGLangXavierHandoff

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
        cache = {"address": actor.address, "uid": "xavier-test", "role": "prefill"}
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
            # Exercise the actual Xinference generation wrappers and handoff,
            # with independent P/D engines and an empty decode-local cache.
            prompt = (
                "The Xavier prefill replica computes this prompt before decode. " * 70
            )
            models = []
            for role, engine in (("prefill", producer), ("decode", consumer)):
                model = object.__new__(SGLANGModel)
                model.model_uid = "sglang-pd-" + role
                model._active_request_ids = set()
                model._engine = engine
                model._xavier_handoff = SGLangXavierHandoff(
                    {**cache, "role": role}, config["page_size"], tokenizer
                )
                models.append(model)
            sampling = {"temperature": 0, "max_tokens": 16, "ignore_eos": True}
            prefill = await models[0].async_generate(
                prompt,
                generate_config={
                    **sampling,
                    "_pd_kv_transfer_params": {"do_remote_decode": True},
                },
            )
            transfer = prefill["_pd_kv_transfer_params"]
            assert transfer["sglang_xavier"]["cached_tokens"] > 0
            results["pd_handoff"] = transfer
            before_decode = await actor.get_stats()
            assert before_decode["active_handoffs"] == 1
            results["pd_decode"] = await models[1].async_generate(
                prompt,
                generate_config={
                    **sampling,
                    "_pd_kv_transfer_params": transfer,
                },
            )
            results["pd_after_decode"] = await actor.get_stats()
            assert (
                results["pd_after_decode"]["read_pages"] > before_decode["read_pages"]
            )
            assert results["pd_after_decode"]["active_handoffs"] == 0
            baseline = await models[0].async_generate(prompt, generate_config=sampling)
            assert (
                results["pd_decode"]["choices"][0]["text"]
                == baseline["choices"][0]["text"]
            )

            # Independent salted PD requests must all restore their complete
            # prefixes under concurrent admission and a bounded host pool.
            async def pd_request(index):
                text = f"Request {index}: " + prompt
                p = await models[0].async_generate(
                    text,
                    generate_config={
                        **sampling,
                        "_pd_kv_transfer_params": {"do_remote_decode": True},
                    },
                )
                d = await models[1].async_generate(
                    text,
                    generate_config={
                        **sampling,
                        "_pd_kv_transfer_params": p["_pd_kv_transfer_params"],
                    },
                )
                return d

            results["pd_concurrent"] = await asyncio.gather(
                *(pd_request(i) for i in range(4))
            )
            assert (await actor.get_stats())["active_handoffs"] == 0
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
