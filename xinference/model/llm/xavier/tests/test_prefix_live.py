# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
"""Opt-in independent-request cross-engine reuse in fresh engine processes.

Set XINFERENCE_TEST_SHARED_PREFIX=1, XINFERENCE_TEST_PD_MODEL_PATH and
XINFERENCE_TEST_{VLLM,SGLANG}_PYTHON on a reserved CUDA host.
"""

import asyncio
import json
import os

import pytest
import xoscar as xo

from ..backends.bytes.storage import XavierBytesCacheActor

pytestmark = pytest.mark.skipif(
    os.environ.get("XINFERENCE_TEST_SHARED_PREFIX") != "1",
    reason="Requires explicitly reserved CUDA resources and local engine environments",
)

SCRIPT = r"""
import asyncio
import json
import sys
import time
from transformers import AutoTokenizer

engine_name, model_path, address, dtype, mode = sys.argv[1:]
cache = {"address": address, "uid": "canonical-prefix", "shared_prefix": True,
         "logical_dtype": dtype, "capacity_bytes": 512 * 1024**2}
tokenizer = AutoTokenizer.from_pretrained(model_path)
prompt = "The capital of France is Paris. " * 96
tokens = tokenizer.encode(prompt, add_special_tokens=False)
if mode != "public":
    tokens = tokens[:1025]
if engine_name == "vllm":
    import os
    # Trusted test-only callbacks instrument packing inside the worker process.
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from xinference.model.llm.vllm.xavier.cross_engine import configure_cross_engine
    config = {"max_model_len": 2048, "gpu_memory_utilization": 0.25,
              "enforce_eager": True, "max_num_seqs": 8, "max_num_batched_tokens": 256}
    configure_cross_engine(model_path, config, cache)
    engine = LLM(model=model_path, **config, kv_transfer_config=KVTransferConfig(
        kv_connector="PrefixConnector", kv_connector_module_path="xinference.model.llm.vllm.xavier.prefix",
        kv_role="kv_both", kv_connector_extra_config={"xavier_config": cache}))
    def count_packing(worker):
        from xinference.model.llm.vllm.xavier import prefix
        pack = prefix.pack_pages
        prefix._test_packed_pages = 0
        def counted(views, slots, contract):
            prefix._test_packed_pages += len(slots)
            return pack(views, slots, contract)
        prefix.pack_pages = counted
    def packed_pages(worker):
        from xinference.model.llm.vllm.xavier import prefix
        return prefix._test_packed_pages
    def generate(salt):
        output = engine.generate({"prompt_token_ids": tokens, "cache_salt": salt},
                                 SamplingParams(temperature=0, max_tokens=16, ignore_eos=True), use_tqdm=False)[0]
        return {"text": output.outputs[0].text, "tokens": list(output.outputs[0].token_ids), "cached": getattr(output, "num_cached_tokens", None)}
elif engine_name == "sglang":
    import requests
    import sglang as sgl
    from xinference.model.llm.sglang.xavier.config import configure_xavier
    config = {"context_length": 2048, "mem_fraction_static": 0.25, "max_total_tokens": 4096,
              "disable_cuda_graph": True, "enable_cache_report": True,
              "hicache_ratio": 1.1, "log_level": "warning"}
    configure_xavier(model_path, config, cache)
    engine = sgl.Runtime(model_path=model_path, **config)
    def generate(salt):
        response = requests.post(engine.url + "/generate", json={"input_ids": tokens,
            "cache_salt": salt, "sampling_params": {"temperature": 0, "max_new_tokens": 16, "ignore_eos": True},
            "return_logprob": True}, timeout=120)
        response.raise_for_status()
        result = response.json()
        return {"text": result["text"], "tokens": [x[1] for x in result["meta_info"]["output_token_logprobs"]],
                "cached": result["meta_info"].get("cached_tokens_details")}
elif engine_name == "trt-llm":
    from xinference.model.llm.llm_family import match_llm
    from xinference.model.llm.trt_llm.core import TRTLLMModel
    engine = TRTLLMModel("prefix-live", match_llm("qwen3", "pytorch", "0_6", "none"), model_path, {
        "max_seq_len": 2048, "dtype": "bfloat16", "max_batch_size": 4, "max_num_tokens": 2048,
        "cuda_graph_config": None, "kv_cache_config": {"max_tokens": 4096}, "_xavier_cache_config": cache})
    engine.load()
    def generate(salt):
        # Public TRT salt remains unsupported. Each TRT run is a fresh process.
        result = asyncio.run(engine.async_generate(prompt,
            {"max_tokens": 16, "temperature": 0, "ignore_eos": True, "stop": [], "stop_token_ids": []}))
        return {"text": result["choices"][0]["text"]}

async def stats():
    import xoscar as xo
    return await (await xo.actor_ref(address=address, uid="canonical-prefix")).get_stats()
try:
    if engine_name == "vllm":
        engine.collective_rpc(count_packing, timeout=10)
    before = asyncio.run(stats())
    if mode == "public":
        before_target = before
        target = generate(None)
        after_target = asyncio.run(stats())
        baseline = other = target
        after_other = after_target
    else:
        baseline = generate("baseline-" + engine_name)
        before_target = asyncio.run(stats())
        target = generate("customer-a")
        after_target = asyncio.run(stats())
        other = generate("customer-b-" + engine_name)
        after_other = asyncio.run(stats())
    # Allow SGLang's background publication to acknowledge all complete pages.
    time.sleep(1)
    after = asyncio.run(stats())
    if engine_name == "vllm":
        packed_before = engine.collective_rpc(packed_pages, timeout=10)
        repeated = generate(None if mode == "public" else "customer-a")
        assert repeated["text"] == target["text"]
        assert engine.collective_rpc(packed_pages, timeout=10) == packed_before, "full hit re-copied GPU pages"
        expected = (3 if mode != "public" else 1) * ((len(tokens) - 1) // 64)
        assert sum(packed_before) <= expected, "chunked prefill re-copied earlier pages"
        print("SHARED_PREFIX_PACKED_PAGES=" + json.dumps(packed_before))
    print("SHARED_PREFIX_RESULT=" + json.dumps(dict(engine=engine_name, baseline=baseline, target=target,
          other=other, before=before, before_target=before_target, after_target=after_target,
          after_other=after_other, after=after)))
finally:
    if engine_name == "vllm":
        engine.llm_engine.engine_core.shutdown()
    elif engine_name == "sglang":
        engine.shutdown()
    elif engine_name == "trt-llm":
        engine.stop()
"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "producer,consumer",
    [("vllm", "sglang"), ("sglang", "vllm")]
    + [
        pytest.param(
            a,
            b,
            marks=pytest.mark.skipif(
                not os.environ.get("XINFERENCE_TEST_TRT_LLM_PYTHON"),
                reason="Requires TRT-LLM environment",
            ),
        )
        for a, b in [
            ("vllm", "trt-llm"),
            ("trt-llm", "vllm"),
            ("sglang", "trt-llm"),
            ("trt-llm", "sglang"),
        ]
    ],
)
async def test_independent_requests_share_prefix_between_engines(
    tmp_path, producer, consumer
):
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        actor = await xo.create_actor(
            XavierBytesCacheActor,
            512 * 1024**2,
            address=pool.external_address,
            uid="canonical-prefix",
        )
        results = []
        try:
            for name in (producer, consumer):
                log_path = tmp_path / f"{name}.log"
                with log_path.open("wb") as output:
                    process = await asyncio.create_subprocess_exec(
                        os.environ[
                            f"XINFERENCE_TEST_{name.upper().replace('-', '_')}_PYTHON"
                        ],
                        "-c",
                        SCRIPT,
                        name,
                        os.environ["XINFERENCE_TEST_PD_MODEL_PATH"],
                        actor.address,
                        "bfloat16" if "trt-llm" in (producer, consumer) else "float16",
                        "public" if "trt-llm" in (producer, consumer) else "salted",
                        stdout=output,
                        stderr=asyncio.subprocess.STDOUT,
                    )
                    try:
                        await asyncio.wait_for(process.wait(), timeout=600)
                    finally:
                        if process.returncode is None:
                            process.kill()
                            await process.wait()
                log = log_path.read_text(errors="replace")
                assert process.returncode == 0, log[-12000:]
                result = json.loads(
                    next(
                        line.split("=", 1)[1]
                        for line in log.splitlines()
                        if line.startswith("SHARED_PREFIX_RESULT=")
                    )
                )
                assert (
                    result["baseline"]["text"]
                    == result["target"]["text"]
                    == result["other"]["text"]
                )
                results.append(result)
            warm = results[1]
            assert (
                warm["after_target"]["read_pages"] > warm["before_target"]["read_pages"]
            ), warm
            assert (
                warm["after_other"]["read_pages"] == warm["after_target"]["read_pages"]
            ), warm
            if "trt-llm" in (producer, consumer):
                assert results[0]["target"]["text"] == warm["target"]["text"]
            if consumer in ("vllm", "trt-llm"):
                assert (
                    warm["after_target"]["imported_pages"]
                    > warm["before_target"]["imported_pages"]
                ), warm
            else:
                assert warm["target"]["cached"]["storage"] > 0, warm
            (tmp_path / "result.json").write_text(json.dumps(results, indent=2))
        finally:
            await xo.destroy_actor(actor)
