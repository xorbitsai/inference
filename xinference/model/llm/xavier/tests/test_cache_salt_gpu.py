# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
"""Opt-in serving regression: XINFERENCE_TEST_CACHE_SALT_GPU=1."""
import os

import pytest

from ...vllm.xavier.test.test_pd_gpu import pd_cluster  # noqa: F401

pytestmark = pytest.mark.skipif(
    os.environ.get("XINFERENCE_TEST_CACHE_SALT_GPU") != "1",
    reason="Requires two CUDA GPUs, vLLM/SGLang and local FP16 weights",
)


@pytest.mark.parametrize("backend", ["xavier"])
@pytest.mark.parametrize(
    "engines",
    [("vLLM", "vLLM"), ("SGLang", "SGLang"), ("vLLM", "SGLang"), ("SGLang", "vLLM")],
)
def test_salted_pd_engine_matrix(pd_cluster, backend, engines):  # noqa: F811
    import asyncio

    import openai
    import xoscar as xo

    from xinference.client import Client

    endpoint, _ = pd_cluster
    client = Client(endpoint)
    worker = client.get_workers_info()[0]["work-ip"]
    uid = "salted-pd"
    options = {
        "vLLM": dict(
            max_model_len=2048,
            gpu_memory_utilization=0.25,
            enforce_eager=True,
            dtype="float16",
        ),
        "SGLang": dict(
            context_length=2048,
            mem_fraction_static=0.25,
            max_total_tokens=4096,
            disable_cuda_graph=True,
            dtype="float16",
        ),
    }
    loop = asyncio.new_event_loop()
    try:
        client.launch_model(
            model_uid=uid,
            model_name="qwen2.5-instruct",
            model_engine=engines[0],
            model_size_in_billions="0_5",
            model_format="pytorch",
            quantization="none",
            model_path=os.environ["XINFERENCE_TEST_PD_MODEL_PATH"],
            replica=2,
            enable_virtual_env=False,
            vllm_transfer_backend_type=backend,
            replica_config=[
                dict(
                    role=role,
                    model_engine=engine,
                    engine_config=options[engine],
                    devices=[dict(worker_ip=worker, n_gpu=1, gpu_idx=[gpu])],
                )
                for gpu, (role, engine) in enumerate(
                    zip(("prefill", "decode"), engines)
                )
            ],
        )
        api = openai.OpenAI(base_url=endpoint + "/v1", api_key="unused", timeout=180)
        prompt = (
            "Water evaporates and condenses in clouds. " * 40
            + "\nExplain how rain forms:"
        )
        for salt, stream in [
            (None, False),
            ("scope-a", False),
            ("scope-a", True),
            ("scope-b", False),
            ("scope-b", True),
        ]:
            result = api.completions.create(
                model=uid,
                prompt=prompt,
                max_tokens=8,
                temperature=0,
                stream=stream,
                stream_options={"include_usage": True} if stream else None,
                extra_body={"cache_salt": salt} if salt else {},
            )
            if stream:
                chunks = list(result)
                text = "".join(
                    choice.text for chunk in chunks for choice in chunk.choices
                )
                usage = next(chunk.usage for chunk in reversed(chunks) if chunk.usage)
            else:
                text, usage = result.choices[0].text, result.usage
            assert text.strip(), (engines, stream)
            if engines == ("vLLM", "vLLM"):
                assert usage.prompt_tokens_details.cached_tokens > 0
            else:

                async def stats():
                    ref = await xo.actor_ref(
                        address=client._get_supervisor_internal_address(),
                        uid=f"xavier-cache-{uid}",
                    )
                    return await ref.get_stats()

                metrics = loop.run_until_complete(stats())
                assert metrics["gpu_bytes"] > 0 and metrics["active_handoffs"] == 0

            print(
                dict(
                    engines=engines,
                    salted=salt is not None,
                    stream=stream,
                    cached=usage.prompt_tokens_details.cached_tokens,
                ),
                flush=True,
            )
    finally:
        if uid in client.list_models():
            client.terminate_model(uid)
        loop.close()


@pytest.mark.parametrize("backend", ["xavier"])
@pytest.mark.parametrize("engine", ["vLLM", "SGLang"])
def test_salted_prefix_reuse_across_real_replicas(
    pd_cluster,  # noqa: F811
    backend,
    engine,
):
    import asyncio
    import json

    import xoscar as xo

    from xinference.client import Client
    from xinference.core.worker import WorkerActor

    endpoint, _ = pd_cluster
    client = Client(endpoint)
    worker = client.get_workers_info()[0]["work-ip"]
    uid = "salted-prefix"
    options = (
        dict(
            max_model_len=2048,
            gpu_memory_utilization=0.25,
            enforce_eager=True,
            enable_prefix_caching=True,
            kv_cache_memory_bytes=64 * 1024 * 1024,
        )
        if engine == "vLLM"
        else dict(
            context_length=2048,
            mem_fraction_static=0.25,
            max_total_tokens=4096,
            disable_cuda_graph=True,
            enable_cache_report=True,
        )
    )
    if engine == "SGLang":
        options["xavier_cache_bytes"] = 64 * 1024 * 1024
    loop = asyncio.new_event_loop()

    async def exercise():
        worker_ref = await xo.actor_ref(address=worker, uid=WorkerActor.default_uid())
        replicas = [await worker_ref.get_model(f"{uid}-rep{i}") for i in range(2)]
        prompt = "The quick brown fox jumps over the lazy dog. " * 100
        store = None
        if engine == "SGLang":
            store = await xo.actor_ref(
                address=client._get_supervisor_internal_address(),
                uid=f"xavier-cache-{uid}",
            )

        async def generate(replica, salt):
            result = await replicas[replica].generate(
                prompt,
                {
                    "max_tokens": 8,
                    "temperature": 0,
                    **({"cache_salt": salt} if salt else {}),
                },
            )
            result = json.loads(result) if isinstance(result, (bytes, str)) else result
            cached = result["usage"]["prompt_tokens_details"]["cached_tokens"]
            assert result["choices"][0]["text"].strip()
            return cached

        async def wait_pages(minimum):
            if store is None:
                return 0
            for _ in range(100):
                stats = await store.get_stats()
                if stats["stored_pages"] > minimum:
                    return stats["stored_pages"]
                await asyncio.sleep(0.1)
            pytest.fail("SGLang did not publish shared prefix pages")

        assert await generate(0, "scope-a") == 0
        pages = await wait_pages(0)
        assert await generate(1, "scope-b") == 0
        await wait_pages(pages)
        assert await generate(1, "scope-a") > 0
        assert await generate(0, "scope-b") > 0
        assert await generate(0, "scope-a") > 0
        assert await generate(1, None) == 0
        print(
            dict(
                engine=engine, salt_isolation=True, remote_reuse=True, local_reuse=True
            ),
            flush=True,
        )

    try:
        client.launch_model(
            model_uid=uid,
            model_name="qwen2.5-instruct",
            model_engine=engine,
            model_size_in_billions="0_5",
            model_format="pytorch",
            quantization="none",
            model_path=os.environ["XINFERENCE_TEST_PD_MODEL_PATH"],
            replica=2,
            enable_virtual_env=False,
            enable_xavier=True,
            dtype="float16",
            replica_config=[
                dict(devices=[dict(worker_ip=worker, n_gpu=1, gpu_idx=[gpu])])
                for gpu in (0, 1)
            ],
            **options,
        )
        loop.run_until_complete(exercise())
    finally:
        if uid in client.list_models():
            client.terminate_model(uid)
        loop.close()
