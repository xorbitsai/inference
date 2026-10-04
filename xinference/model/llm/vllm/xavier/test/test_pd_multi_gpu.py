# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Exercise multiple P/D engines sharing two physical GPUs."""
import asyncio
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from .test_pd_gpu import pd_cluster  # noqa: F401

pytestmark = pytest.mark.skipif(
    os.environ.get("XINFERENCE_TEST_PD_MULTI_GPU") != "1",
    reason="requires two CUDA GPUs with room for two small engines each",
)


@pytest.mark.parametrize("backend", ["xavier", "nixl"])
@pytest.mark.parametrize("num_p,num_d", [(2, 1), (1, 2), (2, 2)])
def test_multi_pd_gpu(pd_cluster, backend, num_p, num_d):  # noqa: F811
    import openai
    import torch
    import xoscar as xo

    from xinference.client import Client
    from xinference.core.pd_model import PDModelActor

    assert torch.cuda.device_count() >= 2
    endpoint, log_path = pd_cluster
    client = Client(endpoint)
    worker = client.get_workers_info()[0]["work-ip"]
    uid = f"pd-multi-{backend}-{num_p}p{num_d}d"
    loop = asyncio.new_event_loop()
    api = openai.OpenAI(base_url=endpoint + "/v1", api_key="unused", timeout=180)

    def logs(offset):
        with open(log_path, "rb") as log:
            log.seek(offset)
            return log.read().decode(errors="replace")

    def chat(index, stream=False):
        # Different answers detect cross-request cache contamination. Long shared
        # prefixes exercise history independently of the engine's prefix cache.
        word = ["orange", "purple", "silver", "yellow"][index % 4]
        prompt = (
            "Read the following background, then answer the final question. "
            + "Water evaporates and condenses in clouds. " * 40
            + f"\nThe secret word is {word}. What is the secret word? "
            "Reply with only the secret word."
        )
        result = api.chat.completions.create(
            model=uid,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            temperature=0,
            stream=stream,
        )
        if stream:
            chunks = list(result)
            text = "".join(
                chunk.choices[0].delta.content or ""
                for chunk in chunks
                if chunk.choices
            )
        else:
            text = result.choices[0].message.content
        assert word in text.lower(), (word, text)
        assert not any(
            other in text.lower()
            for other in ("orange", "purple", "silver", "yellow")
            if other != word
        ), text
        return text

    def check_transfers(evidence, count):
        if backend == "xavier":
            producers = set(
                re.findall(r"Register Xavier direct handoff: request=(\S+)", evidence)
            )
            loaded = set(
                re.findall(r"Finished Xavier async KV load: request=(\S+)", evidence)
            )
            assert len(producers) == count, len(producers)
            assert len(loaded - producers) == count, len(loaded - producers)
        else:
            loaded = set(
                re.findall(r"with remote block size \d+ for req (\S+)", evidence)
            )
            assert len(loaded) == count, len(loaded)
            assert re.search(r"and [1-9]\d* requests done recving", evidence)

    try:
        client.launch_model(
            model_uid=uid,
            vllm_transfer_backend_type=backend,
            model_name=os.environ.get(
                "XINFERENCE_TEST_PD_MODEL_NAME", "qwen2.5-instruct"
            ),
            model_engine="vLLM",
            model_size_in_billions=os.environ.get(
                "XINFERENCE_TEST_PD_MODEL_SIZE", "0_5"
            ),
            model_path=os.environ.get("XINFERENCE_TEST_PD_MODEL_PATH"),
            model_format="pytorch",
            quantization="none",
            replica=num_p + num_d,
            enable_virtual_env=False,
            max_model_len=2048,
            max_num_seqs=16,
            gpu_memory_utilization=0.35,
            enforce_eager=True,
            dtype="float16",
            enable_prefix_caching=False,
            replica_config=[
                {
                    "role": role,
                    "devices": [{"worker_ip": worker, "n_gpu": 1, "gpu_idx": [gpu]}],
                }
                for role, gpu in [("prefill", 0)] * num_p + [("decode", 1)] * num_d
            ],
        )
        pd = loop.run_until_complete(
            xo.actor_ref(
                address=client._get_supervisor_internal_address(),
                uid=f"{uid}-{PDModelActor.default_uid()}",
            )
        )
        info = loop.run_until_complete(pd.get_pd_info())
        assert (info["prefill_count"], info["decode_count"]) == (num_p, num_d)
        start = Path(log_path).stat().st_size
        # Both producers see the same prefix; repeated requests must restore it.
        for i in range(4):
            chat(i, bool(i % 2))
        offset = Path(log_path).stat().st_size
        for i in range(4):
            chat(i, bool(i % 2))
        warm_evidence = logs(offset)
        check_transfers(warm_evidence, 4)
        if backend == "xavier":
            assert re.search(r"Restored Xavier history: blocks=[1-9]\d*", warm_evidence)

        def concurrent():
            offset = Path(log_path).stat().st_size
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(chat, i, bool(i % 2)) for i in range(8)]
                for future in futures:
                    future.result(timeout=180)
            check_transfers(logs(offset), 8)

        concurrent()
        requests = 16
        if num_p == num_d == 2:
            # Equal-length round robins only visit two diagonal pairs. Removing
            # and re-registering D0 rotates D, exercising the other two pairs.
            decoder = info["decode_replica_uids"][0]
            ref = loop.run_until_complete(pd.get_decode_actor(decoder))
            loop.run_until_complete(pd.remove_decode_actor(decoder))
            loop.run_until_complete(pd.add_decode_actor(decoder, ref))
            concurrent()
            requests += 8
        evidence = logs(start)
        routes = re.findall(
            r"PD route: request=(\S+) prefill=(\S+) decode=(\S+) backend=(\S+)",
            evidence,
        )
        assert len(routes) == requests, routes
        pairs = {(p, d) for _, p, d, _ in routes}
        assert len(pairs) == num_p * num_d, pairs
        assert {b for _, _, _, b in routes} == {backend}
        summary = {
            "backend": backend,
            "topology": f"{num_p}P{num_d}D",
            "requests": requests,
            "pairs": sorted(pairs),
            "history_restores": len(re.findall(r"Restored Xavier history:", evidence)),
        }
        print("MULTI_PD_RESULT " + json.dumps(summary), flush=True)
        client.terminate_model(uid)
        assert uid not in client.list_models()
    finally:
        api.close()
        try:
            if uid in client.list_models():
                client.terminate_model(uid)
        finally:
            loop.close()
