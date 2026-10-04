# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Qwen3.5 text P/D must match a standalone engine, with actual transfers."""
import os
from pathlib import Path

import pytest

from . import test_pd_gpu

pd_cluster = test_pd_gpu.pd_cluster

pytestmark = pytest.mark.skipif(
    os.environ.get("XINFERENCE_TEST_PD_RECURRENT_GPU") != "1",
    reason="requires two CUDA GPUs, vLLM 0.21+ and Qwen3.5-0.8B",
)


@pytest.mark.parametrize("backend", ["xavier", "nixl"])
def test_recurrent_pd_matches_standalone(pd_cluster, backend):
    from concurrent.futures import ThreadPoolExecutor

    import openai

    from xinference.client import Client

    if backend == "nixl":
        from packaging.version import Version
        from vllm import __version__

        if Version(__version__) < Version("0.22.0"):
            pytest.skip("Native NIXL GDN transfer requires vLLM >= 0.22.0")

    endpoint, log_path = pd_cluster
    client = Client(endpoint)
    worker = client.get_workers_info()[0]["work-ip"]
    uid = "recurrent-pd"
    config = dict(
        model_uid=uid,
        model_name="qwen3.5",
        model_size_in_billions="0_8",
        model_format="pytorch",
        quantization="none",
        model_engine="vLLM",
        model_path=os.environ["XINFERENCE_TEST_PD_RECURRENT_MODEL_PATH"],
        enable_virtual_env=False,
        max_model_len=2048,
        gpu_memory_utilization=0.4,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=False,
        language_model_only=True,
        mamba_cache_mode="none",
        async_scheduling=False,
        disable_hybrid_kv_cache_manager=False,
        envs={"VLLM_SSM_CONV_STATE_LAYOUT": "DS"},
    )
    api = openai.OpenAI(base_url=endpoint + "/v1", api_key="unused", timeout=180)
    prompts = [
        "Write exactly this sentence, with no extra text: The answer is four.",
        "Water evaporates and condenses in clouds. " * 100 + "Explain rain.",
    ]

    def generate(prompt):
        response = api.chat.completions.create(
            model=uid,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32,
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        assert response.usage.completion_tokens > 1
        return response.choices[0].message.content

    try:
        client.launch_model(**config, n_gpu=1, gpu_idx=[0])
        expected = [generate(prompt) for prompt in prompts]
        client.terminate_model(uid)
        client.launch_model(
            **config,
            replica=2,
            vllm_transfer_backend_type=backend,
            replica_config=[
                {
                    "role": role,
                    "devices": [{"worker_ip": worker, "n_gpu": 1, "gpu_idx": [i]}],
                }
                for i, role in enumerate(["prefill", "decode"])
            ],
        )
        offset = Path(log_path).stat().st_size
        for _ in range(2):
            assert [generate(prompt) for prompt in prompts] == expected
        with ThreadPoolExecutor(max_workers=2) as executor:
            assert list(executor.map(generate, prompts)) == expected
        with open(log_path, "rb") as log:
            log.seek(offset)
            evidence = log.read().decode(errors="replace")
        if backend == "xavier":
            assert "Register Xavier direct handoff" in evidence
            assert "Finished Xavier async KV load" in evidence
            assert "Restored Xavier history" not in evidence
        else:
            assert "calling _read_blocks" in evidence
        print(f"{backend}: six recurrent PD outputs match standalone", flush=True)
    finally:
        if uid in client.list_models():
            client.terminate_model(uid)
