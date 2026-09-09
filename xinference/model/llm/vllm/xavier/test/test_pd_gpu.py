# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Two-GPU end-to-end PD regression; opt in with XINFERENCE_TEST_PD_GPU=1."""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("XINFERENCE_TEST_PD_GPU") != "1",
    reason="requires two CUDA GPUs and vLLM 0.21+",
)


@pytest.fixture
def pd_cluster(monkeypatch, tmp_path):
    import json
    import multiprocessing
    from copy import deepcopy

    import xoscar as xo

    from xinference.api.restful_api import run_in_subprocess as start_api
    from xinference.conftest import (
        TEST_FILE_LOGGING_CONF,
        TEST_LOG_FILE_PATH,
        api_health_check,
    )
    from xinference.deploy.local import health_check, run_in_subprocess

    # Real subprocesses are required for per-replica CUDA_VISIBLE_DEVICES.
    # The normal test:// actor pool runs models in the same process.
    multiprocessing.set_start_method("spawn", force=True)
    monkeypatch.setenv("XINFERENCE_AUTH_ADVANCED", "false")
    # vLLM spawns its own engine process, outside xoscar's logging setup.
    # Capture connector logs there as well, including the GPU cache write.
    logging_config = deepcopy(TEST_FILE_LOGGING_CONF)
    logging_config["loggers"]["vllm"] = {
        "handlers": ["stream_handler"],
        "level": "INFO",
        "propagate": False,
    }
    config_path = tmp_path / "vllm-logging.json"
    config_path.write_text(json.dumps(logging_config))
    monkeypatch.setenv("VLLM_LOGGING_CONFIG_PATH", str(config_path))
    address = f"127.0.0.1:{xo.utils.get_next_port()}"
    cluster = run_in_subprocess(address, None, None, deepcopy(TEST_FILE_LOGGING_CONF))
    api = None
    try:
        assert health_check(address=address, max_attempts=20, sleep_interval=1)
        port = xo.utils.get_next_port()
        endpoint = f"http://127.0.0.1:{port}"
        api = start_api(
            address,
            host="127.0.0.1",
            port=port,
            logging_conf=deepcopy(TEST_FILE_LOGGING_CONF),
        )
        assert api_health_check(endpoint, max_attempts=20, sleep_interval=1)
        yield endpoint, TEST_LOG_FILE_PATH
    finally:
        if api is not None:
            api.kill()
            api.join(timeout=10)
        cluster.kill()
        cluster.join(timeout=10)


def test_pd_gpu(pd_cluster):
    import re
    from concurrent.futures import ThreadPoolExecutor

    import openai
    import torch

    from xinference.client import Client

    assert torch.cuda.device_count() >= 2, "PD GPU CI needs at least two CUDA GPUs"
    endpoint, log_path = pd_cluster
    client = Client(endpoint)
    worker = client.get_workers_info()[0]["work-ip"]
    uid = "pd-gpu-regression"
    try:
        client.launch_model(
            model_uid=uid,
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
            replica=2,
            enable_virtual_env=False,
            max_model_len=2048,
            gpu_memory_utilization=0.5,
            dtype="float16",
            replica_config=[
                {
                    "role": role,
                    "devices": [{"worker_ip": worker, "n_gpu": 1, "gpu_idx": [i]}],
                }
                for i, role in enumerate(["prefill", "decode"])
            ],
        )
        api = openai.OpenAI(base_url=endpoint + "/v1", api_key="unused", timeout=180)
        prompt = (
            "Explain how rain forms. "
            + "Water evaporates and condenses in clouds. " * 20
        )

        def chat(prefix, stream):
            result = api.chat.completions.create(
                model=uid,
                messages=[{"role": "user", "content": f"{prefix}: {prompt}"}],
                max_tokens=32,
                temperature=0,
                stream=stream,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            if stream:
                chunks = list(result)
                text = "".join(
                    chunk.choices[0].delta.content or ""
                    for chunk in chunks
                    if chunk.choices
                )
                assert len(chunks) > 1
            else:
                text = result.choices[0].message.content
                assert result.usage.completion_tokens > 1
            assert text.strip()
            print(f"PD response prefix={prefix} stream={stream}: {text}", flush=True)
            return text

        def log_since(offset):
            with open(log_path, "rb") as log:
                log.seek(offset)
                return log.read().decode(errors="replace")

        responses = {}
        for stream in (False, True):
            offset = Path(log_path).stat().st_size
            responses[stream] = chat(str(stream), stream)
            evidence = log_since(offset)
            # Require actual KV transfer, not merely a successful recomputation.
            assert "Stage Xavier V1 blocks" in evidence
            assert "Load Xavier V1 blocks" in evidence
        # Repeated prompts may hit decode's local prefix cache, but must still
        # yield the same deterministic answer without stale or corrupted KV.
        for stream in (False, True):
            assert chat(str(stream), stream) == responses[stream]

        offset = Path(log_path).stat().st_size
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(chat, f"Concurrent request {i}", bool(i % 2))
                for i in range(4)
            ]
            for future in futures:
                future.result(timeout=180)
        loaded_requests = set(
            re.findall(r"Load Xavier V1 blocks: request=(\S+)", log_since(offset))
        )
        assert len(loaded_requests) == 4
        print("PD concurrent requests: 4 completed with remote KV loads", flush=True)
        client.terminate_model(uid)
        assert uid not in client.list_models()
    finally:
        if uid in client.list_models():
            client.terminate_model(uid)
