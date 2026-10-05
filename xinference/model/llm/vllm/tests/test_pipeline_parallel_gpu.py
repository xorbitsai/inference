# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Real V1 PP serving; opt in with XINFERENCE_TEST_VLLM_PP_GPU=1."""

import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or os.environ.get("XINFERENCE_TEST_VLLM_PP_GPU") != "1",
    reason="requires two CUDA GPUs and vLLM V1",
)


@pytest.fixture(
    params=[(1, 2, 1, False), (1, 1, 2, False), (2, 1, 2, False), (2, 1, 2, True)],
    ids=["tp-regression", "single-worker-pp", "two-workers-pp", "two-workers-prefix"],
)
def pp_cluster(request, tmp_path):
    import requests
    import torch
    import xoscar as xo

    from xinference.client import Client

    assert torch.cuda.device_count() >= 2
    n_worker, tp, pp, prefix_caching = request.param
    port = xo.utils.get_next_port()
    endpoint = f"http://127.0.0.1:{port}"
    processes = []
    logs = []
    env = dict(
        os.environ,
        XINFERENCE_AUTH_ADVANCED="false",
        XINFERENCE_ENABLE_VIRTUAL_ENV="0",
        XINFERENCE_MODEL_ACTOR_AUTO_RECOVER_LIMIT="0",
        XINFERENCE_LOG_MAX_BYTES="0",
        VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="15",
    )

    def start(name, command, devices):
        log = (tmp_path / f"{name}.log").open("w")
        logs.append(log)
        child_env = dict(
            env,
            CUDA_VISIBLE_DEVICES=devices,
            XINFERENCE_HOME=str(tmp_path / name),
        )
        process = subprocess.Popen(
            [sys.executable, "-c", command[0], *command[1:]],
            env=child_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(process)

    def wait_for(predicate):
        for _ in range(120):
            assert all(p.poll() is None for p in processes), f"See logs in {tmp_path}"
            try:
                if predicate():
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        pytest.fail(f"Cluster startup timed out; see logs in {tmp_path}")

    try:
        start(
            "supervisor",
            [
                "from xinference.deploy.cmdline import supervisor; supervisor()",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            "-1",
        )
        wait_for(
            lambda: requests.get(endpoint + "/status", timeout=2).status_code == 200
        )
        for rank in range(n_worker):
            start(
                f"worker-{rank}",
                [
                    "from xinference.deploy.cmdline import worker; worker()",
                    "--host",
                    "127.0.0.1",
                    "--endpoint",
                    endpoint,
                    "--metrics-exporter-port",
                    str(xo.utils.get_next_port()),
                ],
                "0,1" if n_worker == 1 else str(rank),
            )
        client = Client(endpoint)
        wait_for(lambda: len(client.get_workers_info()) == n_worker)
        yield client, n_worker, tp, pp, prefix_caching, tmp_path
    finally:
        for process in reversed(processes):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        for log in logs:
            log.close()


def test_pipeline_parallel_serving_and_rank_failure(pp_cluster):
    import asyncio

    import openai
    import xoscar as xo

    from xinference.core.supervisor import SupervisorActor

    client, n_worker, tp, pp, prefix_caching, log_dir = pp_cluster
    uid = "vllm-pp-regression"
    client.launch_model(
        model_uid=uid,
        model_name=os.environ.get("XINFERENCE_TEST_PP_MODEL_NAME", "qwen2.5-instruct"),
        model_size_in_billions=os.environ.get("XINFERENCE_TEST_PP_MODEL_SIZE", "0_5"),
        model_path=os.environ.get("XINFERENCE_TEST_PP_MODEL_PATH"),
        model_format="pytorch",
        quantization="none",
        model_engine="vLLM",
        xinference_vllm_executor_backend="xoscar",
        n_gpu=2 // n_worker,
        n_worker=n_worker,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        enable_virtual_env=False,
        max_model_len=1024,
        max_num_batched_tokens=128,
        max_num_seqs=2,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_prefix_caching=prefix_caching,
    )
    api = openai.OpenAI(
        base_url=client.base_url + "/v1", api_key="unused", timeout=60, max_retries=0
    )

    def chat(index=0, stream=False):
        response = api.chat.completions.create(
            model=uid,
            messages=[{"role": "user", "content": f"Question {index}: What is rain?"}],
            temperature=0,
            max_tokens=24,
            stream=stream,
            seed=0,
        )
        if stream:
            with response:
                chunks = list(response)
            text = "".join(
                chunk.choices[0].delta.content or ""
                for chunk in chunks
                if chunk.choices
            )
            assert len(chunks) > 1
        else:
            text = response.choices[0].message.content
            assert response.usage.completion_tokens > 1
        assert text.strip()
        return text

    try:
        reference = chat()
        repeated = chat()
        streamed = chat(stream=True)
        # Cached and uncached prefill shapes can change floating-point
        # rounding and greedy output. Check equality with fixed computation.
        if not prefix_caching:
            assert repeated == reference
        assert streamed == repeated
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda i: chat(i, bool(i % 2)), range(6)))
        assert len(results) == 6

        # Closing an active stream must release its request and leave PP usable.
        response = api.chat.completions.create(
            model=uid,
            messages=[
                {"role": "user", "content": "Explain the water cycle in detail."}
            ],
            max_tokens=256,
            stream=True,
        )
        with response:
            next(response)
        after_cancel = chat()
        if not prefix_caching:
            assert after_cancel == reference

        async def first_rank_pid():
            supervisor = await xo.actor_ref(
                client._get_supervisor_internal_address(), SupervisorActor.default_uid()
            )
            model = await supervisor.get_model(uid)
            addresses = await model.get_pool_addresses()
            rank = await xo.actor_ref(addresses[0], "VllmWorker_0")
            return await rank.execute_method(lambda _: os.getpid())

        # Kill only this test's first GPU rank. The final stage must not leave
        # the API hanging while waiting for tensors from the failed rank.
        pid = asyncio.run(first_rank_pid())
        os.kill(pid, signal.SIGKILL)
        started = time.monotonic()
        with pytest.raises(openai.APIError):
            chat()
        assert time.monotonic() - started < 45
        print(
            f"TP={tp}, PP={pp}, workers={n_worker}: serving and rank failure passed; {log_dir}"
        )
    finally:
        api.close()
        if uid in client.list_models():
            client.terminate_model(uid)
