# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Opt-in LAN reuse between fresh CUDA engines and native Metal generation."""

import asyncio
import json
import os
import shlex
import sys

import pytest
import xoscar as xo

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("XINFERENCE_TEST_PREFIX_MLX_GPU") != "1",
    reason="Requires a reserved Mac/CUDA LAN deployment with identical checkpoints",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("mlx_producer", [False, True])
async def test_independent_metal_and_cuda_requests_share_prefix(
    tmp_path, engine, mlx_producer
):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    from ...xavier.backends.bytes.storage import XavierBytesCacheActor
    from ...xavier.tests.test_prefix_live import SCRIPT
    from ..xavier import configure_xavier

    path = os.environ["XINFERENCE_TEST_PREFIX_MLX_MODEL"]
    model, tokenizer = load(path)
    tokens = tokenizer.encode(
        "The capital of France is Paris. " * 96, add_special_tokens=False
    )
    pool = await xo.create_actor_pool(
        os.environ["XINFERENCE_TEST_PREFIX_HOST"], n_process=0
    )
    async with pool:
        actor = await xo.create_actor(
            XavierBytesCacheActor,
            512 * 1024**2,
            address=pool.external_address,
            uid="canonical-prefix",
        )
        client = configure_xavier(
            model,
            path,
            {"context_length": 2048},
            dict(
                address=actor.address,
                uid="canonical-prefix",
                shared_prefix=True,
                logical_dtype="float16",
            ),
            1,
        )
        await client.initialize()

        async def metal(salt):
            cache, count = await client.fetch(tokens, cache_salt=salt)
            cache = cache if cache is not None else make_prompt_cache(model)
            outputs = [
                int(token)
                for token, _ in generate_step(
                    mx.array(tokens[count:]), model, max_tokens=16, prompt_cache=cache
                )
            ]
            write = client.publish(cache, tokens[:-1], count, salt)
            await client.flush([write])
            return dict(tokens=outputs, text=tokenizer.decode(outputs), cached=count)

        async def cuda():
            command = shlex.join(
                [
                    "env",
                    "PYTHONPATH=" + os.environ["XINFERENCE_TEST_PREFIX_GPU_SOURCE"],
                    "PATH="
                    + os.path.dirname(os.environ["XINFERENCE_TEST_VLLM_PYTHON"])
                    + ":/usr/local/cuda/bin:/usr/bin:/bin",
                    "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
                    "CUDA_VISIBLE_DEVICES=0",
                    "HF_HUB_OFFLINE=1",
                    "OMP_NUM_THREADS=2",
                    "SGLANG_PLUGINS=__xinference_test__",
                    os.environ[f"XINFERENCE_TEST_{engine.upper()}_PYTHON"],
                    "-c",
                    SCRIPT,
                    engine,
                    os.environ["XINFERENCE_TEST_PREFIX_GPU_MODEL"],
                    actor.address,
                    "float16",
                    "salted",
                ]
            )
            process = await asyncio.create_subprocess_exec(
                "ssh",
                os.environ.get("XINFERENCE_TEST_PREFIX_GPU_SSH", "gpu"),
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            output, _ = await asyncio.wait_for(process.communicate(), 600)
            log = output.decode(errors="replace")
            (tmp_path / "cuda.log").write_text(log)
            assert process.returncode == 0, log[-12000:]
            return json.loads(
                next(
                    line.split("=", 1)[1]
                    for line in log.splitlines()
                    if line.startswith("SHARED_PREFIX_RESULT=")
                )
            )

        baseline = await metal("metal-baseline")
        if mlx_producer:
            cold = await metal("customer-a")
            remote = await cuda()
            assert (
                remote["after_target"]["read_pages"]
                > remote["before_target"]["read_pages"]
            )
            assert remote["baseline"]["tokens"] == remote["target"]["tokens"]
            assert cold["tokens"] == baseline["tokens"]
        else:
            remote = await cuda()
            assert await client.fetch(tokens, cache_salt="absent-customer") == (None, 0)
            warm = await metal("customer-a")
            assert warm["cached"] > 0
            assert warm["tokens"] == baseline["tokens"]
        (tmp_path / "result.json").write_text(
            json.dumps(
                dict(
                    mlx_producer=mlx_producer,
                    engine=engine,
                    remote=remote,
                    baseline=baseline,
                    metal=cold if mlx_producer else warm,
                ),
                indent=2,
            )
        )
        await xo.destroy_actor(actor)
