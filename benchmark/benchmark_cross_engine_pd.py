# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Controlled monolithic/native/heterogeneous PD comparisons on two CUDA GPUs."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from importlib.metadata import version
from pathlib import Path

import requests
import xoscar as xo
from benchmark_pd import measure, summarize
from benchmark_sglang_pd import GPUIsolation, stop_server
from openai import AsyncOpenAI
from xoscar.utils import get_next_port

from xinference.client import Client


def engine_options(engine, role, args):
    if engine == "vllm":
        return dict(
            dtype="float16",
            max_model_len=8192,
            block_size=64,
            gpu_memory_utilization=0.3,
            num_gpu_blocks_override=args.kv_tokens // 64,
            max_num_seqs=32,
            enforce_eager=True,
            enable_prefix_caching=role != "decode",
        )
    return dict(
        dtype="float16",
        context_length=8192,
        page_size=64,
        mem_fraction_static=0.3,
        max_total_tokens=args.kv_tokens,
        disable_cuda_graph=True,
        stream_interval=1,
        triton_attention_reduce_in_fp32=False,
    )


async def snapshot(worker, uid, heterogeneous):
    ref = await xo.actor_ref(address=worker, uid="xavier-cache-" + uid)
    directory = await ref.get_stats()
    peers = {}
    actor_uid = (
        "xavier-cross-engine-transfer" if heterogeneous else "sglang-xavier-transfer"
    )
    for rank, address in directory["peers"].items():
        actor = await xo.actor_ref(address=address, uid=f"{actor_uid}-{rank}")
        peers[rank] = await actor.get_stats()
    return dict(directory=directory, peers=peers)


def bodies():
    return [
        dict(
            messages=[
                dict(
                    role="user",
                    content="Water evaporates and condenses. " * repeats
                    + "Explain this process.",
                )
            ],
            temperature=0,
            max_tokens=32,
            stop=["__never_stop__"],
            extra_body=dict(ignore_eos=True, repetition_penalty=1.0),
        )
        for repeats in (1, 20, 100, 400)
    ]


async def requests_and_measure(endpoint, uid, root, worker, mode, args):
    api = AsyncOpenAI(base_url=endpoint + "/v1", api_key="unused", timeout=120)
    results = dict(mode=mode, correctness=[], performance=[])
    try:
        for body in bodies():
            regular = await api.chat.completions.create(model=uid, **body)
            response = await api.chat.completions.create(
                model=uid, **body, stream=True, stream_options=dict(include_usage=True)
            )
            content, usage = [], None
            async with response:
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content.append(chunk.choices[0].delta.content)
                    if chunk.usage:
                        usage = chunk.usage.model_dump()
            results["correctness"].append(
                dict(
                    body=body,
                    text=regular.choices[0].message.content,
                    stream_text="".join(content),
                    usage=regular.usage.model_dump(),
                    stream_usage=usage,
                )
            )
            assert regular.choices[0].message.content == "".join(content)
            assert (
                regular.usage.completion_tokens == 32
                and usage["completion_tokens"] == 32
            )
        (root / "correctness.json").write_text(
            json.dumps(results["correctness"], indent=2)
        )
        for prompt in (
            "Explain the water cycle.",
            "Evaporation and condensation. " * 100,
        ):
            body = dict(
                prompt=prompt,
                temperature=0,
                max_tokens=32,
                stop=["__never_stop__"],
                extra_body=dict(ignore_eos=True),
            )
            regular = await api.completions.create(model=uid, **body)
            stream = await api.completions.create(model=uid, **body, stream=True)
            pieces = []
            async with stream:
                async for chunk in stream:
                    if chunk.choices:
                        pieces.append(chunk.choices[0].text)
            assert regular.choices[0].text == "".join(pieces)
            assert regular.usage.completion_tokens == 32
            results.setdefault("completion_correctness", []).append(
                dict(
                    prompt=prompt,
                    text=regular.choices[0].text,
                    stream_text="".join(pieces),
                )
            )
        for concurrency in args.concurrency:
            # Repeated prefixes keep native P caching available in every mode.
            requests_list = [bodies()[i % 4] for i in range(args.requests)]
            records, elapsed, _ = await measure(
                endpoint, uid, requests_list, concurrency, 1
            )
            assert all(
                "error" not in row and row["output_tokens"] == 32 for row in records
            )
            results["performance"].append(
                dict(
                    concurrency=concurrency,
                    elapsed_s=elapsed,
                    summary=summarize(records, elapsed, 1.0, 0.05),
                    records=records,
                )
            )
        if mode in ("vllm-sglang", "sglang-vllm"):
            results["transfers"] = await snapshot(worker, uid, True)
            assert results["transfers"]["directory"]["imported_tokens"] > 0
            assert all(
                peer["cpu_batches"] == 0
                for peer in results["transfers"]["peers"].values()
            )
            assert all(
                peer["active_transfers"] == 0
                for peer in results["transfers"]["peers"].values()
            )
            # Disconnect after the first streaming token, then verify fresh work.
            response = await api.chat.completions.create(
                model=uid, **{**bodies()[3], "max_tokens": 1024}, stream=True
            )
            async with response:
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        break
            await asyncio.sleep(1)
            await api.chat.completions.create(model=uid, **bodies()[0])
            await asyncio.sleep(0.1)
            results["after_cancel"] = await snapshot(worker, uid, True)
            assert results["after_cancel"]["directory"]["active_handoffs"] == 0
            assert all(
                peer["active_transfers"] == 0
                for peer in results["after_cancel"]["peers"].values()
            )
            results["early_cancels"] = []
            for attempt in range(3):
                # Cancel a prepared room while its long prefill is still running.
                request_id = uuid.uuid4().hex
                body = bodies()[3]
                body["messages"][0]["content"] = (
                    f"Trial {attempt}. "
                    + "A new uncached prefill. " * 1000
                    + "Explain evaporation."
                )
                body["max_tokens"] = 1024
                body["extra_body"]["request_id"] = request_id
                pending = asyncio.create_task(
                    api.chat.completions.create(model=uid, **body)
                )
                directory = await xo.actor_ref(
                    address=worker, uid="xavier-cache-" + uid
                )
                for _ in range(1000):
                    if (await directory.get_stats())["active_handoffs"]:
                        break
                    assert (
                        not pending.done()
                    ), "Request finished before the cancellation probe"
                    await asyncio.sleep(0.001)
                else:
                    raise AssertionError(
                        "Cancellation probe did not reach a prepared room"
                    )
                results["early_cancel"] = await asyncio.to_thread(
                    Client(endpoint).abort_request, uid, request_id
                )
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                await asyncio.sleep(1)
                await api.chat.completions.create(model=uid, **bodies()[0])
                await asyncio.sleep(0.1)
                results["after_early_cancel"] = await snapshot(worker, uid, True)
                assert (
                    results["after_early_cancel"]["directory"]["active_handoffs"] == 0
                )
                assert all(
                    peer["active_transfers"] == 0
                    for peer in results["after_early_cancel"]["peers"].values()
                )
                results["early_cancels"].append(results["after_early_cancel"])
    finally:
        await api.close()
        (root / "partial-results.json").write_text(json.dumps(results, indent=2))
    (root / "results.json").write_text(json.dumps(results, indent=2))
    return results


def run(args):
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    isolation = GPUIsolation(args.idle_pid, args.idle_timeout)
    isolation.wait_idle()
    port = get_next_port()
    endpoint = f"http://127.0.0.1:{port}"
    env = dict(
        os.environ,
        XINFERENCE_HOME=str(root / "home"),
        XINFERENCE_AUTH_ADVANCED="false",
        XINFERENCE_ENABLE_VIRTUAL_ENV="0",
    )
    mode = args.mode
    engines = (
        mode.split("-")
        if mode in ("vllm-sglang", "sglang-vllm")
        else [mode.split("-")[-1]]
    )
    with (root / "server.log").open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from xinference.deploy.cmdline import local; local()",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "INFO",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            for _ in range(120):
                if proc.poll() is not None:
                    raise RuntimeError("Xinference exited; inspect server.log")
                try:
                    client = Client(endpoint)
                    workers = client.get_workers_info()
                    if workers:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("Server readiness timeout")
            worker = workers[0]["work-ip"]
            uid = "cross-engine-benchmark"
            launch = dict(
                model_uid=uid,
                model_name="qwen2.5-instruct",
                model_size_in_billions="0_5",
                model_format="pytorch",
                quantization="none",
                model_path=str(args.model_path),
                model_engine=engines[0],
                enable_virtual_env=False,
                envs={
                    key: env[key]
                    for key in ("PATH", "PYTHONPATH", "LD_PRELOAD")
                    if key in env
                },
            )
            if mode.startswith("mono-"):
                launch.update(
                    n_gpu=1, gpu_idx=[0], **engine_options(engines[0], "hybrid", args)
                )
            else:
                if mode.startswith("native-"):
                    engines = [engines[0]] * 2
                    launch["transfer_backend_type"] = "nixl"
                launch["replica"] = 2
                launch["replica_config"] = [
                    dict(
                        role=role,
                        model_engine=engine,
                        engine_config=engine_options(engine, role, args),
                        devices=[dict(worker_ip=worker, n_gpu=1, gpu_idx=[index])],
                    )
                    for index, (role, engine) in enumerate(
                        zip(("prefill", "decode"), engines)
                    )
                ]
            (root / "launch.json").write_text(json.dumps(launch, indent=2))
            (root / "versions.json").write_text(
                json.dumps(
                    {
                        name: version(name)
                        for name in ("vllm", "sglang", "torch", "xoscar")
                    },
                    indent=2,
                )
            )
            client.launch_model(**launch)
            print("launched", mode, flush=True)
            with isolation:
                result = asyncio.run(
                    requests_and_measure(endpoint, uid, root, worker, mode, args)
                )
            (root / "isolation.json").write_text(
                json.dumps(
                    dict(
                        samples=isolation.samples, interference=isolation.interference
                    ),
                    indent=2,
                )
            )
            if isolation.interference:
                raise RuntimeError("Foreign GPU interference; discard measurements")
            client.terminate_model(uid)
            print(
                json.dumps(
                    dict(
                        mode=mode,
                        performance=[row["summary"] for row in result["performance"]],
                    )
                ),
                flush=True,
            )
        finally:
            (root / "isolation.json").write_text(
                json.dumps(
                    dict(
                        samples=isolation.samples, interference=isolation.interference
                    ),
                    indent=2,
                )
            )
            stop_server(proc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "mono-vllm",
            "mono-sglang",
            "native-vllm",
            "native-sglang",
            "vllm-sglang",
            "sglang-vllm",
        ],
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--kv-tokens", type=int, default=32768)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--idle-pid", type=int, action="append", default=[])
    parser.add_argument("--idle-timeout", type=int, default=7200)
    run(parser.parse_args())
