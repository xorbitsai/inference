# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Compare ordinary replicas, shared Xavier, and Xavier P/D on one Apple GPU.

Use a local unquantized FP16 Qwen2.5-0.5B-Instruct checkpoint. Each mode starts
its own Xinference server and two independent MLX model processes. This measures
API overhead and cache reuse on one Mac, not multi-host network performance.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import requests
import xoscar as xo
from openai import AsyncOpenAI
from safetensors import safe_open
from xoscar.utils import get_next_port

from xinference.client import Client


async def measure(endpoint, uid, messages, count, max_tokens):
    records = []
    async with AsyncOpenAI(
        base_url=endpoint + "/v1", api_key="unused", timeout=180, max_retries=0
    ) as api:
        for _ in range(count):
            start = time.perf_counter()
            first, usage, text = None, None, []
            stream = await api.chat.completions.create(
                model=uid,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            async with stream:
                async for chunk in stream:
                    if chunk.usage:
                        usage = chunk.usage
                    if chunk.choices and chunk.choices[0].delta.content:
                        first = first or time.perf_counter()
                        text.append(chunk.choices[0].delta.content)
            if first is None or usage is None:
                raise RuntimeError("Missing streamed text or token usage")
            duration = time.perf_counter() - start
            records.append(
                dict(
                    ttft_ms=(first - start) * 1000,
                    latency_ms=duration * 1000,
                    output_tps=usage.completion_tokens / duration,
                    prompt_tokens=usage.prompt_tokens,
                    cached_tokens=(
                        usage.prompt_tokens_details.cached_tokens
                        if usage.prompt_tokens_details
                        else 0
                    ),
                    output_tokens=usage.completion_tokens,
                    text="".join(text),
                )
            )
    return records


def stop_server(proc):
    # Only terminate the server and descendants created by this benchmark.
    try:
        children = psutil.Process(proc.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    _, alive = psutil.wait_procs(children, timeout=10)
    for child in alive:
        child.kill()


async def stats(address, uid):
    ref = await xo.actor_ref(address=address, uid=f"xavier-cache-{uid}")
    return await ref.get_stats()


def run_mode(args, mode):
    root = args.output / mode
    root.mkdir(parents=True, exist_ok=True)
    endpoint = f"http://127.0.0.1:{get_next_port()}"
    uid = f"mlx-{mode}"
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),
        XINFERENCE_HOME=str(root / "home"),
        XINFERENCE_ENABLE_VIRTUAL_ENV="0",
        XINFERENCE_AUTH_ADVANCED="false",
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
                endpoint.rsplit(":", 1)[1],
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        client, launched = None, False
        try:
            for _ in range(120):
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited; inspect {root / 'server.log'}")
                try:
                    client = Client(endpoint)
                    workers = client.get_workers_info()
                    if workers:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("Server readiness timed out")
            launch = dict(
                model_uid=uid,
                model_name="qwen2.5-instruct",
                model_size_in_billions="0_5",
                model_engine="MLX",
                model_format="mlx",
                quantization="none",
                model_path=str(args.model_path),
                replica=2,
                enable_virtual_env=False,
                prompt_cache_size=2,
                batch_size=4,
            )
            if mode == "shared":
                launch["enable_xavier"] = True
            elif mode == "pd":
                launch["replica_config"] = [
                    dict(
                        role=role,
                        devices=[dict(worker_ip=workers[0]["work-ip"], n_gpu="auto")],
                    )
                    for role in ("prefill", "decode")
                ]
            (root / "launch.json").write_text(json.dumps(launch, indent=2))
            client.launch_model(**launch)
            launched = True
            # Prime kernels with an unrelated short prompt, leaving measured
            # document prefixes cold. Both ordinary replicas receive warm-up.
            asyncio.run(
                measure(
                    endpoint, uid, [{"role": "user", "content": "Say hello."}], 2, 8
                )
            )
            messages = [
                dict(
                    role="user",
                    content=(
                        "Document: "
                        + "Water evaporates in sunlight and condenses into clouds. "
                        * args.repeats
                        + " Explain this process in detail."
                    ),
                )
            ]
            cold = asyncio.run(measure(endpoint, uid, messages, 1, args.max_tokens))
            warm = asyncio.run(
                measure(endpoint, uid, messages, args.requests, args.max_tokens)
            )
            response = requests.post(
                endpoint + "/v1/chat/completions",
                json=dict(
                    model=uid,
                    messages=messages,
                    temperature=0,
                    max_tokens=args.max_tokens,
                ),
                timeout=180,
            )
            response.raise_for_status()
            nonstream = response.json()
            processes = [
                dict(pid=int(pid), model_uid=model_uid)
                for pid, model_uid in re.findall(
                    r"pid:(\d+).*?ModelActor\(([^)]+)\) loaded",
                    (root / "server.log").read_text(),
                )
            ]
            if len({p["pid"] for p in processes}) != 2:
                raise RuntimeError("Expected two independent MLX model processes")
            result = dict(
                mode=mode,
                launch=launch,
                messages=messages,
                cold=cold,
                warm=warm,
                nonstream=nonstream,
                cache_stats=(
                    asyncio.run(stats(workers[0]["work-ip"], uid))
                    if mode != "ordinary"
                    else None
                ),
                model_processes=processes,
            )
            if mode != "ordinary":
                cache = result["cache_stats"]
                if cache["active_handoffs"] or not cache["read_pages"]:
                    raise RuntimeError("Missing cache reads or leaked handoffs")
                if any(r["cached_tokens"] != r["prompt_tokens"] - 1 for r in warm):
                    raise RuntimeError("Warm requests did not reuse their full prefix")
            (root / "results.json").write_text(json.dumps(result, indent=2))
            print(mode, json.dumps(result["cold"] + result["warm"]), flush=True)
            return result
        finally:
            if launched:
                try:
                    client.terminate_model(uid)
                except Exception:
                    pass
            stop_server(proc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    if min(args.requests, args.repeats, args.max_tokens) <= 0:
        parser.error("requests, repeats and max-tokens must be positive")
    args.model_path = args.model_path.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    weights = list(args.model_path.glob("*.safetensors"))
    if not weights:
        parser.error("model-path must contain FP16 safetensors")
    for file in weights:
        with safe_open(file, framework="np") as checkpoint:
            if any(
                checkpoint.get_slice(key).get_dtype() != "F16"
                for key in checkpoint.keys()
            ):
                parser.error("all modes require an unquantized FP16 checkpoint")
    result = dict(
        platform=platform.platform(),
        versions={
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlx-lm", "xoscar")
        },
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        diff_sha256=hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"])
        ).hexdigest(),
        note="One Mac with a shared Metal GPU; no multi-host or NVIDIA-to-Mac measurements.",
        modes=[],
    )
    for mode in ("ordinary", "shared", "pd"):
        result["modes"].append(run_mode(args, mode))
        (args.output / "results.json").write_text(json.dumps(result, indent=2))
    texts = [
        row["text"] for mode in result["modes"] for row in mode["cold"] + mode["warm"]
    ] + [
        mode["nonstream"]["choices"][0]["message"]["content"]
        for mode in result["modes"]
    ]
    result["all_texts_match"] = len(set(texts)) == 1
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    if not result["all_texts_match"]:
        raise RuntimeError("Greedy outputs differ; inspect the per-request records")


if __name__ == "__main__":
    main()
