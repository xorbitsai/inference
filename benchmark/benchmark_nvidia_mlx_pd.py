# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Measure both directions of NVIDIA / Mac MLX PD on an existing LAN cluster.

Both workers must run this checkout, with vLLM/SGLang on NVIDIA and MLX on Mac.
Use identical original checkpoint assets for P/D. The optional local baseline
must contain the exact same effective weights, cast to FP16. No server processes
are created or stopped here; only this benchmark's model deployments are owned.
"""

import argparse
import asyncio
import json
import statistics
import time
import uuid
from pathlib import Path

import xoscar as xo

from xinference.client import Client

from benchmark_mlx_xavier import measure


def launch_options(args, mode, uid):
    gpu_engine = mode.removeprefix("local-")
    options = dict(
        model_uid=uid,
        model_name=args.model_name,
        model_size_in_billions=args.model_size,
        model_engine="MLX" if mode == "local" else gpu_engine,
        model_format="mlx" if mode == "local" else "pytorch",
        quantization="none",
        enable_virtual_env=False,
    )
    if mode == "local":
        return dict(
            options,
            replica=1,
            worker_ip=args.mac_worker,
            model_path=args.baseline_model_path,
            prompt_cache_size=0,
        )
    config = dict(model_path=args.gpu_model_path, dtype="float16")
    if gpu_engine == "vLLM":
        config.update(
            max_model_len=args.context_length,
            gpu_memory_utilization=0.3,
            enforce_eager=True,
        )
    else:
        config.update(
            context_length=args.context_length,
            mem_fraction_static=0.3,
            max_total_tokens=args.context_length,
            disable_cuda_graph=True,
            page_size=64,
            stream_interval=1,
        )
    if mode.startswith("local-"):
        config.update(
            {"enable_prefix_caching": False}
            if gpu_engine == "vLLM"
            else {"disable_radix_cache": True}
        )
        return dict(
            options,
            **config,
            replica=1,
            worker_ip=args.gpu_worker,
            n_gpu=1,
            gpu_idx=[args.gpu],
        )
    reverse = getattr(args, "direction", "nvidia-prefill") == "mlx-prefill"
    options.update(
        replica=2,
        replica_config=[
            dict(
                role="decode" if reverse else "prefill",
                model_engine=mode,
                engine_config=config,
                devices=[dict(worker_ip=args.gpu_worker, n_gpu=1, gpu_idx=[args.gpu])],
            ),
            dict(
                role="prefill" if reverse else "decode",
                model_engine="MLX",
                engine_config=dict(
                    model_path=args.mac_model_path,
                    model_format="mlx",
                    context_length=args.context_length,
                    prompt_cache_size=0,
                ),
                devices=[dict(worker_ip=args.mac_worker, n_gpu="auto")],
            ),
        ],
    )
    if reverse:
        options["replica_config"].reverse()
    return options


async def directory_stats(address, uid):
    ref = await xo.actor_ref(address=address, uid="xavier-cache-" + uid)
    return await ref.get_stats()


def isolation_samples(path):
    samples = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # The monitor may still be writing its last line.
    return samples


async def measured_run(args, uid, messages, use_gpu, is_pd):
    path = args.gpu_isolation_log if use_gpu else None
    deadline = time.monotonic() + 1800
    before = []
    if path:
        while True:
            before = isolation_samples(path)
            if (
                before
                and not before[-1]["foreign"]
                and time.time() - path.stat().st_mtime < 2
            ):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("No isolated GPU measurement window")
            await asyncio.sleep(1)
    state = await directory_stats(args.supervisor_address, uid) if is_pd else None
    runs = await measure(args.endpoint, uid, messages, 1, args.max_tokens)
    if state is not None:
        after = await directory_stats(args.supervisor_address, uid)
        if after["completed_requests"] != state["completed_requests"] + 1:
            raise RuntimeError("Request did not complete one KV handoff")
        runs[0]["imported_tokens"] = after["imported_tokens"] - state["imported_tokens"]
        runs[0]["host_bytes"] = after["host_bytes"] - state["host_bytes"]
        if not runs[0]["host_bytes"] or after["gpu_bytes"] != 0:
            raise RuntimeError("Request did not use the host transport")
    if path:
        # The external monitor samples at least twice per second. Inspect the
        # samples spanning this request, without including this wait in TTFT.
        await asyncio.sleep(0.6)
        after = isolation_samples(path)
        window = after[max(0, len(before) - 1) :]
        if len(after) <= len(before) or any(s["foreign"] for s in window):
            print(
                "Discarding a GPU run with interference; retrying with a new prompt",
                flush=True,
            )
            return None
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "endpoint",
        "supervisor-address",
        "gpu-worker",
        "mac-worker",
        "gpu-model-path",
        "mac-model-path",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--baseline-model-path", help="Same effective weights in FP16")
    parser.add_argument("--model-name", default="qwen2.5-instruct")
    parser.add_argument("--model-size", default="0_5")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--direction", choices=["nvidia-prefill", "mlx-prefill", "both"], default="both"
    )
    parser.add_argument("--gpu-baseline", action="store_true")
    parser.add_argument(
        "--gpu-isolation-log",
        type=Path,
        help="External JSONL monitor sampled at >=2 Hz; each record has a foreign PID list",
    )
    parser.add_argument(
        "--engines", nargs="+", choices=["vLLM", "SGLang"], default=["vLLM", "SGLang"]
    )
    parser.add_argument("--repeats", nargs="+", type=int, default=[1, 20, 100, 400])
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = Client(args.endpoint)
    result = {}
    directions = (
        ["nvidia-prefill", "mlx-prefill"]
        if args.direction == "both"
        else [args.direction]
    )
    modes = [("local", "local")] if args.baseline_model_path else []
    if args.gpu_baseline:
        modes += [("local-" + engine, "local") for engine in args.engines]
    modes += [
        (engine, direction) for direction in directions for engine in args.engines
    ]
    for mode, direction in modes:
        args.direction = direction
        label = mode if direction == "local" else f"{direction}:{mode}"
        uid = "nvidia-mlx-bench-" + uuid.uuid4().hex[:8]
        try:
            client.launch_model(**launch_options(args, mode, uid))
            records = []
            trial = 0
            for repeats in args.repeats:
                content = (
                    "Water evaporates and condenses. " * repeats
                    + "Explain this process."
                )
                warmup = [dict(role="user", content=f"Warmup {repeats}: " + content)]
                # Vary the early user prefix so P cannot reuse complete pages;
                # use the same trial sequence for every engine and local MLX.
                asyncio.run(measure(args.endpoint, uid, warmup, 1, args.max_tokens))
                runs = []
                while len(runs) < args.count:
                    messages = [
                        dict(role="user", content=f"Trial {trial:08d}: " + content)
                    ]
                    measured = asyncio.run(
                        measured_run(
                            args, uid, messages, mode != "local", direction != "local"
                        )
                    )
                    if measured is not None:
                        for record in measured:
                            record["trial"] = trial
                        runs.extend(measured)
                    trial += 1
                for run in runs:
                    expected = (
                        run["prompt_tokens"]
                        - (0 if direction == "mlx-prefill" and mode == "SGLang" else 1)
                        if direction != "local"
                        else 0
                    )
                    # NVIDIA usage currently omits cached-token details. The
                    # directory's completed import counter establishes reuse.
                    if direction != "local" and run["imported_tokens"] != expected:
                        raise RuntimeError("Decode did not import the expected prefix")
                records.append(
                    dict(
                        repeats=repeats,
                        prompt_tokens=runs[0]["prompt_tokens"],
                        median_ttft_ms=statistics.median(r["ttft_ms"] for r in runs),
                        runs=runs,
                    )
                )
            state = (
                None
                if direction == "local"
                else asyncio.run(directory_stats(args.supervisor_address, uid))
            )
            if state and state["active_handoffs"]:
                raise RuntimeError("Completed requests left active handoffs")
            result[label] = dict(records=records, directory=state)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2))
            print(
                label,
                [(r["prompt_tokens"], round(r["median_ttft_ms"], 1)) for r in records],
                flush=True,
            )
        finally:
            client.terminate_model(uid)


if __name__ == "__main__":
    main()
