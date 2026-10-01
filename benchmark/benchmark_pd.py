# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Compare hybrid replicas, Xavier PD and NIXL PD using identical requests.

Run against an otherwise idle Xinference server. Launch JSON supplies the model,
placements, dtype, memory budget and engine settings; each mode runs sequentially.
JSONL workloads contain OpenAI chat bodies (messages, max_tokens, etc.).
"""
import argparse
import asyncio
import json
import math
import subprocess
import time
from copy import deepcopy
from pathlib import Path

from openai import AsyncOpenAI

from xinference.client import Client


def percentile(values, fraction):
    values = sorted(values)
    return (
        values[min(len(values) - 1, math.ceil(len(values) * fraction) - 1)]
        if values
        else None
    )


def summarize(records, elapsed, ttft_slo, tpot_slo):
    successful = [r for r in records if "error" not in r]
    eligible = [
        r for r in successful if r["ttft_s"] is not None and r["tpot_s"] is not None
    ]
    good = [r for r in eligible if r["ttft_s"] <= ttft_slo and r["tpot_s"] <= tpot_slo]
    result = {
        "requests": len(records),
        "successful": len(successful),
        "elapsed_s": elapsed,
        "goodput_req_s": len(good) / elapsed,
        "output_tokens_s": sum(r["output_tokens"] for r in successful) / elapsed,
        "slo_eligible_requests": len(eligible),
    }
    for metric in ("ttft_s", "tpot_s", "latency_s"):
        values = [r[metric] for r in successful if r[metric] is not None]
        result[metric] = {
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
        }
    return result


async def measure(
    endpoint, uid, workload, concurrency, repeats, sample_gpu=False, server_pid=None
):
    semaphore = asyncio.Semaphore(concurrency)
    async with AsyncOpenAI(
        base_url=endpoint.rstrip("/") + "/v1",
        api_key="unused",
        timeout=600,
        max_retries=0,
    ) as api:

        async def request(index, body):
            async with semaphore:
                start = time.perf_counter()
                first = last = None
                usage = None
                text = []
                try:
                    body = deepcopy(body)
                    body.update(
                        model=uid, stream=True, stream_options={"include_usage": True}
                    )
                    stream = await api.chat.completions.create(**body)
                    async with stream:
                        async for chunk in stream:
                            if chunk.usage:
                                usage = chunk.usage
                            if chunk.choices:
                                delta = chunk.choices[0].delta
                                content = delta.content or getattr(
                                    delta, "reasoning_content", None
                                )
                                if content:
                                    last = time.perf_counter()
                                    first = first or last
                                    text.append(content)
                    if usage is None:
                        raise RuntimeError(
                            "Server did not return streaming token usage"
                        )
                    tokens = usage.completion_tokens
                    return {
                        "index": index,
                        "ttft_s": first - start if first else None,
                        "tpot_s": (last - first) / (tokens - 1)
                        if first and tokens > 1
                        else None,
                        "latency_s": time.perf_counter() - start,
                        "input_tokens": usage.prompt_tokens,
                        "output_tokens": tokens,
                        "text": "".join(text),
                    }
                except Exception as exc:
                    return {
                        "index": index,
                        "error": str(exc),
                        "latency_s": time.perf_counter() - start,
                    }

        samples = []

        async def sample_resources():
            while True:
                result = await asyncio.to_thread(
                    subprocess.run,
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used,utilization.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                process_rss = None
                if server_pid:
                    import psutil

                    parent = psutil.Process(server_pid)
                    process_rss = sum(
                        process.memory_info().rss
                        for process in [parent, *parent.children(recursive=True)]
                        if process.is_running()
                    )
                samples.append(
                    {
                        "server_tree_rss_bytes": process_rss,
                        "time": time.time(),
                        "gpu_csv": result.stdout.strip(),
                        "error": result.stderr.strip(),
                    }
                )
                await asyncio.sleep(1)

        sampler = asyncio.create_task(sample_resources()) if sample_gpu else None
        start = time.perf_counter()
        records = await asyncio.gather(
            *(request(i, body) for i, body in enumerate(workload * repeats))
        )
        elapsed = time.perf_counter() - start
        if sampler is not None:
            sampler.cancel()
            await asyncio.gather(sampler, return_exceptions=True)
        return records, elapsed, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("hybrid", "xavier", "nixl"),
        default=["hybrid", "xavier", "nixl"],
    )
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument(
        "--sample-gpu",
        action="store_true",
        help="Sample local nvidia-smi; run on the GPU host",
    )
    parser.add_argument(
        "--server-pid",
        type=int,
        help="Optional local server PID for process-tree RSS sampling",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ttft-slo", type=float, default=2)
    parser.add_argument("--tpot-slo", type=float, default=0.05)
    args = parser.parse_args()
    if args.repeats < 1 or any(c < 1 for c in args.concurrency):
        parser.error("repeats and concurrency must be positive")
    launch = json.loads(args.launch.read_text())
    workload = [
        json.loads(line)
        for line in args.workload.read_text().splitlines()
        if line.strip()
    ]
    if not workload:
        parser.error("workload must not be empty")
    client = Client(args.endpoint)
    report = {
        "launch": launch,
        "workload": workload,
        "ttft_slo_s": args.ttft_slo,
        "tpot_slo_s": args.tpot_slo,
        "repeats": args.repeats,
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        config = deepcopy(launch)
        uid = config.pop("model_uid", "pd-benchmark") + "-" + mode
        if uid in client.list_models():
            raise RuntimeError(
                f"Model {uid} already exists; choose another benchmark UID"
            )
        config.pop("vllm_transfer_backend_type", None)
        config.pop("transfer_backend_type", None)
        config.pop("enable_xavier", None)
        if mode == "hybrid":
            for replica in config["replica_config"]:
                replica["role"] = "hybrid"
        else:
            config["vllm_transfer_backend_type"] = mode
        try:
            client.launch_model(model_uid=uid, **config)
            # Disjoint prompt warms kernels without priming the measured prefixes.
            warm, _, _ = asyncio.run(
                measure(
                    args.endpoint,
                    uid,
                    [
                        {
                            "messages": [
                                {"role": "user", "content": "Warmup: count to ten."}
                            ],
                            "max_tokens": 16,
                        }
                    ],
                    1,
                    len(config["replica_config"]),
                )
            )
            if any("error" in record for record in warm):
                raise RuntimeError(f"Warmup failed: {warm}")
            for concurrency in args.concurrency:
                records, elapsed, samples = asyncio.run(
                    measure(
                        args.endpoint,
                        uid,
                        workload,
                        concurrency,
                        args.repeats,
                        args.sample_gpu,
                        args.server_pid,
                    )
                )
                report["runs"].append(
                    {
                        "mode": mode,
                        "concurrency": concurrency,
                        "records": records,
                        "gpu_samples": samples,
                        "summary": summarize(
                            records, elapsed, args.ttft_slo, args.tpot_slo
                        ),
                    }
                )
                args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
                print(mode, concurrency, report["runs"][-1]["summary"], flush=True)
        finally:
            if uid in client.list_models():
                client.terminate_model(uid)


if __name__ == "__main__":
    main()
