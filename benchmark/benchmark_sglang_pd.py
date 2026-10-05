# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Fresh-server SGLang P/D controls through the same Xinference API.

Run on a two-GPU Linux host with SGLang and Xinference installed. Each backend
uses the same launch settings, warm workload, and cold arrivals during decode.
"""

import argparse
import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import requests
from benchmark_pd import measure, summarize
from openai import AsyncOpenAI
from xoscar.utils import get_next_port

from xinference.client import Client


class GPUInterference(RuntimeError):
    pass


def workload() -> list[dict]:
    bodies = []
    for size, repeats in (("short", 20), ("long", 400)):
        for index in range(4):
            prompt = (
                f"{size} document {index}: "
                + "Water evaporates in sunlight and condenses into clouds. " * repeats
                + " Explain this process in detail."
            )
            bodies.append(
                dict(
                    messages=[dict(role="user", content=prompt)],
                    max_tokens=64,
                    temperature=0,
                    extra_body=dict(ignore_eos=True, repetition_penalty=1.0),
                )
            )
    bodies.extend([bodies[4], bodies[4]])
    for index in range(2):
        bodies.append(
            dict(
                messages=[
                    dict(role="user", content="Explain evaporation."),
                    dict(
                        role="assistant",
                        content="Evaporation turns liquid water into vapor.",
                    ),
                    dict(
                        role="user",
                        content=f"How does this relate to rain? Example {index}.",
                    ),
                ],
                max_tokens=64,
                temperature=0,
                extra_body=dict(ignore_eos=True, repetition_penalty=1.0),
            )
        )
    # Override chat-family stop strings as well as EOS, keeping decode work fixed.
    for body in bodies:
        body["stop"] = ["__xinference_pd_benchmark_never_stop__"]
    return bodies


def gpu_processes() -> list[dict]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        dict(gpu=gpu.strip(), pid=int(pid), memory_mib=int(memory))
        for gpu, pid, memory in (
            line.split(",") for line in output.splitlines() if line.strip()
        )
    ]


class GPUIsolation:
    def __init__(self, idle_pids: list[int], timeout: int):
        self.idle_pids = set(idle_pids)
        self.timeout = timeout
        self.samples: list[dict] = []
        self.interference: list[dict] = []
        self.stop = threading.Event()

    def snapshot(self) -> dict:
        owner = psutil.Process()
        owned = {owner.pid, *(child.pid for child in owner.children(recursive=True))}
        processes = gpu_processes()
        external = [
            row for row in processes if row["pid"] not in owned | self.idle_pids
        ]
        return dict(time=time.time(), processes=processes, external=external)

    def wait_idle(self) -> None:
        deadline = time.monotonic() + self.timeout
        idle_since = None
        last_notice = 0.0
        while True:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "GPU isolation timeout; no performance result was accepted"
                )
            now = time.monotonic()
            if self.snapshot()["external"]:
                idle_since = None
            elif idle_since is None:
                idle_since = now
            elif now - idle_since >= 30:
                return
            if now - last_notice >= 30:
                print(
                    "Waiting for 30 seconds without unrelated GPU workloads", flush=True
                )
                last_notice = now
            time.sleep(1)

    def __enter__(self):
        self.samples.clear()
        self.interference.clear()
        self.stop.clear()

        def monitor():
            while not self.stop.is_set():
                try:
                    sample = self.snapshot()
                except Exception as error:
                    sample = dict(time=time.time(), monitor_error=repr(error))
                    self.interference.append(sample)
                self.samples.append(sample)
                if sample.get("external"):
                    self.interference.append(sample)
                self.stop.wait(1)

        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        sample = self.snapshot()
        self.samples.append(sample)
        if sample["external"]:
            self.interference.append(sample)


async def overlap(endpoint: str, uid: str, round_index: int) -> dict:
    """Inject 32 distinct long prompts after eight background decodes start."""
    ready = [asyncio.Event() for _ in range(8)]
    records = []
    async with AsyncOpenAI(
        base_url=endpoint + "/v1", api_key="unused", timeout=300, max_retries=0
    ) as api:

        async def request(index, prompt, tokens, gate=None):
            start = time.perf_counter()
            stamps, text, usage = [], [], None
            stream = await api.chat.completions.create(
                model=uid,
                messages=[dict(role="user", content=prompt)],
                max_tokens=tokens,
                temperature=0,
                stop=["__xinference_pd_benchmark_never_stop__"],
                stream=True,
                stream_options=dict(include_usage=True),
                extra_body=dict(ignore_eos=True, repetition_penalty=1.0),
            )
            async with stream:
                async for chunk in stream:
                    if chunk.usage:
                        usage = chunk.usage
                    if chunk.choices and chunk.choices[0].delta.content:
                        stamps.append(time.perf_counter())
                        text.append(chunk.choices[0].delta.content)
                        if gate:
                            gate.set()
            if usage is None or not stamps:
                raise RuntimeError("Missing streaming content or token usage")
            records.append(
                dict(
                    index=index,
                    start=start,
                    stamps=stamps,
                    text="".join(text),
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    expected_output_tokens=tokens,
                )
            )

        started = time.perf_counter()
        background = [
            asyncio.create_task(
                request(
                    f"decode-{i}",
                    f"Round {round_index} example {i}: Explain the water cycle in detail.",
                    1024,
                    ready[i],
                )
            )
            for i in range(8)
        ]
        injections = []
        try:
            await asyncio.wait_for(asyncio.gather(*(gate.wait() for gate in ready)), 90)
            for i in range(32):
                prompt = (
                    f"Incoming round {round_index} document {i}: "
                    + "Water evaporates in sunlight and condenses into clouds. " * 400
                    + " Explain this process in detail."
                )
                injections.append(
                    asyncio.create_task(request(f"incoming-{i}", prompt, 64))
                )
                await asyncio.sleep(0.1)
            await asyncio.gather(*background, *injections)
        finally:
            for task in background + injections:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*background, *injections, return_exceptions=True)
        return dict(elapsed_s=time.perf_counter() - started, records=records)


def stop_server(proc) -> None:
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


def run_backend(
    args, backend: str, trial: int, root: Path, isolation: GPUIsolation
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    endpoint = f"http://127.0.0.1:{get_next_port()}"
    env = dict(
        os.environ,
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
                "--log-level",
                "INFO",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        client, launched = None, False
        uid = f"sglang-controlled-{backend}"
        try:
            for _ in range(180):
                if proc.poll() is not None:
                    raise RuntimeError("Xinference server exited; inspect server.log")
                try:
                    client = Client(endpoint)
                    workers = client.get_workers_info()
                    if workers:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("Xinference server readiness timed out")
            launch = dict(
                model_uid=uid,
                model_name="qwen2.5-instruct",
                model_size_in_billions="0_5",
                model_format="pytorch",
                quantization="none",
                model_engine="SGLang",
                model_path=str(args.model_path),
                replica=2,
                enable_virtual_env=False,
                transfer_backend_type=backend,
                envs={
                    key: env[key]
                    for key in ("PATH", "PYTHONPATH", "LD_PRELOAD")
                    if key in env
                },
                replica_config=[
                    dict(
                        role=role,
                        devices=[
                            dict(worker_ip=workers[0]["work-ip"], n_gpu=1, gpu_idx=[i])
                        ],
                    )
                    for i, role in enumerate(("prefill", "decode"))
                ],
                tp_size=1,
                dtype="float16",
                context_length=8192,
                mem_fraction_static=args.memory_fraction,
                max_total_tokens=args.kv_tokens,
                disable_cuda_graph=True,
                page_size=64,
                stream_interval=1,
                enable_cache_report=True,
                triton_attention_reduce_in_fp32=False,
                trust_remote_code=False,
            )
            (root / "launch.json").write_text(json.dumps(launch, indent=2))
            client.launch_model(**launch)
            launched = True
            bodies = workload()
            (root / "workload.json").write_text(json.dumps(bodies, indent=2))
            # Match the historical vLLM serial initialization, priming all twelve bodies.
            initial, _, _ = asyncio.run(measure(endpoint, uid, bodies, 1, 1))
            if any("error" in row for row in initial):
                raise RuntimeError(f"Initialization failed: {initial}")
            result = dict(
                backend=backend,
                trial=trial,
                launch=launch,
                initial=initial,
                throughput=[],
                overlap=[],
                gpu_samples=[],
            )
            with isolation:
                for concurrency in (16, 32):
                    records, elapsed, _ = asyncio.run(
                        measure(
                            endpoint,
                            uid,
                            bodies,
                            concurrency,
                            args.requests // len(bodies),
                        )
                    )
                    row = dict(
                        concurrency=concurrency,
                        records=records,
                        summary=summarize(records, elapsed, 2, 0.05),
                    )
                    result["throughput"].append(row)
                    (root / "results.json").write_text(json.dumps(result, indent=2))
                    print(backend, trial, concurrency, row["summary"], flush=True)
                    if any(
                        "error" in record or record["output_tokens"] != 64
                        for record in records
                    ):
                        raise RuntimeError(
                            "Throughput requests failed or returned fewer than 64 tokens"
                        )
                for round_index in range(args.overlap_rounds):
                    probe = asyncio.run(overlap(endpoint, uid, round_index))
                    result["overlap"].append(probe)
                    (root / "results.json").write_text(json.dumps(result, indent=2))
                    mismatches = [
                        (
                            row["index"],
                            row["output_tokens"],
                            row["expected_output_tokens"],
                        )
                        for row in probe["records"]
                        if row["output_tokens"] != row["expected_output_tokens"]
                    ]
                    if mismatches:
                        raise RuntimeError(
                            f"Overlap output lengths differ (index, actual, expected): {mismatches}"
                        )
                    print(
                        backend, trial, "overlap", round_index, "complete", flush=True
                    )
            result["gpu_samples"] = list(isolation.samples)
            result["interference"] = list(isolation.interference)
            if isolation.interference:
                (root / "rejected.json").write_text(json.dumps(result, indent=2))
                raise GPUInterference(
                    "Unrelated GPU work overlapped this trial; results rejected"
                )
            if backend == "xavier":
                address = client._get_supervisor_internal_address()

                async def counters():
                    import xoscar as xo

                    directory = await xo.actor_ref(
                        address=address, uid=f"xavier-cache-{uid}"
                    )
                    stats = await directory.get_stats()
                    peers = {}
                    for rank, peer in stats["peers"].items():
                        actor = await xo.actor_ref(
                            address=peer, uid=f"sglang-xavier-transfer-{rank}"
                        )
                        peers[rank] = await actor.get_stats()
                    return dict(directory=stats, peers=peers)

                result["counters"] = asyncio.run(counters())
            (root / "results.json").write_text(json.dumps(result, indent=2))
            return result
        finally:
            if launched and client is not None:
                try:
                    client.terminate_model(uid)
                except Exception:
                    pass
            stop_server(proc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--overlap-rounds", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=0.6)
    parser.add_argument("--kv-tokens", type=int, default=524288)
    parser.add_argument("--idle-pid", type=int, action="append", default=[])
    parser.add_argument("--idle-timeout", type=int, default=1800)
    args = parser.parse_args()
    if (
        args.requests < 12
        or args.requests % 12
        or args.trials < 1
        or args.overlap_rounds < 0
        or args.kv_tokens < 8192
    ):
        parser.error(
            "requests must be a positive multiple of 12; trials positive; "
            "overlap-rounds nonnegative; kv-tokens at least 8192"
        )
    if args.output_dir.exists():
        parser.error(
            "output-dir already exists; use a fresh path to preserve prior measurements"
        )
    args.output_dir.mkdir(parents=True)
    import importlib.metadata

    report = dict(
        versions={
            name: importlib.metadata.version(name)
            for name in ("sglang", "torch", "nixl", "xoscar")
        },
        source_commit=args.source_commit,
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                *Path("xinference/model/llm/sglang").rglob("*.py"),
                Path("xinference/core/pd_model.py"),
                Path("xinference/api/restful_api.py"),
                Path("xinference/core/supervisor.py"),
                Path("xinference/types.py"),
                Path("xinference/model/llm/xavier/backends/torch/gpu_transfer.py"),
                Path("xinference/model/llm/xavier/backends/torch/direct_handoff.py"),
            ]
        },
        gpu=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv",
            ],
            text=True,
        ),
        args={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        runs=[],
    )
    isolation = GPUIsolation(args.idle_pid, args.idle_timeout)
    for trial in range(args.trials):
        order = ("nixl", "xavier") if trial % 2 == 0 else ("xavier", "nixl")
        for backend in order:
            for attempt in range(3):
                isolation.wait_idle()
                root = args.output_dir / f"trial-{trial}-{backend}-attempt-{attempt}"
                try:
                    result = run_backend(args, backend, trial, root, isolation)
                except GPUInterference:
                    print("Discarded interrupted trial", root, flush=True)
                    if attempt == 2:
                        raise
                else:
                    report["runs"].append(result)
                    break
            (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print("All controlled trials completed", flush=True)


if __name__ == "__main__":
    main()
