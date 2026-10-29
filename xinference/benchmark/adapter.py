"""Runner contract and the built-in fixed-concurrency adapter."""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import threading
from dataclasses import asdict
from typing import Any, Callable, Dict, Protocol

import aiohttp
import numpy as np

from .runner import ConcurrentBenchmarkRunner

# Bump when metric semantics change, so historical comparisons retain provenance.
RUNNER_VERSION = "1"
METRICS_VERSION = "1"


def distribution(values):
    if not values:
        return None
    return {
        "mean": float(np.mean(values)),
        **{f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)},
    }


class Runner(Protocol):
    def capabilities(self) -> Dict[str, Any]: ...

    async def run(self, payload: dict, emit: Callable[[dict], None]) -> None: ...

    async def cancel(self) -> None: ...


class StageRunner(ConcurrentBenchmarkRunner):
    def __init__(self, *args, warmup_requests, progress, **kwargs):
        super().__init__(*args, **kwargs)
        self._warmup_requests = warmup_requests
        self._progress = progress

    async def send_request(self, request: tuple, warming_up: bool = False):
        output = await super().send_request(request, warming_up=warming_up)
        if output.http_status in (401, 403):
            raise RuntimeError(
                f"Benchmark authorization failed (HTTP {output.http_status}); "
                "the current concurrency point was discarded"
            )
        return output

    async def warm_up(self, num_requests=5):
        for request in self._warmup_requests:
            await self.send_request(request, warming_up=True)

    async def _run(self):
        requests = iter(self.input_requests)

        async def worker():
            for request in requests:
                await self.send_request(request)
                self._progress(len(self.outputs))

        workers = [
            asyncio.create_task(worker())
            for _ in range(min(self.concurrency, len(self.input_requests)))
        ]
        try:
            await asyncio.gather(*workers)
        finally:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    def result(self):
        successful = [r for r in self.outputs if r.success]
        timed = [r for r in successful if r.last_token_latency > 0]
        usage_complete = bool(successful) and all(r.usage_received for r in successful)
        duration = self.benchmark_time
        return {
            "concurrency": self.concurrency,
            "duration_seconds": duration,
            "successful": len(successful),
            "failed": len(self.outputs) - len(successful),
            "request_throughput": len(successful) / duration,
            "input_throughput": (
                sum(r.prompt_len for r in successful) / duration
                if usage_complete
                else None
            ),
            "output_throughput": (
                sum(r.completion_tokens for r in successful) / duration
                if usage_complete
                else None
            ),
            "prompt_tokens": (
                distribution([r.prompt_len for r in successful])
                if usage_complete
                else None
            ),
            "output_tokens": (
                distribution([r.completion_tokens for r in successful])
                if usage_complete
                else None
            ),
            "latency_ms": distribution([r.latency * 1000 for r in successful]),
            "ttft_ms": distribution([r.ttft * 1000 for r in timed]),
            # TPOT ends at the last content/reasoning chunk, excluding the
            # trailing usage/DONE frames. ITL samples gaps between chunks.
            "tpot_ms": distribution(
                [
                    (r.last_token_latency - r.ttft) * 1000 / (r.completion_tokens - 1)
                    for r in timed
                    if r.usage_received and r.completion_tokens > 1 and r.itl
                ]
            ),
            "itl_ms": distribution([gap * 1000 for r in timed for gap in r.itl]),
            "usage_complete": usage_complete,
            "raw_requests": [asdict(r) for r in self.outputs],
        }


class XinferenceRunner:
    def __init__(self):
        self._task = None

    def capabilities(self):
        return {
            "name": "xinference",
            "version": RUNNER_VERSION,
            "metrics_version": METRICS_VERSION,
            "load_modes": ["concurrency"],
            "datasets": ["random"],
            "metrics": [
                "latency_ms",
                "ttft_ms",
                "tpot_ms",
                "itl_ms",
                "request_throughput",
                "input_throughput",
                "output_throughput",
            ],
        }

    async def cancel(self):
        if self._task is not None:
            self._task.cancel()

    async def run(self, payload, emit):
        self._task = asyncio.current_task()
        config = payload["config"]
        endpoint = payload["endpoint"]
        token = payload.get("token")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        headers["anthropic-version"] = "2023-06-01"
        # Calibrate on the serving model's tokenizer. No model files or optional
        # tokenizer dependencies are needed on the API/benchmark host.
        # Calibration must not prime any measured prompt when warmup is zero.
        rng = random.Random(config["seed"] ^ 0x5A17CA1B)
        words = [
            str(rng.randrange(100000, 999999)) for _ in range(config["input_tokens"])
        ]
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            count_api_supported = True

            async def count(size):
                nonlocal count_api_supported
                messages = [{"role": "user", "content": " ".join(words[:size])}]
                if count_api_supported:
                    async with session.post(
                        endpoint + "/v1/messages/count_tokens",
                        headers=headers,
                        json={
                            "model": config["model_uid"],
                            "messages": messages,
                            "max_tokens": config["output_tokens"],
                        },
                    ) as response:
                        if response.status == 200:
                            return (await response.json())["input_tokens"]
                        if response.status not in (404, 405, 501):
                            raise RuntimeError(
                                f"Prompt token calibration failed (HTTP {response.status})"
                            )
                        count_api_supported = False
                        emit(
                            {
                                "type": "log",
                                "message": "Token-count API unavailable; calibrating with untimed single-token chat requests.",
                            }
                        )
                # Engines such as llama.cpp serve chat and report prompt usage
                # without implementing the separate Anthropic count endpoint.
                # These probes are preparation only, outside every stage timer.
                async with session.post(
                    endpoint + "/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": config["model_uid"],
                        "messages": messages,
                        "max_tokens": 1,
                        "temperature": 0,
                        "stream": False,
                    },
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"Chat prompt calibration failed (HTTP {response.status})"
                        )
                    data = await response.json()
                    count = (data.get("usage") or {}).get("prompt_tokens")
                    if type(count) is not int or count <= 0:
                        raise RuntimeError(
                            "Prompt calibration requires usage.prompt_tokens in chat responses when the token-count API is unavailable"
                        )
                    return count

            low, high = 1, len(words)
            minimum = await count(low)
            if minimum > config["input_tokens"]:
                raise ValueError(
                    f"Input target is below the chat template's minimum ({minimum} tokens)"
                )
            while low < high:
                middle = (low + high + 1) // 2
                if await count(middle) <= config["input_tokens"]:
                    low = middle
                else:
                    high = middle - 1
            words_per_prompt = low
        emit(
            {
                "type": "log",
                "message": "Prompt calibration complete; actual lengths are taken from response usage.",
            }
        )
        for index, concurrency in enumerate(config["concurrency"]):
            # Independent data per stage; warmup never replays measured prompts.
            stage_rng = random.Random(config["seed"] + index * 100003)

            def sample():
                return (
                    " ".join(
                        str(stage_rng.randrange(100000, 999999))
                        for _ in range(words_per_prompt)
                    ),
                    config["input_tokens"],
                    config["output_tokens"],
                )

            warmup = [sample() for _ in range(config["warmup"])]
            inputs = [sample() for _ in range(config["requests"])]

            def progress(done):
                emit({"type": "progress", "stage": index, "completed": done})

            emit({"type": "log", "message": f"Running concurrency {concurrency}"})
            runner = StageRunner(
                endpoint + "/v1/chat/completions",
                config["model_uid"],
                inputs,
                True,
                concurrency,
                api_key=token,
                warmup_requests=warmup,
                progress=progress,
            )
            await runner.run()
            result = runner.result()
            emit({"type": "stage", "result": result})
            if result["successful"] == 0:
                raise RuntimeError(
                    "All requests failed at this concurrency; see request errors in the results"
                )


# New tools register an adapter here; task storage and the UI use the same schema.
RUNNERS = {"xinference": XinferenceRunner}


def main():
    payload = json.loads(sys.stdin.readline())

    # Keep stdin open as a parent-liveness pipe. A hard API-server crash closes
    # it; the daemon exits even when inference is stalled and emits no progress.
    def watch_parent():
        while os.read(sys.stdin.fileno(), 1):
            pass
        os._exit(1)

    threading.Thread(target=watch_parent, daemon=True).start()

    def emit(event):
        print(json.dumps(event, allow_nan=False), flush=True)

    try:
        asyncio.run(RUNNERS[payload["config"]["runner"]]().run(payload, emit))
    except Exception as exc:
        emit({"type": "log", "message": f"Error: {exc}"})
        raise
