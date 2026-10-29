# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import aiohttp
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=3 * 3600)


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    return text.strip()


@dataclass
class RequestOutput:
    success: bool = False
    prompt_len: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    ttft: float = 0.0
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    error: str = ""
    usage_received: bool = False
    last_token_latency: float = 0.0


class BenchmarkRunner:
    def __init__(
        self,
        api_url: str,
        model_uid: str,
        input_requests: List[Tuple[str, int, int]],
        stream: bool,
        api_key: Optional[str] = None,
        print_error: bool = False,
        ignore_eos: bool = False,
    ):
        self.api_url = api_url
        self.model_uid = model_uid
        self.input_requests = input_requests
        self.outputs: List[RequestOutput] = []
        self.benchmark_time = None
        self.stream = stream
        self.api_key = api_key
        self.print_error = print_error
        self.ignore_eos = ignore_eos
        self._session: Optional[aiohttp.ClientSession] = None

    async def run(self):
        async with aiohttp.ClientSession(
            timeout=AIOHTTP_TIMEOUT, connector=aiohttp.TCPConnector(limit=0)
        ) as session:
            self._session = session
            try:
                await self.warm_up()
                start_time = time.perf_counter()
                await self._run()
                self.benchmark_time = time.perf_counter() - start_time
            finally:
                self._session = None

    async def warm_up(self, num_requests: int = 5):
        logger.info("Warming up...")
        for i in range(min(num_requests, len(self.input_requests))):
            request = self.input_requests[i]
            await self.send_request(request, warming_up=True)
        logger.info("Warm-up completed.")

    async def _run(self):
        pass

    async def send_request(self, request: tuple, warming_up: bool = False):
        prompt, prompt_len, output_len = request
        payload = {
            "model": self.model_uid,
            "n": 1,
            "temperature": 0.6,
            "top_p": 0.9,
            "max_tokens": output_len,
            "stream": self.stream,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.stream:
            payload["stream_options"] = {"include_usage": True}
        if self.ignore_eos:
            payload["ignore_eos"] = True
        headers = {"User-Agent": "Xinference Benchmark"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        output = RequestOutput(prompt_len=prompt_len)
        started = time.perf_counter()
        previous_token = None
        usage = None
        finished = False
        try:
            if self._session is None:
                raise RuntimeError("Benchmark requests must run inside run()")
            async with self._session.post(
                self.api_url, headers=headers, json=payload
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"HTTP {response.status}: {(await response.text())[:1000]}"
                    )
                if self.stream:
                    # aiohttp iterates complete lines, including when TCP splits
                    # an SSE frame. Empty role/usage events carry no token timing.
                    async for line in response.content:
                        line = line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == b"[DONE]":
                            finished = True
                            break
                        data = json.loads(raw)
                        if data.get("error"):
                            raise RuntimeError(str(data["error"])[:1000])
                        usage = data.get("usage") or usage
                        choices = data.get("choices") or []
                        delta = choices[0].get("delta", {}) if choices else {}
                        if any(
                            delta.get(key)
                            for key in ("content", "reasoning_content", "reasoning")
                        ):
                            now = time.perf_counter()
                            if previous_token is None:
                                output.ttft = now - started
                            else:
                                output.itl.append(now - previous_token)
                            output.last_token_latency = now - started
                            previous_token = now
                    if not finished:
                        raise RuntimeError("Stream ended before [DONE]")
                else:
                    data = await response.json()
                    if data.get("error"):
                        raise RuntimeError(str(data["error"])[:1000])
                    usage = data.get("usage")
                output.latency = time.perf_counter() - started
                if isinstance(usage, dict):
                    output.usage_received = (
                        type(usage.get("prompt_tokens")) is int
                        and usage["prompt_tokens"] >= 0
                        and type(usage.get("completion_tokens")) is int
                        and usage["completion_tokens"] >= 0
                    )
                    if output.usage_received:
                        output.prompt_len = usage["prompt_tokens"]
                        output.completion_tokens = usage["completion_tokens"]
                # Legacy CLI retains its estimate when usage is unavailable;
                # the structured adapter explicitly reports missing token data.
                if (
                    not output.usage_received
                    and self.stream
                    and previous_token is not None
                ):
                    output.completion_tokens = len(output.itl) + 1
                output.success = True
        except Exception as exc:
            output.latency = time.perf_counter() - started
            output.error = str(exc)[:1000]
        if not warming_up:
            self.outputs.append(output)

    def print_stats(self):
        total_time = self.benchmark_time

        if self.stream:
            # Initialize variables for metrics
            total_input = 0
            completed = 0
            actual_output_lens = []
            itls = []
            tpots = []
            ttfts = []

            for output in self.outputs:
                if output.success:
                    actual_output_lens.append(output.completion_tokens)
                    total_input += output.prompt_len
                    if output.completion_tokens > 1:
                        tpots.append(
                            (output.latency - output.ttft)
                            / (output.completion_tokens - 1)
                        )
                    itls += output.itl
                    ttfts.append(output.ttft)
                    completed += 1
                else:
                    actual_output_lens.append(0)

            if completed == 0:
                warnings.warn(
                    "All requests failed. This is likely due to a misconfiguration "
                    "on the benchmark arguments.",
                    stacklevel=2,
                )

            # Calculate statistics
            total_output = sum(actual_output_lens)
            request_throughput = completed / total_time if total_time > 0 else 0
            input_throughput = total_input / total_time if total_time > 0 else 0
            output_throughput = total_output / total_time if total_time > 0 else 0

            mean_ttft = np.mean(ttfts) * 1000 if ttfts else 0
            median_ttft = np.median(ttfts) * 1000 if ttfts else 0
            std_ttft = np.std(ttfts) * 1000 if ttfts else 0
            p99_ttft = np.percentile(ttfts, 99) * 1000 if ttfts else 0

            mean_tpot = np.mean(tpots) * 1000 if tpots else 0
            median_tpot = np.median(tpots) * 1000 if tpots else 0
            std_tpot = np.std(tpots) * 1000 if tpots else 0
            p99_tpot = np.percentile(tpots, 99) * 1000 if tpots else 0

            mean_itl = np.mean(itls) * 1000 if itls else 0
            median_itl = np.median(itls) * 1000 if itls else 0
            std_itl = np.std(itls) * 1000 if itls else 0
            p99_itl = np.percentile(itls, 99) * 1000 if itls else 0

            # Print benchmark results
            print("{s:{c}^{n}}".format(s=" Benchmark Result ", n=50, c="="))
            print("{:<40} {:<10}".format("Successful requests:", completed))
            print("{:<40} {:<10.2f}".format("Benchmark duration (s):", total_time))
            print("{:<40} {:<10}".format("Total input tokens:", total_input))
            print("{:<40} {:<10}".format("Total generated tokens:", total_output))
            print(
                "{:<40} {:<10.2f}".format(
                    "Request throughput (req/s):", request_throughput
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Input token throughput (tok/s):", input_throughput
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Output token throughput (tok/s):", output_throughput
                )
            )

            print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
            print("{:<40} {:<10.4f}".format("Mean TTFT (ms):", mean_ttft))
            print("{:<40} {:<10.4f}".format("Median TTFT (ms):", median_ttft))
            print("{:<40} {:<10.4f}".format("Std TTFT (ms):", std_ttft))
            print("{:<40} {:<10.4f}".format("P99 TTFT (ms):", p99_ttft))

            print(
                "{s:{c}^{n}}".format(
                    s="Time per Output Token (excl. 1st token)", n=50, c="-"
                )
            )
            print("{:<40} {:<10.4f}".format("Mean TPOT (ms):", mean_tpot))
            print("{:<40} {:<10.4f}".format("Median TPOT (ms):", median_tpot))
            print("{:<40} {:<10.4f}".format("Std TPOT (ms):", std_tpot))
            print("{:<40} {:<10.4f}".format("P99 TPOT (ms):", p99_tpot))

            print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
            print("{:<40} {:<10.4f}".format("Mean ITL (ms):", mean_itl))
            print("{:<40} {:<10.4f}".format("Median ITL (ms):", median_itl))
            print("{:<40} {:<10.4f}".format("Std ITL (ms):", std_itl))
            print("{:<40} {:<10.4f}".format("P99 ITL (ms):", p99_itl))

            print("=" * 50)
        else:
            # Initialize variables for metrics
            total_input = 0
            completed = 0
            actual_output_lens = []
            latencies = []
            per_token_latencies = []
            per_output_token_latencies = []

            for output in self.outputs:
                if output.success:
                    actual_output_lens.append(output.completion_tokens)
                    total_input += output.prompt_len
                    latencies.append(output.latency)
                    per_token_latencies.append(
                        output.latency / (output.prompt_len + output.completion_tokens)
                    )
                    if output.completion_tokens > 0:
                        per_output_token_latencies.append(
                            output.latency / output.completion_tokens
                        )
                    completed += 1
                else:
                    actual_output_lens.append(0)

            if completed == 0:
                warnings.warn(
                    "All requests failed. This is likely due to a misconfiguration "
                    "on the benchmark arguments.",
                    stacklevel=2,
                )

            # Calculate statistics
            total_output = sum(actual_output_lens)
            request_throughput = len(self.outputs) / total_time if total_time > 0 else 0
            input_throughput = total_input / total_time if total_time > 0 else 0
            output_throughput = total_output / total_time if total_time > 0 else 0

            mean_latency = np.mean(latencies) if latencies else 0
            mean_per_token_latency = (
                np.mean(per_token_latencies) if per_token_latencies else 0
            )
            mean_per_output_token_latency = (
                np.mean(per_output_token_latencies) if per_output_token_latencies else 0
            )

            # Print benchmark results
            print("{s:{c}^{n}}".format(s=" Benchmark Result ", n=50, c="="))
            print("{:<40} {:<10}".format("Successful requests:", completed))
            print("{:<40} {:<10.2f}".format("Benchmark duration (s):", total_time))
            print("{:<40} {:<10}".format("Total input tokens:", total_input))
            print("{:<40} {:<10}".format("Total generated tokens:", total_output))
            print(
                "{:<40} {:<10.2f}".format(
                    "Request throughput (req/s):", request_throughput
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Input token throughput (tok/s):", input_throughput
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Output token throughput (tok/s):", output_throughput
                )
            )

            print("{s:{c}^{n}}".format(s="Latency Statistics", n=50, c="-"))
            print("{:<40} {:<10.4f}".format("Mean latency (s):", mean_latency))
            print(
                "{:<40} {:<10.4f}".format(
                    "Mean latency per token (s):", mean_per_token_latency
                )
            )
            print(
                "{:<40} {:<10.4f}".format(
                    "Mean latency per output token (s):", mean_per_output_token_latency
                )
            )

            print("=" * 50)

            print(f"Total time: {total_time:.2f} s")
            print(f"Throughput: {len(self.outputs) / total_time:.2f} requests/s")

        if completed < len(self.input_requests):
            if self.print_error:
                logger.info("Errors encountered during benchmark:")
                for output in self.outputs:
                    if not output.success:
                        print(
                            f"Error for prompt with length {output.prompt_len}: {output.error}"
                        )
            else:
                logger.info(
                    "Errors were encountered during the benchmark. Run with --print-error to see detailed error messages."
                )


class ConcurrentBenchmarkRunner(BenchmarkRunner):
    def __init__(
        self,
        api_url: str,
        model_uid: str,
        input_requests: List[Tuple[str, int, int]],
        stream: bool,
        concurrency: int,
        api_key: Optional[str] = None,
        print_error: bool = False,
        ignore_eos: bool = False,
    ):
        super().__init__(
            api_url,
            model_uid,
            input_requests,
            stream,
            api_key,
            print_error,
            ignore_eos,
        )
        self.concurrency = concurrency
        self.left = len(input_requests)

    async def worker(self):
        pass
