# Copyright 2022-2023 XProbe Inc.
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

import aiohttp
import json
import sys
import traceback
import warnings
import logging
from dataclasses import dataclass, field
import time
from typing import List, Optional, Tuple

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


class BenchmarkRunner:
    def __init__(
        self,
        api_url: str,
        model_uid: str,
        input_requests: List[Tuple[str, int, int]],
        stream: bool,
        api_key: Optional[str]=None,
    ):
        self.api_url = api_url
        self.model_uid = model_uid
        self.input_requests = input_requests
        self.outputs: List[RequestOutput] = []
        self.benchmark_time = None
        self.stream = stream
        self.api_key = api_key

    async def run(self):
        await self.warm_up()
        start_time = time.time()
        await self._run()
        end_time = time.time()
        self.benchmark_time = end_time - start_time

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

        if self.stream:
            pload = {
                "model": self.model_uid,
                "n": 1,
                "temperature": 0.6,
                "top_p": 0.9,
                "max_tokens": output_len,
                "stream": True,
                "messages": [{"role": "user", "content": prompt}],
                "stream_options": {"include_usage": True},
            }
        else:
            pload = {
                "model": self.model_uid,
                "n": 1,
                "temperature": 0.6,
                "top_p": 0.9,
                "max_tokens": output_len,
                "stream": False,
                "messages": [{"role": "user", "content": prompt}],
            }

        headers = {"User-Agent": "Benchmark Client"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            output = RequestOutput(prompt_len=prompt_len)
            ttft = 0.0
            st = time.perf_counter()
            most_recent_timestamp = st

            try:
                async with session.post(
                    self.api_url, headers=headers, json=pload
                ) as response:
                    if response.status == 200:
                        if self.stream:
                            async for chunk_bytes in response.content:
                                # {
                                #     "id": "chataec79465-dfea-46af-81b9-c28124063fc0",
                                #     "model": "llama-3-instruct",
                                #     "created": 1721202668,
                                #     "object": "chat.completion.chunk",
                                #     "choices": [
                                #         {
                                #             "index": 0,
                                #             "delta": {"role": "assistant", "content": ""},
                                #             "finish_reason": null,
                                #         }
                                #     ],
                                # }
                                chunk_bytes = chunk_bytes.strip()
                                if not chunk_bytes:
                                    continue

                                chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data:")

                                if chunk == "[DONE]":
                                    latency = time.perf_counter() - st
                                else:
                                    timestamp = time.perf_counter()
                                    data = json.loads(chunk)

                                    # First token
                                    if ttft == 0.0:
                                        ttft = time.perf_counter() - st
                                        output.ttft = ttft

                                    # Decoding phase
                                    else:
                                        output.itl.append(timestamp - most_recent_timestamp)

                                    most_recent_timestamp = timestamp

                            output.latency = latency
                            output.success = True
                            output.completion_tokens = data["usage"]["completion_tokens"]
                        else:
                            resp = await response.json()
                            output.latency = time.perf_counter() - st
                            output.success = True
                            output.completion_tokens = resp["usage"]["completion_tokens"]
            except Exception:
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))

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
            print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
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


class ConcurrentBenchmarkRunner(BenchmarkRunner):
    def __init__(
        self,
        api_url: str,
        model_uid: str,
        input_requests: List[Tuple[str, int, int]],
        stream: bool,
        concurrency: int,
        api_key: Optional[str]=None,
    ):
        super().__init__(api_url, model_uid, input_requests, stream, api_key)
        self.concurrency = concurrency
        self.left = len(input_requests)

    async def worker(self):
        pass
