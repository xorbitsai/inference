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

import argparse
import asyncio
import logging
import random
import time
from typing import Dict, List, Optional

import aiohttp
import numpy as np
from benchmark_runner import ConcurrentBenchmarkRunner, RequestOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingBenchmarkRunner(ConcurrentBenchmarkRunner):
    def __init__(
        self,
        api_url: str,
        model_uid: str,
        input_requests: List[Dict],
        stream: bool,
        concurrency: int,
        api_key: Optional[str] = None,
        print_error: bool = False,
    ):
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if not input_requests:
            raise ValueError("input_requests must not be empty")
        self._session: Optional[aiohttp.ClientSession] = None
        super().__init__(
            api_url,
            model_uid,
            input_requests,
            stream,
            concurrency,
            api_key,
            print_error,
        )

    async def run(self):
        self.outputs.clear()
        self.left = len(self.input_requests)
        headers = {"User-Agent": "Benchmark Client"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        connector = aiohttp.TCPConnector(limit=self.concurrency)
        async with aiohttp.ClientSession(
            timeout=timeout, headers=headers, connector=connector
        ) as session:
            self._session = session
            try:
                await self.warm_up()
                start_time = time.perf_counter()
                await self._run()
                self.benchmark_time = time.perf_counter() - start_time
            finally:
                self._session = None

    async def _run(self):
        tasks = [
            asyncio.create_task(self.worker(i))
            for i in range(min(self.concurrency, len(self.input_requests)))
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def worker(self, i: int):
        while self.left > 0:
            # Claim each input exactly once before yielding to another worker.
            index = len(self.input_requests) - self.left
            self.left -= 1
            await self.send_request(self.input_requests[index])

    async def send_request(self, request, warming_up: bool = False):
        assert self._session is not None
        payload = {"model": self.model_uid, "input": request["sentence"]}
        output = RequestOutput()
        start_time = time.perf_counter()
        try:
            async with self._session.post(self.api_url, json=payload) as response:
                if response.status == 200:
                    await response.json()
                    output.success = True
                else:
                    output.error = f"HTTP {response.status}: {await response.text()}"
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            output.error = str(exc)
        output.latency = time.perf_counter() - start_time
        if not output.success:
            logger.error("Embedding request failed")
            if self.print_error:
                logger.error("%s", output.error)
        if not warming_up:
            self.outputs.append(output)

    def print_stats(self):
        successful = sum(output.success for output in self.outputs)
        failed = len(self.outputs) - successful
        duration = self.benchmark_time or 0.0
        throughput = successful / duration if duration > 0 else 0.0
        print(f"Successful requests: {successful}")
        print(f"Failed requests: {failed}")
        print(f"Total time: {duration:.2f} s")
        print(f"Throughput: {throughput:.2f} requests/s")


def main(args: argparse.Namespace):
    from datasets import load_dataset

    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/v1/embeddings"
    model_uid = args.model_uid

    logger.info("Preparing for benchmark.")
    dataset = load_dataset(args.dataset, args.subset)
    input_requests = dataset["test"].to_list()
    if args.num_query > 0:
        input_requests = input_requests[: args.num_query]
    else:
        args.num_query = len(input_requests)

    logger.info("Benchmark starts.")

    benchmark = EmbeddingBenchmarkRunner(
        api_url,
        model_uid,
        input_requests,
        args.stream,
        concurrency=args.concurrency,
        api_key=args.api_key,
        print_error=args.print_error,
    )
    asyncio.run(benchmark.run())

    benchmark.print_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test the embedding model.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9997)
    parser.add_argument(
        "--dataset",
        type=str,
        default="clue",
        help="Name to the dataset.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="tnews",
        help="Subset to the dataset.",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=256,
        help="Set the concurrency of request to send",
    )
    parser.add_argument(
        "--num-query",
        "-q",
        type=int,
        default=-1,
        help="Set the query dataset count, default is all",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface.",
    )
    parser.add_argument(
        "--model-uid", type=str, required=True, help="Xinference model UID."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming responses."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Authorization api key",
    )
    parser.add_argument(
        "--print-error",
        action="store_true",
        help="Print detailed error messages if any errors encountered.",
    )
    args = parser.parse_args()
    main(args)
