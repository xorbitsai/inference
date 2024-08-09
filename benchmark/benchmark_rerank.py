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

import argparse
import asyncio
import logging
import random
import time
import aiohttp
from typing import List, Dict, Optional
from datasets import load_dataset
import numpy as np
from benchmark_runner import ConcurrentBenchmarkRunner


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RerankBenchmarkRunner(ConcurrentBenchmarkRunner):
    def __init__(
        self,
        api_url: str,
        model_uid: str,
        input_requests: List[Dict],
        stream: bool,
        top_n: int,
        concurrency: int,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            api_url,
            model_uid,
            input_requests,
            stream,
            concurrency,
            api_key,
        )
        self.top_n = top_n

    async def _run(self):
        tasks = []
        for i in range(self.concurrency):
            tasks.append(asyncio.create_task(self.worker(i)))

        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    async def worker(self, i: int):
        r = random.Random(i)
        index = r.randint(0, len(self.input_requests) - 1)
        while self.left > 0:
            request = self.input_requests[index]
            index += 1
            index = index % len(self.input_requests)
            await self.send_request(request)
            self.left -= 1
            # pring longer space to overwrite the previous when left decrease
            print("\rdone_request, left %d    " % (self.left), end="")
        # The last one
        print("")

    async def send_request(self, request, warming_up: bool = False):
        prompt, documents = request["query"], request["positive"]
        request_start_time = time.time()

        pload = {
            "model": self.model_uid,
            "top_n": self.top_n,
            "query": prompt,
            "documents": documents,
        }

        headers = {"User-Agent": "Benchmark Client"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.api_url, headers=headers, json=pload
            ) as response:
                resp = await response.json()
                if response.status == 200:
                    request_end_time = time.time()
                    request_latency = request_end_time - request_start_time
                    if not warming_up:
                        self.outputs.append(request_latency)
                else:
                    logger.error(f"Failed to create chat completion: {resp}")


def main(args: argparse.Namespace):
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/v1/rerank"
    model_uid = args.model_uid

    logger.info("Preparing for benchmark.")
    dataset = load_dataset(args.dataset)
    input_requests = dataset["test"].remove_columns("negative").to_list()
    if args.num_query > 0:
        input_requests = input_requests[: args.num_query]
    else:
        args.num_query = len(input_requests)

    logger.info("Benchmark starts.")

    benchmark = RerankBenchmarkRunner(
        api_url,
        model_uid,
        input_requests,
        args.stream,
        top_n=args.top_n,
        concurrency=args.concurrency,
        api_key=args.api_key,
    )
    asyncio.run(benchmark.run())

    # TODO: Print the results of request_latency in detail.
    # benchmark.print_stats() needs to be overridden
    print(f"Total time: {benchmark.benchmark_time:.2f} s")
    print(f"Throughput: {args.num_query / benchmark.benchmark_time:.2f} requests/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test the rerank model.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9997)
    parser.add_argument(
        "--dataset",
        type=str,
        default="mteb/scidocs-reranking",
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=16,
        help="Set the concurrency of request to send",
    )
    parser.add_argument(
        "--top-n",
        "-n",
        type=int,
        default=5,
        help="Set the top n to the rerank",
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
        "--api-key", type=str, default=None, help="Authorization api key",
    )
    args = parser.parse_args()
    main(args)
