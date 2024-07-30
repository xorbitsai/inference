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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUEST_LATENCY: List[float] = []


class BenchmarkRunner:
    def __init__(
        self,
        api_url: str,
        model_uid: str,
        input_requests: List[Dict],
        top_n: int,
        concurrency: int,
        api_key: Optional[str]=None,
    ):
        self.api_url = api_url
        self.model_uid = model_uid
        self.input_requests = input_requests
        self.top_n = top_n
        self.concurrency = concurrency
        self.sent = 0
        self.left = len(input_requests)
        self.api_key = api_key

    async def run(self):
        tasks = []
        for i in range(0, self.concurrency):
            tasks.append(asyncio.create_task(self.worker(i)))
        await asyncio.gather(*tasks)

    async def worker(self, i: int):
        r = random.Random(i)
        index = r.randint(0, len(self.input_requests) - 1)
        while self.sent < len(self.input_requests):
            item = self.input_requests[index]
            prompt, documents = item["query"], item["positive"]
            index += 1
            self.sent += 1
            index = index % len(self.input_requests)
            await self.send_request(
                self.api_url,
                self.model_uid,
                prompt,
                documents,
            )
            self.left -= 1
            # pring longer space to overwrite the previous when left decrease
            print("\rdone_request, left %d    " % (self.left), end="")
        # The last one
        print("")

    async def send_request(
        self, api_url: str, model_uid: str, prompt: str, documents: List[str],
            api_key: Optional[str]=None,
    ):
        request_start_time = time.time()

        pload = {
            "model": model_uid,
            "top_n": self.top_n,
            "query": prompt,
            "documents": documents,
        }

        headers = {"User-Agent": "Benchmark Client"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(api_url, headers=headers, json=pload) as response:
                resp = await response.json()
                if response.status == 200:
                    request_end_time = time.time()
                    request_latency = request_end_time - request_start_time
                    REQUEST_LATENCY.append(request_latency)
                else:
                    logger.error(f"Failed to create chat completion: {resp}")


def main(args: argparse.Namespace):
    print(args)

    api_url = f"http://{args.host}:{args.port}/v1/rerank"
    model_uid = args.model_uid

    logger.info("Preparing for benchmark.")
    dataset = load_dataset("mteb/scidocs-reranking")
    input_requests = dataset["test"].remove_columns("negative").to_list()
    if args.num_query > 0:
        input_requests = input_requests[: args.num_query]
    else:
        args.num_query = len(input_requests)

    logger.info("Benchmark starts.")
    benchmark_start_time = time.time()

    benchmark = BenchmarkRunner(
        api_url,
        model_uid,
        input_requests,
        top_n=args.top_n,
        concurrency=args.concurrency,
        api_key=args.api_key,
    )
    asyncio.run(benchmark.run())
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {args.num_query / benchmark_time:.2f} requests/s")
    # TODO(codingl2k1): We should calculate the tokens / s in the future.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test the rerank model.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9997)
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
    parser.add_argument("--model-uid", type=str, help="Xinference model UID.")
    parser.add_argument(
        "--api-key", type=str, default=None, help="Authorization api key",
    )
    args = parser.parse_args()
    main(args)
