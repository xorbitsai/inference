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

import numpy as np

from utils import generate_sorting_prompts, get_tokenizer
from benchmark_runner import ConcurrentBenchmarkRunner


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LongBenchmarkRunner(ConcurrentBenchmarkRunner):
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


def main(args: argparse.Namespace):
    if args.concurrency > args.num_prompts:
        print("Fix concurrency with num_prompts %d" % (args.num_prompts))
        args.concurrency = args.num_prompts
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/v1/chat/completions"
    model_uid = args.model_uid

    logger.info("Preparing for benchmark.")
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    # XXX: generate_sorting_prompts() currently only generate prompts 1/2 to 2/3 of context_length,
    # because tokenizers vary by models, consider improve in the future.
    input_requests = generate_sorting_prompts(
        args.concurrency, args.context_length, args.context_length / 2 - 20, tokenizer
    )

    logger.info("Benchmark starts.")

    benchmark = LongBenchmarkRunner(
        api_url,
        model_uid,
        input_requests,
        args.stream,
        concurrency=args.concurrency,
        api_key=args.api_key,
    )
    asyncio.run(benchmark.run())

    benchmark.print_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput with long context."
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9997)
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Name or path of the tokenizer."
    )
    parser.add_argument(
        "--context-length", type=int, default=32768, help="model context_length."
    )
    parser.add_argument(
        "--num-prompts", type=int, default=16, help="Number of prompts to process."
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=16,
        help="Set the concurrency of request to send",
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming responses."
    )
    args = parser.parse_args()
    main(args)
