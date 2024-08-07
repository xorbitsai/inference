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
from utils import get_tokenizer, sample_requests
from benchmark_runner import BenchmarkRunner


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LatencyBenchmarkRunner(BenchmarkRunner):
    async def _run(self):
        for request in self.input_requests:
            await self.send_request(request)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/v1/chat/completions"
    model_uid = args.model_uid

    logger.info("Preparing for benchmark.")
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    logger.info("Benchmark starts.")

    benchmark = LatencyBenchmarkRunner(
        api_url,
        model_uid,
        input_requests,
        args.stream,
        args.api_key,
    )
    asyncio.run(benchmark.run())

    benchmark.print_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the latency of processing a single batch of requests."
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9997)
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Name or path of the tokenizer."
    )
    parser.add_argument(
        "--num-prompts", type=int, default=100, help="Number of prompts to process."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface.",
    )
    parser.add_argument("--model-uid", type=str, help="Xinference model UID.")
    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming responses."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Authorization api key",
    )

    args = parser.parse_args()
    main(args)
