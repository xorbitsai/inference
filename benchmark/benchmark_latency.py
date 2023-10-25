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
from typing import List, Tuple

import numpy as np
from utils import get_tokenizer, sample_requests, send_request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUEST_LATENCY: List[Tuple[int, int, float]] = []


async def benchmark(
    api_url: str,
    model_uid: str,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
) -> None:
    for request in input_requests:
        prompt, prompt_len, output_len = request
        await send_request(
            api_url, model_uid, prompt, prompt_len, output_len, best_of, REQUEST_LATENCY
        )


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/v1"
    model_uid = args.model_uid

    logger.info("Preparing for benchmark.")
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    logger.info("Benchmark starts.")
    benchmark_start_time = time.time()

    asyncio.run(
        benchmark(
            api_url,
            model_uid,
            input_requests,
            args.best_of,
        )
    )

    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {len(REQUEST_LATENCY) / benchmark_time:.2f} requests/s")

    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean(
        [
            latency / (prompt_len + output_len)
            for prompt_len, output_len, latency in REQUEST_LATENCY
        ]
    )
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency in REQUEST_LATENCY]
    )
    print("Average latency per output token: " f"{avg_per_output_token_latency:.2f} s")


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
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and " "returns the best one.",
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

    args = parser.parse_args()
    main(args)
