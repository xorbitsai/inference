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
import json
import logging
import random
from typing import List, Optional, Tuple

import numpy as np
from benchmark_runner import ConcurrentBenchmarkRunner
from utils import get_tokenizer, sample_random_requests, sample_requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _validate_random_range_ratio(value: float, name: str) -> float:
    if not (0.0 <= value < 1.0):
        raise argparse.ArgumentTypeError(f"{name} must be in [0, 1), got {value}.")
    return value


def parse_random_range_ratio(value):
    try:
        return _validate_random_range_ratio(float(value), name="--random-range-ratio")
    except ValueError:
        pass

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "--random-range-ratio must be a float or a JSON object with "
            "'input' and 'output' keys."
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            "--random-range-ratio JSON value must be an object."
        )
    try:
        return {
            "input": _validate_random_range_ratio(
                float(parsed["input"]), name="--random-range-ratio input"
            ),
            "output": _validate_random_range_ratio(
                float(parsed["output"]), name="--random-range-ratio output"
            ),
        }
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "--random-range-ratio JSON object must contain 'input' and 'output'."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "--random-range-ratio input and output values must be numbers."
        ) from exc


class ServingBenchmarkRunner(ConcurrentBenchmarkRunner):
    def __init__(
        self,
        api_url: str,
        model_uid: str,
        input_requests: List[Tuple[str, int, int]],
        stream: bool,
        concurrency: int,
        request_rate: float,
        api_key: Optional[str] = None,
        print_error: bool = False,
        ignore_eos: bool = False,
    ):
        super().__init__(
            api_url,
            model_uid,
            input_requests,
            stream,
            concurrency,
            api_key,
            print_error,
            ignore_eos,
        )
        self.request_rate = request_rate
        self.queue = None  # delay the creation of the queue

    async def _run(self):
        tasks = []

        for _ in range(self.concurrency):
            tasks.append(asyncio.create_task(self.worker()))

        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    async def warm_up(self, num_requests: int = 5):
        if self.queue is None:
            self.queue = asyncio.Queue(len(self.input_requests))

        logger.info(f"Enqueuing {len(self.input_requests)} requests.")
        for req in iter(self.input_requests):
            await self.queue.put(req)
        await super().warm_up(num_requests)

    async def worker(self):
        """
        wait request dispatch by run(), and then send_request.
        When all request is done, most worker will hang on self.queue,
        but at least one worker will exit"""
        while self.left > 0:
            request = await self.queue.get()
            await self.send_request(request)
            self.left -= 1
            print("\rdone_request, left %d    " % (self.left), end="")

            if self.request_rate != float("inf"):
                # If the request rate is infinity, then we don't need to wait.
                # Sample the request interval from the exponential distribution.
                interval = np.random.exponential(1.0 / self.request_rate)
                # The next request will be sent after the interval.
                await asyncio.sleep(interval)
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
    if args.request_rate <= 0.0:
        raise ValueError("--request-rate must be positive.")

    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    if args.dataset_name == "sharegpt":
        if not args.dataset:
            raise ValueError("--dataset is required when --dataset-name=sharegpt.")
        input_requests = sample_requests(
            args.dataset,
            args.num_prompts,
            tokenizer,
            prompt_len_limit=args.prompt_len_limit,
        )
    elif args.dataset_name == "random":
        input_requests = sample_random_requests(
            args.num_prompts,
            tokenizer,
            input_len=args.input_len,
            output_len=args.output_len,
            range_ratio=args.random_range_ratio,
            prefix_len=args.random_prefix_len,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}")

    logger.info("Benchmark starts.")

    benchmark = ServingBenchmarkRunner(
        api_url,
        model_uid,
        input_requests,
        args.stream,
        request_rate=args.request_rate,
        concurrency=args.concurrency,
        api_key=args.api_key,
        print_error=args.print_error,
        ignore_eos=args.ignore_eos,
    )
    asyncio.run(benchmark.run())

    benchmark.print_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9997)
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=["sharegpt", "random"],
        default="sharegpt",
        help="Dataset source to use for benchmark requests.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the dataset. Required when --dataset-name=sharegpt.",
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Name or path of the tokenizer."
    )
    parser.add_argument(
        "--num-prompts", type=int, default=100, help="Number of prompts to process."
    )
    parser.add_argument(
        "--prompt-len-limit", type=int, default=1024, help="Prompt length limitation."
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=1024,
        help="Input token length for random dataset requests.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Output token length for random dataset requests.",
    )
    parser.add_argument(
        "--random-input-len",
        dest="input_len",
        type=int,
        help="Alias for --input-len.",
    )
    parser.add_argument(
        "--random-output-len",
        dest="output_len",
        type=int,
        help="Alias for --output-len.",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=parse_random_range_ratio,
        default=0.0,
        help="Sample random input/output lengths from length * (1 +/- ratio). "
        'Can also be a JSON object like \'{"input": 0.3, "output": 0.5}\'.',
    )
    parser.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="Fixed prefix token length prepended to random dataset prompts.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Authorization api key",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=100,
        help="Set the concurrency of request to send",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
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
        "--ignore-eos",
        action="store_true",
        help="Ask the backend to ignore EOS so random outputs approach max_tokens.",
    )
    parser.add_argument(
        "--print-error",
        action="store_true",
        help="Print detailed error messages if any errors encountered.",
    )
    args = parser.parse_args()
    main(args)
