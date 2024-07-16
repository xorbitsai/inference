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

import asyncio
import aiohttp
import json
import logging
import random
import time
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import openai
from transformers import AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    **kwargs,
) -> "PreTrainedTokenizerBase":
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    if (
        "llama" in tokenizer_name.lower()
        and kwargs.get("use_fast", True)
        and tokenizer_name != _FAST_LLAMA_TOKENIZER
    ):
        logger.info(
            "For some LLaMA-based models, initializing the fast tokenizer may "
            "take a long time. To eliminate the initialization time, consider "
            f"using '{_FAST_LLAMA_TOKENIZER}' instead of the original "
            "tokenizer."
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, *args, trust_remote_code=trust_remote_code, **kwargs
        )
    except TypeError as e:
        # The LLaMA tokenizer causes a protobuf error in some environments.
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA-based "
            f"model, use '{_FAST_LLAMA_TOKENIZER}' instead of the original "
            "tokenizer."
        )
        raise RuntimeError(err_msg) from e
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if not trust_remote_code and (
            "does not exist or is not currently imported." in str(e)
            or "requires you to execute the tokenizer file" in str(e)
        ):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        logger.warning(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )
    return tokenizer


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: "PreTrainedTokenizerBase",
    prompt_len_limit: int = 1024,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if (
            prompt_len > prompt_len_limit
            or prompt_len + output_len > prompt_len_limit * 2
        ):
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


def generate_sorting_prompts(
    num_prompts: int,
    context_length: int,
    prompt_len_limit: int,
    tokenizer: "PreTrainedTokenizerBase",
) -> List[Tuple[str, int, int]]:
    prompts = []
    for i in range(0, num_prompts):
        random_nums = []
        _prompt_len = 0
        while True:
            r_str = "%s" % random.randint(0, 99)
            r_len = len(r_str) + 1
            if r_len + _prompt_len > prompt_len_limit:
                break
            random_nums.append(r_str)
            _prompt_len += r_len
        prompt = "Sort the numbers:" + ",".join(random_nums)
        prompts.append(prompt)
    prompt_token_ids = tokenizer(prompts).input_ids
    dataset = []
    for i in range(0, len(prompts)):
        prompt_len = len(prompt_token_ids[i])
        dataset.append((prompts[i], prompt_len, context_length - prompt_len))
    return dataset


class BenchmarkRunner:
    def __init__(
        self,
        api_url: str,
        model_uid: str,
        input_requests: List[Tuple[str, int, int]],
    ):
        self.api_url = api_url
        self.model_uid = model_uid
        self.input_requests = input_requests
        self.request_latency = []
        self.benchmark_time = None

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
            await self.send_request(request)
        logger.info("Warm-up completed.")

    async def _run(self):
        pass

    async def send_request(
        self,
        request: tuple,
    ) -> None:
        prompt, prompt_len, output_len = request
        request_start_time = time.time()

        pload = {
            "model": self.model_uid,
            "n": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        }

        headers = {"User-Agent": "Benchmark Client"}

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.api_url, headers=headers, json=pload
            ) as response:
                resp = await response.json()
                if response.status == 200:
                    completion_tokens = resp["usage"]["completion_tokens"]
                    request_end_time = time.time()
                    request_latency = request_end_time - request_start_time
                    self.request_latency.append(
                        (prompt_len, completion_tokens, request_latency)
                    )
                else:
                    logger.error(f"Failed to create chat completion: {resp}")

    def print_stats(self):
        total_time = self.benchmark_time

        # Calculate latencies
        latencies = [latency for _, _, latency in self.request_latency]
        prompt_output_lengths = [prompt_len + output_len for prompt_len, output_len, _ in self.request_latency]

        # Calculate additional latency statistics
        mean_ttft = np.mean(latencies)
        median_ttft = np.median(latencies)
        p99_ttft = np.percentile(latencies, 99)

        mean_tpot = np.mean([latency / output_len for _, output_len, latency in self.request_latency])
        median_tpot = np.median([latency / output_len for _, output_len, latency in self.request_latency])
        p99_tpot = np.percentile([latency / output_len for _, output_len, latency in self.request_latency], 99)

        mean_itl = np.mean([latency / length for latency, length in zip(latencies, prompt_output_lengths)])
        median_itl = np.median([latency / length for latency, length in zip(latencies, prompt_output_lengths)])
        p99_itl = np.percentile([latency / length for latency, length in zip(latencies, prompt_output_lengths)], 99)

        # Print benchmark results
        print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
        print("{:<40} {:<10}".format("Successful requests:", len(self.request_latency)))
        print("{:<40} {:<10.2f}".format("Benchmark duration (s):", total_time))
        print("{:<40} {:<10}".format("Total input tokens:", sum(prompt_len for prompt_len, _, _ in self.request_latency)))
        print("{:<40} {:<10}".format("Total generated tokens:", sum(output_len for _, output_len, _ in self.request_latency)))
        print("{:<40} {:<10.2f}".format("Request throughput (req/s):", len(self.request_latency) / total_time))
        print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):", sum(prompt_len for prompt_len, _, _ in self.request_latency) / total_time))
        print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", sum(output_len for _, output_len, _ in self.request_latency) / total_time))
        
        print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
        print("{:<40} {:<10.2f}".format("Mean TTFT (s):", mean_ttft))
        print("{:<40} {:<10.2f}".format("Median TTFT (s):", median_ttft))
        print("{:<40} {:<10.2f}".format("P99 TTFT (s):", p99_ttft))
        
        print("{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-"))
        print("{:<40} {:<10.2f}".format("Mean TPOT (s):", mean_tpot))
        print("{:<40} {:<10.2f}".format("Median TPOT (s):", median_tpot))
        print("{:<40} {:<10.2f}".format("P99 TPOT (s):", p99_tpot))
        
        print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
        print("{:<40} {:<10.2f}".format("Mean ITL (s):", mean_itl))
        print("{:<40} {:<10.2f}".format("Median ITL (s):", median_itl))
        print("{:<40} {:<10.2f}".format("P99 ITL (s):", p99_itl))
        
        print("=" * 50)


class ConcurrentBenchmarkRunner(BenchmarkRunner):
    def __init__(
        self,
        api_url: str,
        model_uid: str,
        input_requests: List[Tuple[str, int, int]],
        concurrency: int,
    ):
        super().__init__(api_url, model_uid, input_requests)
        self.concurrency = concurrency
        self.left = len(input_requests)

    async def worker(self):
        pass
