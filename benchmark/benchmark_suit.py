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

import json

import argparse
import asyncio
import logging
import random
import time

import numpy as np
import pandas as pd
import datetime
import os
import textwrap

from typing import List, Tuple
from utils import sample_requests, get_tokenizer, send_request

from benchmark_serving import BenchmarkRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoBenchmarkRunner(BenchmarkRunner):


    def __init__(self, args):
        self.inf = args.inf
        self.json_filename:str = args.file
        self.folder:str = args.folder
        self.configs = []

        # Benchmark Result
        self.REQUEST_LATENCY: List[Tuple[int, int, float]] = [] 
        
        self.throughput_request = None
        self.throughput_token = None
        self.avg_latency = None
        self.avg_per_token_latency = None
        self.avg_per_output_token_latency = None
        

        # Benchmark Info
        self.host:str = None
        self.port:int = None
        self.dataset:str = None
        self.trust_remote_code:bool = True
        self.seed:int = None
        self.num_prompts:int = None
        self.concurrency:int = None
        self.request_rate:float = None

        self.api_url:str = None
        self.input_requests = None
        self.queue = None
        self.left = None


        # MUST provide
        self.tokenizer:str = None
        self.model_uid:str = None

    async def worker(self):
        """
        wait request dispatch by run(), and then send_request.
        When all request is done, most worker will hang on self.queue,
        but at least one worker will exit"""
        while self.left > 0:
            prompt, prompt_len, output_len = await self.queue.get()
            await send_request(
                self.api_url,
                self.model_uid,
                prompt,
                prompt_len,
                output_len,
                self.REQUEST_LATENCY,
            )

            self.left -= 1
            # pring longer space to overwrite the previous when left decrease
            print("\rdone_request, left %d    " % (self.left), end="")
        # The last one
        print("")

    def traverse_json_configs(self):
        print(f"Searching Folder{self.folder!r}...")
        for root, dirs, files in os.walk(self.folder):
            for file_name in files:
                if file_name.endswith('.json'):
                    file_path = os.path.join(root, file_name)
                    self.configs.append(file_path)
        print(f"Found file: {self.configs!r}")

    def write_result(self):
        dict = {'Model': self.model_uid, 
                'throughput_request': self.throughput_request , 
                'throughput_token': self.throughput_token, 
                'avg_latency' : self.avg_latency,
                'avg_per_token_latency': self.avg_per_token_latency,
                'avg_per_output_token_latency': self.avg_per_output_token_latency}
        
        dataframe = pd.DataFrame.from_dict(dict, orient='index')
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        csv_filename = f'{self.model_uid}_output_{timestamp}.csv'
        dataframe.to_csv(csv_filename)
    
    def read_file(self, filename:str):
        if filename.endswith('.json'):
            self.configs.append(filename)
        else:
            logger.error("Invalid config file")
    
    def read_json(self):
        if self.inf:
            logger.info("Infinite Benchmark Enabled!")
            # Never Pop() for infinite benchmark
            self.json_filename = self.configs[0]
        else:
            self.json_filename = self.configs.pop()
        
        print(f"Loading file: {self.json_filename!r}")
        with open(self.json_filename, "r") as f:
            data = json.load(f)
        
        print(f"Read from {self.json_filename!r}\n")

        self.host = data.get("host", "localhost")
        self.port = data.get("port", "9997")
        self.dataset:str = data.get("dataset")
        self.trust_remote_code = data.get("trust_remote_code", True)
        self.seed = data.get("seed", 0)
        self.num_prompts = data.get("num_prompt", 100)
        self.concurrency = data.get("concurrency", 100)
        self.request_rate = data.get("request_rate", float("inf"))
        self.tokenizer = data.get("tokenizer")
        self.model_uid = data.get("model_uid")
        self.api_url = f"http://{self.host}:{self.port}/v1/chat/completions"

        print(f"Tokenizer: {self.tokenizer!r}")
        print(f"Model_UID: {self.model_uid!r}")
        
        # Get tokenizer
        self.tokenizer = get_tokenizer(self.tokenizer, trust_remote_code=self.trust_remote_code)
        self.input_requests = sample_requests(self.dataset, self.num_prompts, self.tokenizer)

        self.left = len(self.input_requests)

        # Fix Concurrency
        if self.concurrency > self.num_prompts:
            print("Fix concurrency with num_prompts %d" % (self.num_prompts))
            self.concurrency = self.num_prompts


        self.queue = asyncio.Queue(self.concurrency or 100)



def main(args: argparse.Namespace):
    benchmark = AutoBenchmarkRunner(args)
    if args.file is None and args.folder is not None:
        benchmark.traverse_json_configs()
    elif args.file is not None and args.folder is None:
        benchmark.read_file(args.file)
    elif args.file is None and args.folder is None:
        logger.error("Please provide a folder or a file parameter.")
    else:
        logger.error("Cannot provide both folder and file parameters at the same time.")

    
    while benchmark.configs != []:
        benchmark.read_json()
        logger.info("Preparing for benchmark.")
        
        # Set random seeds
        random.seed(benchmark.seed)
        np.random.seed(benchmark.seed)


        logger.info("Benchmark starts.")
        benchmark_start_time = time.time()

        # Start Benchmark
        asyncio.run(benchmark.run())
        
        benchmark_end_time = time.time()
        benchmark_time = benchmark_end_time - benchmark_start_time

        print(f"Benchmark Time: {benchmark_time!r}")

        print("Generating Result...")

        benchmark.throughput_request = benchmark.num_prompts / benchmark_time
        
        print(f"Benchmark Throughput Request: {benchmark.throughput_request!r}")

        benchmark.throughput_token = (
            sum([output_len for _, output_len, _ in benchmark.REQUEST_LATENCY]) / benchmark_time
        )


        print(f"Benchmark Throughput Token: {benchmark.throughput_token!r}")
        
        benchmark.avg_latency = np.mean([latency for _, _, latency in benchmark.REQUEST_LATENCY])
        
        print(f"Benchmark avg latency: {benchmark.avg_latency!r}")
        
        
        benchmark.avg_per_token_latency = np.mean(
            [
                latency / (prompt_len + output_len)
                for prompt_len, output_len, latency in benchmark.REQUEST_LATENCY
            ]
        )
        
        print(f"Benchmark avg per token latency: {benchmark.avg_per_token_latency!r}")

        benchmark.avg_per_output_token_latency = np.mean(
            [latency / output_len for _, output_len, latency in benchmark.REQUEST_LATENCY]
        )

        print(f"Benchmark avg per token output latency: {benchmark.avg_per_output_token_latency!r}")
        print("Result generation finished\n")
        
        benchmark.write_result()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
        Xinfernece Benchmark Suit
        -------------------------
                                    
        Benchmark Suit for online inference serving.
        ''')
        )
    parser.add_argument('-f', '--file',
                        type=str,
                        help="Set the config file to benchmark"
                        )
    
    parser.add_argument('-F', '--folder',
                        type=str, 
                        help="Set the folder containing test configurations"
                        )
    
    parser.add_argument('-I', '--inf',
                        action='store_true',
                        help='Allow infinite benchmark. Should provide a file argument.'
                        )
    args = parser.parse_args()
    main(args)