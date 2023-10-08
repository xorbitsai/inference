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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..spec_decoding import speculative_generate_stream
from ..utils import generate_stream

import logging

logging.basicConfig(level=logging.DEBUG)


def test_spec_decoding():
    # model = AutoModelForCausalLM.from_pretrained(
    #     "openlm-research/open_llama_7b_v2",
    #     # load_in_8bit=True,
    #     device_map="auto",
    #     torch_dtype=torch.float16,
    # )
    draft_model = AutoModelForCausalLM.from_pretrained(
        "openlm-research/open_llama_3b_v2",
        # load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b_v2")
    prompt = "Q: What is the largest animal?\nA:"

    print("\nSpeculative decoding:")
    for completion_chunk, completion_usage in speculative_generate_stream(
        draft_model=draft_model,
        model=draft_model,
        tokenizer=tokenizer,
        prompt=prompt,
        device="cuda",
        generate_config={"model": "test", "temperature": 0, "max_tokens": 64},
    ):
        print(completion_chunk["choices"][0]["text"])

    print("\nRegular decoding:")
    for completion_chunk, completion_usage in generate_stream(
        model=draft_model,
        tokenizer=tokenizer,
        prompt=prompt,
        device="cuda",
        generate_config={"model": "test", "temperature": 0}
    ):
        pass
    print(completion_chunk['choices'][0]['text'])

    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # input_ids = input_ids.to("cuda")
    # generation_output = draft_model.generate(
    #     input_ids=input_ids, max_new_tokens=256
    # )
    # print(tokenizer.decode(generation_output[0]))
