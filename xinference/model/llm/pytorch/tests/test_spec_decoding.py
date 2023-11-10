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

import logging

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..spec_decoding_utils import speculative_generate_stream

logging.basicConfig(level=logging.DEBUG)


@pytest.mark.skip(reason="Temporary disabled")
def test_spec_decoding():
    """
    Use the draft model itself as the target model. If the decoding works, all the draft tokens
    should be accepted, and the result of speculative decoding should be the same as the regular
    decoding, which starts with "The largest animal ever recorded is the Tyrannosaurus Rex".
    """

    model_id = "PY007/TinyLlama-1.1B-Chat-v0.3"
    draft_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = "What is the largest animal?"
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    for completion_chunk, completion_usage in speculative_generate_stream(
        model_uid=model_id,
        draft_model=draft_model,
        model=draft_model,
        tokenizer=tokenizer,
        prompt=formatted_prompt,
        generate_config={"model": "test", "temperature": 0, "max_tokens": 64},
    ):
        pass

    completion = completion_chunk["choices"][0]["text"]
    assert completion.startswith(
        "The largest animal ever recorded is the Tyrannosaurus Rex"
    )
