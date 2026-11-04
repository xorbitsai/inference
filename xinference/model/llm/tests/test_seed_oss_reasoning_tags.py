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

# CI: linux
from ..llm_family import BUILTIN_LLM_FAMILIES
from ..reasoning_parser import ReasoningParser

# The format in ByteDance-Seed/Seed-OSS-36B-Instruct's model card. The tags are
# added tokens with "special": false, so decoding keeps them in the text.
OUTPUT = ["<seed:think>", "Two", " plus two.", "</seed:think>", "It", " is 4."]


def _parser():
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "seed-oss")
    return ReasoningParser(True, family.reasoning_start_tag, family.reasoning_end_tag)


def test_seed_oss_reasoning_is_split_from_the_answer():
    assert _parser().extract_reasoning_content("".join(OUTPUT)) == (
        "Two plus two.",
        "It is 4.",
    )


def test_seed_oss_streamed_reasoning_is_split_from_the_answer():
    parser = _parser()
    previous = ""
    reasoning = content = ""
    for delta in OUTPUT:
        current = previous + delta
        chunk = parser.extract_reasoning_content_streaming(previous, current, delta)
        reasoning += chunk.get("reasoning_content") or ""
        content += chunk.get("content") or ""
        previous = current
    assert (reasoning, content) == ("Two plus two.", "It is 4.")
