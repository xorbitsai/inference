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

import pytest
import torch

from ....scheduler.request import InferenceRequest
from ..utils import _get_token_from_logits


def _make_request() -> InferenceRequest:
    request = InferenceRequest("prompt", object(), True, "generate", {})
    request.prompt_tokens = [1, 2, 3]
    return request


@pytest.mark.parametrize(
    "temperature,repetition_penalty,top_p,top_k",
    [
        (0.0, 1.0, 1.0, -1),
        (0.0, 1.0, 1.0, 2),
    ],
)
def test_get_token_from_batched_logits(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
):
    logits = torch.zeros((2, 1, 4))
    logits[1, -1, 3] = 10

    token = _get_token_from_logits(
        _make_request(),
        1,
        logits,
        temperature,
        repetition_penalty,
        top_p,
        top_k,
    )

    assert token == 3
