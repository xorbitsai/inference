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


import pytest

from xinference.client import Client


@pytest.mark.asyncio
async def test_generate(setup):
    endpoint, _ = setup

    client = Client(endpoint)
    model_uid = client.launch_model(
        "facebook/opt-125m",
        model_size_in_billions=1,
        model_format="pytorch",
        device="cpu",
    )
    model = client.get_model(model_uid)

    prompt = "Once upon a time, there was a very old computer."
    completion = model.generate(prompt, generate_config={"max_tokens": 1024})

    assert "text" in completion["choices"][0]
