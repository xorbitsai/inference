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


"""
# skip unittest in temporary

import pytest
from click.testing import CliRunner

from ...client import Client
from ..cmdline import model_chat, model_generate


@pytest.mark.asyncio
async def test_generate(setup):
    pool = setup
    address = pool.external_address
    client = Client(address)
    model_uid = client.launch_model("wizardlm-v1.0", quantization="q2_K")
    assert model_uid is not None

    runner = CliRunner()
    result = runner.invoke(
        model_generate,
        [
            "--model-uid",
            model_uid,
            "--prompt",
            "You are a helpful AI assistant. USER: write a poem. ASSISTANT:",
        ],
    )

    assert len(result.stdout) != 0
    assert result.exit_code == 0


async def test_chat(setup):
    pool = setup
    address = pool.external_address
    client = Client(address)
    model_uid = client.launch_model("wizardlm-v1.0", quantization="q4_0")
    assert model_uid is not None

    runner = CliRunner()
    result = runner.invoke(
        model_chat,
        [
            "--model-uid",
            model_uid,
        ],
        input="Write a poem.\nexit\n",
    )

    assert len(result.stdout) != 0
    assert result.exit_code == 0
"""
