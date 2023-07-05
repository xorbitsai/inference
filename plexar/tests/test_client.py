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

from typing import TYPE_CHECKING

import pytest
import xoscar as xo

from ..client import Client

if TYPE_CHECKING:
    from ..core import ModelActor


@pytest.mark.asyncio
async def test_sync_client(setup):
    client = Client(setup.external_address)

    assert len(client.list_models()) == 0

    # TODO: use q2_K
    model_uid = client.launch_model(model_name="wizardlm-v1.0", quantization="q4_0")
    assert len(client.list_models()) == 1

    model_ref: xo.ActorRefType["ModelActor"] = client.get_model(model_uid=model_uid)

    completion = await model_ref.chat("write a poem.")
    assert "content" in completion["choices"][0]["message"]

    client.terminate_model(model_uid=model_uid)
    assert len(client.list_models()) == 0


def test_async_client():
    pass
