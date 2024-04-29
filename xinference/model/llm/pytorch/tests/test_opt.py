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
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

import pytest
import xoscar

from .....client import Client
from .....client.restful.restful_client import RESTfulGenerateModelHandle
from .....core.model import ModelActor
from ... import BUILTIN_LLM_FAMILIES
from ..core import PytorchModel


class MockNonPytorchModel(object):
    def __init__(self):
        self._test_dict = {}

    def generate(self, prompt: str, generate_config=None):
        tid = threading.get_ident()
        self._test_dict[tid] = True
        time.sleep(1)
        self._test_dict.pop(tid, None)
        return len(self._test_dict)


class MockPytorchModel(MockNonPytorchModel, PytorchModel):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("quantization", ["none"])
async def test_opt_pytorch_model(setup, quantization):
    endpoint, _ = setup
    client = Client(endpoint)
    assert len(client.list_models()) == 0

    if quantization == "4-bit":
        with pytest.raises(ValueError):
            client.launch_model(
                model_name="opt",
                model_size_in_billions=1,
                model_format="pytorch",
                quantization=quantization,
                device="cpu",
            )
    else:
        model_uid = client.launch_model(
            model_name="opt",
            model_size_in_billions=1,
            model_format="pytorch",
            quantization=quantization,
            device="cpu",
        )
        assert len(client.list_models()) == 1

        model = client.get_model(model_uid=model_uid)
        assert isinstance(model, RESTfulGenerateModelHandle)

        # Test concurrent generate is OK.
        def _check():
            completion = model.generate(
                "Once upon a time, there was a very old computer"
            )
            assert isinstance(completion, dict)
            assert "text" in completion["choices"][0]

        results = []
        with ThreadPoolExecutor() as executor:
            for _ in range(3):
                r = executor.submit(_check)
                results.append(r)
        for r in results:
            r.result()

        client.terminate_model(model_uid=model_uid)
        assert len(client.list_models()) == 0

        # check for cached revision
        home_address = str(Path.home())
        snapshot_address = (
            home_address
            + "/.cache/huggingface/hub/models--facebook--opt-125m/snapshots"
        )
        actual_revision = os.listdir(snapshot_address)
        model_name = "opt"
        expected_revision: Union[str, None] = ""  # type: ignore

        for family in BUILTIN_LLM_FAMILIES:
            if model_name != family.model_name:
                continue
            for spec in family.model_specs:
                expected_revision = spec.model_revision

        assert [expected_revision] == actual_revision


@pytest.mark.asyncio
async def test_concurrent_pytorch_model(setup):
    pool = await xoscar.create_actor_pool("127.0.0.1", n_process=1)
    async with pool:
        mock_torch_model = MockPytorchModel()
        model_torch_actor = await xoscar.create_actor(
            ModelActor,
            pool.external_address,
            mock_torch_model,
            address=next(iter(pool.sub_processes.keys())),
        )
        coros = []
        for _ in range(3):
            co = model_torch_actor.generate(
                "Once upon a time, there was a very old computer"
            )
            coros.append(co)
        r = await asyncio.gather(*coros)
        assert any(r)

        mock_non_torch_model = MockNonPytorchModel()
        model_non_torch_actor = await xoscar.create_actor(
            ModelActor,
            pool.external_address,
            mock_non_torch_model,
            address=next(iter(pool.sub_processes.keys())),
        )
        coros = []
        for _ in range(3):
            co = model_non_torch_actor.generate(
                "Once upon a time, there was a very old computer"
            )
            coros.append(co)
        r = await asyncio.gather(*coros)
        r = [json.loads(i) for i in r]
        assert not any(r)
