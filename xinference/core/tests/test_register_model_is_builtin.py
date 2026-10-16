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
from unittest.mock import AsyncMock

import pytest

from ...model.llm.llm_family import CustomLLMFamilyV2, LlamaCppLLMSpecV2
from ..supervisor import SupervisorActor
from ..worker import WorkerActor


def _custom_llm_json() -> bytes:
    # A client-submitted custom LLM registration that also claims
    # "is_builtin": true. Config.extra = "allow" means parse_raw()
    # accepts the field like any other; nothing about the payload
    # itself distinguishes it from a real registration.
    spec = LlamaCppLLMSpecV2(
        model_format="ggufv2",
        model_size_in_billions=2,
        quantization="q4_0",
        model_id="example/TestModel",
        model_hub="huggingface",
        model_revision="123",
        model_file_name_template="TestModel.{quantization}.bin",
    )
    family = CustomLLMFamilyV2(
        version=2,
        model_type="LLM",
        model_name="test_is_builtin_llm",
        model_lang=["en"],
        model_ability=["chat", "generate"],
        model_specs=[spec],
        model_family="glm4-chat",
        chat_template="glm4-chat",
        is_builtin=True,
    )
    return bytes(family.json(), "utf8")


class _DummySupervisor:
    register_model = SupervisorActor.register_model

    def __init__(self):
        self._custom_register_type_to_cls = {
            "LLM": (
                CustomLLMFamilyV2,
                self._register_fn,
                AsyncMock(),
                lambda model_spec: {},
            )
        }
        self._cache_tracker_ref = None
        self.registered_spec = None

    async def get_model_registration(self, model_type, model_name):
        raise ValueError(f"Model {model_name} not found")

    def _register_fn(self, model_spec, persist):
        self.registered_spec = model_spec

    async def _sync_register_model(self, model_type, model, persist, model_name):
        return None


class _DummyWorker:
    register_model = WorkerActor.register_model

    def __init__(self):
        self._custom_register_type_to_cls = {
            "LLM": (
                CustomLLMFamilyV2,
                self._register_fn,
                AsyncMock(),
                lambda model_spec: {},
            )
        }
        self._cache_tracker_ref = None
        self.registered_spec = None

    def _register_fn(self, model_spec, persist):
        self.registered_spec = model_spec


@pytest.mark.asyncio
async def test_supervisor_register_model_resets_client_is_builtin():
    supervisor = _DummySupervisor()
    await supervisor.register_model("LLM", _custom_llm_json(), persist=False)
    assert supervisor.registered_spec is not None
    assert supervisor.registered_spec.is_builtin is False


@pytest.mark.asyncio
async def test_worker_register_model_resets_client_is_builtin():
    worker = _DummyWorker()
    await worker.register_model("LLM", _custom_llm_json(), persist=False)
    assert worker.registered_spec is not None
    assert worker.registered_spec.is_builtin is False
