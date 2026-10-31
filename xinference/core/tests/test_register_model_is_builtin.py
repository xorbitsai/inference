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

from types import SimpleNamespace
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
    control_address = SupervisorActor.control_address
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "actor_cls, service_address",
    [
        (_DummySupervisor, None),
        (_DummySupervisor, "public-supervisor"),
        (_DummyWorker, None),
    ],
)
async def test_custom_registration_tags_versions_for_legacy_tracker(
    actor_cls, service_address
):
    actor = actor_cls()
    actor.address = "node"
    actor._service_address = service_address
    recorded = []

    # Match the tracker RPC signature from before cache metadata was added.
    async def record_model_version(versions, address):
        recorded.append((versions, address))

    actor._cache_tracker_ref = SimpleNamespace(
        record_model_version=record_model_version
    )
    schema, register, unregister, _ = actor._custom_register_type_to_cls["LLM"]
    actor._custom_register_type_to_cls["LLM"] = (
        schema,
        register,
        unregister,
        lambda spec: {
            spec.model_name: [
                {
                    "model_version": "custom",
                    "cache_status": False,
                    "model_file_location": None,
                }
            ]
        },
    )
    await actor.register_model("LLM", _custom_llm_json(), persist=False)
    assert len(recorded) == 1
    versions, address = recorded[0]
    assert address == (service_address or "node")
    assert versions["test_is_builtin_llm"][0]["model_type"] == "LLM"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name, model_hub",
    [
        ("speech_campplus_sv_zh-cn_16k-common", "modelscope"),
        ("speech_campplus_sv_zh_en_16k-common_advanced", "modelscope"),
        ("whisper-tiny", "huggingface"),
    ],
)
async def test_worker_get_builtin_audio_registration(model_name: str, model_hub: str):
    worker = WorkerActor.__new__(WorkerActor)
    registration = await worker.get_model_registration("audio", model_name)

    assert registration is not None
    assert registration.model_name == model_name
    assert registration.model_hub == model_hub


@pytest.mark.asyncio
async def test_worker_listed_builtin_audio_registrations_can_be_described():
    from ...model.audio import BUILTIN_AUDIO_MODELS

    worker = WorkerActor.__new__(WorkerActor)
    registrations = await worker.list_model_registrations("audio")
    builtin_names = {
        registration["model_name"]
        for registration in registrations
        if registration["is_builtin"]
    }
    assert builtin_names == set(BUILTIN_AUDIO_MODELS)
    for model_name in sorted(builtin_names):
        registration = await worker.get_model_registration("audio", model_name)
        assert registration is not None, model_name
        assert registration.model_name == model_name
