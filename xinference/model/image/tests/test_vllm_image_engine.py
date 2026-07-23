# Copyright 2022-2026 XProbe Inc.
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
import dataclasses
import queue
import sys
import types
from typing import Any, Optional
from unittest.mock import patch

import pytest
from PIL import Image

from .. import BUILTIN_IMAGE_MODELS, _install
from ..engine import VLLMImageModel
from ..engine import platform as engine_platform
from ..vllm.core import VLLM_SUPPORTED_IMAGE_MODELS, VLLMDiffusionModel


@pytest.fixture(scope="module", autouse=True)
def setup_builtin_models():
    _install()


def _get_spec(model_name: str):
    return BUILTIN_IMAGE_MODELS[model_name][0]


def test_match_on_linux_with_cuda():
    with patch.object(engine_platform, "system", return_value="Linux"), patch.object(
        sys.modules[VLLMImageModel.__module__], "has_cuda_device", return_value=True
    ):
        for model_name in VLLM_SUPPORTED_IMAGE_MODELS:
            assert VLLMImageModel.match(_get_spec(model_name)) is True
        assert VLLMImageModel.match(_get_spec("sd3-medium")) is False


def test_match_rejects_non_linux():
    with patch.object(engine_platform, "system", return_value="Darwin"), patch.object(
        sys.modules[VLLMImageModel.__module__], "has_cuda_device", return_value=True
    ):
        assert VLLMImageModel.match(_get_spec("Z-Image")) is False


def test_match_rejects_without_cuda():
    with patch.object(engine_platform, "system", return_value="Linux"), patch.object(
        sys.modules[VLLMImageModel.__module__], "has_cuda_device", return_value=False
    ):
        assert VLLMImageModel.match(_get_spec("Z-Image")) is False


def test_abilities_restricted_to_text2image():
    spec = _get_spec("Z-Image")
    original_abilities = list(spec.model_ability)
    assert "image2image" in original_abilities

    model = VLLMDiffusionModel("uid", "/path", model_spec=spec)
    assert model.model_ability == ["text2image"]
    assert model.model_family.model_ability == ["text2image"]
    # the shared builtin spec must stay untouched
    assert spec.model_ability == original_abilities


def test_instance_is_picklable():
    # the worker pickles the constructed model into the actor subprocess;
    # dispatcher locks must not cross that boundary
    import pickle

    spec = _get_spec("Z-Image")
    model = VLLMDiffusionModel("uid", "/path", model_spec=spec)
    restored = pickle.loads(pickle.dumps(model))
    assert restored._submit_lock is not None
    assert restored._generate_lock is not None
    assert restored.model_ability == ["text2image"]


def test_constructor_rejects_unsupported_features():
    spec = _get_spec("Z-Image")
    with pytest.raises(ValueError, match="GGUF"):
        VLLMDiffusionModel("uid", "/path", model_spec=spec, gguf_model_path="/gguf")
    with pytest.raises(ValueError, match="Lightning"):
        VLLMDiffusionModel(
            "uid", "/path", model_spec=spec, lightning_model_path="/lightning"
        )
    with pytest.raises(ValueError, match="LoRA"):
        VLLMDiffusionModel("uid", "/path", model_spec=spec, lora_model=[object()])
    with pytest.raises(ValueError, match="Controlnet"):
        VLLMDiffusionModel("uid", "/path", model_spec=spec, controlnet=("canny",))


@pytest.fixture
def fake_vllm_omni_sampling_params():
    @dataclasses.dataclass
    class FakeSamplingParams:
        negative_prompt: Optional[str] = None
        width: Optional[int] = None
        height: Optional[int] = None
        num_inference_steps: Optional[int] = None
        guidance_scale: float = 1.0
        true_cfg_scale: float = 1.0
        seed: Optional[int] = None
        num_outputs_per_prompt: int = 1

    data_mod = types.ModuleType("vllm_omni.inputs.data")
    data_mod.OmniDiffusionSamplingParams = FakeSamplingParams
    fake_modules = {
        "vllm_omni": types.ModuleType("vllm_omni"),
        "vllm_omni.inputs": types.ModuleType("vllm_omni.inputs"),
        "vllm_omni.inputs.data": data_mod,
    }
    to_restore = {name: sys.modules.get(name) for name in fake_modules}
    sys.modules.update(fake_modules)
    try:
        yield FakeSamplingParams
    finally:
        for name, orig in to_restore.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


def test_build_sampling_params(fake_vllm_omni_sampling_params):
    spec = _get_spec("Z-Image")
    model = VLLMDiffusionModel("uid", "/path", model_spec=spec)

    params = model._build_sampling_params(
        2,
        512,
        768,
        {
            "true_cfg_scale": 4.0,
            "num_inference_steps": 20,
            "seed": 123,
            "save_output": False,  # sglang-only, should be dropped
        },
    )
    assert isinstance(params, fake_vllm_omni_sampling_params)
    assert params.width == 512
    assert params.height == 768
    assert params.num_outputs_per_prompt == 2
    assert params.seed == 123
    assert params.num_inference_steps == 20
    assert params.true_cfg_scale == 4.0
    assert not hasattr(params, "save_output")


@pytest.mark.parametrize("seed", [None, -1])
def test_build_sampling_params_random_seed(fake_vllm_omni_sampling_params, seed):
    spec = _get_spec("Z-Image")
    model = VLLMDiffusionModel("uid", "/path", model_spec=spec)
    params = model._build_sampling_params(1, 512, 512, {"seed": seed})
    assert isinstance(params.seed, int)
    assert params.seed >= 0


def test_extract_images():
    @dataclasses.dataclass
    class FakeOutput:
        images: Any = None
        request_output: Any = None

    image = Image.new("RGB", (8, 8))
    images = VLLMDiffusionModel._extract_images([FakeOutput(images=[image])])
    assert len(images) == 1
    assert images[0].size == (8, 8)

    # images nested inside request_output
    images = VLLMDiffusionModel._extract_images(
        [FakeOutput(request_output=FakeOutput(images=[image, image]))]
    )
    assert len(images) == 2

    with pytest.raises(RuntimeError, match="failed"):
        VLLMDiffusionModel._extract_images([])

    with pytest.raises(RuntimeError, match="no images"):
        VLLMDiffusionModel._extract_images([FakeOutput(images=[])])

    with pytest.raises(RuntimeError, match="Unexpected image type"):
        VLLMDiffusionModel._extract_images([FakeOutput(images=[object()])])


# --- concurrent dispatch ---------------------------------------------------


@dataclasses.dataclass
class _FakeEngineOutputs:
    images: Any = None
    error: Any = None


@dataclasses.dataclass
class _FakeOutputMessage:
    request_id: str
    engine_outputs: Any
    finished: bool
    stage_id: int = 0


@dataclasses.dataclass
class _FakeErrorMessage:
    error: str
    fatal: bool = False
    request_id: Optional[str] = None


class _FakeEngine:
    def __init__(self):
        self.output_queue: "queue.Queue" = queue.Queue()
        self.requests: list = []

    def add_request(
        self,
        request_id,
        prompt,
        sampling_params_list,
        final_stage_id,
        final_output_stage_ids,
    ):
        self.requests.append(
            {
                "request_id": request_id,
                "prompt": prompt,
                "sampling_params_list": sampling_params_list,
                "final_stage_id": final_stage_id,
                "final_output_stage_ids": final_output_stage_ids,
            }
        )

    def try_get_output(self, timeout=0.5):
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class _FakeOmni:
    def __init__(self):
        self.engine = _FakeEngine()


@pytest.fixture
def fake_vllm_omni_messages():
    messages_mod = types.ModuleType("vllm_omni.engine.messages")
    messages_mod.OutputMessage = _FakeOutputMessage
    messages_mod.ErrorMessage = _FakeErrorMessage
    fake_modules = {
        "vllm_omni": types.ModuleType("vllm_omni"),
        "vllm_omni.engine": types.ModuleType("vllm_omni.engine"),
        "vllm_omni.engine.messages": messages_mod,
    }
    to_restore = {name: sys.modules.get(name) for name in fake_modules}
    sys.modules.update(fake_modules)
    try:
        yield
    finally:
        for name, orig in to_restore.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


def _make_concurrent_model():
    model = VLLMDiffusionModel("uid", "/path", model_spec=_get_spec("Z-Image"))
    model._model = _FakeOmni()
    return model, model._model.engine


async def _wait_for_requests(engine, count, timeout=5.0):
    async def _poll():
        while len(engine.requests) < count:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout)


def test_allow_batch_enabled():
    # unlocks ModelActor serialization; concurrency safety comes from the
    # dispatcher in VLLMDiffusionModel, see class comment
    assert VLLMDiffusionModel.allow_batch is True


@pytest.mark.asyncio
async def test_concurrent_requests_route_outputs_by_request_id(
    fake_vllm_omni_messages,
):
    model, engine = _make_concurrent_model()
    tasks = [
        asyncio.create_task(model._submit_and_wait(f"prompt-{i}", object()))
        for i in range(3)
    ]
    await _wait_for_requests(engine, 3)
    # every submission carries its own sampling params and request id
    assert len({r["request_id"] for r in engine.requests}) == 3

    # answer in reverse submission order with interleaved partial outputs
    by_prompt = {r["prompt"]: r["request_id"] for r in engine.requests}
    for i in reversed(range(3)):
        rid = by_prompt[f"prompt-{i}"]
        engine.output_queue.put(
            _FakeOutputMessage(
                request_id=rid,
                engine_outputs=_FakeEngineOutputs(images=[f"partial-{i}"]),
                finished=False,
            )
        )
        engine.output_queue.put(
            _FakeOutputMessage(
                request_id=rid,
                engine_outputs=_FakeEngineOutputs(images=[f"final-{i}"]),
                finished=True,
            )
        )
    results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
    for i, outputs in enumerate(results):
        assert [o.images for o in outputs] == [[f"partial-{i}"], [f"final-{i}"]]
    # dispatcher exits once idle
    thread = model._dispatcher_thread
    if thread is not None:
        thread.join(timeout=5.0)
    assert not model._waiters


@pytest.mark.asyncio
async def test_dispatcher_restarts_after_idle(fake_vllm_omni_messages):
    model, engine = _make_concurrent_model()
    for round_no in range(2):
        task = asyncio.create_task(model._submit_and_wait(f"p{round_no}", object()))
        await _wait_for_requests(engine, round_no + 1)
        rid = engine.requests[-1]["request_id"]
        engine.output_queue.put(
            _FakeOutputMessage(
                request_id=rid,
                engine_outputs=_FakeEngineOutputs(images=["img"]),
                finished=True,
            )
        )
        outputs = await asyncio.wait_for(task, timeout=5.0)
        assert outputs[0].images == ["img"]
        thread = model._dispatcher_thread
        if thread is not None:
            thread.join(timeout=5.0)


@pytest.mark.asyncio
async def test_request_error_fails_only_that_request(fake_vllm_omni_messages):
    model, engine = _make_concurrent_model()
    ok = asyncio.create_task(model._submit_and_wait("ok", object()))
    bad = asyncio.create_task(model._submit_and_wait("bad", object()))
    await _wait_for_requests(engine, 2)
    by_prompt = {r["prompt"]: r["request_id"] for r in engine.requests}
    engine.output_queue.put(
        _FakeOutputMessage(
            request_id=by_prompt["bad"],
            engine_outputs=_FakeEngineOutputs(error="boom"),
            finished=False,
        )
    )
    engine.output_queue.put(
        _FakeOutputMessage(
            request_id=by_prompt["ok"],
            engine_outputs=_FakeEngineOutputs(images=["img"]),
            finished=True,
        )
    )
    with pytest.raises(RuntimeError, match="boom"):
        await asyncio.wait_for(bad, timeout=5.0)
    outputs = await asyncio.wait_for(ok, timeout=5.0)
    assert outputs[0].images == ["img"]


@pytest.mark.asyncio
async def test_fatal_engine_error_fails_all_requests(fake_vllm_omni_messages):
    model, engine = _make_concurrent_model()
    tasks = [
        asyncio.create_task(model._submit_and_wait(f"p{i}", object())) for i in range(2)
    ]
    await _wait_for_requests(engine, 2)
    engine.output_queue.put(_FakeErrorMessage(error="engine died", fatal=True))
    for task in tasks:
        with pytest.raises(RuntimeError, match="engine died"):
            await asyncio.wait_for(task, timeout=5.0)
    assert not model._waiters


@pytest.mark.asyncio
async def test_serial_fallback_without_engine_internals(
    fake_vllm_omni_sampling_params,
):
    @dataclasses.dataclass
    class FakeOutput:
        images: Any = None

    generate_calls = []

    class _LegacyOmni:
        # no `engine` attribute → concurrent dispatch unavailable
        def generate(self, prompt, sampling_params_list):
            # the actor no longer serializes calls, so the fallback itself
            # must hold the model-level lock while Omni.generate runs
            assert model._generate_lock.locked()
            generate_calls.append(prompt)
            return [FakeOutput(images=[Image.new("RGB", (4, 4))])]

    model = VLLMDiffusionModel("uid", "/path", model_spec=_get_spec("Z-Image"))
    model._model = _LegacyOmni()
    assert model._concurrency_available() is False

    result = await model.text_to_image(
        "a cat", n=1, size="64*64", response_format="b64_json"
    )
    assert generate_calls == ["a cat"]
    assert len(result["data"]) == 1
    assert result["data"][0]["b64_json"]


def test_builtin_specs_have_vllm_virtualenv_marker():
    for model_name in VLLM_SUPPORTED_IMAGE_MODELS:
        for spec in BUILTIN_IMAGE_MODELS[model_name]:
            packages = spec.virtualenv.packages if spec.virtualenv else []
            omni = [
                pkg
                for pkg in packages
                if pkg.startswith("vllm-omni") and '#engine# == "vLLM"' in pkg
            ]
            assert omni, f"{model_name} misses vllm-omni virtualenv marker"
            # vllm-omni does not declare its vllm dependency and requires a
            # vllm with the same major.minor version, so the spec must pin a
            # matching vllm alongside it
            vllm = [
                pkg
                for pkg in packages
                if pkg.startswith("vllm==") and '#engine# == "vLLM"' in pkg
            ]
            assert vllm, f"{model_name} misses paired vllm virtualenv pin"
            omni_ver = omni[0].split(";")[0].strip().split("==")[1]
            vllm_ver = vllm[0].split(";")[0].strip().split("==")[1]
            assert omni_ver == vllm_ver, (
                f"{model_name}: vllm-omni ({omni_ver}) and vllm ({vllm_ver}) "
                "pins must share the same version"
            )
