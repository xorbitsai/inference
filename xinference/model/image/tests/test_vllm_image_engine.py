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

# CI: linux

import asyncio
import dataclasses
import importlib.metadata
import queue
import sys
import threading
import types
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from PIL import Image

from ....core.exceptions import InvalidModelRequestError
from ....core.replica_load import EngineLoad
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
    with (
        patch.object(engine_platform, "system", return_value="Linux"),
        patch.object(
            sys.modules[VLLMImageModel.__module__], "has_cuda_device", return_value=True
        ),
    ):
        for model_name in VLLM_SUPPORTED_IMAGE_MODELS:
            assert VLLMImageModel.match(_get_spec(model_name)) is True
        assert VLLMImageModel.match(_get_spec("sd3-medium")) is False


def test_match_rejects_non_linux():
    with (
        patch.object(engine_platform, "system", return_value="Darwin"),
        patch.object(
            sys.modules[VLLMImageModel.__module__], "has_cuda_device", return_value=True
        ),
    ):
        assert VLLMImageModel.match(_get_spec("Z-Image")) is False


def test_match_rejects_without_cuda():
    with (
        patch.object(engine_platform, "system", return_value="Linux"),
        patch.object(
            sys.modules[VLLMImageModel.__module__],
            "has_cuda_device",
            return_value=False,
        ),
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


def test_load_lets_omni_detect_the_pipeline(monkeypatch):
    calls = []

    def omni(**kwargs):
        calls.append(kwargs)
        assert "mode" not in kwargs
        return object()

    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.entrypoints.omni",
        types.SimpleNamespace(Omni=omni),
    )
    model = VLLMDiffusionModel(
        "uid", "/path", model_spec=_get_spec("Z-Image"), max_num_seqs=2
    )
    model.load()
    assert calls == [{"model": "/path", "max_num_seqs": 2}]
    assert model._model is not None


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
            "use_lightning": False,
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
    assert not hasattr(params, "use_lightning")
    with pytest.raises(InvalidModelRequestError, match="Launch the model"):
        model._build_sampling_params(2, 512, 768, {"use_lightning": True})


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
    custom_output: Any = None


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


@dataclasses.dataclass
class _LifecycleSamplingParams:
    emit_request_lifecycle: bool = False


@pytest.fixture
def fake_diffusion_request_lifecycle(monkeypatch):
    data = types.ModuleType("vllm_omni.diffusion.data")
    data.is_diffusion_request_started_output = lambda output: (
        (output.custom_output or {}).get("_diffusion_request_lifecycle") == "started"
    )
    params = types.ModuleType("vllm_omni.inputs.data")
    params.OmniDiffusionSamplingParams = _LifecycleSamplingParams
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.data", data)
    monkeypatch.setitem(sys.modules, "vllm_omni.inputs.data", params)


def test_requires_request_lifecycle_for_load_reporting(fake_vllm_omni_messages):
    model, _ = _make_concurrent_model()
    with pytest.raises(ImportError, match="vllm-omni>=0.32.0"):
        model.enable_engine_load_reporting()


@pytest.mark.asyncio
async def test_reports_natively_queued_work_before_completion(
    fake_vllm_omni_messages, fake_diffusion_request_lifecycle
):
    model, engine = _make_concurrent_model()
    model.enable_engine_load_reporting()
    model._model.check_health = lambda: None
    model._model.resolve_sampling_params_list = lambda params: [
        dataclasses.replace(params[0], emit_request_lifecycle=False)
    ]
    engine._running_counter = types.SimpleNamespace(value=0)
    engine._engines_waiting_counter = types.SimpleNamespace(value=0)
    assert model.get_engine_load() == EngineLoad()

    params = _LifecycleSamplingParams()
    tasks = [asyncio.create_task(model._submit_and_wait("running", params))]
    try:
        await _wait_for_requests(engine, 1)
        running_id = engine.requests[0]["request_id"]
        engine._running_counter.value = 1
        started = _FakeEngineOutputs(
            custom_output={"_diffusion_request_lifecycle": "started"}
        )
        model._route_message(_FakeOutputMessage(running_id, started, finished=False))
        assert model.get_engine_load() == EngineLoad(running=1)

        tasks.append(asyncio.create_task(model._submit_and_wait("queued", params)))
        await _wait_for_requests(engine, 2)
        queued_id = engine.requests[1]["request_id"]
        engine._running_counter.value = 2
        load = model.get_engine_load()
        assert (load.waiting, load.running) == (1, 1)
        assert load.backlog_seconds > 0
        assert not any(task.done() for task in tasks)
        assert all(
            request["sampling_params_list"][0].emit_request_lifecycle
            for request in engine.requests
        )
        assert not params.emit_request_lifecycle

        output = _FakeEngineOutputs(images=["image"])
        model._route_message(_FakeOutputMessage(running_id, output, finished=True))
        assert await tasks[0] == [output]
        model._route_message(_FakeOutputMessage(queued_id, started, finished=False))
        assert model.get_engine_load() == EngineLoad(running=1)
        model._route_message(_FakeErrorMessage(error="failed", request_id=queued_id))
        with pytest.raises(RuntimeError, match="failed"):
            await tasks[1]
        assert model.get_engine_load() == EngineLoad()
    finally:
        model._fail_all_waiters(RuntimeError("test cleanup"))
        await asyncio.gather(*tasks, return_exceptions=True)


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
@pytest.mark.parametrize("fatal", [True, False])
async def test_unattributed_engine_error_closes_autoscaled_model(
    fake_vllm_omni_messages, fake_diffusion_request_lifecycle, fatal
):
    class _ClosableOmni(_FakeOmni):
        def __init__(self):
            super().__init__()
            self.closed = False
            self.close_complete = threading.Event()

        def close(self):
            self.closed = True
            self.close_complete.set()

    model = VLLMDiffusionModel("uid", "/path", model_spec=_get_spec("Z-Image"))
    model.enable_engine_load_reporting()
    omni = model._model = _ClosableOmni()
    engine = omni.engine

    params = _LifecycleSamplingParams()
    task = asyncio.create_task(model._submit_and_wait("p", params))
    await _wait_for_requests(engine, 1)
    engine.output_queue.put(_FakeErrorMessage(error="engine died", fatal=fatal))
    with pytest.raises(RuntimeError, match="engine died"):
        await asyncio.wait_for(task, timeout=5.0)

    # Fatal errors reach request futures before background engine cleanup ends.
    assert await asyncio.to_thread(omni.close_complete.wait, 5.0)
    # the model is in a terminal state: the dead engine has been shut down
    # and subsequent requests fail immediately instead of hanging
    assert omni.closed is True
    assert model._model is None
    assert model.get_engine_load() is None
    with pytest.raises(RuntimeError, match="fatal error"):
        await asyncio.wait_for(model._submit_and_wait("p2", params), timeout=5.0)
    with pytest.raises(RuntimeError, match="fatal error"):
        await model.text_to_image("prompt")
    assert not engine.requests[1:]


@pytest.mark.asyncio
async def test_dispatcher_failure_closes_model(fake_vllm_omni_messages):
    model, engine = _make_concurrent_model()

    def _broken_try_get_output(timeout=0.5):
        raise RuntimeError("output queue broken")

    engine.try_get_output = _broken_try_get_output
    task = asyncio.create_task(model._submit_and_wait("p", object()))
    with pytest.raises(RuntimeError, match="output queue broken"):
        await asyncio.wait_for(task, timeout=5.0)
    with pytest.raises(RuntimeError, match="fatal error"):
        await asyncio.wait_for(model._submit_and_wait("p2", object()), timeout=5.0)


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


@pytest.mark.parametrize(
    "hub,revision", [("huggingface", "main"), ("modelscope", "master")]
)
def test_minimax_factory_selects_native_checkpoint(
    monkeypatch, tmp_path, hub, revision
):
    from ..cache_manager import ImageCacheManager
    from ..core import create_image_model_instance
    from ..engine_family import IMAGE_ENGINES, generate_engine_config_by_model_name

    spec = next(
        s for s in BUILTIN_IMAGE_MODELS["MiniMax-H3-Image"] if s.model_hub == hub
    )
    original = spec.dict()
    monkeypatch.setattr(engine_platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        sys.modules[VLLMImageModel.__module__], "has_cuda_device", lambda: True
    )
    engines = {}
    generate_engine_config_by_model_name(spec, target_engines=engines)
    monkeypatch.setitem(IMAGE_ENGINES, spec.model_name, engines[spec.model_name])
    cached = []

    def cache(manager):
        cached.append(manager._model_family)
        assert manager.get_cache_dir() != ImageCacheManager(spec).get_cache_dir()
        return str(tmp_path)

    monkeypatch.setattr(ImageCacheManager, "cache", cache)
    model = create_image_model_instance(
        "minimax-vllm",
        spec.model_name,
        model_engine="vLLM",
        download_hub=hub,
        enable_virtual_env=False,
    )
    assert isinstance(model, VLLMImageModel)
    assert model.model_ability == ["text2image"]
    assert cached[0].model_hub == hub
    assert cached[0].model_revision == revision
    assert cached[0].cache_name == "MiniMax-H3-Image-vllm"
    assert set(cached[0].cache_config["allow_patterns"]) == {
        "model_index.json",
        "pdd_*",
        "transformer/*",
        "text_encoder/*",
        "tokenizer/*",
        "processor/*",
        "vae/*",
    }
    assert model.model_family.model_engine == "vLLM"
    assert spec.dict() == original


@pytest.mark.parametrize(
    "offload,request_flags,expected_flags",
    [
        (None, {}, {"enable_cpu_offload": True}),
        ({"mode": "layer", "components": ["dit", "text_encoder"]}, {}, {}),
        (
            {"mode": "layer", "components": ["dit"]},
            {"cpu_offload": True},
            {"enable_cpu_offload": True},
        ),
        (
            {"mode": "layer", "components": ["dit"]},
            {"enable_cpu_offload": True},
            {"enable_cpu_offload": True},
        ),
    ],
)
def test_minimax_factory_offload_precedence(
    monkeypatch, tmp_path, offload, request_flags, expected_flags
):
    from ..core import create_image_model_instance
    from ..engine_family import IMAGE_ENGINES, generate_engine_config_by_model_name

    spec = _get_spec("MiniMax-H3-Image")
    monkeypatch.setattr(engine_platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        sys.modules[VLLMImageModel.__module__], "has_cuda_device", lambda: True
    )
    engines = {}
    generate_engine_config_by_model_name(spec, target_engines=engines)
    monkeypatch.setitem(IMAGE_ENGINES, spec.model_name, engines[spec.model_name])
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.34.0")
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.entrypoints.omni",
        types.SimpleNamespace(Omni=lambda **kwargs: calls.append(kwargs)),
    )

    model = create_image_model_instance(
        "minimax-vllm",
        spec.model_name,
        model_engine="vLLM",
        model_path=str(tmp_path),
        enable_virtual_env=False,
        diffusion_offload_config=offload,
        **request_flags,
    )
    assert isinstance(model, VLLMImageModel)
    model.load()

    assert calls == [
        {
            "model": str(tmp_path),
            "dtype": "bfloat16",
            "diffusion_offload_config": offload,
            **expected_flags,
        }
    ]
    assert spec.default_model_config["cpu_offload"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cached_engine,distributed",
    [(None, False), ("diffusers", False), ("vLLM", False), ("vLLM", True)],
)
async def test_minimax_detailed_registration_reports_engine_caches(
    monkeypatch, tmp_path, cached_engine, distributed
):
    from functools import partial

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from .... import constants
    from ....api.restful_api import RESTfulAPI
    from ....core.supervisor import SupervisorActor
    from ....core.worker import WorkerActor

    name = "MiniMax-H3-Image"
    cache_root = tmp_path / "worker-cache"
    if cached_engine:
        cache_name = name if cached_engine == "diffusers" else f"{name}-vllm"
        (cache_root / "v2" / cache_name).mkdir(parents=True)

    def worker(root):
        async def registrations(model_type, detailed):
            with monkeypatch.context() as scoped:
                scoped.setattr(constants, "XINFERENCE_CACHE_DIR", str(root))
                return await WorkerActor.list_model_registrations(
                    types.SimpleNamespace(), model_type, detailed
                )

        return types.SimpleNamespace(list_model_registrations=registrations)

    workers = {"cached-worker": worker(cache_root)}
    if distributed:
        workers = {"first-worker": worker(tmp_path / "empty"), **workers}
    supervisor = types.SimpleNamespace(_worker_address_to_worker=workers)
    supervisor.list_model_registrations = partial(
        SupervisorActor.list_model_registrations, supervisor
    )
    api = object.__new__(RESTfulAPI)
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    app = FastAPI()
    app.get("/v1/model_registrations/{model_type}")(api.list_model_registrations)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/v1/model_registrations/image?detailed=true")
    assert response.status_code == 200
    registration = next(item for item in response.json() if item["model_name"] == name)
    assert {
        spec.get("model_engine"): spec["cache_status"]
        for spec in registration["model_specs"]
    } == {
        "diffusers": cached_engine == "diffusers",
        "vLLM": cached_engine == "vLLM",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("hub", ["huggingface", "modelscope"])
@pytest.mark.parametrize("first_engine", ["diffusers", "vLLM"])
@pytest.mark.parametrize("removed_engine", ["diffusers", "vLLM"])
async def test_minimax_cache_lifecycle(
    monkeypatch, tmp_path, hub, first_engine, removed_engine
):
    from fnmatch import fnmatchcase
    from functools import partial
    from importlib import import_module

    import huggingface_hub

    from .... import constants
    from ....core import worker as worker_module
    from ....core.cache_metadata import get_model_version_infos
    from ....core.cache_tracker import CacheTrackerActor
    from ....core.worker import DownloadInfo, WorkerActor
    from ..core import IMAGE_MODEL_DESCRIPTIONS
    from ..engine import MiniMaxH3DiffusersImageModel
    from ..engine_family import IMAGE_ENGINES, generate_engine_config_by_model_name
    from ..ocr.ocr_family import OCR_ENGINES

    name = "MiniMax-H3-Image"
    versions = {"diffusers": name, "vLLM": f"{name}-vllm"}
    shared = {"tokenizer/tokenizer.json", "text_encoder/model.safetensors"}
    unique = {
        "diffusers": {"modular_model_index.json", "transformer_diffusers/weights"},
        "vLLM": {"model_index.json", "pdd_config.json", "transformer/weights"},
    }
    cache_root = tmp_path / "cache"
    snapshot = tmp_path / "hub" / "snapshot"
    for filename in shared | unique["diffusers"] | unique["vLLM"]:
        # The downloader below materializes only requested files into this
        # shared snapshot and leaves files from the first engine in place.
        (snapshot / filename).parent.mkdir(parents=True, exist_ok=True)

    def download(repo_id, revision, allow_patterns, **kwargs):
        assert repo_id == (
            "MiniMaxAI/MiniMax-H3-Image"
            if hub == "huggingface"
            else "MiniMax/MiniMax-H3-Image"
        )
        assert revision == ("main" if hub == "huggingface" else "master")
        for filename in shared | unique["diffusers"] | unique["vLLM"]:
            if any(fnmatchcase(filename, pattern) for pattern in allow_patterns):
                (snapshot / filename).write_text(filename)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(
        import_module("modelscope.hub.snapshot_download"), "snapshot_download", download
    )
    monkeypatch.setattr(constants, "XINFERENCE_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(worker_module, "XINFERENCE_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(engine_platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        sys.modules[VLLMImageModel.__module__], "has_cuda_device", lambda: True
    )
    engine_registry = {}
    generate_engine_config_by_model_name(
        _get_spec(name), target_engines=engine_registry
    )
    monkeypatch.setitem(IMAGE_ENGINES, name, engine_registry[name])
    monkeypatch.setattr(worker_module, "XINFERENCE_LOG_CONSOLE", True)
    monkeypatch.setattr(MiniMaxH3DiffusersImageModel, "preflight", Mock())
    supported = {engine: [{"model_name": name}] for engine in versions}
    monkeypatch.setattr(
        worker_module, "get_engine_params_by_name", Mock(return_value=supported)
    )
    monkeypatch.setattr(
        worker_module,
        "get_engine_params_by_name_with_virtual_env",
        Mock(return_value=supported),
    )

    worker = types.SimpleNamespace(
        address="worker:9978",
        _check_model_is_valid=Mock(),
        _download_artifact_cleanup_lock=asyncio.Lock(),
        _cache_uid_to_download_info={},
        _launch_semaphore=asyncio.Semaphore(1),
        _get_progressor=AsyncMock(return_value=MagicMock()),
        _upload_download_progress=Mock(),
    )
    worker.update_cache_status = partial(WorkerActor.update_cache_status, worker)
    worker._download_model_files = partial(WorkerActor._download_model_files, worker)
    worker.list_deletable_models = partial(WorkerActor.list_deletable_models, worker)

    def register_inventory():
        tracker = CacheTrackerActor()
        with (
            patch.dict(IMAGE_MODEL_DESCRIPTIONS),
            patch.dict(IMAGE_ENGINES),
            patch.dict(OCR_ENGINES),
        ):
            _install()
            tracker.record_model_version(get_model_version_infos(), worker.address)
        worker._cache_tracker_ref = types.SimpleNamespace(
            **{
                method: AsyncMock(side_effect=getattr(tracker, method))
                for method in (
                    "update_cache_status",
                    "list_cached_models",
                    "list_deletable_models",
                    "confirm_and_remove_model",
                )
            }
        )

    async def listed():
        return {
            row["model_version"]: (row["model_engine"], row["path"])
            for row in await WorkerActor.list_cached_models(worker, name)
        }

    register_inventory()
    other_engine = next(engine for engine in versions if engine != first_engine)
    common = dict(
        model_name=name,
        model_size_in_billions=None,
        model_format="diffusers",
        quantization="none",
        model_type="image",
        download_hub=hub,
    )
    # Exercise the cache-only endpoint and the download step of normal launch.
    await WorkerActor.cache_builtin_model(
        worker,
        cache_uid="prefetch",
        model_engine=first_engine,
        enable_virtual_env=False,
        **common,
    )
    expected = {
        versions[first_engine]: (
            [first_engine],
            str(cache_root / "v2" / versions[first_engine]),
        )
    }
    assert await listed() == expected
    register_inventory()
    assert await listed() == expected
    await worker._download_model_files(
        operation_uid="launch",
        request_id="launch",
        download_info=DownloadInfo(),
        progressor=MagicMock(),
        model_engine=other_engine,
        peft_model_config=None,
        model_path=None,
        model_kwargs={"enable_virtual_env": False},
        completion_stage="loading",
        **common,
    )
    expected[versions[other_engine]] = (
        [other_engine],
        str(cache_root / "v2" / versions[other_engine]),
    )
    assert await listed() == expected
    register_inventory()
    assert await listed() == expected

    for engine, version in versions.items():
        folder = cache_root / "v2" / version
        assert {
            path.relative_to(folder).as_posix()
            for path in folder.rglob("*")
            if path.is_file()
        } == shared | unique[engine]

    removed_version = versions[removed_engine]
    retained_engine = next(engine for engine in versions if engine != removed_engine)
    deletable = await worker.list_deletable_models(removed_version)
    assert set(deletable) == {
        expected[removed_version][1],
        *(str(snapshot / filename) for filename in unique[removed_engine]),
    }
    assert await WorkerActor.confirm_and_remove_model(worker, removed_version)
    assert not (cache_root / "v2" / removed_version).exists()
    for filename in unique[removed_engine]:
        assert not (snapshot / filename).exists()
    for filename in shared | unique[retained_engine]:
        assert (
            cache_root / "v2" / versions[retained_engine] / filename
        ).read_text() == filename
    del expected[removed_version]
    assert await listed() == expected
    register_inventory()
    assert await listed() == expected


@pytest.mark.asyncio
async def test_minimax_native_cache_version_launches_vllm():
    from ....core.supervisor import SupervisorActor

    supervisor = types.SimpleNamespace(launch_builtin_model=AsyncMock())
    version = "MiniMax-H3-Image-vllm"
    await SupervisorActor.launch_model_by_version(
        supervisor, "minimax", "image", None, version
    )
    launch = supervisor.launch_builtin_model.call_args.kwargs
    assert launch["model_name"] == "MiniMax-H3-Image"
    assert launch["model_engine"] == "vLLM"
    assert launch["model_version"] == version

    with pytest.raises(ValueError, match="requires the vLLM engine"):
        await SupervisorActor.launch_model_by_version(
            supervisor, "minimax", "image", "diffusers", version
        )
    supervisor.launch_builtin_model.assert_awaited_once()


def test_minimax_virtualenv_dependencies_are_engine_scoped():
    from xoscar.virtualenv.uv import UVVirtualEnvManager

    spec = _get_spec("MiniMax-H3-Image")
    omni = UVVirtualEnvManager.process_packages(spec.virtualenv.packages, engine="vLLM")
    assert "vllm-omni==0.34.*" in omni
    assert "vllm==0.34.*" in omni
    assert not any(p.startswith(("torch", "numpy", "diffusers")) for p in omni)
    diffusers = UVVirtualEnvManager.process_packages(
        spec.virtualenv.packages, engine="diffusers"
    )
    assert "diffusers>=0.43.0" in diffusers
    assert not any(p.startswith("vllm") for p in diffusers)


@pytest.mark.parametrize("version", ["0.32.0", "0.33.0rc1", "0.34.0"])
def test_minimax_load_checks_release_and_maps_offload(monkeypatch, version):
    calls = []
    monkeypatch.setattr(importlib.metadata, "version", lambda _: version)
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.entrypoints.omni",
        types.SimpleNamespace(Omni=lambda **kwargs: calls.append(kwargs)),
    )
    model = VLLMDiffusionModel(
        "minimax-vllm",
        "/native",
        model_spec=_get_spec("MiniMax-H3-Image"),
        torch_dtype="bfloat16",
        cpu_offload=True,
    )
    if version != "0.34.0":
        with pytest.raises(
            ImportError, match="MiniMax-H3-Image requires vllm-omni>=0.34.0"
        ):
            model.load()
        assert not calls
    else:
        model.load()
        assert calls == [
            {"model": "/native", "dtype": "bfloat16", "enable_cpu_offload": True}
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seed,expected", [(42, [42, 43]), ([7, 11], [7, 11]), (None, None), (-1, None)]
)
async def test_minimax_multiple_outputs_use_single_image_requests(
    fake_vllm_omni_sampling_params, seed, expected
):
    calls = []

    class Omni:
        def generate(self, prompt, *, sampling_params_list):
            params = sampling_params_list[0]
            assert params.num_outputs_per_prompt == 1
            assert params.num_inference_steps == 6
            assert params.guidance_scale == 1.0
            calls.append(params)
            return [types.SimpleNamespace(images=[Image.new("RGB", (8, 8))])]

    model = VLLMDiffusionModel(
        "minimax-vllm", "/native", model_spec=_get_spec("MiniMax-H3-Image")
    )
    model._model = Omni()
    result = await model.text_to_image(
        "a teapot", n=2, seed=seed, size="512*512", response_format="b64_json"
    )
    assert len(result["data"]) == len(calls) == 2
    if expected is not None:
        assert [call.seed for call in calls] == expected
