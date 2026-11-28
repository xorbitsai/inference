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

import asyncio
import base64
import importlib.metadata
import inspect
import json
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from PIL import Image

from .. import load_model_family_from_json
from ..core import create_image_model_instance
from ..engine import DiffusersImageModel, MiniMaxH3DiffusersImageModel
from ..minimax_h3 import MiniMaxH3ImageModel
from .test_flux3 import edit_client  # noqa: F401


@pytest.fixture(params=["huggingface", "modelscope"])
def spec(request):
    families = {}
    load_model_family_from_json(
        str(Path(__file__).parents[1] / "models" / "MiniMax-H3-Image.json"),
        families,
    )
    return next(
        item for item in families["MiniMax-H3-Image"] if item.model_hub == request.param
    )


def test_catalog_and_dispatch(monkeypatch, spec, tmp_path):
    from xoscar.virtualenv.uv import UVVirtualEnvManager

    from ..engine_family import IMAGE_ENGINES

    assert spec.model_id == (
        "MiniMaxAI/MiniMax-H3-Image"
        if spec.model_hub == "huggingface"
        else "MiniMax/MiniMax-H3-Image"
    )
    assert spec.model_revision == (
        "main" if spec.model_hub == "huggingface" else "master"
    )
    assert spec.model_ability == ["text2image", "image2image"]
    assert not spec.default_generate_config
    assert MiniMaxH3DiffusersImageModel.match(spec)
    assert not DiffusersImageModel.match(spec)
    assert "diffusers" in IMAGE_ENGINES[spec.model_name]
    assert "diffusers>=0.43.0" in UVVirtualEnvManager.process_packages(
        spec.virtualenv.packages, engine="diffusers"
    )
    # Only download the modular checkpoint, without a second copy of the native
    # transformer/VAE or the audio/video components of MiniMax-H3.
    assert set(spec.cache_config["allow_patterns"]) == {
        "modular_model_index.json",
        "scheduler/*",
        "transformer_diffusers/*",
        "text_encoder/*",
        "tokenizer/*",
        "processor/*",
        "vae_diffusers/*",
    }
    monkeypatch.setattr(
        MiniMaxH3DiffusersImageModel, "preflight", Mock(), raising=False
    )
    model = create_image_model_instance(
        "minimax-image",
        spec.model_name,
        model_path=str(tmp_path),
        download_hub=spec.model_hub,
        enable_virtual_env=False,
    )
    assert isinstance(model, MiniMaxH3DiffusersImageModel)
    assert model.model_family.model_hub == spec.model_hub
    assert model.model_family.model_engine == "diffusers"


@pytest.fixture
def launch_runtime(monkeypatch, tmp_path):
    from ...cache_manager import CacheManager
    from ..cache_manager import ImageCacheManager
    from ..engine_family import ImageEngineModel

    calls = SimpleNamespace(cache=Mock(return_value=str(tmp_path)), backend=Mock())
    monkeypatch.setattr(ImageCacheManager, "cache", calls.cache)
    monkeypatch.setattr(CacheManager, "cache", calls.cache)
    monkeypatch.setattr(ImageEngineModel, "check_lib", classmethod(lambda cls: True))
    installed_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.43.0" if name == "diffusers" else installed_version(name),
    )
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(ComponentsManager=Mock(), MiniMaxH3ImageModularPipeline=Mock()),
    )
    monkeypatch.setitem(
        sys.modules,
        "diffusers.utils.torch_utils",
        SimpleNamespace(TorchDeviceBackend=calls.backend),
    )
    return calls


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("automatic_device", [False, True])
@pytest.mark.parametrize("enable_virtual_env", [False, True])
def test_factory_rejects_unsupported_offload_before_cache(
    monkeypatch, spec, launch_runtime, device, automatic_device, enable_virtual_env
):
    from .... import device_utils

    calls = launch_runtime
    calls.backend.return_value.mem_get_info.side_effect = NotImplementedError(device)
    monkeypatch.setattr(device_utils, "get_available_device", lambda: device)
    with pytest.raises(ValueError, match="cpu_offload=False"):
        create_image_model_instance(
            "minimax-image",
            spec.model_name,
            download_hub=spec.model_hub,
            enable_virtual_env=enable_virtual_env,
            device=None if automatic_device else device,
        )
    calls.cache.assert_not_called()


def test_factory_rejects_old_diffusers_before_cache(monkeypatch, spec, launch_runtime):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.33.1")
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace())
    with pytest.raises(ImportError, match="diffusers>=0.43.0"):
        create_image_model_instance(
            "minimax-image",
            spec.model_name,
            download_hub=spec.model_hub,
            enable_virtual_env=False,
            cpu_offload=False,
        )
    launch_runtime.cache.assert_not_called()


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_factory_caches_after_successful_preflight(spec, launch_runtime, cpu_offload):
    create_image_model_instance(
        "minimax-image",
        spec.model_name,
        download_hub=spec.model_hub,
        enable_virtual_env=False,
        device="cuda:1",
        cpu_offload=cpu_offload,
    )
    launch_runtime.cache.assert_called_once_with()
    if cpu_offload:
        launch_runtime.backend.assert_called_once_with("cuda:1")
    else:
        launch_runtime.backend.assert_not_called()


@pytest.fixture
def launch_worker(monkeypatch):
    from ....core import worker as worker_module

    manager = SimpleNamespace(env_path="/model-env")
    worker = SimpleNamespace(
        address="127.0.0.1:0",
        _supervisor_address="127.0.0.1:1",
        _event_collector_ref=None,
        _model_uid_to_model={},
        _model_uid_launching_guard={},
        _launch_semaphore=asyncio.Semaphore(1),
        _launch_waiting=0,
        _launch_active=0,
        get_supervisor_ref=AsyncMock(),
        _check_model_is_valid=Mock(),
        get_model_launch_status=Mock(return_value=None),
        _create_virtual_env_manager=Mock(return_value=manager),
        _allocate_subpool_devices=AsyncMock(
            return_value=({"CUDA_VISIBLE_DEVICES": "3"}, ["3"])
        ),
        _prepare_virtual_env=Mock(return_value="fingerprint"),
        _reserve_virtual_env_usage=Mock(),
        _release_virtual_env_usage=Mock(),
        _activate_virtual_env_usage=Mock(),
        release_devices=Mock(),
        _get_progressor=AsyncMock(return_value=Mock()),
        _download_model_files=AsyncMock(),
        _update_model_state=AsyncMock(),
    )
    monkeypatch.setattr(
        worker_module, "build_subpool_envs_for_virtual_env", lambda *args: {}
    )
    monkeypatch.setattr(
        worker_module,
        "resolve_virtualenv_python_path",
        lambda env: "/model-env/bin/python",
    )
    monkeypatch.setattr(
        worker_module.WorkerActor, "_finish_launch_progress", AsyncMock()
    )
    return worker


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_virtual_env", [False, True])
@pytest.mark.parametrize("engine", ["diffusers", "Diffusers"])
async def test_worker_checks_target_environment_before_cache(
    monkeypatch, spec, launch_worker, launch_runtime, enable_virtual_env, engine
):
    import subprocess

    from ....core.worker import WorkerActor

    worker = launch_worker
    monkeypatch.setenv("XINFERENCE_MODEL_SRC", spec.model_hub)
    if not enable_virtual_env:
        worker._create_virtual_env_manager.return_value = None
        monkeypatch.setattr(
            "xinference.core.worker.resolve_virtualenv_python_path", lambda env: None
        )
    probe = Mock(
        return_value=SimpleNamespace(returncode=1, stderr="unsupported target runtime")
    )
    monkeypatch.setattr(subprocess, "run", probe)
    with pytest.raises(ValueError, match="unsupported target runtime"):
        await inspect.unwrap(WorkerActor.launch_builtin_model)(
            worker,
            model_uid="minimax-image-0",
            model_name=spec.model_name,
            model_size_in_billions=None,
            model_format="diffusers",
            quantization="none",
            model_engine=engine,
            model_type="image",
            n_gpu=None,
            download_hub="auto",
            enable_virtual_env=enable_virtual_env,
            virtual_env_packages=["diffusers==0.43.0"],
        )
    command = probe.call_args.args[0]
    assert command[0] == (
        "/model-env/bin/python" if enable_virtual_env else sys.executable
    )
    assert probe.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert json.loads(probe.call_args.kwargs["input"]) == {
        "device": None,
        "cpu_offload": True,
    }
    if enable_virtual_env:
        worker._prepare_virtual_env.assert_called_once()
        assert worker._prepare_virtual_env.call_args.args[2] == ["diffusers==0.43.0"]
    else:
        worker._prepare_virtual_env.assert_not_called()
    worker._download_model_files.assert_not_called()
    launch_runtime.cache.assert_not_called()
    worker.release_devices.assert_called_once_with(model_uid="minimax-image-0")
    worker._release_virtual_env_usage.assert_called_once_with("minimax-image-0")


@pytest.mark.asyncio
async def test_worker_uses_model_runtime_when_worker_runtime_is_missing(
    monkeypatch, spec, launch_worker, launch_runtime
):
    import subprocess

    from ....core.worker import WorkerActor

    worker = launch_worker

    class ReachedSubpool(Exception):
        pass

    worker._spawn_subpool = AsyncMock(side_effect=ReachedSubpool)
    monkeypatch.setattr(
        MiniMaxH3DiffusersImageModel,
        "check_lib",
        classmethod(lambda cls: pytest.fail("Worker dependencies must not be checked")),
    )
    monkeypatch.setattr(
        subprocess, "run", Mock(return_value=SimpleNamespace(returncode=0))
    )

    async def download(**kwargs):
        model = create_image_model_instance(
            kwargs["operation_uid"],
            kwargs["model_name"],
            download_hub=kwargs["download_hub"],
            **kwargs["model_kwargs"],
        )
        return model, SimpleNamespace(cancelled=False)

    worker._download_model_files = download
    with pytest.raises(ReachedSubpool):
        await inspect.unwrap(WorkerActor.launch_builtin_model)(
            worker,
            model_uid="minimax-image-0",
            model_name=spec.model_name,
            model_size_in_billions=None,
            model_format="diffusers",
            quantization="none",
            model_engine="diffusers",
            model_type="image",
            n_gpu=None,
            download_hub=spec.model_hub,
            enable_virtual_env=True,
        )
    launch_runtime.cache.assert_called_once_with()
    worker._prepare_virtual_env.assert_called_once()


@pytest.mark.asyncio
async def test_cancelling_dependency_setup_does_not_cache_weights(
    monkeypatch, spec, launch_worker, launch_runtime
):
    import subprocess

    from ....core.worker import WorkerActor

    worker = launch_worker

    def cancel_setup(*args, **kwargs):
        worker._model_uid_launching_guard["minimax-image-0"].cancel_event.set()
        return "fingerprint"

    worker._prepare_virtual_env.side_effect = cancel_setup
    probe = Mock(side_effect=AssertionError("Launch was cancelled"))
    monkeypatch.setattr(subprocess, "run", probe)
    with pytest.raises(RuntimeError, match="cancelled before runtime preflight"):
        await inspect.unwrap(WorkerActor.launch_builtin_model)(
            worker,
            model_uid="minimax-image-0",
            model_name=spec.model_name,
            model_size_in_billions=None,
            model_format="diffusers",
            quantization="none",
            model_engine="diffusers",
            model_type="image",
            n_gpu=None,
            download_hub=spec.model_hub,
            enable_virtual_env=True,
        )
    probe.assert_not_called()
    worker._download_model_files.assert_not_called()
    launch_runtime.cache.assert_not_called()
    worker._release_virtual_env_usage.assert_called_once_with("minimax-image-0")


@pytest.mark.asyncio
async def test_cache_only_does_not_require_image_runtime(
    monkeypatch, spec, launch_runtime
):
    from ....core.worker import WorkerActor

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.33.1")
    worker = SimpleNamespace(
        address="127.0.0.1:0",
        _check_model_is_valid=Mock(),
        _download_artifact_cleanup_lock=asyncio.Lock(),
        _cache_uid_to_download_info={},
        _launch_semaphore=asyncio.Semaphore(1),
        _get_progressor=AsyncMock(),
    )

    async def download(**kwargs):
        model = create_image_model_instance(
            kwargs["operation_uid"],
            kwargs["model_name"],
            download_hub=kwargs["download_hub"],
            **kwargs["model_kwargs"],
        )
        return model, SimpleNamespace(cancelled=False)

    worker._download_model_files = download
    await inspect.unwrap(WorkerActor.cache_builtin_model)(
        worker,
        cache_uid="minimax-cache",
        model_name=spec.model_name,
        model_size_in_billions=None,
        model_format="diffusers",
        quantization="none",
        model_engine="diffusers",
        model_type="image",
        download_hub=spec.model_hub,
        enable_virtual_env=False,
    )
    launch_runtime.cache.assert_called_once_with()
    launch_runtime.backend.assert_not_called()


@pytest.mark.parametrize("virtualenv_aware", [False, True])
@pytest.mark.parametrize("enable_virtual_env", [False, True])
@pytest.mark.parametrize("version", ["0.33.1", "0.42.0", "0.43.0", "0.44.0"])
def test_engine_discovery_checks_diffusers_version(
    monkeypatch, spec, virtualenv_aware, enable_virtual_env, version
):
    from ...utils import (
        get_engine_params_by_name,
        get_engine_params_by_name_with_virtual_env,
    )
    from ..engine_family import ImageEngineModel

    monkeypatch.setattr(ImageEngineModel, "check_lib", classmethod(lambda cls: True))
    installed_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: version if name == "diffusers" else installed_version(name),
    )
    query = (
        get_engine_params_by_name_with_virtual_env
        if virtualenv_aware
        else get_engine_params_by_name
    )
    result = query("image", spec.model_name, enable_virtual_env=enable_virtual_env)
    needs_upgrade = version in ("0.33.1", "0.42.0")
    if needs_upgrade and not (virtualenv_aware and enable_virtual_env):
        assert isinstance(result["diffusers"], str)
        assert "diffusers>=0.43.0" in result["diffusers"]
        assert version in result["diffusers"]
    else:
        assert result["diffusers"][0]["model_name"] == spec.model_name
        if needs_upgrade:
            assert result["diffusers"][0]["virtualenv_required"] is True
            assert "diffusers>=0.43.0" in result["diffusers"][0]["virtualenv_reason"]
        else:
            assert not result["diffusers"][0].get("virtualenv_required", False)


@pytest.fixture
def loaded_model(monkeypatch, tmp_path, spec):
    from ...cache_manager import CacheManager

    calls = SimpleNamespace(loads=[], generated=[], offloads=[], devices=[], edits=[])
    model_path = tmp_path / "model"
    model_path.mkdir()

    class Pipeline:
        pretrained_component_names = [
            "text_encoder",
            "tokenizer",
            "processor",
            "vae",
            "scheduler",
            "transformer",
        ]
        blocks = SimpleNamespace(
            input_names=[
                "prompt",
                "image",
                "height",
                "width",
                "generator",
                "min_generation_area",
            ]
        )

        @classmethod
        def from_pretrained(cls, path, components_manager):
            assert path == str(model_path)
            return cls()

        def get_component_spec(self, name):
            # ModelScope's index also points to this Hugging Face namespace.
            return SimpleNamespace(
                pretrained_model_name_or_path="MiniMaxAI/MiniMax-H3-Image",
                revision=None,
            )

        def load_components(self, **kwargs):
            calls.loads.append(kwargs)
            assert set(kwargs["pretrained_model_name_or_path"].values()) == {
                str(model_path)
            }
            for name in kwargs.get("names") or self.pretrained_component_names:
                setattr(self, name, object())

        def to(self, device):
            calls.devices.append(device)

        def __call__(self, *, output, **kwargs):
            assert output == "images"
            assert "num_images_per_prompt" not in kwargs
            generator = kwargs.get("generator")
            assert generator is None or isinstance(generator, torch.Generator)
            calls.generated.append(kwargs)
            return [Image.new("RGB", (32, 48))]

    class Manager:
        def enable_auto_cpu_offload(self, **kwargs):
            calls.offloads.append(kwargs)

    def unexpected_download(*args, **kwargs):
        pytest.fail(
            "All MiniMax-H3-Image components must use the selected local snapshot"
        )

    monkeypatch.setattr(CacheManager, "cache", unexpected_download)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            ComponentsManager=Manager, MiniMaxH3ImageModularPipeline=Pipeline
        ),
    )
    model = MiniMaxH3ImageModel(
        "minimax-image", str(model_path), spec, device="cpu", cpu_offload=False
    )
    model.load()
    return model, calls


def test_loading_and_offload(monkeypatch, loaded_model):
    model, calls = loaded_model
    load = calls.loads[0]
    assert load["dtype"] == {"vae": torch.float32, "default": torch.bfloat16}
    assert load["local_files_only"] is True
    assert load["revision"] is None
    assert not calls.offloads
    assert calls.devices == ["cpu"]
    backend = Mock()
    monkeypatch.setitem(
        sys.modules,
        "diffusers.utils.torch_utils",
        SimpleNamespace(TorchDeviceBackend=backend),
    )
    model._device = "cuda:1"
    model._kwargs["cpu_offload"] = True
    model.load()
    backend.assert_called_once_with("cuda:1")
    backend.return_value.mem_get_info.assert_called_once_with()
    assert calls.offloads == [{"device": "cuda:1"}]
    assert calls.devices == ["cpu"]


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("automatic_device", [False, True])
def test_unsupported_offload_fails_before_loading_components(
    monkeypatch, loaded_model, device, automatic_device
):
    from .... import device_utils

    loaded, calls = loaded_model
    backend = Mock()
    backend.return_value.mem_get_info.side_effect = NotImplementedError(
        f"No free-memory query for {device}"
    )
    monkeypatch.setitem(
        sys.modules,
        "diffusers.utils.torch_utils",
        SimpleNamespace(TorchDeviceBackend=backend),
    )
    monkeypatch.setattr(device_utils, "get_available_device", lambda: device)
    from_pretrained = Mock(side_effect=AssertionError("Must check offload first"))
    monkeypatch.setattr(type(loaded._model), "from_pretrained", from_pretrained)
    model = MiniMaxH3ImageModel(
        "minimax-image",
        loaded._model_path,
        loaded.model_family,
        device=None if automatic_device else device,
    )
    with pytest.raises(ValueError, match="cpu_offload=False") as error:
        model.load()
    assert device in str(error.value)
    backend.assert_called_once_with(device)
    backend.return_value.mem_get_info.assert_called_once_with()
    from_pretrained.assert_not_called()
    assert len(calls.loads) == 1
    assert not calls.offloads


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_disabled_offload_skips_memory_query(monkeypatch, loaded_model, device):
    model, calls = loaded_model
    backend = Mock(side_effect=AssertionError("Offload is disabled"))
    monkeypatch.setitem(
        sys.modules,
        "diffusers.utils.torch_utils",
        SimpleNamespace(TorchDeviceBackend=backend),
    )
    model._device = device
    model.load()
    backend.assert_not_called()
    assert calls.devices[-1] == device
    assert not calls.offloads


def test_component_failure_surfaces_at_load(monkeypatch, loaded_model):
    model, _ = loaded_model
    monkeypatch.setattr(type(model._model), "load_components", lambda *a, **kw: None)
    with pytest.raises(
        RuntimeError, match="Failed to load MiniMax-H3-Image components"
    ):
        model.load()


def test_component_quantization_config(loaded_model):
    model, calls = loaded_model
    config = {
        "text_encoder": {"quant_method": "bitsandbytes", "load_in_8bit": True},
        "transformer": {"quant_method": "bitsandbytes", "load_in_8bit": True},
    }
    model._kwargs["quantization_config"] = config
    model.load()
    assert calls.loads[-2]["names"] == ["text_encoder"]
    assert set(calls.loads[-1]["names"]) == {
        "tokenizer",
        "processor",
        "vae",
        "scheduler",
        "transformer",
    }
    assert calls.loads[-1]["quantization_config"] is config
    assert calls.loads[-1]["dtype"]["vae"] == torch.float32


@pytest.mark.parametrize("n,seeds", [(1, [7]), (2, [7, 11])])
def test_generation_count_seeds_and_fixed_sampling(loaded_model, n, seeds):
    model, calls = loaded_model
    progress = []
    result = model.text_to_image(
        "a fox",
        n=n,
        size="768x512",
        seed=seeds,
        progressor=SimpleNamespace(set_progress=progress.append),
        response_format="b64_json",
        min_generation_area=1024,
        num_inference_steps=50,
        guidance_scale=3.5,
        negative_prompt="blurry",
        request_id="request-id",
    )
    assert len(calls.generated) == n
    assert [call["generator"].initial_seed() for call in calls.generated] == seeds
    for call in calls.generated:
        assert (call["width"], call["height"]) == (768, 512)
        assert call["min_generation_area"] == 1024
        assert not set(call) & {
            "num_inference_steps",
            "guidance_scale",
            "negative_prompt",
            "request_id",
        }
    assert progress == [0.0] + [(i + 1) / n for i in range(n)]
    assert len(result["data"]) == n
    assert Image.open(
        BytesIO(base64.b64decode(result["data"][0]["b64_json"]))
    ).size == (32, 48)


@pytest.mark.parametrize("seed", [None, -1, 42])
def test_multiple_images_with_scalar_seed(loaded_model, seed):
    model, calls = loaded_model
    images = model.text_to_image("a fox", n=2, seed=seed, _return_images=True)
    assert len(images) == len(calls.generated) == 2
    if seed == 42:
        # Keep advancing one generator rather than resetting its seed per image.
        assert calls.generated[0]["generator"] is calls.generated[1]["generator"]
        assert calls.generated[0]["generator"].initial_seed() == 42
    else:
        assert all("generator" not in call for call in calls.generated)


def test_sixteen_ordered_references_and_url_output(monkeypatch, loaded_model, tmp_path):
    from .. import utils

    monkeypatch.setattr(utils, "XINFERENCE_IMAGE_DIR", str(tmp_path))
    model, calls = loaded_model
    images = [Image.new("RGB", (16 + i, 32)) for i in range(16)]
    result = model.image_to_image(
        images[0],
        prompt=["combine"],
        reference_images=images[1:],
        response_format="url",
    )
    assert calls.generated[0]["image"] == images
    assert calls.generated[0]["prompt"] == "combine"
    assert "width" not in calls.generated[0]
    assert Path(result["data"][0]["url"]).is_file()


@pytest.mark.parametrize("images", [[], [Image.new("RGB", (16, 16))] * 17, ["invalid"]])
def test_invalid_references(loaded_model, images):
    model, calls = loaded_model
    with pytest.raises((TypeError, ValueError), match="reference images"):
        model.image_to_image(images, prompt="edit")
    assert not calls.generated


@pytest.mark.parametrize(
    "option", ["lora_model", "gguf_model_path", "lightning_model_path"]
)
def test_unsupported_weights(loaded_model, option):
    model, _ = loaded_model
    model._kwargs[option] = "/unsupported"
    with pytest.raises(ValueError, match=option):
        model.load()


@pytest.mark.parametrize(
    "method,size",
    [
        ("image_edit", None),
        ("image_edit", "original"),
        ("image_edit", "1024x768"),
        ("image_to_image", None),
        ("image_to_image", "1024x768"),
    ],
)
@pytest.mark.parametrize("image_count", [1, 2])
def test_public_edit_routes(loaded_model, request, method, size, image_count):
    handle, calls = request.getfixturevalue("edit_client")
    buffer = BytesIO()
    Image.new("RGB", (64, 48)).save(buffer, format="PNG")
    images = [buffer.getvalue()] * image_count
    image_input = images[0] if image_count == 1 else images
    options = {"prompt": "Edit", "n": 2, "response_format": "b64_json", "seed": [7, 11]}
    if size is not None:
        options["size"] = size
    if method == "image_edit":
        result = handle.image_edit(images=image_input, **options)
        assert calls.edits[0]["prompt"] == "Edit"
    else:
        result = handle.image_to_image(image=image_input, **options)
        assert calls.edits[0]["prompt"] == ["Edit"]
    assert len(calls.generated) == 2
    for call in calls.generated:
        assert call["prompt"] == "Edit"
        assert [image.size for image in call["image"]] == [(64, 48)] * image_count
        if size == "1024x768":
            assert (call["width"], call["height"]) == (1024, 768)
        else:
            assert "width" not in call and "height" not in call
    assert [call["generator"].initial_seed() for call in calls.generated] == [7, 11]
    assert len(result["data"]) == 2
    assert all(image["b64_json"] for image in result["data"])


def test_old_diffusers_has_actionable_error(monkeypatch, spec):
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace())
    model = MiniMaxH3ImageModel("minimax-image", "/unused", spec)
    with pytest.raises(ImportError, match="diffusers>=0.43.0"):
        model.load()
