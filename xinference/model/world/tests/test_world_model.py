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

import asyncio
import importlib
import sys
from pathlib import Path

import pytest

from ...core import create_model_instance
from ...utils import (
    get_engine_params_by_name,
    get_engine_params_by_name_with_virtual_env,
)
from .. import BUILTIN_WORLD_MODELS
from ..core import (
    create_world_model_instance,
    match_world_model,
    resolve_world_model_engine,
)
from ..engine import PyTorchAstraModel, PyTorchHYWorldPlayModel, PyTorchMatrixGameModel
from ..engine_family import WORLD_ENGINES


@pytest.mark.parametrize(
    ("model_name", "model_class"),
    [
        ("Matrix-Game-3.0-5B", PyTorchMatrixGameModel),
        ("HY-WorldPlay-5B", PyTorchHYWorldPlayModel),
        ("Astra", PyTorchAstraModel),
    ],
)
def test_world_model_engine_registry(model_name, model_class, monkeypatch):
    assert resolve_world_model_engine(model_name) == "PyTorch"
    assert resolve_world_model_engine(model_name, "pytorch") == "PyTorch"
    assert WORLD_ENGINES[model_name]["PyTorch"][0]["world_class"] is model_class
    monkeypatch.setattr(model_class, "check_lib", classmethod(lambda cls: True))
    monkeypatch.setattr(model_class, "check_host", classmethod(lambda cls: True))
    assert get_engine_params_by_name("world", model_name, False) == {
        "PyTorch": [
            {
                "model_name": model_name,
                "model_format": "pytorch",
            }
        ]
    }

    model = create_world_model_instance(
        "world-uid",
        model_name,
        model_path="/unused/model/path",
        model_engine="pytorch",
        enable_virtual_env=False,
    )
    assert isinstance(model, model_class)
    assert model.model_family.model_engine == "PyTorch"


def test_world_model_rejects_unknown_engine():
    with pytest.raises(ValueError, match="cannot be run on engine unknown"):
        create_world_model_instance(
            "world-uid",
            "Matrix-Game-3.0-5B",
            model_path="/unused/model/path",
            model_engine="unknown",
            enable_virtual_env=False,
        )


@pytest.mark.parametrize(
    ("model_name", "model_class"),
    [
        ("Matrix-Game-3.0-5B", PyTorchMatrixGameModel),
        ("HY-WorldPlay-5B", PyTorchHYWorldPlayModel),
        ("Astra", PyTorchAstraModel),
    ],
)
def test_world_engine_can_be_prepared_in_virtualenv(
    model_name, model_class, monkeypatch
):
    monkeypatch.setattr(
        model_class,
        "check_lib",
        classmethod(lambda cls: (False, "torch is not installed")),
    )
    monkeypatch.setattr(model_class, "check_host", classmethod(lambda cls: True))

    engines = get_engine_params_by_name_with_virtual_env(
        "world",
        model_name,
        enable_virtual_env=True,
    )

    assert isinstance(engines["PyTorch"], list)
    assert engines["PyTorch"][0]["virtualenv_required"] is True


def test_astra_does_not_install_unused_controlnet_runtime():
    model_spec = BUILTIN_WORLD_MODELS["Astra"][0]

    # Astra's inference entry point imports the annotator class but never
    # instantiates it.  Installing controlnet-aux would unnecessarily make uv
    # resolve torch from the package index, which breaks when the inherited
    # CUDA torch build has a local version suffix such as ``+cu130``.
    assert model_spec.virtualenv is not None
    assert not any(
        package.startswith("controlnet-aux")
        for package in model_spec.virtualenv.packages
    )


def test_generic_model_factory_preserves_world_engine_selection(monkeypatch):
    monkeypatch.setattr(
        PyTorchMatrixGameModel, "check_host", classmethod(lambda cls: True)
    )
    model = create_model_instance(
        "world-uid",
        "world",
        "Matrix-Game-3.0-5B",
        "pytorch",
        model_path="/unused/model/path",
        enable_virtual_env=False,
    )
    assert isinstance(model, PyTorchMatrixGameModel)
    assert model.model_family.model_engine == "PyTorch"


def test_world_engine_rejects_cpu_only_host_before_virtualenv(monkeypatch):
    reason = "The PyTorch world engine requires an NVIDIA CUDA GPU"
    monkeypatch.setattr(
        PyTorchMatrixGameModel,
        "check_host",
        classmethod(lambda cls: (False, reason)),
    )

    engines = get_engine_params_by_name_with_virtual_env(
        "world", "Matrix-Game-3.0-5B", enable_virtual_env=True
    )

    assert engines["PyTorch"] == reason
    with pytest.raises(ValueError, match="requires an NVIDIA CUDA GPU"):
        create_world_model_instance(
            "world-uid",
            "Matrix-Game-3.0-5B",
            model_path="/unused/model/path",
            model_engine="PyTorch",
            enable_virtual_env=True,
        )


@pytest.mark.parametrize(
    ("model_name", "model_id"),
    [
        ("Matrix-Game-3.0-5B", "Skywork/Matrix-Game-3.0"),
        ("HY-WorldPlay-5B", "Tencent-Hunyuan/HY-WorldPlay"),
        ("Astra", "Xorbits/Astra"),
    ],
)
def test_world_models_have_modelscope_sources(model_name, model_id):
    model_spec = match_world_model(model_name, "modelscope")

    assert model_spec.model_hub == "modelscope"
    assert model_spec.model_id == model_id
    assert model_spec.model_revision == "master"


def test_matrix_game_generation_builds_official_runner_command(tmp_path, monkeypatch):
    from .. import model as world_model_module

    model_spec = BUILTIN_WORLD_MODELS["Matrix-Game-3.0-5B"][0]
    model = PyTorchMatrixGameModel("matrix", "/weights/matrix", model_spec)
    model._code_path = str(tmp_path)
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    output_root = tmp_path / "responses"
    monkeypatch.setattr(world_model_module, "XINFERENCE_WORLD_DIR", str(output_root))
    captured = {}

    def fake_run(command, cwd, env, log_path, request_id=None):
        captured.update(
            command=command,
            cwd=cwd,
            env=env,
            log_path=log_path,
            request_id=request_id,
        )
        output_dir = command[command.index("--output_dir") + 1]
        Path(output_dir, "world.mp4").write_bytes(b"video")

    monkeypatch.setattr(model, "_run_command", fake_run)
    monkeypatch.setattr(model, "_gpu_count", lambda: 2)
    result = model.world_generate(
        "move forward",
        image=str(image_path),
        generation_config={"num_frames": 97},
        model_kwargs={"sample_shift": 4.0},
        request_id="matrix-request",
    )

    command = captured["command"]
    assert command[command.index("--num_iterations") + 1] == "2"
    assert command[command.index("--sample_shift") + 1] == "4.0"
    assert command[command.index("--prompt") + 1] == "move forward"
    assert command[command.index("--ulysses_size") + 1] == "2"
    assert "--dit_fsdp" in command
    assert "--t5_fsdp" in command
    assert captured["request_id"] == "matrix-request"
    assert result["data"][0]["url"] is not None
    assert Path(result["data"][0]["url"]).read_bytes() == b"video"


def test_worldplay_generation_passes_model_specific_kwargs(tmp_path, monkeypatch):
    from .. import model as world_model_module

    model_spec = BUILTIN_WORLD_MODELS["HY-WorldPlay-5B"][0]
    model = PyTorchHYWorldPlayModel("worldplay", "/weights/worldplay", model_spec)
    model._code_path = str(tmp_path)
    model._base_model_path = "/weights/wan"
    output_root = tmp_path / "responses"
    monkeypatch.setattr(world_model_module, "XINFERENCE_WORLD_DIR", str(output_root))
    captured = {}

    def fake_run(command, cwd, env, log_path, request_id=None):
        captured.update(
            command=command,
            cwd=cwd,
            env=env,
            log_path=log_path,
            request_id=request_id,
        )
        output_dir = command[command.index("--out") + 1]
        Path(output_dir, "generated.mp4").write_bytes(b"worldplay")

    monkeypatch.setattr(model, "_run_command", fake_run)
    result = model.world_generate(
        "README.md@example.com",
        model_kwargs={"pose": "d-8", "num_chunk": 2},
        request_id="worldplay-request",
    )

    command = captured["command"]
    assert any(part.endswith("hy_worldplay_runner.py") for part in command)
    assert "--input" not in command
    assert command[command.index("--prompt") + 1] == "README.md@example.com"
    assert command[command.index("--pose") + 1] == "d-8"
    assert command[command.index("--num_chunk") + 1] == "2"
    assert command[command.index("--model_id") + 1] == "/weights/wan"
    assert captured["request_id"] == "worldplay-request"
    assert result["data"][0]["url"] is not None
    assert Path(result["data"][0]["url"]).read_bytes() == b"worldplay"


def test_astra_generation_builds_single_gpu_runner_command(tmp_path, monkeypatch):
    from .. import model as world_model_module

    model_spec = BUILTIN_WORLD_MODELS["Astra"][0]
    model_path = tmp_path / "astra"
    checkpoint = (
        model_path / "models" / "Astra" / "checkpoints" / "diffusion_pytorch_model.ckpt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    model = PyTorchAstraModel("astra", str(model_path), model_spec)
    model._code_path = str(tmp_path)
    model._base_model_path = "/weights/wan-1.3b"
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    output_root = tmp_path / "responses"
    monkeypatch.setattr(world_model_module, "XINFERENCE_WORLD_DIR", str(output_root))
    captured = {}

    progress_updates = []

    class FakeProgressor:
        def set_progress(self, progress, info=None):
            progress_updates.append((progress, info))

    def fake_run(command, cwd, env, log_path, progress_callback=None, request_id=None):
        captured.update(
            command=command,
            cwd=cwd,
            env=env,
            log_path=log_path,
            request_id=request_id,
        )
        assert progress_callback is not None
        for line in (
            "Starting MoE FramePack sliding window generation...\n",
            "Loading initial condition frames...\n",
            "Generation step 1\n",
            "  Denoising step 1/50\n",
            "Generation step 2\n",
            "  Denoising step 41/50\n",
            "Decoding generated video...\n",
            "Saving video to output.mp4 ...\n",
        ):
            progress_callback(line)
        output_path = command[command.index("--output_path") + 1]
        Path(output_path).write_bytes(b"astra")

    monkeypatch.setattr(model, "_run_command", fake_run)
    result = model.world_generate(
        "walk through the garden",
        image=str(image_path),
        generation_config={"total_frames_to_generate": 16},
        model_kwargs={"cam_type": 4, "add_icons": True},
        request_id="astra-request",
        progressor=FakeProgressor(),
    )

    command = captured["command"]
    assert command[:2] == [world_model_module.sys.executable, "scripts/infer_demo.py"]
    assert "torch.distributed.run" not in command
    assert command[command.index("--cam_type") + 1] == "4"
    assert command[command.index("--total_frames_to_generate") + 1] == "16"
    assert command[command.index("--wan_model_path") + 1] == "/weights/wan-1.3b"
    assert command[command.index("--dit_path") + 1] == str(checkpoint)
    assert "--add_icons" in command
    assert captured["request_id"] == "astra-request"
    assert Path(result["data"][0]["url"]).read_bytes() == b"astra"
    assert progress_updates[0] == (0.01, "Starting Astra runner")
    assert (0.92, "Decoding video") in progress_updates
    assert progress_updates[-1] == (0.98, "Saving video")


def test_astra_loads_pinned_wan_base_model(tmp_path, monkeypatch):
    import huggingface_hub

    from .. import model as world_model_module

    model_spec = BUILTIN_WORLD_MODELS["Astra"][0]
    checkpoint = (
        tmp_path / "models" / "Astra" / "checkpoints" / "diffusion_pytorch_model.ckpt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    model = PyTorchAstraModel("astra", str(tmp_path), model_spec)
    captured = {}

    def fake_world_load(self):
        self._code_path = "/code/astra"

    def fake_snapshot_download(model_id, **kwargs):
        captured.update(model_id=model_id, **kwargs)
        return "/weights/wan-1.3b"

    monkeypatch.setattr(world_model_module.WorldModel, "load", fake_world_load)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    model.load()

    assert model._base_model_path == "/weights/wan-1.3b"
    assert captured == {
        "model_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "revision": "37ec512624d61f7aa208f7ea8140a131f93afc9a",
        "allow_patterns": [
            "diffusion_pytorch_model.safetensors",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.1_VAE.pth",
            "google/umt5-xxl/*",
        ],
    }


def test_astra_loads_wan_base_model_from_modelscope(tmp_path, monkeypatch):
    modelscope_snapshot_download = importlib.import_module(
        "modelscope.hub.snapshot_download"
    )

    from .. import model as world_model_module

    model_spec = match_world_model("Astra", "modelscope")
    checkpoint = (
        tmp_path / "models" / "Astra" / "checkpoints" / "diffusion_pytorch_model.ckpt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    model = PyTorchAstraModel("astra", str(tmp_path), model_spec)
    captured = {}

    def fake_world_load(self):
        self._code_path = "/code/astra"

    def fake_snapshot_download(model_id, **kwargs):
        captured.update(model_id=model_id, **kwargs)
        return "/weights/wan-1.3b"

    monkeypatch.setattr(world_model_module.WorldModel, "load", fake_world_load)
    monkeypatch.setattr(
        modelscope_snapshot_download, "snapshot_download", fake_snapshot_download
    )

    model.load()

    assert model._base_model_path == "/weights/wan-1.3b"
    assert captured == {
        "model_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "revision": "master",
        "allow_patterns": [
            "diffusion_pytorch_model.safetensors",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.1_VAE.pth",
            "google/umt5-xxl/*",
        ],
    }


def test_astra_rejects_unsupported_inputs_and_camera_type():
    model_spec = BUILTIN_WORLD_MODELS["Astra"][0]
    model = PyTorchAstraModel("astra", "/weights/astra", model_spec)
    model._code_path = "/unused/code/path"
    model._base_model_path = "/weights/wan-1.3b"

    with pytest.raises(ValueError, match="requires an input image"):
        model.world_generate("move forward")
    with pytest.raises(ValueError, match="does not support video input"):
        model.world_generate("move forward", image="image", video="video")
    with pytest.raises(ValueError, match="cam_type must be between 1 and 7"):
        model.world_generate(
            "move forward", image="image", model_kwargs={"cam_type": 8}
        )


def test_generation_config_and_model_kwargs_must_not_overlap():
    model_spec = BUILTIN_WORLD_MODELS["Matrix-Game-3.0-5B"][0]
    model = PyTorchMatrixGameModel("matrix", "/weights/matrix", model_spec)
    model._code_path = "/unused/code/path"

    with pytest.raises(ValueError, match="duplicate keys: seed"):
        model.world_generate(
            "move forward",
            image="unused",
            generation_config={"seed": 1},
            model_kwargs={"seed": 2},
        )


@pytest.mark.parametrize(
    ("model_name", "model_class", "needs_image"),
    [
        ("Matrix-Game-3.0-5B", PyTorchMatrixGameModel, True),
        ("HY-WorldPlay-5B", PyTorchHYWorldPlayModel, False),
        ("Astra", PyTorchAstraModel, True),
    ],
)
def test_invalid_response_format_is_rejected_before_runner(
    model_name, model_class, needs_image, monkeypatch
):
    model_spec = BUILTIN_WORLD_MODELS[model_name][0]
    model = model_class("world", "/weights/world", model_spec)
    model._code_path = "/unused/code/path"
    if isinstance(model, (PyTorchHYWorldPlayModel, PyTorchAstraModel)):
        model._base_model_path = "/weights/wan"
    monkeypatch.setattr(
        model,
        "_run_command",
        lambda *args, **kwargs: pytest.fail("runner must not be called"),
    )

    with pytest.raises(ValueError, match="Unsupported response_format"):
        model.world_generate(
            "move forward",
            image="unused" if needs_image else None,
            generation_config={"response_format": "base64"},
        )


@pytest.mark.asyncio
async def test_world_runner_is_terminated_on_abort(tmp_path):
    from ...scheduler.core import AbortRequestMessage

    model_spec = BUILTIN_WORLD_MODELS["Matrix-Game-3.0-5B"][0]
    model = PyTorchMatrixGameModel("matrix", "/weights/matrix", model_spec)
    run_task = asyncio.create_task(
        asyncio.to_thread(
            model._run_command,
            [
                sys.executable,
                "-c",
                "import time; print('ready', flush=True); time.sleep(60)",
            ],
            str(tmp_path),
            {},
            str(tmp_path / "runner.log"),
            request_id="abort-me",
        )
    )
    deadline = asyncio.get_running_loop().time() + 10
    while True:
        with model._process_lock:
            if "abort-me" in model._running_processes:
                break
        if run_task.done():
            await run_task
            pytest.fail("runner exited before it was registered")
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail("runner process was not registered within 10 seconds")
        await asyncio.sleep(0.05)

    assert await model.abort_request("abort-me") == AbortRequestMessage.DONE.name
    with pytest.raises(RuntimeError, match="runner exited with code"):
        await asyncio.wait_for(run_task, timeout=5)
    with model._process_lock:
        assert "abort-me" not in model._running_processes
