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
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import field, make_dataclass
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ....core.exceptions import InvalidModelRequestError
from .. import _install, minimax_h3, native
from ..core import create_video_model_instance, match_diffusion
from ..engine_family import VIDEO_ENGINES
from ..minimax_h3 import SGLangMiniMaxH3Model, VLLMMiniMaxH3Model


@pytest.fixture(autouse=True)
def registry(monkeypatch, tmp_path):
    monkeypatch.setattr(native.platform, "system", lambda: "Linux")
    monkeypatch.setattr(native, "has_cuda_device", lambda: True)
    monkeypatch.setattr(minimax_h3, "XINFERENCE_VIDEO_DIR", str(tmp_path))
    sampling_fields = (
        "prompt",
        "task",
        "conditions",
        "target",
        "num_outputs_per_prompt",
        "save_output",
        "output_path",
        "output_mode",
        "return_frames",
        "seed",
        "num_inference_steps",
        "flow_shift",
        "audio_flow_shift",
    )
    params_cls = make_dataclass(
        "MiniMaxH3SamplingParams",
        [(name, object, field(default=None)) for name in sampling_fields]
        + [("guidance_scale_2", object, field(default=None, init=False))],
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.multimodal_gen.configs.sample.minimax_h3",
        SimpleNamespace(MiniMaxH3SamplingParams=params_cls),
    )
    _install()
    yield
    monkeypatch.undo()
    _install()


@pytest.fixture
def sglang_lightning_bridge(monkeypatch):
    package = "sglang.multimodal_gen"
    context = None
    registered = {}

    @contextmanager
    def set_forward_context(current_timestep, attn_metadata, forward_batch):
        nonlocal context
        previous = context
        context = SimpleNamespace(forward_batch=forward_batch)
        try:
            yield
        finally:
            context = previous

    class Stage:
        def __init__(self, transformer, pipeline):
            self.transformer = transformer

        def forward(self, batch, server_args):
            refinement_scale = (
                context.forward_batch.runtime_lora_scale if context else 1
            )
            with set_forward_context(0, None, batch):
                return refinement_scale, self._maybe_get_bcg_runner(self.transformer)

        def _maybe_get_bcg_runner(self, current_model):
            return "lightning graph"

    class Pipeline:
        pipeline_name = "MiniMaxH3Pipeline"
        pipeline_config_cls = object

        def __init__(self):
            self.stages = []

        def create_pipeline_stages(self, server_args):
            self.add_stage(Stage(None, self))

        def add_stage(self, stage, stage_name=None):
            self.stages.append(stage)

        def get_stage(self, stage_name):
            return next(
                (stage for stage in self.stages if type(stage).__name__ == stage_name),
                None,
            )

    def register_pipeline(pipeline, **kwargs):
        registered.update(pipeline=pipeline, **kwargs)

    modules = {
        "registry": SimpleNamespace(register_pipeline=register_pipeline),
        "runtime.managers.forward_context": SimpleNamespace(
            get_forward_context_or_none=lambda: context,
            set_forward_context=set_forward_context,
        ),
        "runtime.pipelines.minimax_h3_pipeline": SimpleNamespace(
            MiniMaxH3Pipeline=Pipeline
        ),
        "runtime.pipelines_core": SimpleNamespace(Req=SimpleNamespace),
        "runtime.pipelines_core.stages.model_specific_stages.minimax_h3": (
            SimpleNamespace(MiniMaxH3DenoisingStage=Stage)
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, f"{package}.{name}", module)
    params_module = sys.modules[f"{package}.configs.sample.minimax_h3"]
    monkeypatch.setattr(
        params_module.MiniMaxH3SamplingParams,
        "apply_request_extra",
        lambda self, req: None,
        raising=False,
    )
    module_name = f"{minimax_h3.__package__}.sglang_h3_lightning"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    try:
        yield registered
    finally:
        sys.modules.pop(module_name, None)


def make_model(cls, lightning_version=None, **kwargs):
    spec = match_diffusion("MiniMax-H3", "huggingface", cls.engine_name)
    path = (
        "/adapters/"
        + spec.lightning_model_file_name_template.format(
            lightning_version=lightning_version
        )
        if lightning_version
        else None
    )
    return cls(
        "uid",
        "/model",
        spec,
        lightning_version=lightning_version,
        lightning_model_path=path,
        **kwargs,
    )


@pytest.mark.parametrize(
    "engine,cls", [("vLLM", VLLMMiniMaxH3Model), ("SGLang", SGLangMiniMaxH3Model)]
)
def test_registry_factory_and_dependencies(engine, cls):
    assert list(VIDEO_ENGINES["MiniMax-H3"]) == ["diffusers", "vLLM", "SGLang"]
    model = create_video_model_instance(
        "uid",
        "MiniMax-H3",
        download_hub="huggingface",
        model_engine=engine.lower(),
        model_path="/model",
        enable_virtual_env=False,
        lightning_version="4step_v0.1",
        lightning_model_path="/adapters/minimax_h3_fl2v_turbo_4step_v0.1.safetensors",
    )
    assert isinstance(model, cls)
    assert model.allow_batch
    assert model.model_ability == ["text2video", "image2video", "firstlastframe2video"]
    assert model.model_family.model_engine == engine
    spec = match_diffusion("MiniMax-H3", "modelscope", engine)
    assert spec.model_id == "MiniMax/MiniMax-H3"
    assert spec.lightning_model_id == "lightx2v/Minimax-h3-Turbo"
    assert any(engine in p for p in spec.virtualenv.packages)
    if engine == "SGLang":
        assert any(p.startswith("numpy<2.3") for p in spec.virtualenv.packages)
        assert any(p.startswith("pandas<3") for p in spec.virtualenv.packages)


@pytest.mark.parametrize("cls", [SGLangMiniMaxH3Model, VLLMMiniMaxH3Model])
def test_native_h3_drops_reused_diffusers_offload_options(cls):
    model = make_model(cls, group_offload=False, offload_device="cuda", num_gpus=1)
    assert "group_offload" not in model._kwargs
    assert "offload_device" not in model._kwargs
    assert model._kwargs["num_gpus"] == 1


def test_vllm_h3_load_passes_only_native_engine_options(monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda _: "0.32.0")
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.entrypoints.omni",
        SimpleNamespace(
            Omni=lambda **kwargs: calls.append(kwargs)
            or SimpleNamespace(
                engine=SimpleNamespace(collective_rpc=lambda *a, **k: None)
            )
        ),
    )
    model = make_model(
        VLLMMiniMaxH3Model,
        "4step_v1.0_768p_bf16",
        group_offload=False,
        offload_device="cuda",
        quantization_config="fp8",
        num_gpus=1,
    )
    model.load()
    assert len(calls) == 1
    config = calls[0]
    assert {"mode", "group_offload", "offload_device"}.isdisjoint(config)
    assert config["quantization_config"] == "fp8"
    assert config["task_type"] == "fl2va"
    assert config["model"] == "/model"
    assert config["lora_path"] == model._lightning_model_path


@pytest.mark.parametrize("cls", [SGLangMiniMaxH3Model, VLLMMiniMaxH3Model])
@pytest.mark.parametrize(
    "version", ["4step_v0.1", "8step_v1.0_bf16", "4step_v1.0_768p_bf16"]
)
def test_lightning_schedule_defaults_and_overrides(cls, version):
    model = make_model(cls, version)
    expected = model._model_spec.lightning_version_configs[version]
    width, height, duration, config = model._sampling_config(None, {"seed": 12})
    assert (width, height, duration) == (1344, 768, 124 / 24)
    assert config["num_inference_steps"] == expected["num_inference_steps"]
    assert config["flow_shift"] == expected["video_shift"]
    assert config["audio_flow_shift"] == expected["audio_shift"]
    assert config["seed"] == 12
    with pytest.raises(ValueError, match="requires num_inference_steps"):
        model._sampling_config(50, {})
    with pytest.raises(ValueError, match="requires flow_shift"):
        model._sampling_config(None, {"flow_shift": 99})
    with pytest.raises(InvalidModelRequestError, match="requires audio_flow_shift"):
        model._sampling_config(None, {"audio_flow_shift": 99})
    assert model._model_spec.default_generate_config["num_frames"] == 124


@pytest.mark.parametrize(
    "cls,parent,package_version",
    [
        (SGLangMiniMaxH3Model, minimax_h3.SGLangDiffusionModel, "0.5.22"),
        (VLLMMiniMaxH3Model, minimax_h3.VLLMDiffusionModel, "0.32.0"),
    ],
)
@pytest.mark.parametrize(
    "lightning_version", [None, "4step_v0.1", "8step_v1.0_bf16", "4step_v1.0_768p_bf16"]
)
def test_load_configures_the_partition_and_adapter(
    monkeypatch,
    cls,
    parent,
    package_version,
    lightning_version,
    sglang_lightning_bridge,
):
    monkeypatch.setattr("importlib.metadata.version", lambda _: package_version)
    calls = []

    def load(self):
        calls.append(dict(self._kwargs))
        if cls is SGLangMiniMaxH3Model and lightning_version:
            assert os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"].endswith(
                ".sglang_h3_lightning"
            )
            assert sglang_lightning_bridge["overwrite"] is True
        self._model = SimpleNamespace(
            engine=SimpleNamespace(collective_rpc=lambda *a, **k: None)
        )

    monkeypatch.setattr(parent, "load", load)
    model = make_model(cls, lightning_version)
    model.load()
    config = calls[0]
    if lightning_version:
        assert config["lora_path"] == model._lightning_model_path
    else:
        assert "lora_path" not in config
    if cls is SGLangMiniMaxH3Model:
        assert config["model_variant"] == "fl2va"
        assert config["pipeline_class_name"] == "MiniMaxH3Pipeline"
        assert config["backend"] == "sglang"
        if lightning_version:
            expected = model._model_spec.lightning_version_configs[lightning_version]
            assert config["lora_alpha"] == expected["lora_alpha"]
            assert config["lora_merge_mode"] == "dynamic"
            assert (
                model.model_family.to_description()["lightning_inference_steps"]
                == expected["num_inference_steps"]
            )
    else:
        assert config["task_type"] == "fl2va"
        if lightning_version:
            assert config["lora_backend"] == "peft"
    assert "group_offload" not in config
    assert "quantization" not in config


@pytest.mark.parametrize("cls", [SGLangMiniMaxH3Model, VLLMMiniMaxH3Model])
def test_native_request_lightning_changes_schedule_without_changing_default(cls):
    model = make_model(cls, "4step_v1.0_768p_bf16")
    _, _, _, base = model._sampling_config(None, {"use_lightning": False})
    assert base["use_lightning"] is False
    assert (
        base["num_inference_steps"],
        base["flow_shift"],
        base["audio_flow_shift"],
    ) == (
        50,
        12.0,
        3.0,
    )
    _, _, _, lightning = model._sampling_config(None, {})
    assert lightning["use_lightning"] is True
    assert lightning["num_inference_steps"] == 4
    assert lightning["flow_shift"] == 6.0
    with pytest.raises(InvalidModelRequestError, match="must be a boolean"):
        model._sampling_config(None, {"use_lightning": "false"})
    with pytest.raises(InvalidModelRequestError, match="Launch the model"):
        make_model(cls)._sampling_config(None, {"use_lightning": True})


def test_sglang_request_lightning_covers_prompt_refinement_and_graph(
    sglang_lightning_bridge,
):
    bridge = import_module(f"{minimax_h3.__package__}.sglang_h3_lightning")
    pipeline = sglang_lightning_bridge["pipeline"]()
    pipeline.create_pipeline_stages(None)
    stage = pipeline.stages[0]
    for enabled, expected in [
        (True, (1, "lightning graph")),
        (False, (0, None)),
        (True, (1, "lightning graph")),
    ]:
        params = bridge.RequestLightningSamplingParams(use_lightning=enabled)
        request = SimpleNamespace()
        params.apply_request_extra(request)
        assert stage.forward(request, None) == expected
        assert bridge.get_forward_context_or_none() is None
    assert not bridge.RequestLightningSamplingParams.__dataclass_fields__[
        "use_lightning"
    ].metadata.get("batch_sig_exclude")


@pytest.mark.parametrize(
    "cls,old", [(SGLangMiniMaxH3Model, "0.5.20"), (VLLMMiniMaxH3Model, "0.28.0")]
)
def test_old_engine_versions_fail_before_initialization(monkeypatch, cls, old):
    monkeypatch.setattr("importlib.metadata.version", lambda _: old)
    with pytest.raises(ImportError, match="MiniMax-H3 requires"):
        make_model(cls).load()


def test_vllm_rejects_mismatched_engine_versions(monkeypatch):
    monkeypatch.setattr(
        "importlib.metadata.version",
        lambda package: "0.32.0" if package == "vllm-omni" else "0.28.0",
    )
    with pytest.raises(ImportError, match="matching"):
        make_model(VLLMMiniMaxH3Model).load()


@pytest.mark.parametrize(
    "cls,key,version",
    [
        (SGLangMiniMaxH3Model, "model_variant", "0.5.22"),
        (VLLMMiniMaxH3Model, "task_type", "0.32.0"),
    ],
)
def test_load_rejects_unsupported_partition(monkeypatch, cls, key, version):
    monkeypatch.setattr("importlib.metadata.version", lambda _: version)
    with pytest.raises(ValueError, match=f"require {key}=fl2va"):
        make_model(cls, **{key: "ref2va"}).load()


def test_vllm_rejects_renamed_lightning_checkpoint(monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda _: "0.32.0")
    model = make_model(VLLMMiniMaxH3Model, "4step_v0.1")
    model._lightning_model_path = "/adapters/renamed.safetensors"
    with pytest.raises(ValueError, match="requires the published filename"):
        model.load()


@pytest.mark.parametrize("cls", [SGLangMiniMaxH3Model, VLLMMiniMaxH3Model])
@pytest.mark.parametrize(
    "kwargs,field_name",
    [
        ({"width": "960"}, "width and height"),
        ({"height": {}}, "width and height"),
        ({"width": True}, "width and height"),
        ({"height": 768.5}, "width and height"),
        ({"width": 0}, "width and height"),
        ({"num_frames": "124"}, "num_frames"),
        ({"num_frames": False}, "num_frames"),
        ({"num_frames": 124.5}, "num_frames"),
        ({"duration": "5"}, "duration"),
        ({"duration": []}, "duration"),
        ({"duration": True}, "duration"),
        ({"duration": float("nan")}, "duration"),
        ({"duration": float("inf")}, "duration"),
        ({"duration": 3.9}, "duration"),
        ({"duration": 15.1}, "duration"),
    ],
)
def test_invalid_sampling_values_are_request_errors(cls, kwargs, field_name):
    with pytest.raises(InvalidModelRequestError, match=field_name):
        make_model(cls)._sampling_config(None, kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["text", "image", "firstlast"])
@pytest.mark.parametrize("use_lightning", [False, True])
async def test_sglang_preserves_h3_fields_and_audio_file(
    tmp_path, mode, use_lightning, sglang_lightning_bridge
):
    model = make_model(SGLangMiniMaxH3Model, "4step_v1.0_768p_bf16")
    bridge = import_module(f"{minimax_h3.__package__}.sglang_h3_lightning")
    calls = []
    image = Image.new("RGB", (32, 32), "red")
    last = Image.new("RGB", (32, 32), "blue")

    def generate(sampling_params_kwargs):
        params = sampling_params_kwargs
        bridge.RequestLightningSamplingParams(**params)
        calls.append(params)
        assert params["save_output"] is True
        assert params["output_mode"] == "decoded_files"
        assert params["task"] == ("t2va" if mode == "text" else "fl2va")
        assert "num_frames" not in params and "fps" not in params
        for index, condition in enumerate(params["conditions"]):
            path = Path(condition["uri"].removeprefix("file://"))
            with Image.open(path) as frame:
                assert frame.getpixel((0, 0)) == [(255, 0, 0), (0, 0, 255)][index]
        path = Path(params["output_path"]) / "joint.mp4"
        path.write_bytes(b"video-and-stereo-audio")
        return SimpleNamespace(output_file_path=str(path))

    model._model = SimpleNamespace(generate=generate)
    kwargs = dict(
        seed=42,
        width=960,
        height=544,
        num_frames=124,
        response_format="url",
        unsupported_option=True,
        guidance_scale_2=2,
        use_lightning=use_lightning,
    )
    if mode == "text":
        result = await model.text_to_video("prompt", **kwargs)
    elif mode == "image":
        result = await model.image_to_video(image, "prompt", **kwargs)
    else:
        result = await model.firstlastframe_to_video(image, last, "prompt", **kwargs)
    params = calls[0]
    assert params["target"] == dict(
        short_edge=544,
        aspect_ratio="16:9" if mode == "text" else "960:544",
        duration_seconds=124 / 24,
    )
    assert "unsupported_option" not in params and "guidance_scale_2" not in params
    assert params["use_lightning"] is use_lightning
    assert params["num_inference_steps"] == (4 if use_lightning else 50)
    assert params["flow_shift"] == (6 if use_lightning else 12)
    assert params["audio_flow_shift"] == 3
    assert [c["frame_index"] for c in params["conditions"]] == (
        [] if mode == "text" else [0] if mode == "image" else [0, -1]
    )
    assert Path(result["data"][0]["url"]).read_bytes() == b"video-and-stereo-audio"
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "width,height,ratio",
    [
        (1344, 768, "16:9"),
        (544, 960, "9:16"),
        (768, 768, "1:1"),
        (1024, 768, "4:3"),
        (768, 1024, "3:4"),
        (1792, 768, "21:9"),
    ],
)
async def test_sglang_text_uses_named_ratio(width, height, ratio):
    model = make_model(SGLangMiniMaxH3Model)

    def generate(sampling_params_kwargs):
        assert sampling_params_kwargs["target"]["aspect_ratio"] == ratio
        assert sampling_params_kwargs["target"]["short_edge"] == min(width, height)
        path = Path(sampling_params_kwargs["output_path"]) / "joint.mp4"
        path.write_bytes(b"audio-video")
        return SimpleNamespace(output_file_path=str(path))

    model._model = SimpleNamespace(generate=generate)
    await model.text_to_video("prompt", width=width, height=height)


@pytest.mark.asyncio
async def test_sglang_concurrent_requests_keep_modes_and_files_separate(
    tmp_path, sglang_lightning_bridge
):
    model = make_model(SGLangMiniMaxH3Model, "4step_v0.1")
    barrier = threading.Barrier(2, timeout=5)
    directories = []

    def generate(sampling_params_kwargs):
        params = sampling_params_kwargs
        directories.append(params["output_path"])
        barrier.wait()
        assert params["use_lightning"] is (params["prompt"] == "first")
        results = []
        for index in range(params["num_outputs_per_prompt"]):
            path = Path(params["output_path"]) / f"{index}.mp4"
            path.write_bytes(params["prompt"].encode())
            results.append(SimpleNamespace(output_file_path=str(path)))
        return results

    model._model = SimpleNamespace(generate=generate)
    first, second = await asyncio.gather(
        model.text_to_video("first", n=2),
        model.text_to_video("second", use_lightning=False),
    )
    assert [base64.b64decode(item["b64_json"]) for item in first["data"]] == [
        b"first",
        b"first",
    ]
    assert base64.b64decode(second["data"][0]["b64_json"]) == b"second"
    assert len(set(directories)) == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_vllm_reports_worker_steps_before_encoding(monkeypatch, tmp_path):
    from ..minimax_h3_progress import PROGRESS_PATH_KEY

    model = make_model(VLLMMiniMaxH3Model)
    model._progress_enabled = True
    updates = []
    progressor = SimpleNamespace(
        request_id="request", set_progress=lambda value, **kw: updates.append(value)
    )
    monkeypatch.setattr(model, "_concurrency_available", lambda: True)
    monkeypatch.setattr(model, "_build_sampling_params", lambda n, w, h, cfg: cfg)

    async def submit(payload, params):
        path = Path(params["extra_args"][PROGRESS_PATH_KEY])
        assert path.parent.parent == tmp_path
        for value in (0.275, 0.45, 0.625, 0.8):
            path.write_text(str(value))
            await asyncio.sleep(0.3)
        return ["complete"]

    def save(outputs, workdir):
        assert outputs == ["complete"]
        assert max(updates) == 0.95
        path = Path(workdir) / "video.mp4"
        path.write_bytes(b"audio-video")
        return [str(path)]

    monkeypatch.setattr(model, "_submit_and_wait", submit)
    monkeypatch.setattr(model, "_save_outputs", save)
    await model.text_to_video(
        "prompt",
        progressor=progressor,
        extra_args={PROGRESS_PATH_KEY: "/untrusted/path"},
    )
    assert all(value in updates for value in (0.275, 0.45, 0.625, 0.8, 0.95, 1.0))
    assert updates == sorted(updates)
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_cancelled_vllm_progress_poll_does_not_leave_a_task(
    monkeypatch, tmp_path
):
    model = make_model(VLLMMiniMaxH3Model)
    model._progress_enabled = True
    entered, cancelled = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(model, "_concurrency_available", lambda: True)
    monkeypatch.setattr(model, "_build_sampling_params", lambda n, w, h, cfg: cfg)

    async def submit(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(model, "_submit_and_wait", submit)
    progressor = SimpleNamespace(
        request_id="request", set_progress=lambda *a, **k: None
    )
    request = asyncio.create_task(model.text_to_video("prompt", progressor=progressor))
    await asyncio.wait_for(entered.wait(), 5)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert cancelled.is_set()
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["text", "image", "firstlast"])
@pytest.mark.parametrize("aspect_ratio", [None, "9:16"])
@pytest.mark.parametrize("output_layout", ["primary_video", "legacy"])
@pytest.mark.parametrize("use_lightning", [None, False])
async def test_vllm_selects_lightning_per_request_and_muxes_every_sample(
    monkeypatch, tmp_path, mode, aspect_ratio, output_layout, use_lightning
):
    model = make_model(VLLMMiniMaxH3Model, "8step_v1.0_bf16")
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.lora.request",
        SimpleNamespace(LoRARequest=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.lora.utils",
        SimpleNamespace(stable_lora_int_id=lambda path: 123),
    )
    received = []
    muxed = []
    image = Image.new("RGB", (32, 32), "red")
    last = Image.new("RGB", (32, 32), "blue")
    frames = np.ones((2, 32, 32, 3), dtype=np.uint8)
    audio = np.ones((2, 320), dtype=np.float32)

    def mux(video_frames, audio_waveform, **kwargs):
        muxed.append((video_frames, audio_waveform, kwargs))
        return b"muxed-audio-video"

    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.utils.media_utils",
        SimpleNamespace(mux_video_audio_bytes=mux),
    )
    monkeypatch.setattr(model, "_concurrency_available", lambda: True)
    monkeypatch.setattr(
        model,
        "_build_sampling_params",
        lambda n, w, h, cfg: dict(n=n, width=w, height=h, **cfg),
    )

    async def submit(payload, params):
        received.append((payload, params))
        media = {"audio": [audio, audio], "audio_sample_rate": 32000}
        if output_layout == "legacy":
            media["video"] = [frames, frames]
        return [
            SimpleNamespace(
                finished=False,
                images=[],
                multimodal_output={},
                custom_output={"_diffusion_request_lifecycle": "started"},
            ),
            SimpleNamespace(
                finished=True,
                images=[frames, frames] if output_layout == "primary_video" else [],
                multimodal_output=media,
            ),
        ]

    monkeypatch.setattr(model, "_submit_and_wait", submit)
    kwargs = {"extra_args": {"aspect_ratio": aspect_ratio}} if aspect_ratio else {}
    if use_lightning is False:
        kwargs["use_lightning"] = False
    if mode == "text":
        result = await model.text_to_video("prompt", n=2, seed=42, **kwargs)
    elif mode == "image":
        result = await model.image_to_video(image, "prompt", n=2, seed=42, **kwargs)
    else:
        result = await model.firstlastframe_to_video(
            image, last, "prompt", n=2, seed=42, **kwargs
        )
    payload, params = received[0]
    if mode == "text" or aspect_ratio:
        assert params["extra_args"]["aspect_ratio"] == (aspect_ratio or "16:9")
    else:
        assert "aspect_ratio" not in params["extra_args"]
    assert "use_lightning" not in params
    if use_lightning is False:
        assert params["num_inference_steps"] == 50
        assert "lora_request" not in params and "lora_scale" not in params
    else:
        assert params["num_inference_steps"] == 8
        assert params["lora_request"].lora_path == model._lightning_model_path
        assert params["lora_scale"] == 1.0
    assert params["extra_args"]["flow_shift"] == 12
    assert params["extra_args"]["audio_flow_shift"] == 3
    assert params["extra_args"]["task"] == ("t2va" if mode == "text" else "fl2va")
    if mode != "text":
        assert payload["multi_modal_data"]["image"] == (
            [image] if mode == "image" else [image, last]
        )
        assert params["extra_args"]["frame_indices"] == (
            [0] if mode == "image" else [0, -1]
        )
    assert len(muxed) == 2
    assert all(item[2] == dict(fps=24.0, audio_sample_rate=32000) for item in muxed)
    assert [base64.b64decode(item["b64_json"]) for item in result["data"]] == [
        b"muxed-audio-video"
    ] * 2
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["exception", "missing", "count"])
async def test_sglang_failure_cleans_up_request_files(monkeypatch, tmp_path, error):
    model = make_model(SGLangMiniMaxH3Model)

    def generate(sampling_params_kwargs):
        path = Path(sampling_params_kwargs["output_path"]) / "partial.mp4"
        path.write_bytes(b"partial")
        if error == "exception":
            raise RuntimeError("engine failed")
        if error == "missing":
            return SimpleNamespace(output_file_path=None)
        return [SimpleNamespace(output_file_path=str(path))] * 2

    model._model = SimpleNamespace(generate=generate)
    with pytest.raises(RuntimeError):
        await model.image_to_video(Image.new("RGB", (32, 32)), "prompt")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("media", [{}, {"video": [1]}, {"video": [1], "audio": [None]}])
def test_vllm_rejects_missing_audio(tmp_path, monkeypatch, media):
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.utils.media_utils",
        SimpleNamespace(mux_video_audio_bytes=None),
    )
    with pytest.raises(RuntimeError, match="incomplete|no MiniMax-H3 audio"):
        VLLMMiniMaxH3Model._save_outputs(
            [SimpleNamespace(finished=True, multimodal_output=media)], str(tmp_path)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("cancels", [1, 3])
@pytest.mark.parametrize("engine_fails", [False, True])
async def test_cancelled_sglang_keeps_keyframes_until_engine_finishes(
    tmp_path, cancels, engine_fails
):
    model = make_model(SGLangMiniMaxH3Model)
    started, release = threading.Event(), threading.Event()
    references = []

    def generate(sampling_params_kwargs):
        config = sampling_params_kwargs
        references.append(Path(config["conditions"][0]["uri"].removeprefix("file://")))
        started.set()
        assert release.wait(timeout=5)
        assert references[0].is_file()
        if engine_fails:
            raise RuntimeError("engine failed after cancellation")
        path = Path(config["output_path"]) / "joint.mp4"
        path.write_bytes(b"video-and-audio")
        return SimpleNamespace(output_file_path=str(path))

    model._model = SimpleNamespace(generate=generate)
    request = asyncio.create_task(
        model.image_to_video(Image.new("RGB", (32, 32)), "prompt")
    )
    try:
        assert await asyncio.to_thread(started.wait, 5)
        for _ in range(cancels):
            request.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not request.done()
            assert references[0].exists()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("cls", [SGLangMiniMaxH3Model, VLLMMiniMaxH3Model])
def test_h3_base_steps_and_maximum_temporal_bucket(cls):
    model = make_model(cls)
    _, _, duration, config = model._sampling_config(None, {"num_frames": 362})
    assert duration == 15
    assert config["num_inference_steps"] == 50
    _, _, _, config = model._sampling_config(1, {})
    assert config["num_inference_steps"] == 1
    with pytest.raises(ValueError, match="between 4 and 15"):
        model._sampling_config(None, {"num_frames": 400})
