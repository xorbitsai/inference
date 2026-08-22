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

import base64
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import PIL.Image
import pytest

from ...utils import flatten_model_src
from ..core import VideoModelFamilyV2
from ..ltx2 import LTX2VideoModel

MODEL_SPEC_DIR = Path(__file__).parents[1] / "models"
MODEL_SPEC_FILES = ("LTX-2.5.json", "LTX-2.5-Dev.json")
OFFICIAL_MODEL_ID = "Lightricks/LTX-2.5-Diffusers"
OFFICIAL_MODEL_REVISION = "a6de4b5354f078db24d9cf4778c14846788aea3d"
OFFICIAL_MODELSCOPE_REVISION = "master"
DIFFUSERS_VERSION = "0.40.0"
DISTILLED_SIGMAS = [
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
]
STAGE_2_SIGMAS = [0.909375, 0.725, 0.421875]


def _raw_ltx_specs() -> List[Dict[str, Any]]:
    specs = []
    for filename in MODEL_SPEC_FILES:
        with (MODEL_SPEC_DIR / filename).open(encoding="utf-8") as model_spec_file:
            specs.extend(json.load(model_spec_file))
    return specs


def _model_spec(
    model_name: str = "LTX-2.5", model_hub: str = "huggingface"
) -> VideoModelFamilyV2:
    raw_spec = next(
        spec for spec in _raw_ltx_specs() if spec["model_name"] == model_name
    )
    flattened_specs = flatten_model_src(raw_spec)
    flattened_spec = next(
        spec for spec in flattened_specs if spec["model_hub"] == model_hub
    )
    return VideoModelFamilyV2(**flattened_spec)


def _model(model_name: str = "LTX-2.5") -> LTX2VideoModel:
    return LTX2VideoModel("ltx-test", "/tmp/official-ltx-2.5", _model_spec(model_name))


def _install_fake_ltx_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in ("diffusers", "diffusers.pipelines", "diffusers.pipelines.ltx2"):
        if module_name not in sys.modules:
            module = types.ModuleType(module_name)
            module.__path__ = []  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, module_name, module)

    utils_module = types.ModuleType("diffusers.pipelines.ltx2.utils")
    setattr(utils_module, "DEFAULT_NEGATIVE_PROMPT", "official negative prompt")
    setattr(utils_module, "DISTILLED_SIGMA_VALUES", DISTILLED_SIGMAS)
    setattr(utils_module, "STAGE_2_DISTILLED_SIGMA_VALUES", STAGE_2_SIGMAS)
    monkeypatch.setitem(sys.modules, "diffusers.pipelines.ltx2.utils", utils_module)


def test_specs_only_reference_official_bf16_sources() -> None:
    specs = _raw_ltx_specs()

    assert {spec["model_name"] for spec in specs} == {
        "LTX-2.5",
        "LTX-2.5-Dev",
    }
    assert {spec["model_variant"] for spec in specs} == {"distilled", "dev"}

    for spec in specs:
        assert set(spec["model_src"]) == {"huggingface", "modelscope"}
        assert spec["model_src"]["huggingface"] == {
            "model_id": OFFICIAL_MODEL_ID,
            "model_revision": OFFICIAL_MODEL_REVISION,
        }
        assert spec["model_src"]["modelscope"] == {
            "model_id": OFFICIAL_MODEL_ID,
            "model_revision": OFFICIAL_MODELSCOPE_REVISION,
        }
        packages = spec["virtualenv"]["packages"]
        assert f"diffusers=={DIFFUSERS_VERSION}" in packages
        assert not any("git+" in package for package in packages)
        serialized = json.dumps(spec).lower()
        assert "gguf" not in serialized
        assert "int8" not in serialized
        assert "nvfp4" not in serialized
        assert "load_in_4bit" not in serialized
        assert (
            "connectors/diffusion_pytorch_model.safetensors"
            in spec["cache_config"]["ignore_patterns"]
        )
        assert (
            "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
            in spec["cache_config"]["ignore_patterns"]
        )

    distilled = next(spec for spec in specs if spec["model_variant"] == "distilled")
    assert (
        "transformer/diffusion_pytorch_model-*-of-00008.safetensors"
        in distilled["cache_config"]["ignore_patterns"]
    )
    assert "latent_upsampler/*" not in distilled["cache_config"]["ignore_patterns"]


def test_factory_dispatches_ltx_and_rejects_quantization_before_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from .. import core
    from ..cache_manager import VideoCacheManager

    spec = _model_spec()
    monkeypatch.setattr(core, "match_diffusion", lambda *args, **kwargs: spec)
    monkeypatch.setattr(
        VideoCacheManager,
        "cache",
        lambda self: "/tmp/official-ltx-2.5",
    )

    model = core.create_video_model_instance("ltx-test", "LTX-2.5")
    assert isinstance(model, LTX2VideoModel)

    monkeypatch.setattr(
        VideoCacheManager,
        "cache",
        lambda self: pytest.fail("rejected options must not start a model download"),
    )
    with pytest.raises(ValueError, match="official BF16"):
        core.create_video_model_instance(
            "ltx-test",
            "LTX-2.5",
            quantization="int8",
        )

    for option, value in (
        ("quantization_config", {"load_in_8bit": True}),
        ("load_in_4bit", True),
        ("load_in_8bit", "true"),
        ("layerwise_cast", True),
        ("variant", "fp16"),
    ):
        with pytest.raises(ValueError, match="official BF16"):
            model_kwargs: Dict[str, Any] = {option: value}
            core.create_video_model_instance("ltx-test", "LTX-2.5", **model_kwargs)

    with pytest.raises(ValueError, match="custom model_path"):
        core.create_video_model_instance(
            "ltx-test", "LTX-2.5", model_path="/tmp/unverified-ltx"
        )


def test_factory_rejects_unofficial_ltx_spec_before_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from .. import core
    from ..cache_manager import VideoCacheManager

    unofficial = _model_spec().copy(
        update={"model_id": "third-party/LTX-2.5", "model_revision": "main"}
    )
    monkeypatch.setattr(core, "match_diffusion", lambda *args, **kwargs: unofficial)
    monkeypatch.setattr(
        VideoCacheManager,
        "cache",
        lambda self: pytest.fail("unofficial sources must not start a model download"),
    )

    with pytest.raises(ValueError, match="official Lightricks"):
        core.create_video_model_instance("ltx-test", "LTX-2.5")


def test_factory_keeps_non_ltx_dispatch_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from .. import core
    from .. import engine_family
    from ..engine import DiffusersVideoEngineModel

    spec = _model_spec().copy(
        update={"model_name": "other-video", "model_family": "Wan"}
    )
    monkeypatch.setattr(core, "match_diffusion", lambda *args, **kwargs: spec)
    monkeypatch.setattr(
        engine_family,
        "check_engine_by_model_name_and_engine",
        lambda *args, **kwargs: DiffusersVideoEngineModel,
    )
    monkeypatch.setattr(
        engine_family,
        "check_engine_by_model_name_and_engine_with_virtual_env",
        lambda *args, **kwargs: DiffusersVideoEngineModel,
    )

    model = core.create_video_model_instance(
        "video-test",
        "other-video",
        model_path="/tmp/other-video",
    )

    assert type(model) is DiffusersVideoEngineModel


def test_distilled_generation_uses_official_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_ltx_modules(monkeypatch)
    model = _model()
    model._model = SimpleNamespace(_execution_device="cpu")

    kwargs, frame_rate, two_stage, progressor = model._prepare_generate_kwargs(
        2, None, {}
    )

    assert kwargs["sigmas"] == DISTILLED_SIGMAS
    assert kwargs["guidance_scale"] == 1.0
    assert kwargs["audio_guidance_scale"] == 1.0
    assert kwargs["stg_scale"] == 0.0
    assert kwargs["audio_stg_scale"] == 0.0
    assert kwargs["modality_scale"] == 1.0
    assert kwargs["audio_modality_scale"] == 1.0
    assert kwargs["guidance_rescale"] == 0.0
    assert kwargs["audio_guidance_rescale"] == 0.0
    assert kwargs["negative_prompt"].endswith("official negative prompt")
    assert "has_subtitles" in kwargs["negative_prompt"]
    assert kwargs["num_videos_per_prompt"] == 2
    assert kwargs["output_type"] == "np"
    assert "num_inference_steps" not in kwargs
    assert frame_rate == 24.0
    assert two_stage is False
    assert progressor is None


def test_string_false_does_not_enable_two_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_ltx_modules(monkeypatch)
    model = _model()
    model._model = SimpleNamespace(_execution_device="cpu")

    _, _, two_stage, _ = model._prepare_generate_kwargs(1, None, {"two_stage": "false"})

    assert two_stage is False


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"num_inference_steps": 8}, "num_inference_steps"),
        ({"sigmas": [1.0, 0.5]}, "official fixed schedule"),
        ({"num_frames": 120}, "num_frames % 8"),
        ({"height": 545}, "divisible by 32"),
        ({"fps": True}, "frame_rate must be at least 1"),
        ({"fps": 0.5}, "frame_rate must be at least 1"),
        ({"fps": 23.976}, "frame_rate must be an integer"),
        ({"guidance_scale": 2.0}, "requires guidance_scale=1.0"),
        ({"two_stage": True}, "divisible by 64"),
    ],
)
def test_distilled_rejects_incompatible_generation_options(
    monkeypatch: pytest.MonkeyPatch,
    overrides: Dict[str, Any],
    message: str,
) -> None:
    _install_fake_ltx_modules(monkeypatch)
    model = _model()
    model._model = SimpleNamespace(_execution_device="cpu")

    with pytest.raises(ValueError, match=message):
        model._prepare_generate_kwargs(1, None, overrides)


def test_dev_keeps_guided_scheduler_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ltx_modules(monkeypatch)
    model = _model("LTX-2.5-Dev")
    model._model = SimpleNamespace(_execution_device="cpu")

    kwargs, _, _, _ = model._prepare_generate_kwargs(1, None, {})

    assert kwargs["num_inference_steps"] == 30
    assert kwargs["guidance_scale"] == 3.0
    assert kwargs["audio_guidance_scale"] == 7.0
    assert "sigmas" not in kwargs

    with pytest.raises(ValueError, match="positive integer"):
        model._prepare_generate_kwargs(1, 0, {})
    with pytest.raises(ValueError, match="cannot preserve audio_stg_scale=0"):
        model._prepare_generate_kwargs(1, None, {"audio_stg_scale": 0.0})


def test_two_stage_runs_half_resolution_then_refines_target_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_ltx_modules(monkeypatch)
    model = _model()
    calls: List[Dict[str, Any]] = []

    class FakePipeline:
        def __init__(self) -> None:
            self.vae = SimpleNamespace(tiling_enabled=False)
            self.vae.enable_tiling = lambda: setattr(self.vae, "tiling_enabled", True)

        def __call__(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            if kwargs["output_type"] == "latent":
                return "video-latents", "audio-latents"
            return "final-output"

    class FakeUpsampler:
        def to(self, device: str) -> None:
            raise AssertionError("CPU tests must not move the upsampler")

    class FakeUpsamplePipeline:
        latent_upsampler = FakeUpsampler()

        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        def __call__(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return ("upscaled-video-latents",)

    upsample_pipeline = FakeUpsamplePipeline()
    model._model = FakePipeline()
    model._device = "cpu"
    monkeypatch.setattr(
        model, "_get_latent_upsample_pipeline", lambda: upsample_pipeline
    )

    generate_kwargs, _, two_stage, progressor = model._prepare_generate_kwargs(
        1,
        None,
        {"width": 1920, "height": 1088, "two_stage": True},
    )
    generate_kwargs["prompt"] = "A cinematic tracking shot"
    result = model._run_two_stage(["condition"], generate_kwargs, progressor)

    assert two_stage is True
    assert result == "final-output"
    assert model._model.vae.tiling_enabled is True
    assert calls[0]["conditions"] == ["condition"]
    assert calls[0]["width"] == 960
    assert calls[0]["height"] == 544
    assert calls[0]["output_type"] == "latent"
    assert upsample_pipeline.calls == [
        {
            "latents": "video-latents",
            "height": 544,
            "width": 960,
            "num_frames": 121,
            "output_type": "latent",
            "return_dict": False,
        }
    ]
    assert calls[1]["width"] == 1920
    assert calls[1]["height"] == 1088
    assert calls[1]["latents"] == "upscaled-video-latents"
    assert calls[1]["audio_latents"] == "audio-latents"
    assert calls[1]["sigmas"] == STAGE_2_SIGMAS
    assert calls[1]["noise_scale"] == STAGE_2_SIGMAS[0]
    assert calls[1]["conditions"] == ["condition"]


def test_image_and_first_last_frame_conditions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_ltx_modules(monkeypatch)

    condition_module = types.ModuleType(
        "diffusers.pipelines.ltx2.pipeline_ltx2_condition"
    )

    class FakeCondition:
        def __init__(self, **kwargs: Any) -> None:
            self.index: int = kwargs["index"]
            self.strength: float = kwargs["strength"]
            self.crf: Optional[int] = kwargs["crf"]

    setattr(condition_module, "LTX2VideoCondition", FakeCondition)
    monkeypatch.setitem(
        sys.modules,
        "diffusers.pipelines.ltx2.pipeline_ltx2_condition",
        condition_module,
    )

    model = _model()
    captured: List[List[FakeCondition]] = []

    def fake_generate(*args: Any, **kwargs: Any) -> str:
        captured.append(kwargs["conditions"])
        return "video"

    monkeypatch.setattr(model, "_generate", fake_generate)
    image = PIL.Image.new("RGB", (64, 64))

    assert (
        model.image_to_video(
            image,
            "prompt",
            condition_strength=0.75,
            image_crf=18,
        )
        == "video"
    )
    assert captured[0][0].index == 0
    assert captured[0][0].strength == 0.75
    assert captured[0][0].crf == 18

    assert model.firstlastframe_to_video(image, image, "prompt") == "video"
    assert [condition.index for condition in captured[1]] == [0, -1]

    with pytest.raises(ValueError, match="between 0 and 1"):
        model.image_to_video(image, "prompt", condition_strength=1.1)
    with pytest.raises(ValueError, match="image_crf"):
        model.image_to_video(image, "prompt", image_crf=52)


def test_muxes_each_audio_track_and_cleans_base64_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_ltx_modules(monkeypatch)
    from .. import ltx2

    encoded_calls: List[Dict[str, Any]] = []

    def encode_video(video: Any, **kwargs: Any) -> None:
        encoded_calls.append({"video": video, **kwargs})
        Path(kwargs["output_path"]).write_bytes(f"muxed-{video}".encode("utf-8"))

    diffusers_utils = types.ModuleType("diffusers.utils")
    setattr(diffusers_utils, "encode_video", encode_video)
    monkeypatch.setitem(sys.modules, "diffusers.utils", diffusers_utils)
    monkeypatch.setattr(ltx2, "XINFERENCE_VIDEO_DIR", str(tmp_path))
    monkeypatch.setattr(ltx2, "empty_cache", lambda: None)

    class FakeWaveform:
        def float(self) -> "FakeWaveform":
            return self

        def cpu(self) -> "FakeWaveform":
            return self

    model = _model()
    model._model = SimpleNamespace(
        vocoder=SimpleNamespace(config=SimpleNamespace(output_sampling_rate=48000))
    )
    output = SimpleNamespace(
        frames=["frames-1", "frames-2"],
        audio=[FakeWaveform(), FakeWaveform()],
    )

    result = model._output_to_audio_video(output, 24.0, "b64_json")

    assert len(result["data"]) == 2
    assert base64.b64decode(result["data"][0]["b64_json"]) == b"muxed-frames-1"
    assert [call["audio_sample_rate"] for call in encoded_calls] == [48000, 48000]
    assert [call["fps"] for call in encoded_calls] == [24, 24]
    assert list(tmp_path.iterdir()) == []


def test_loads_distilled_and_dev_from_expected_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from .. import ltx2

    monkeypatch.setattr(torch, "__version__", "2.7.0")
    monkeypatch.setattr(ltx2, "get_available_device", lambda: "cuda")

    pipeline_calls: List[Dict[str, Any]] = []
    transformer_calls: List[Dict[str, Any]] = []
    scheduler_calls: List[Dict[str, Any]] = []

    class FakePipeline:
        def __init__(self) -> None:
            self.transformer = object()
            self.vae = SimpleNamespace(enable_tiling=lambda: None)
            self.scheduler = SimpleNamespace(config={"original": True})
            self.offload_device: Optional[str] = None

        @classmethod
        def from_pretrained(cls, model_path: str, **kwargs: Any) -> "FakePipeline":
            pipeline_calls.append({"model_path": model_path, **kwargs})
            return cls()

        def enable_model_cpu_offload(self, device: str) -> None:
            self.offload_device = device

        def enable_sequential_cpu_offload(self, device: str) -> None:
            raise AssertionError("sequential offload was not requested")

        def to(self, device: str) -> None:
            raise AssertionError("model CPU offload should own placement")

    class FakeTransformer:
        @classmethod
        def from_pretrained(cls, model_path: str, **kwargs: Any) -> object:
            transformer_calls.append({"model_path": model_path, **kwargs})
            return object()

    class FakeScheduler:
        @classmethod
        def from_config(cls, config: Dict[str, Any], **kwargs: Any) -> object:
            scheduler_calls.append({"config": config, **kwargs})
            return object()

    diffusers_module = types.ModuleType("diffusers")
    setattr(diffusers_module, "LTX2ConditionPipeline", FakePipeline)
    setattr(diffusers_module, "LTX2VideoTransformer3DModel", FakeTransformer)
    setattr(diffusers_module, "FlowMatchEulerDiscreteScheduler", FakeScheduler)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers_module)

    distilled = _model()
    distilled.load()
    assert pipeline_calls[0]["model_path"] == "/tmp/official-ltx-2.5"
    assert pipeline_calls[0]["dtype"] is torch.bfloat16
    assert pipeline_calls[0]["local_files_only"] is True
    assert pipeline_calls[0]["use_safetensors"] is True
    assert distilled._model.offload_device == "cuda"

    dev = _model("LTX-2.5-Dev")
    dev.load()
    assert transformer_calls[0]["subfolder"] == "transformer_full"
    assert pipeline_calls[1]["transformer"] is not None
    assert scheduler_calls == [
        {
            "config": {"original": True},
            "use_dynamic_shifting": True,
            "shift_terminal": 0.1,
        }
    ]


def test_load_rejects_device_and_device_map_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(torch, "__version__", "2.7.0")

    class Unused:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("validation must happen before weight loading")

    diffusers_module = types.ModuleType("diffusers")
    setattr(diffusers_module, "LTX2ConditionPipeline", Unused)
    setattr(diffusers_module, "LTX2VideoTransformer3DModel", Unused)
    setattr(diffusers_module, "FlowMatchEulerDiscreteScheduler", Unused)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers_module)

    model = LTX2VideoModel(
        "ltx-test",
        "/tmp/official-ltx-2.5",
        _model_spec(),
        device="cpu",
        device_map="balanced",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        model.load()

    dev_model = LTX2VideoModel(
        "ltx-dev-test",
        "/tmp/official-ltx-2.5",
        _model_spec("LTX-2.5-Dev"),
        device_map="balanced",
    )
    with pytest.raises(ValueError, match="transformer_full"):
        dev_model.load()


def test_distilled_device_map_is_passed_without_extra_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(torch, "__version__", "2.7.0")

    calls: List[Dict[str, Any]] = []

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_path: str, **kwargs: Any) -> "FakePipeline":
            calls.append(kwargs)
            return cls()

        def to(self, device: str) -> None:
            raise AssertionError("device_map must own pipeline placement")

    class Unused:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("distilled load does not need this component")

    diffusers_module = types.ModuleType("diffusers")
    setattr(diffusers_module, "LTX2ConditionPipeline", FakePipeline)
    setattr(diffusers_module, "LTX2VideoTransformer3DModel", Unused)
    setattr(diffusers_module, "FlowMatchEulerDiscreteScheduler", Unused)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers_module)

    model = LTX2VideoModel(
        "ltx-test",
        "/tmp/official-ltx-2.5",
        _model_spec(),
        device_map="balanced",
    )
    model.load()

    assert calls[0]["device_map"] == "balanced"
    assert model._device == "device_map"


def test_rejects_unpinned_or_non_official_model_spec() -> None:
    official_spec = _model_spec()
    unofficial = official_spec.copy(
        update={"model_id": "third-party/LTX-2.5", "model_revision": "main"}
    )

    model = LTX2VideoModel("ltx-test", "/tmp/model", unofficial)
    with pytest.raises(ValueError, match="official Lightricks"):
        model._validate_official_model()


def test_accepts_official_modelscope_model_spec() -> None:
    model = LTX2VideoModel(
        "ltx-test", "/tmp/model", _model_spec(model_hub="modelscope")
    )

    model._validate_official_model()
