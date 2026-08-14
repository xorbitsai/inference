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
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

from .. import diffusers as diffusers_module
from .. import load_model_family_from_json
from ..core import VideoModelFamilyV2, match_diffusion
from ..diffusers import DiffusersVideoModel


def _model_spec():
    return VideoModelFamilyV2(
        version=2,
        model_name="MiniMax-H3",
        model_family="MiniMax-H3",
        model_id="MiniMaxAI/MiniMax-H3",
        model_revision="test-revision",
        model_ability=["text2video", "image2video", "firstlastframe2video"],
        default_model_config={
            "quantization": "int4",
            "group_offload": True,
            "torch_dtype": "bfloat16",
        },
        default_generate_config={"num_frames": 124},
        cache_config=None,
        virtualenv={"packages": []},
    )


def test_minimax_h3_selects_modelscope_source(monkeypatch):
    import xinference.model.video as video_module

    model_families = {}
    load_model_family_from_json("model_spec.json", model_families)
    monkeypatch.setattr(video_module, "BUILTIN_VIDEO_MODELS", model_families)
    monkeypatch.setenv("XINFERENCE_MODEL_SRC", "modelscope")

    model_spec = match_diffusion("MiniMax-H3")

    assert model_spec.model_hub == "modelscope"
    assert model_spec.model_id == "MiniMax/MiniMax-H3"
    assert model_spec.model_revision == "master"
    assert model_spec.default_model_config["quantization"] == "int4"


def test_minimax_h3_loads_modular_pipeline(monkeypatch):
    calls = {}

    class FakeComponentsManager:
        def enable_auto_cpu_offload(self, **kwargs):
            calls["offload"] = kwargs

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls["from_pretrained"] = (model_path, kwargs)
            return cls()

        def load_components(self, **kwargs):
            calls["load_components"] = kwargs

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.ComponentsManager = FakeComponentsManager
    fake_diffusers.ModularPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    model = DiffusersVideoModel(
        "mock",
        "/tmp/minimax-h3",
        _model_spec(),
        quantization="none",
        onload_device="cuda:1",
    )
    model.load()

    assert calls["offload"] == {
        "device": "cuda:1",
        "memory_reserve_margin": "12GB",
    }
    model_path, from_pretrained_kwargs = calls["from_pretrained"]
    assert model_path == "/tmp/minimax-h3"
    assert list(from_pretrained_kwargs) == ["components_manager"]
    assert isinstance(model._model, FakePipeline)
    assert calls["load_components"]["workflow"] == "t2va"
    assert (
        calls["load_components"]["pretrained_model_name_or_path"] == "/tmp/minimax-h3"
    )


def test_minimax_h3_defaults_to_int4_and_group_offload(monkeypatch):
    import torch

    calls = {}

    class FakeInt8WeightOnlyConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeInt4WeightOnlyConfig(FakeInt8WeightOnlyConfig):
        pass

    class FakeQuantizationConfig:
        def __init__(self, quant_type, modules_to_not_convert):
            self.quant_type = quant_type
            self.modules_to_not_convert = modules_to_not_convert

    class FakeComponent:
        def __init__(self, name):
            self.name = name

        def requires_grad_(self, value):
            calls.setdefault("requires_grad", []).append((self.name, value))

        def enable_group_offload(self, **kwargs):
            calls["transformer_group_offload"] = kwargs

        def to(self, device):
            calls.setdefault("to", []).append((self.name, device))
            return self

    class FakeTextEncoder(FakeComponent):
        def __init__(self):
            super().__init__("text_encoder")
            self.model = FakeComponent("text_encoder.model")

    class FakeTransformer:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls["transformer_from_pretrained"] = (model_path, kwargs)
            return FakeComponent("transformer")

    class FakeTextEncoderClass:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls["text_encoder_from_pretrained"] = (model_path, kwargs)
            return FakeTextEncoder()

    class FakePipeline:
        def __init__(self):
            self.vae = FakeComponent("vae")
            self.audio_vae = FakeComponent("audio_vae")

        @classmethod
        def from_pretrained(cls, model_path):
            calls["from_pretrained"] = model_path
            return cls()

        def update_components(self, **kwargs):
            calls["update_components"] = kwargs
            self.transformer = kwargs["transformer"]
            self.text_encoder = kwargs["text_encoder"]

        def load_components(self, **kwargs):
            calls["load_components"] = kwargs

    def fake_apply_group_offloading(module, **kwargs):
        calls["text_encoder_group_offload"] = (module, kwargs)

    fake_hooks = types.ModuleType("diffusers.hooks")
    fake_hooks.apply_group_offloading = fake_apply_group_offloading
    fake_diffusers_modeling_utils = types.ModuleType("diffusers.models.modeling_utils")
    original_diffusers_dispatch = object()
    fake_diffusers_modeling_utils.dispatch_model = original_diffusers_dispatch
    fake_diffusers_models = types.ModuleType("diffusers.models")
    fake_diffusers_models.modeling_utils = fake_diffusers_modeling_utils
    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.ModularPipeline = FakePipeline
    fake_diffusers.MiniMaxH3Transformer3DModel = FakeTransformer
    fake_diffusers.TorchAoConfig = FakeQuantizationConfig
    fake_diffusers.hooks = fake_hooks
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Qwen3VLForConditionalGeneration = FakeTextEncoderClass
    fake_transformers.TorchAoConfig = FakeQuantizationConfig
    fake_transformers_modeling_utils = types.ModuleType("transformers.modeling_utils")
    original_transformers_dispatch = object()
    fake_transformers_modeling_utils.accelerate_dispatch = (
        original_transformers_dispatch
    )
    fake_transformers.modeling_utils = fake_transformers_modeling_utils
    fake_torchao = types.ModuleType("torchao")
    fake_torchao_quantization = types.ModuleType("torchao.quantization")
    fake_torchao_quantization.Int4WeightOnlyConfig = FakeInt4WeightOnlyConfig
    fake_torchao_quantization.Int8WeightOnlyConfig = FakeInt8WeightOnlyConfig
    fake_torchao.quantization = fake_torchao_quantization
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.hooks", fake_hooks)
    monkeypatch.setitem(sys.modules, "diffusers.models", fake_diffusers_models)
    monkeypatch.setitem(
        sys.modules, "diffusers.models.modeling_utils", fake_diffusers_modeling_utils
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(
        sys.modules, "transformers.modeling_utils", fake_transformers_modeling_utils
    )
    monkeypatch.setitem(sys.modules, "torchao", fake_torchao)
    monkeypatch.setitem(sys.modules, "torchao.quantization", fake_torchao_quantization)

    model = DiffusersVideoModel("mock", "/tmp/minimax-h3", _model_spec())
    model.load()

    transformer_path, transformer_kwargs = calls["transformer_from_pretrained"]
    assert transformer_path == "/tmp/minimax-h3"
    assert transformer_kwargs["dtype"] == torch.bfloat16
    assert transformer_kwargs["low_cpu_mem_usage"] is True
    assert transformer_kwargs["device_map"][""] == torch.device("cuda")
    assert transformer_kwargs["device_map"]["token_refiner"] == "cpu"
    assert transformer_kwargs["device_map"]["transformer_blocks.46"] == "cpu"
    assert transformer_kwargs["device_map"]["transformer_blocks.49"] == "cpu"
    assert "transformer_blocks.45" not in transformer_kwargs["device_map"]
    assert transformer_kwargs["quantization_config"].quant_type.kwargs == {
        "group_size": 128,
        "int4_packing_format": "tile_packed_to_4d",
        "version": 2,
    }
    assert "proj_in" in transformer_kwargs["quantization_config"].modules_to_not_convert

    text_encoder_kwargs = calls["text_encoder_from_pretrained"][1]
    assert text_encoder_kwargs["dtype"] == torch.bfloat16
    assert text_encoder_kwargs["low_cpu_mem_usage"] is True
    assert text_encoder_kwargs["device_map"] == {
        "": torch.device("cuda"),
        "model.visual": "cpu",
        "model.language_model.embed_tokens": "cpu",
        "model.language_model.norm": "cpu",
        "lm_head": "cpu",
    }
    assert (
        "model.visual"
        in text_encoder_kwargs["quantization_config"].modules_to_not_convert
    )
    assert calls["load_components"]["workflow"] == "t2va"
    assert (
        calls["load_components"]["pretrained_model_name_or_path"] == "/tmp/minimax-h3"
    )
    assert calls["transformer_group_offload"]["offload_type"] == "block_level"
    assert calls["transformer_group_offload"]["num_blocks_per_group"] == 1
    assert calls["transformer_group_offload"]["use_stream"] is False
    assert calls["text_encoder_group_offload"][0].name == "text_encoder.model"
    assert fake_diffusers_modeling_utils.dispatch_model is original_diffusers_dispatch
    assert (
        fake_transformers_modeling_utils.accelerate_dispatch
        is original_transformers_dispatch
    )
    assert [name for name, _ in calls["to"]].count("transformer") == 1
    assert [name for name, _ in calls["to"]].count("text_encoder") == 1
    assert {name for name, _ in calls["to"]} == {
        "transformer",
        "text_encoder",
        "vae",
        "audio_vae",
    }


def test_minimax_h3_forwards_workflows_and_muxes_audio(monkeypatch, tmp_path):
    calls = []

    class FakeProgressBar:
        def __init__(self, total):
            self.total = total
            self.n = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def update(self, count=1):
            self.n += count

    class MiniMaxH3DenoiseStep:
        model_name = "minimax-h3"

        def progress_bar(self, total):
            return FakeProgressBar(total)

        def loop_step(self):
            pass

    denoise = MiniMaxH3DenoiseStep()

    class FakePipeline:
        _blocks = types.SimpleNamespace(sub_blocks={"denoise": denoise})

        def __call__(self, **kwargs):
            calls.append(kwargs)
            with denoise.progress_bar(total=kwargs["num_inference_steps"]) as bar:
                for _ in range(kwargs["num_inference_steps"]):
                    bar.update()
            return {
                "videos": [[np.zeros((2, 2, 3), dtype=np.uint8)]],
                "audio": np.zeros((1, 2, 4), dtype=np.float32),
                "sampling_rate": 32000,
            }

    def fake_encode_video(
        video, *, fps, output_path, audio, audio_sample_rate, **kwargs
    ):
        assert fps == 24
        assert audio is not None
        assert audio_sample_rate == 32000
        Path(output_path).write_bytes(b"mp4-with-audio")

    fake_utils = types.ModuleType("diffusers.utils")
    fake_utils.encode_video = fake_encode_video
    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.utils = fake_utils
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.utils", fake_utils)
    monkeypatch.setattr(diffusers_module, "XINFERENCE_VIDEO_DIR", str(tmp_path))

    model = DiffusersVideoModel("mock", "/tmp/minimax-h3", _model_spec())
    model._model = FakePipeline()

    class FakeProgressor:
        request_id = "request"

        def __init__(self):
            self.progress = None
            self.progress_updates = []

        def set_progress(self, value):
            self.progress = value
            self.progress_updates.append(value)

    progressor = FakeProgressor()
    result = model.text_to_video(
        "a red fox",
        num_inference_steps=3,
        negative_prompt="ignored",
        guidance_scale=7,
        response_format="b64_json",
        progressor=progressor,
    )

    assert result["data"][0]["b64_json"] is not None
    assert calls[0]["num_frames"] == 124
    assert calls[0]["num_inference_steps"] == 3
    assert calls[0]["output"] == ["videos", "audio", "sampling_rate"]
    assert "negative_prompt" not in calls[0]
    assert "guidance_scale" not in calls[0]
    assert progressor.progress == 1.0
    assert progressor.progress_updates[:3] == [1 / 3, 2 / 3, 1.0]
    assert "progress_bar" not in denoise.__dict__

    image = Image.new("RGB", (2, 2))
    model.image_to_video(image, "animate it", response_format="url")
    assert calls[1]["image"] is image

    last_image = Image.new("RGB", (2, 2), color="white")
    model.firstlastframe_to_video(
        image, last_image, "transition", response_format="url"
    )
    assert calls[2]["image"] is image
    assert calls[2]["last_image"] is last_image


def test_minimax_h3_progress_tolerates_pipeline_without_blocks():
    class FakeProgressor:
        request_id = "request"

    pipeline = types.SimpleNamespace()
    with DiffusersVideoModel._track_minimax_h3_progress(
        pipeline, FakeProgressor(), 0, 1
    ):
        pass
