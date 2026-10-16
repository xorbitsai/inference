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

import base64
import importlib
import os
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from .. import mlx_video as mlx_video_module
from ..core import VideoModelFamilyV2
from ..engine import MLXVideoEngineModel
from ..mlx_video import MLXVideoModel


def _model_spec(
    model_name: str,
    model_family: str,
    abilities,
    default_generate_config=None,
    **kwargs,
):
    return VideoModelFamilyV2(
        version=2,
        model_name=model_name,
        model_family=model_family,
        model_id=f"test/{model_name}",
        model_revision="test-revision",
        model_ability=abilities,
        default_model_config={},
        default_generate_config=default_generate_config or {},
        cache_config=None,
        engine="MLX",
        model_format="mlx",
        cache_name=f"{model_name}-mlx",
        virtualenv={"packages": []},
        **kwargs,
    )


def _write_converted_wan_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for filename in (
        "config.json",
        "model.safetensors",
        "t5_encoder.safetensors",
        "vae.safetensors",
    ):
        (path / filename).write_bytes(b"test")


def test_convert_wan_model_is_atomic_and_reused(tmp_path, monkeypatch):
    from filelock import FileLock

    source_path = tmp_path / "raw-wan"
    source_path.mkdir()
    calls = []
    preserve_lock_file_values = []

    def file_lock(*args, **kwargs):
        preserve_lock_file_values.append(kwargs.get("preserve_lock_file"))
        return FileLock(*args, **kwargs)

    def convert(source, output, dtype):
        calls.append((source, dtype))
        _write_converted_wan_model(Path(output))

    real_import_module = mlx_video_module.importlib.import_module

    def import_module(name):
        if name == "mlx_video.models.wan_2.convert":
            return SimpleNamespace(convert_wan_checkpoint=convert)
        return real_import_module(name)

    monkeypatch.setattr(mlx_video_module.importlib, "import_module", import_module)
    monkeypatch.setattr("filelock.FileLock", file_lock)

    converted_path = MLXVideoModel._convert_wan_model(source_path)
    assert MLXVideoModel._is_wan_mlx_model_dir(converted_path)
    assert preserve_lock_file_values == [True]
    assert calls == [(str(source_path), "bfloat16")]

    assert MLXVideoModel._convert_wan_model(source_path) == converted_path
    assert preserve_lock_file_values == [True, True]
    assert len(calls) == 1


def test_convert_wan_model_rebuilds_when_source_changes(tmp_path, monkeypatch):
    source_path = tmp_path / "raw-wan"
    source_path.mkdir()
    source_weight = source_path / "source.safetensors"
    source_weight.write_bytes(b"first")
    calls = []

    def convert(source, output, dtype):
        calls.append((source, dtype))
        _write_converted_wan_model(Path(output))

    real_import_module = mlx_video_module.importlib.import_module

    def import_module(name):
        if name == "mlx_video.models.wan_2.convert":
            return SimpleNamespace(convert_wan_checkpoint=convert)
        return real_import_module(name)

    monkeypatch.setattr(mlx_video_module.importlib, "import_module", import_module)

    converted_path = MLXVideoModel._convert_wan_model(source_path)
    source_weight.write_bytes(b"changed-source")

    assert MLXVideoModel._convert_wan_model(source_path) == converted_path
    assert calls == [
        (str(source_path), "bfloat16"),
        (str(source_path), "bfloat16"),
    ]


def test_convert_wan_model_refuses_unmanaged_target(tmp_path):
    source_path = tmp_path / "raw-wan"
    source_path.mkdir()
    converted_path = Path(f"{source_path}.mlx-video")
    converted_path.mkdir()
    marker = converted_path / "user-owned.txt"
    marker.write_text("keep")

    with pytest.raises(FileExistsError, match="Refusing to overwrite unmanaged"):
        MLXVideoModel._convert_wan_model(source_path)

    assert marker.read_text() == "keep"


def test_prepare_wan_model_binds_effective_converter_override(tmp_path):
    source_path = tmp_path / "raw-wan"
    source_path.mkdir()
    converted_path = tmp_path / "converted-wan"
    spec = _model_spec("Wan2.1-1.3B", "Wan", ["text2video"])
    spec.virtualenv.packages = [
        "mlx-video @ git+https://example.invalid/mlx-video@default"
    ]
    override = "mlx-video @ git+https://example.invalid/mlx-video@override"
    model = MLXVideoModel(
        "uid",
        str(source_path),
        spec,
        _xinference_enable_virtual_env=True,
        _xinference_virtual_env_packages=[override],
    )

    with (
        patch.object(
            model,
            "_converter_runtime_identity",
            return_value={
                "environment": "virtualenv",
                "distribution_version": "0.1.dev0",
                "direct_url": {"vcs_info": {"commit_id": "override"}},
            },
        ),
        patch.object(
            MLXVideoModel, "_convert_wan_model", return_value=converted_path
        ) as convert,
    ):
        model._prepare_wan_model()

    identity = convert.call_args.args[1]
    assert identity["converter"] == {
        "requirement": override,
        "runtime": {
            "environment": "virtualenv",
            "distribution_version": "0.1.dev0",
            "direct_url": {"vcs_info": {"commit_id": "override"}},
        },
    }
    assert model._runtime_model_path == str(converted_path)


def test_converter_runtime_identity_distinguishes_host_runtime():
    spec = _model_spec("Wan2.1-1.3B", "Wan", ["text2video"])
    model = MLXVideoModel(
        "uid", "/downloaded", spec, _xinference_enable_virtual_env=False
    )
    distribution = SimpleNamespace(
        version="0.2.0",
        read_text=lambda name: (
            '{"url":"file:///host/mlx-video"}' if name == "direct_url.json" else None
        ),
    )

    with patch.object(
        mlx_video_module.importlib.metadata,
        "distribution",
        return_value=distribution,
    ):
        identity = model._converter_runtime_identity()

    assert identity == {
        "environment": "host",
        "distribution_version": "0.2.0",
        "direct_url": {"url": "file:///host/mlx-video"},
    }


def test_wan_text_to_video_maps_request_and_returns_base64(tmp_path, monkeypatch):
    spec = _model_spec(
        "Wan2.1-1.3B",
        "Wan",
        ["text2video"],
        {"width": 832, "height": 480, "num_frames": 81},
    )
    model = MLXVideoModel(
        "uid", "/downloaded", spec, cpu_offload=False, unrelated_launch_option=True
    )
    model._runtime_model_path = "/converted"
    calls = []

    def generate_video(**kwargs):
        calls.append(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"generated-video")

    monkeypatch.setattr(
        mlx_video_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(generate_video=generate_video),
    )
    monkeypatch.setattr(mlx_video_module, "XINFERENCE_VIDEO_DIR", str(tmp_path))

    response = model._generate(
        "a test prompt",
        n=2,
        num_inference_steps=12,
        response_format="b64_json",
        image=None,
        last_image=None,
        request_kwargs={
            "guidance_scale": 4.5,
            "request_id": "request-id",
            "num_videos_per_prompt": 2,
        },
    )

    assert len(response["data"]) == 2
    assert all(
        base64.b64decode(video["b64_json"]) == b"generated-video"
        for video in response["data"]
    )
    assert len(calls) == 2
    for call in calls:
        assert call["model_dir"] == "/converted"
        assert call["prompt"] == "a test prompt"
        assert call["steps"] == 12
        assert call["guide_scale"] == 4.5
        assert call["width"] == 832
        assert call["height"] == 480
        assert "cpu_offload" not in call
        assert "unrelated_launch_option" not in call
        assert "request_id" not in call
        assert not os.path.exists(call["output_path"])


def test_wan_prompt_lists_fan_out_and_pair_negative_prompts(tmp_path, monkeypatch):
    spec = _model_spec(
        "Wan2.1-1.3B",
        "Wan",
        ["text2video"],
        {"width": 512, "height": 512, "num_frames": 17},
    )
    model = MLXVideoModel("uid", "/downloaded", spec)
    model._runtime_model_path = "/converted"
    calls = []

    def generate_video(**kwargs):
        calls.append(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"generated-video")

    monkeypatch.setattr(
        mlx_video_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(generate_video=generate_video),
    )
    monkeypatch.setattr(mlx_video_module, "XINFERENCE_VIDEO_DIR", str(tmp_path))

    response = model._generate(
        ["first prompt", "second prompt"],
        n=2,
        num_inference_steps=4,
        response_format="b64_json",
        image=None,
        last_image=None,
        request_kwargs={"negative_prompt": ["first negative", "second negative"]},
    )

    assert len(response["data"]) == 4
    assert [call["prompt"] for call in calls] == [
        "first prompt",
        "first prompt",
        "second prompt",
        "second prompt",
    ]
    assert [call["negative_prompt"] for call in calls] == [
        "first negative",
        "first negative",
        "second negative",
        "second negative",
    ]


def test_prompt_and_negative_prompt_lists_must_align(tmp_path, monkeypatch):
    spec = _model_spec("Wan2.1-1.3B", "Wan", ["text2video"])
    model = MLXVideoModel("uid", "/downloaded", spec)

    with pytest.raises(
        ValueError, match="negative_prompt list length must match prompt list length"
    ):
        model._generate(
            ["first", "second"],
            n=1,
            num_inference_steps=4,
            response_format="b64_json",
            image=None,
            last_image=None,
            request_kwargs={"negative_prompt": ["only one"]},
        )


def test_ltx_first_last_frame_generation_cleans_temporary_inputs(tmp_path, monkeypatch):
    class PipelineType(Enum):
        DISTILLED = "distilled"
        DEV = "dev"

    spec = _model_spec(
        "LTX-2.3-dev",
        "LTX-2.3",
        ["text2video", "image2video", "firstlastframe2video"],
        {"pipeline": "dev", "width": 512, "height": 512, "num_frames": 33},
    )
    model = MLXVideoModel("uid", "/downloaded", spec)
    model._runtime_model_path = "/ltx-model"
    model._text_encoder_path = "/ltx-text-encoder"
    calls = []

    def generate_video(**kwargs):
        assert Path(kwargs["image"]).is_file()
        assert Path(kwargs["end_image"]).is_file()
        calls.append(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"generated-video")
        Path(kwargs["output_audio_path"]).write_bytes(b"generated-audio")

    generate_module = SimpleNamespace(
        PipelineType=PipelineType, generate_video=generate_video
    )
    monkeypatch.setattr(
        mlx_video_module.importlib,
        "import_module",
        lambda name: generate_module,
    )
    monkeypatch.setattr(mlx_video_module, "XINFERENCE_VIDEO_DIR", str(tmp_path))

    response = model._generate(
        "a test prompt",
        n=1,
        num_inference_steps=20,
        response_format="url",
        image=Image.new("RGB", (8, 8), color="red"),
        last_image=Image.new("RGB", (8, 8), color="blue"),
        request_kwargs={"guidance_scale": 3.5},
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["model_repo"] == "/ltx-model"
    assert call["text_encoder_repo"] == "/ltx-text-encoder"
    assert call["pipeline"] is PipelineType.DEV
    assert call["num_inference_steps"] == 20
    assert call["cfg_scale"] == 3.5
    assert not Path(call["image"]).exists()
    assert not Path(call["end_image"]).exists()
    assert not Path(call["output_audio_path"]).exists()
    assert response["data"][0]["url"] == call["output_path"]
    Path(call["output_path"]).unlink()


def test_ltx_audio_cleanup_error_does_not_mask_generation_error(tmp_path, monkeypatch):
    class PipelineType(Enum):
        DEV = "dev"

    spec = _model_spec(
        "LTX-2.3-dev",
        "LTX-2.3",
        ["text2video"],
        {"pipeline": "dev"},
    )
    model = MLXVideoModel("uid", "/downloaded", spec)
    output_path = str(tmp_path / "output.mp4")
    audio_path = str(tmp_path / "output.wav")

    def generate_video(**kwargs):
        Path(kwargs["output_audio_path"]).write_bytes(b"generated-audio")
        raise RuntimeError("generation failed")

    monkeypatch.setattr(
        mlx_video_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            PipelineType=PipelineType, generate_video=generate_video
        ),
    )

    def fail_audio_cleanup(path):
        assert path == audio_path
        raise PermissionError("cleanup failed")

    monkeypatch.setattr(mlx_video_module.os, "remove", fail_audio_cleanup)

    with pytest.raises(RuntimeError, match="generation failed"):
        model._generate_ltx(
            "a test prompt",
            output_path,
            image_path=None,
            last_image_path=None,
            num_inference_steps=20,
            request_kwargs={},
        )


def test_ltx_23_prepares_pinned_external_text_encoder(tmp_path, monkeypatch):
    model_path = tmp_path / "ltx-model"
    model_path.mkdir()
    text_encoder_path = tmp_path / "text-encoder"
    text_encoder_path.mkdir()
    spec = _model_spec(
        "LTX-2.3-distilled",
        "LTX-2.3",
        ["text2video"],
        text_encoder_model_id="prince-canuma/LTX-2-distilled",
        text_encoder_model_revision="pinned-revision",
    )
    model = MLXVideoModel("uid", str(model_path), spec)
    monkeypatch.setenv("HF_HUB_DOWNLOAD_WORKERS", "2")

    with patch(
        "huggingface_hub.snapshot_download", return_value=str(text_encoder_path)
    ) as snapshot_download:
        model._prepare_ltx_model()

    snapshot_download.assert_called_once_with(
        "prince-canuma/LTX-2-distilled",
        revision="pinned-revision",
        allow_patterns=["text_encoder/**", "tokenizer/**"],
        max_workers=2,
    )
    assert model._runtime_model_path == str(model_path)
    assert model._text_encoder_path == str(text_encoder_path)


def test_ltx_23_prepares_external_text_encoder_from_modelscope(tmp_path):
    model_path = tmp_path / "ltx-model"
    model_path.mkdir()
    text_encoder_path = tmp_path / "text-encoder"
    text_encoder_path.mkdir()
    spec = _model_spec(
        "LTX-2.3-distilled",
        "LTX-2.3",
        ["text2video"],
        model_hub="modelscope",
        text_encoder_model_id="Xorbits/LTX-2-distilled",
        text_encoder_model_revision="master",
    )
    model = MLXVideoModel("uid", str(model_path), spec)
    snapshot_download_module = importlib.import_module(
        "modelscope.hub.snapshot_download"
    )

    with patch.object(
        snapshot_download_module,
        "snapshot_download",
        return_value=str(text_encoder_path),
    ) as snapshot_download:
        model._prepare_ltx_model()

    snapshot_download.assert_called_once_with(
        "Xorbits/LTX-2-distilled",
        revision="master",
        allow_patterns=["text_encoder/**", "tokenizer/**"],
    )
    assert model._runtime_model_path == str(model_path)
    assert model._text_encoder_path == str(text_encoder_path)


def test_mlx_engine_rejects_unrelated_pypi_package_namespace():
    def find_spec(name):
        if name in ("mlx", "mlx_video"):
            return object()
        return None

    with (
        patch.object(MLXVideoEngineModel, "_is_apple_silicon", return_value=True),
        patch("xinference.model.video.engine.sys.version_info", (3, 11)),
        patch("importlib.util.find_spec", side_effect=find_spec),
    ):
        result = MLXVideoEngineModel.check_lib()

    assert result == (
        False,
        "Blaizzy/mlx-video is not installed; the unrelated PyPI package "
        "with the same name is not compatible",
    )
