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
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from packaging.version import Version

from ....core.model import XINFERENCE_BATCHING_ALLOWED_VISION_MODELS
from ....core.utils import filter_virtualenv_packages_by_markers
from ....core.virtual_env_manager import expand_engine_dependency_placeholders
from ...utils import get_engine_params_by_name_with_virtual_env
from ..llm_family import check_engine_by_spec_parameters_with_virtual_env, match_llm
from ..sglang.core import SGLANGVisionModel
from ..transformers.multimodal.minicpmv46 import MiniCPMV46Model
from ..transformers.multimodal.minicpmv47 import MiniCPMV47Model
from ..vllm import core as vllm_core

VARIANTS = [
    (1, "pytorch", "none", ""),
    (1, "awq", "Int4", "-AWQ"),
    (1, "gptq", "Int4", "-GPTQ"),
    (1, "bnb", "4-bit", "-int4"),
    (35, "pytorch", "none", ""),
    (35, "awq", "Int4", "-AWQ"),
    (35, "gptq", "Int4", "-GPTQ"),
    (35, "pytorch", "Int4", "-int4"),
    (35, "fp8", "FP8", "-FP8"),
]


@pytest.fixture
def virtualenv_worker(monkeypatch):
    monkeypatch.setattr(vllm_core, "VLLM_INSTALLED", False)
    monkeypatch.setattr(vllm_core, "_virtual_env_allows_missing_vllm", lambda: True)
    monkeypatch.setattr(
        vllm_core.VLLMModel, "_has_cuda_device", classmethod(lambda cls: True)
    )
    monkeypatch.setattr(vllm_core.VLLMModel, "_is_linux", classmethod(lambda cls: True))


@pytest.mark.parametrize("stream", [False, True])
def test_http_request_preserves_processor_options(stream):
    from ....types import CreateChatCompletion

    options = {"downsample_mode": "4x", "max_num_frames": 2}
    body = CreateChatCompletion.parse_obj(
        {
            "model": "MiniCPM-V-4.7",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            "mm_processor_kwargs": options,
        }
    )
    assert body.dict(exclude_unset=True)["mm_processor_kwargs"] == options


@pytest.mark.parametrize(
    "hub,owner,revision",
    [("huggingface", "openbmb", "main"), ("modelscope", "OpenBMB", "master")],
)
@pytest.mark.parametrize("size,model_format,quantization,suffix", VARIANTS)
def test_official_checkpoints_and_engine_selection(
    virtualenv_worker, hub, owner, revision, size, model_format, quantization, suffix
):
    family = match_llm("MiniCPM-V-4.7", model_format, size, quantization, hub)
    spec = family.model_specs[0]
    label = "1B" if size == 1 else "35B-A3B"
    assert spec.model_id == f"{owner}/MiniCPM-V-4.7-{label}{suffix}"
    assert spec.model_revision == revision
    assert spec.activated_size_in_billions == (3 if size == 35 else None)
    assert family.context_length == 262144
    assert family.model_ability == ["chat", "vision", "hybrid"]
    assert MiniCPMV47Model.match_json(family, spec, quantization) is True
    assert vllm_core.VLLMMultiModel.match_json(family, spec, quantization) is True
    for engine, expected in (
        ("Transformers", MiniCPMV47Model),
        ("vLLM", vllm_core.VLLMMultiModel),
    ):
        assert (
            check_engine_by_spec_parameters_with_virtual_env(
                engine,
                family.model_name,
                model_format,
                size,
                quantization,
                llm_family=family,
            )
            is expected
        )


@pytest.mark.parametrize("size", [1, 35])
@pytest.mark.parametrize("hub", ["huggingface", "modelscope"])
def test_gguf_is_not_registered(size, hub):
    for quantization in ("F16", "Q4_K_M", "mix_q3gud_im"):
        assert match_llm("MiniCPM-V-4.7", "ggufv2", size, quantization, hub) is None
    assert match_llm("MiniCPM-V-4.7", "fp8", 1, "FP8", hub) is None


def test_virtualenv_engine_discovery(virtualenv_worker):
    params = get_engine_params_by_name_with_virtual_env(
        "LLM", "MiniCPM-V-4.7", enable_virtual_env=True
    )
    for engine in ("Transformers", "vLLM"):
        assert {
            (int(entry["model_size_in_billions"]), entry["model_format"], quantization)
            for entry in params[engine]
            for quantization in entry["quantizations"]
        } == {(size, fmt, quant) for size, fmt, quant, _ in VARIANTS}
    assert not isinstance(params.get("SGLang"), list)
    assert not isinstance(params.get("llama.cpp"), list)
    assert "MiniCPM-V-4.7" not in XINFERENCE_BATCHING_ALLOWED_VISION_MODELS


@pytest.mark.parametrize(
    "engine,expected",
    [
        ("Transformers", "transformers>=5.18.0"),
        ("vllm", "vllm==0.31.1rc1.dev281+gc85616efc"),
        ("llama.cpp", None),
    ],
)
def test_virtualenv_requirements(engine, expected):
    family = match_llm("MiniCPM-V-4.7", "pytorch", 1, "none", "huggingface")
    expanded = expand_engine_dependency_placeholders(family.virtualenv.packages, engine)
    prepared = filter_virtualenv_packages_by_markers(expanded, engine, None)
    if expected is None:
        assert prepared == []
        return
    assert expected in prepared
    if engine == "Transformers":
        assert "#system_torch#" in prepared
        assert "#system_torchvision#" in prepared
        assert "#system_numpy#" in prepared
        assert "torchvision" not in prepared
    else:
        assert "#system_torch#" not in prepared
        assert "#system_torchvision#" not in prepared
    if engine == "vllm":
        assert "transformers>=5.18.0" in prepared
        assert "qwen-omni-utils>=0.0.9" in prepared
        assert "audioread" in prepared
        assert "#system_numpy#" in prepared
        assert family.virtualenv.extra_index_url.endswith(
            "/c85616efc6f0eaa50ca1f1a09c32368aedffd3d8"
        )
    else:
        assert "qwen-omni-utils>=0.0.9" not in prepared
        assert "audioread" not in prepared


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("frame_limit", [1, 2])
async def test_vllm_forwards_compression_and_samples_video(
    monkeypatch, stream, frame_limit
):
    import sys
    from types import ModuleType
    from unittest.mock import AsyncMock

    import torch

    frames = torch.arange(4).reshape(4, 1, 1, 1)
    media = ModuleType("qwen_omni_utils")
    media.process_audio_info = Mock()
    media.process_mm_info = Mock()
    media.process_vision_info = Mock(return_value=(["image"], [frames], {"fps": [2.0]}))
    monkeypatch.setitem(sys.modules, "qwen_omni_utils", media)
    model = object.__new__(vllm_core.VLLMMultiModel)
    model.model_uid = "minicpm-v47-test"
    model.model_family = match_llm("MiniCPM-V-4.7", "pytorch", 1, "none", "huggingface")
    model.reasoning_parser = None
    model._get_chat_template_and_tokenizer = AsyncMock(return_value=("template", None))
    model._get_chat_template_kwargs_from_generate_config = Mock(return_value={})
    model.get_full_context = Mock(return_value="prompt")
    model._sanitize_chat_config = Mock(side_effect=lambda config: config)

    async def completion_stream():
        yield {"completion": "ok"}

    model.async_generate = AsyncMock(
        return_value=completion_stream() if stream else {"completion": "ok"}
    )
    model._to_chat_completion = Mock(return_value="answer")
    model._async_to_chat_completion_chunks = Mock(return_value="stream")
    monkeypatch.setattr(
        vllm_core, "materialize_messages_media", lambda messages, workspace: None
    )
    config = {
        "stream": stream,
        "mm_processor_kwargs": {"downsample_mode": "4x", "max_num_frames": frame_limit},
    }
    original = deepcopy(config)
    result = await model.async_chat([{"role": "user", "content": "describe"}], config)
    assert result == ("stream" if stream else "answer")
    inputs, generate_config = model.async_generate.call_args.args[:2]
    assert inputs["mm_processor_kwargs"] == {"downsample_mode": "4x"}
    assert inputs["multi_modal_data"]["image"] == ["image"]
    [video] = inputs["multi_modal_data"]["video"]
    assert video.flatten().tolist() == ([0] if frame_limit == 1 else [0, 3])
    assert "mm_processor_kwargs" not in generate_config
    assert config == original


@pytest.mark.asyncio
@pytest.mark.parametrize("frame_limit", [0, -1, 1.5, True])
async def test_vllm_rejects_invalid_frame_limit(frame_limit):
    model = object.__new__(vllm_core.VLLMMultiModel)
    model.model_family = match_llm("MiniCPM-V-4.7", "pytorch", 1, "none", "huggingface")
    with pytest.raises(ValueError, match="max_num_frames must be a positive integer"):
        await model.async_chat(
            [], {"mm_processor_kwargs": {"max_num_frames": frame_limit}}
        )


@pytest.mark.parametrize(
    "installed_version,accepted",
    [
        ("0.31.0", False),
        ("0.31.1rc1.dev280", False),
        ("0.31.1rc1.dev281+gc85616efc", True),
        ("0.31.1", True),
    ],
)
def test_vllm_nightly_gate(virtualenv_worker, monkeypatch, installed_version, accepted):
    monkeypatch.setattr(vllm_core, "_virtual_env_allows_missing_vllm", lambda: False)
    monkeypatch.setattr(vllm_core, "VLLM_INSTALLED", True)
    monkeypatch.setattr(vllm_core, "VLLM_VERSION", Version(installed_version))
    family = match_llm("MiniCPM-V-4.7", "pytorch", 1, "none", "huggingface")
    result = vllm_core.VLLMMultiModel.match_json(family, family.model_specs[0], "none")
    assert (result is True) is accepted
    if not accepted:
        assert "PR #58674" in result[1]


def test_int4_exception_is_scoped(virtualenv_worker):
    family = match_llm("MiniCPM-V-4.7", "pytorch", 35, "Int4", "huggingface")
    spec = family.model_specs[0]
    family.architectures = ["MiniCPMV4_6ForConditionalGeneration"]
    assert vllm_core.VLLMMultiModel.match_json(family, spec, "Int4")[0] is False
    family.architectures = ["MiniCPMV4_7ForConditionalGeneration"]
    spec.model_size_in_billions = 1
    assert vllm_core.VLLMMultiModel.match_json(family, spec, "Int4")[0] is False


def test_fp8_uses_checkpoint_quantizer(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vllm_core, "VLLM_VERSION", vllm_core.MINICPMV47_MIN_VLLM_VERSION
    )
    family = match_llm("MiniCPM-V-4.7", "fp8", 35, "FP8", "huggingface")
    model = object.__new__(vllm_core.VLLMMultiModel)
    model.model_path = str(tmp_path)
    model.model_family = family
    model.model_spec = family.model_specs[0]
    model._device_count = 1
    model._n_worker = 1
    model._apply_draft_model = Mock()
    assert model._sanitize_model_config({})["quantization"] is None
    assert (
        model._sanitize_model_config({"model_quantization": "compressed-tensors"})[
            "quantization"
        ]
        == "compressed-tensors"
    )
    family.architectures = ["MiniCPMV4_6ForConditionalGeneration"]
    assert model._sanitize_model_config({})["quantization"] == "fp8"


@pytest.fixture
def adapter(monkeypatch):
    import torch
    from transformers import BatchFeature

    monkeypatch.setattr(
        "xinference.model.llm.transformers.multimodal.minicpmv47.materialize_messages_media",
        lambda messages, workspace: None,
    )
    model = object.__new__(MiniCPMV47Model)
    model._pytorch_model_config = {"downsample_mode": "16x", "max_slice_nums": 36}
    model.reasoning_parser = None
    model._tokenizer = Mock()
    inputs = BatchFeature(
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "grids": torch.tensor([[1, 2]]),
            "num_patches_per_image": torch.tensor([3]),
            "target_sizes": torch.tensor([[16, 16]]),
        }
    )
    model._processor = SimpleNamespace(apply_chat_template=Mock(return_value=inputs))

    def generate(**kwargs):
        kwargs["streamer"].on_finalized_text("answer", stream_end=True)

    model._model = SimpleNamespace(device="cpu", generate=Mock(side_effect=generate))
    return model


@pytest.mark.parametrize("mode", ["4x", "16x"])
@pytest.mark.parametrize("modality", ["image", "video", "both", "text"])
def test_multimodal_messages_preserve_canvas_metadata(adapter, mode, modality):
    content = [{"type": "text", "text": "Describe the inputs."}]
    if modality in ("image", "both"):
        content.extend(
            [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,a"}},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/b.png"},
                },
            ]
        )
    if modality in ("video", "both"):
        content.append({"type": "video_url", "video_url": {"url": "clip.mp4"}})
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": content},
    ]
    original = deepcopy(messages)
    stream, prompt_tokens = adapter.build_streaming_iter(
        messages,
        {
            "temperature": 0,
            "max_tokens": 17,
            "mm_processor_kwargs": {"downsample_mode": mode},
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    assert list(stream) == ["answer"]
    assert prompt_tokens == 3
    assert messages == original
    call = adapter._processor.apply_chat_template.call_args
    assert call.args[0][0] == messages[0]
    assert call.kwargs["enable_thinking"] is True
    processor_kwargs = call.kwargs["processor_kwargs"]
    assert ("images_kwargs" in processor_kwargs) == (modality in ("image", "both"))
    assert ("videos_kwargs" in processor_kwargs) == (modality in ("video", "both"))
    if "images_kwargs" in processor_kwargs:
        assert processor_kwargs["images_kwargs"]["downsample_mode"] == mode
        assert processor_kwargs["images_kwargs"]["max_slice_nums"] == 36
    if "videos_kwargs" in processor_kwargs:
        assert processor_kwargs["videos_kwargs"]["downsample_mode"] == mode
        assert processor_kwargs["videos_kwargs"]["max_slice_nums"] == 1
        assert processor_kwargs["videos_kwargs"]["max_num_frames"] == 128
    generation = adapter._model.generate.call_args.kwargs
    assert generation["max_new_tokens"] == 17
    assert generation["do_sample"] is False
    for key in ("grids", "num_patches_per_image", "target_sizes"):
        assert (
            generation[key] is adapter._processor.apply_chat_template.return_value[key]
        )
    if modality == "text":
        assert "downsample_mode" not in generation
    else:
        assert generation["downsample_mode"] == mode


def test_video_options_and_invalid_compression(adapter):
    messages = [
        {
            "role": "user",
            "content": [{"type": "video_url", "video_url": {"url": "clip.mp4"}}],
        }
    ]
    adapter.build_inputs_from_messages(
        messages,
        {
            "mm_processor_kwargs": {
                "max_slice_nums": 2,
                "max_num_frames": 8,
                "stack_frames": 3,
            }
        },
    )
    kwargs = adapter._processor.apply_chat_template.call_args.kwargs[
        "processor_kwargs"
    ]["videos_kwargs"]
    assert kwargs == {
        "downsample_mode": "16x",
        "max_slice_nums": 2,
        "max_num_frames": 8,
        "stack_frames": 3,
    }
    with pytest.raises(ValueError, match="downsample_mode"):
        adapter.build_inputs_from_messages(
            messages, {"mm_processor_kwargs": {"downsample_mode": "8x"}}
        )


def test_generation_errors_reach_caller(adapter):
    adapter._model.generate.side_effect = RuntimeError("generation failed")
    stream, _ = adapter.build_streaming_iter([{"role": "user", "content": "hi"}], {})
    with pytest.raises(RuntimeError, match="generation failed"):
        list(stream)


@pytest.mark.parametrize("processor_fails", [False, True])
def test_base64_video_is_request_scoped(adapter, processor_fails):
    import base64
    from pathlib import Path

    payload = b"video payload"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "data:video/mp4;base64,"
                        + base64.b64encode(payload).decode()
                    },
                }
            ],
        }
    ]
    original = deepcopy(messages)
    paths = []
    inputs = adapter._processor.apply_chat_template.return_value

    def process(converted, **kwargs):
        path = Path(converted[0]["content"][0]["url"])
        assert path.read_bytes() == payload
        paths.append(path)
        if processor_fails:
            raise RuntimeError("processor failed")
        return inputs

    adapter._processor.apply_chat_template.side_effect = process
    if processor_fails:
        with pytest.raises(RuntimeError, match="processor failed"):
            adapter.build_inputs_from_messages(messages, {})
    else:
        adapter.build_inputs_from_messages(messages, {})
    assert messages == original
    assert paths and not paths[0].exists()


def test_older_architecture_and_unsupported_engines():
    family = match_llm("MiniCPM-V-4.6", "pytorch", 1, "none", "huggingface")
    assert MiniCPMV46Model.match_json(family, family.model_specs[0], "none") is True
    assert MiniCPMV47Model.match_json(family, family.model_specs[0], "none")[0] is False
    family = match_llm("MiniCPM-V-4.7", "pytorch", 1, "none", "huggingface")
    assert (
        SGLANGVisionModel.match_json(family, family.model_specs[0], "none")[0] is False
    )
