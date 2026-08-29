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

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from xinference.api.schemas.requests import CreateEmbeddingRequest
from xinference.model.embedding import _install
from xinference.model.utils import get_engine_params_by_name_with_virtual_env

from ..wemm import ensure_wemm_video_reader, iter_wemm_media, normalize_wemm_inputs


def test_wemm_builtin_specs_have_both_sources_and_revisions():
    spec_path = Path(__file__).parents[1] / "model_spec.json"
    specs = {
        item["model_name"]: item
        for item in json.loads(spec_path.read_text())
        if item["model_name"].startswith("WeMM-Embedding-")
    }
    assert set(specs) == {
        "WeMM-Embedding-2B",
        "WeMM-Embedding-4B",
        "WeMM-Embedding-9B",
    }
    assert {name: item["dimensions"] for name, item in specs.items()} == {
        "WeMM-Embedding-2B": 2048,
        "WeMM-Embedding-4B": 2560,
        "WeMM-Embedding-9B": 4096,
    }
    for name, item in specs.items():
        assert item["max_tokens"] == 262144
        sources = item["model_specs"][0]["model_src"]
        assert sources["huggingface"] == {
            "model_id": f"tencent/{name}",
            "model_revision": "main",
            "quantizations": ["none"],
        }
        assert sources["modelscope"] == {
            "model_id": f"Tencent-Hunyuan/{name}",
            "model_revision": "master",
            "quantizations": ["none"],
        }
        packages = item["virtualenv"]["packages"]
        assert any(package.startswith("qwen-vl-utils==0.0.14") for package in packages)
        assert any(package.startswith("#system_torch#") for package in packages)
        assert any(package.startswith("#system_torchvision#") for package in packages)
        assert not any("[decord]" in package for package in packages)


def test_wemm_engines_are_discoverable_with_virtualenv():
    _install()
    for model_name in (
        "WeMM-Embedding-2B",
        "WeMM-Embedding-4B",
        "WeMM-Embedding-9B",
    ):
        params = get_engine_params_by_name_with_virtual_env(
            "embedding", model_name, enable_virtual_env=True
        )
        assert isinstance(params, dict)
        available = {
            engine
            for engine, engine_params in params.items()
            if isinstance(engine_params, list)
        }
        assert available == {"sentence_transformers", "vllm"}, (model_name, params)
        for engine in ("sentence_transformers", "vllm"):
            assert isinstance(params.get(engine), list), (model_name, engine, params)


def test_wemm_api_schema_preserves_mixed_multimodal_batch():
    request = CreateEmbeddingRequest(
        model="wemm",
        input=["text", {"image": "x.jpg"}, {"video": "x.mp4"}],
    )
    assert request.input == ["text", {"image": "x.jpg"}, {"video": "x.mp4"}]


def test_normalize_wemm_interleaved_message_and_openai_media_keys():
    messages_batch = normalize_wemm_inputs(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "image.jpg"}},
                {"type": "text", "text": "find this"},
                {"type": "video_url", "video_url": {"url": "clip.mp4"}},
            ],
        }
    )
    messages = messages_batch[0]
    assert messages[0]["content"] == [
        {"type": "image", "image": "image.jpg"},
        {"type": "text", "text": "find this"},
        {"type": "video", "video": "clip.mp4"},
    ]
    assert list(iter_wemm_media(messages)) == [
        ("image", "image.jpg"),
        ("video", "clip.mp4"),
    ]


def test_normalize_wemm_flat_multimodal_input_preserves_order():
    messages = normalize_wemm_inputs(
        {"image": "one.png", "text": "caption", "video": ["a.mp4", "b.mp4"]}
    )[0]
    assert messages[0]["content"] == [
        {"type": "image", "image": "one.png"},
        {"type": "text", "text": "caption"},
        {"type": "video", "video": "a.mp4"},
        {"type": "video", "video": "b.mp4"},
    ]


def test_normalize_wemm_rejects_audio():
    with pytest.raises(ValueError, match="does not support audio"):
        normalize_wemm_inputs({"audio": "sample.wav"})


def test_normalize_wemm_rejects_empty_media():
    with pytest.raises(ValueError, match="image input cannot be empty"):
        normalize_wemm_inputs({"image_url": {}})


def test_wemm_installs_pyav_fallback_when_torchvision_reader_is_absent(monkeypatch):
    vision_process = SimpleNamespace(io=SimpleNamespace(), VIDEO_READER_BACKENDS={})
    qwen_utils = types.ModuleType("qwen_vl_utils")
    qwen_utils.vision_process = vision_process
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", qwen_utils)

    ensure_wemm_video_reader()

    assert callable(vision_process.VIDEO_READER_BACKENDS["torchvision"])
    assert vision_process._xinference_wemm_pyav_fallback is True
