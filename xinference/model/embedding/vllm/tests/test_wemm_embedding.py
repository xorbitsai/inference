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
from types import SimpleNamespace
from unittest.mock import MagicMock

from ..core import VLLMEmbeddingModel


def test_wemm_vllm_load_uses_memory_default_and_preserves_override(
    monkeypatch, tmp_path
):
    (tmp_path / "embedding_chat_template.jinja").write_text(
        "{{ messages }}", encoding="utf-8"
    )
    calls = []

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        @staticmethod
        def get_tokenizer():
            return object()

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.__version__ = "0.28.0"
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    for kwargs in ({}, {"gpu_memory_utilization": 0.4}):
        model = VLLMEmbeddingModel.__new__(VLLMEmbeddingModel)
        model.model_family = SimpleNamespace(model_name="WeMM-Embedding-2B")
        model._model_path = str(tmp_path)
        model._kwargs = kwargs
        model.load()

    assert calls == [
        {
            "model": str(tmp_path),
            "runner": "pooling",
            "gpu_memory_utilization": 0.6,
        },
        {
            "model": str(tmp_path),
            "runner": "pooling",
            "gpu_memory_utilization": 0.4,
        },
    ]


def test_wemm_vllm_builds_prompt_and_multimodal_data(monkeypatch, tmp_path):
    image_path = tmp_path / "image.png"
    video_path = tmp_path / "video.mp4"
    image_path.write_bytes(b"image")
    video_path.write_bytes(b"video")

    fake_vllm = types.ModuleType("vllm")
    fake_multimodal = types.ModuleType("vllm.multimodal")
    fake_utils = types.ModuleType("vllm.multimodal.utils")
    fake_utils.fetch_image = lambda value: f"image:{value}"
    fake_utils.fetch_video = lambda value: f"video:{value}"
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.multimodal", fake_multimodal)
    monkeypatch.setitem(sys.modules, "vllm.multimodal.utils", fake_utils)

    model = VLLMEmbeddingModel.__new__(VLLMEmbeddingModel)
    model._model = MagicMock()
    model._tokenizer = MagicMock()
    model._tokenizer.apply_chat_template.return_value = "rendered"
    model._chat_template = "template"
    pool_params = object()
    input_value = {
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": "caption"},
            {"type": "video", "video": str(video_path)},
        ],
    }

    model._embed_wemm(input_value, pool_params)

    model._tokenizer.apply_chat_template.assert_called_once_with(
        [input_value],
        tokenize=False,
        add_generation_prompt=False,
        chat_template="template",
    )
    embed_input = model._model.embed.call_args.args[0][0]
    assert embed_input["prompt"] == "rendered"
    assert embed_input["multi_modal_data"] == {
        "image": f"image:file://{image_path}",
        "video": f"video:file://{video_path}",
    }
    assert model._model.embed.call_args.kwargs == {
        "use_tqdm": False,
        "pooling_params": pool_params,
    }


def test_wemm_vllm_match_requires_supported_prefix_and_format(monkeypatch):
    from xinference.model.utils import virtualenv_discovery_var

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = "0.27.0"
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    family = SimpleNamespace(model_name="WeMM-Embedding-2B")
    token = virtualenv_discovery_var.set(False)
    try:
        assert (
            VLLMEmbeddingModel.match_json(
                family, SimpleNamespace(model_format="pytorch"), "none"
            )
            is True
        )
        assert (
            VLLMEmbeddingModel.match_json(
                family, SimpleNamespace(model_format="ggufv2"), "none"
            )[0]
            is False
        )
    finally:
        virtualenv_discovery_var.reset(token)
