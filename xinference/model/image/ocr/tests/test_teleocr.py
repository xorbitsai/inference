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
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from PIL import Image

from ... import load_model_family_from_json
from ...cache_manager import ImageCacheManager
from ...core import (
    BUILTIN_IMAGE_MODELS,
    _select_ocr_model_family,
    create_image_model_instance,
    create_ocr_model_instance,
)
from .. import register_builtin_ocr_engines
from ..ocr_family import OCR_ENGINES, generate_engine_config_by_model_name
from ..teleocr import LlamaCppTeleOCRModel, TeleOCRModel
from ..vllm import VLLMTeleOCRModel


def _families():
    families = {}
    load_model_family_from_json(str(Path(__file__).parents[2] / "models"), families)
    return families["TeleOCR"]


def test_teleocr_sources_and_engine_selection(monkeypatch):
    families = _families()
    pytorch = [s for s in families if s.model_format == "pytorch"]
    gguf = [s for s in families if s.model_format == "ggufv2"]
    assert {(s.model_hub, s.model_id, s.model_revision) for s in pytorch} == {
        ("huggingface", "StarDoc-AI/TeleOCR", "main"),
        ("modelscope", "XingChen-AGI/TeleOCR", "master"),
    }
    assert {(s.model_hub, s.model_id, s.model_revision) for s in gguf} == {
        ("huggingface", "nandraj/NaviDC-OCR-GGUF", "main"),
        ("modelscope", "tardigrade4ai/NaviDC-OCR-GGUF", "master"),
    }
    assert {s.quantization for s in gguf if s.model_hub == "huggingface"} == {
        "Q4_K_M",
        "Q2_K",
        "Q3_K_L",
        "Q3_K_M",
        "Q3_K_S",
        "Q4_K_S",
        "Q5_K_M",
        "Q5_K_S",
        "Q6_K",
        "Q8_0",
        "f16",
    }

    monkeypatch.setitem(BUILTIN_IMAGE_MODELS, "TeleOCR", families)
    for spec in families:
        assert TeleOCRModel.match(spec) == (spec.model_format == "pytorch")
        assert LlamaCppTeleOCRModel.match(spec) == (spec.model_format == "ggufv2")
        assert VLLMTeleOCRModel.match(spec) == (spec.model_format == "pytorch")

    old = OCR_ENGINES.pop("TeleOCR", None)
    try:
        register_builtin_ocr_engines()
        for spec in families:
            generate_engine_config_by_model_name(spec)
        assert OCR_ENGINES["TeleOCR"]["transformers"][0]["ocr_class"] is TeleOCRModel
        assert OCR_ENGINES["TeleOCR"]["vllm"][0]["ocr_class"] is VLLMTeleOCRModel
        assert (
            OCR_ENGINES["TeleOCR"]["llama.cpp"][0]["ocr_class"] is LlamaCppTeleOCRModel
        )
        selected = _select_ocr_model_family(
            "TeleOCR", "llama.cpp", "modelscope", quantization="Q4_K_M"
        )
        assert selected.model_hub == "modelscope"
        assert selected.model_format == "ggufv2"
        monkeypatch.setattr(
            ImageCacheManager, "cache_ocr_gguf", lambda self: "/tmp/model.gguf"
        )
        model = create_ocr_model_instance("teleocr-test", selected, "llama.cpp")
        assert isinstance(model, LlamaCppTeleOCRModel)
        assert model._model_path == "/tmp/model.gguf"
    finally:
        OCR_ENGINES.pop("TeleOCR", None)
        if old is not None:
            OCR_ENGINES["TeleOCR"] = old


def test_teleocr_gguf_downloads_only_model_and_projector(monkeypatch, tmp_path):
    import huggingface_hub

    import xinference.model.utils as utils

    spec = next(
        s
        for s in _families()
        if s.model_format == "ggufv2"
        and s.model_hub == "huggingface"
        and s.quantization == "Q4_K_M"
    )
    manager = ImageCacheManager(spec)
    manager._cache_dir = str(tmp_path / "cache")
    calls = []

    def download(repo_id, filename, revision):
        calls.append((repo_id, filename, revision))
        result = tmp_path / filename
        result.write_bytes(b"gguf")
        return str(result)

    monkeypatch.setattr(
        utils,
        "retry_download",
        lambda fn, name, info, *args, **kwargs: fn(*args, **kwargs),
    )
    monkeypatch.setattr(utils, "IS_NEW_HUGGINGFACE_HUB", True)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    model_path = manager.cache_ocr_gguf()
    assert Path(model_path).is_file()
    assert calls == [
        ("nandraj/NaviDC-OCR-GGUF", "NaviDC-OCR-Q4_K_M.gguf", "main"),
        ("nandraj/NaviDC-OCR-GGUF", "NaviDC-OCR-mmproj-q8_0.gguf", "main"),
    ]
    assert manager.cache_ocr_gguf() == model_path
    assert len(calls) == 2


def test_teleocr_accepts_image_gguf_options(monkeypatch):
    import xinference.model.image.core as core

    pytorch = next(s for s in _families() if s.model_format == "pytorch")
    selected = next(
        s for s in _families() if getattr(s, "quantization", None) == "Q4_K_M"
    )
    calls = {}

    monkeypatch.setattr(core, "match_diffusion", lambda name, hub: pytorch)

    def select(name, engine, hub, **kwargs):
        calls["selection"] = (name, engine, kwargs)
        return selected

    def create(**kwargs):
        calls["instance"] = kwargs
        return "model"

    monkeypatch.setattr(core, "_select_ocr_model_family", select)
    monkeypatch.setattr(core, "create_ocr_model_instance", create)
    assert (
        create_image_model_instance(
            model_uid="teleocr-test",
            model_name="TeleOCR",
            gguf_quantization="Q4_K_M",
            gguf_model_path="/local/model.gguf",
        )
        == "model"
    )
    assert calls["selection"] == (
        "TeleOCR",
        "llama.cpp",
        {"model_format": "ggufv2", "quantization": "Q4_K_M"},
    )
    assert calls["instance"]["model_path"] == "/local/model.gguf"


def test_teleocr_gguf_modelscope_revision(monkeypatch, tmp_path):
    import xinference.model.utils as utils

    spec = next(
        s
        for s in _families()
        if s.model_format == "ggufv2"
        and s.model_hub == "modelscope"
        and s.quantization == "Q4_K_M"
    )
    manager = ImageCacheManager(spec)
    manager._cache_dir = str(tmp_path / "cache")
    calls = []

    def download(repo_id, filename, revision):
        calls.append((repo_id, filename, revision))
        result = tmp_path / filename
        result.write_bytes(b"gguf")
        return str(result)

    file_download = ModuleType("modelscope.hub.file_download")
    file_download.model_file_download = download
    monkeypatch.setitem(sys.modules, "modelscope", ModuleType("modelscope"))
    monkeypatch.setitem(sys.modules, "modelscope.hub", ModuleType("modelscope.hub"))
    monkeypatch.setitem(sys.modules, "modelscope.hub.file_download", file_download)
    monkeypatch.setattr(
        utils,
        "retry_download",
        lambda fn, name, info, *args, **kwargs: fn(*args, **kwargs),
    )

    assert Path(manager.cache_ocr_gguf()).is_file()
    assert calls == [
        ("tardigrade4ai/NaviDC-OCR-GGUF", "NaviDC-OCR-Q4_K_M.gguf", "master"),
        (
            "tardigrade4ai/NaviDC-OCR-GGUF",
            "NaviDC-OCR-mmproj-q8_0.gguf",
            "master",
        ),
    ]


def test_teleocr_gguf_ocr_request(monkeypatch, tmp_path):
    model_path = tmp_path / "NaviDC-OCR-Q4_K_M.gguf"
    projector = tmp_path / "NaviDC-OCR-mmproj-q8_0.gguf"
    model_path.write_bytes(b"gguf")
    projector.write_bytes(b"gguf")

    class Params:
        def __init__(self):
            self.mmproj = SimpleNamespace(path=None)

    class Server:
        def __init__(self, params):
            self.params = params

        def handle_chat_completions(self, request, callback):
            self.request = request
            callback({"choices": [{"message": {"content": "  text  "}}]})

    module = ModuleType("xllamacpp")
    module.CommonParams = Params
    module.Server = Server
    monkeypatch.setitem(sys.modules, "xllamacpp", module)

    model = LlamaCppTeleOCRModel("teleocr-test", str(model_path))
    assert model.ocr(Image.new("RGBA", (4, 4)), max_new_tokens=128) == "text"
    request = model._model.request
    assert model._model.params.mmproj.path == str(projector)
    assert request["max_tokens"] == 128
    assert request["temperature"] == 0
    content = request["messages"][1]["content"]
    assert content[1]["text"] == TeleOCRModel.DEFAULT_PROMPT
    assert base64.b64decode(content[0]["image_url"]["url"].split(",", 1)[1]).startswith(
        b"\x89PNG"
    )
