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
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
from PIL import Image

from xinference.thirdparty.navidc_ocr import MODEL_ARCHITECTURE, MODEL_CLASS, register

from ... import load_model_family_from_json
from .. import register_builtin_ocr_engines
from ..navidc_ocr import NaviDCOCRModel
from ..ocr_family import OCR_ENGINES, generate_engine_config_by_model_name
from ..vllm import VLLMNaviDCOCRModel


def _load_navidc_families():
    families = {}
    spec_path = Path(__file__).parents[2] / "model_spec.json"
    load_model_family_from_json(str(spec_path), families)
    return families["NaviDC-OCR"]


def test_navidc_ocr_metadata_and_engine_registration():
    families = _load_navidc_families()
    specs = {spec.model_hub: spec for spec in families}

    assert specs["huggingface"].model_id == "StarDoc-AI/NaviDC-OCR"
    assert specs["huggingface"].model_revision == "main"
    assert specs["modelscope"].model_id == "jackpi/NaviDC-OCR"
    assert specs["modelscope"].model_revision == "master"
    assert all(spec.model_ability == ["ocr"] for spec in families)
    assert all(NaviDCOCRModel.match(spec) for spec in families)
    assert all(
        "transformers==4.57.1" in spec.virtualenv.packages for spec in families
    )
    assert all("pillow>=11,<12" in spec.virtualenv.packages for spec in families)
    assert all(
        'accelerate==1.12.0 ; #engine# == "transformers"' in spec.virtualenv.packages
        for spec in families
    )
    assert all(
        'vllm==0.11.0 ; #engine# == "vllm"' in spec.virtualenv.packages
        for spec in families
    )

    previous = OCR_ENGINES.pop("NaviDC-OCR", None)
    try:
        register_builtin_ocr_engines()
        generate_engine_config_by_model_name(specs["huggingface"])
        engine_params = OCR_ENGINES["NaviDC-OCR"]["transformers"]
        assert engine_params[0]["ocr_class"] is NaviDCOCRModel
        vllm_engine_params = OCR_ENGINES["NaviDC-OCR"]["vllm"]
        assert vllm_engine_params[0]["ocr_class"] is VLLMNaviDCOCRModel
    finally:
        OCR_ENGINES.pop("NaviDC-OCR", None)
        if previous is not None:
            OCR_ENGINES["NaviDC-OCR"] = previous


def test_navidc_ocr_load_and_infer(monkeypatch):
    class FakeInputs(dict):
        def to(self, **kwargs):
            self.to_kwargs = kwargs
            return self

    class FakeProcessor:
        def __init__(self):
            self.messages = None
            self.process_kwargs = None
            self.decoded_ids = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.chat_template_kwargs = kwargs
            return "rendered prompt"

        def __call__(self, **kwargs):
            self.process_kwargs = kwargs
            self.inputs = FakeInputs(input_ids=torch.tensor([[11, 12]]))
            return self.inputs

        def batch_decode(self, generated_ids, **kwargs):
            self.decoded_ids = generated_ids
            self.decode_kwargs = kwargs
            return ["  recognized text  "]

    class FakeModel:
        device = torch.device("cpu")
        dtype = torch.float32

        def to(self, device):
            self.to_device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            return torch.tensor([[11, 12, 21, 22]])

    processor = FakeProcessor()
    loaded_model = FakeModel()

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            cls.model_path = model_path
            cls.kwargs = kwargs
            return processor

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            cls.model_path = model_path
            cls.kwargs = kwargs
            return loaded_model

    transformers_module = ModuleType("transformers")
    transformers_module.AutoModel = FakeAutoModel
    transformers_module.AutoProcessor = FakeAutoProcessor
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    model_spec = SimpleNamespace(
        model_name="NaviDC-OCR",
        model_ability=["ocr"],
        is_builtin=True,
    )
    model = NaviDCOCRModel(
        model_uid="navidc-test",
        model_path="/models/navidc",
        device="cpu",
        model_spec=model_spec,
    )
    model.load()

    assert FakeAutoProcessor.model_path == "/models/navidc"
    assert FakeAutoProcessor.kwargs == {
        "trust_remote_code": True,
        "use_fast": True,
    }
    assert FakeAutoModel.kwargs["trust_remote_code"] is True
    assert FakeAutoModel.kwargs["torch_dtype"] is torch.float32
    assert loaded_model.to_device == "cpu"
    assert loaded_model.eval_called is True

    result = model.ocr(
        Image.new("RGBA", (8, 8)),
        prompt="Read this document.",
        system_prompt="System prompt",
        max_new_tokens=123,
    )

    assert result == "recognized text"
    assert processor.messages == [
        {"role": "system", "content": "System prompt"},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Read this document."},
            ],
        },
    ]
    assert processor.process_kwargs["images"][0].mode == "RGB"
    assert processor.inputs.to_kwargs == {
        "device": loaded_model.device,
        "dtype": loaded_model.dtype,
    }
    assert loaded_model.generate_kwargs["max_new_tokens"] == 123
    assert loaded_model.generate_kwargs["do_sample"] is False
    assert loaded_model.generate_kwargs["use_cache"] is True
    assert torch.equal(processor.decoded_ids, torch.tensor([[21, 22]]))


def test_navidc_ocr_vllm_plugin_registration(monkeypatch):
    class FakeModelRegistry:
        registrations = []

        @classmethod
        def register_model(cls, architecture, model_class):
            cls.registrations.append((architecture, model_class))

    vllm_module = ModuleType("vllm")
    vllm_module.ModelRegistry = FakeModelRegistry
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)

    register()

    assert FakeModelRegistry.registrations == [(MODEL_ARCHITECTURE, MODEL_CLASS)]
    project_path = Path(__file__).parents[5] / "pyproject.toml"
    assert (
        'xinference_navidc_ocr = "xinference.thirdparty.navidc_ocr:register"'
        in project_path.read_text()
    )


def test_navidc_ocr_vllm_install_dependencies():
    from xinference.core.utils import filter_virtualenv_packages_by_markers
    from xinference.core.virtual_env_manager import (
        expand_engine_dependency_placeholders,
    )

    family = _load_navidc_families()[0]
    expanded = expand_engine_dependency_placeholders(
        family.virtualenv.packages,
        "vllm",
    )
    prepared = filter_virtualenv_packages_by_markers(expanded, "vllm", None)

    assert [package for package in prepared if package.startswith("vllm")] == [
        "vllm==0.11.0"
    ]
    assert [package for package in prepared if package.startswith("transformers")] == [
        "transformers==4.57.1"
    ]
    assert not any(package.startswith("accelerate") for package in prepared)


def test_navidc_ocr_vllm_load_and_infer(monkeypatch):
    from .. import vllm as vllm_ocr

    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.chat_template_kwargs = kwargs
            return "rendered vllm prompt"

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            cls.model_path = model_path
            cls.kwargs = kwargs
            return processor

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeVLLMModel:
        def generate(self, inputs, sampling_params):
            self.inputs = inputs
            self.sampling_params = sampling_params
            return [SimpleNamespace(outputs=[SimpleNamespace(text="  vllm text  ")])]

        def shutdown(self):
            self.shutdown_called = True

    processor = FakeProcessor()
    loaded_model = FakeVLLMModel()
    loaded = {}

    def fake_load(model_path, model_kwargs):
        loaded["model_path"] = model_path
        loaded["model_kwargs"] = model_kwargs
        return loaded_model

    transformers_module = ModuleType("transformers")
    transformers_module.AutoProcessor = FakeAutoProcessor
    vllm_module = ModuleType("vllm")
    vllm_module.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setattr(vllm_ocr, "_load_vllm_model", fake_load)

    model_spec = SimpleNamespace(
        model_name="NaviDC-OCR",
        model_ability=["ocr"],
        is_builtin=True,
    )
    model = VLLMNaviDCOCRModel(
        model_uid="navidc-vllm-test",
        model_path="/models/navidc",
        model_spec=model_spec,
        torch_dtype=torch.bfloat16,
        hf_overrides={"custom": "value"},
    )
    model.load()

    assert loaded["model_path"] == "/models/navidc"
    assert loaded["model_kwargs"]["hf_overrides"] == {
        "custom": "value",
        "architectures": [MODEL_ARCHITECTURE],
    }
    assert loaded["model_kwargs"]["trust_remote_code"] is True
    assert "torch_dtype" not in loaded["model_kwargs"]
    assert FakeAutoProcessor.model_path == "/models/navidc"
    assert FakeAutoProcessor.kwargs == {
        "trust_remote_code": True,
        "use_fast": True,
    }

    result = model.ocr(
        Image.new("RGBA", (8, 8)),
        prompt="Read with vLLM.",
        system_prompt="vLLM system",
        max_new_tokens=123,
        use_cache=True,
    )

    assert result == "vllm text"
    assert processor.messages == [
        {"role": "system", "content": "vLLM system"},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Read with vLLM."},
            ],
        },
    ]
    assert loaded_model.inputs[0]["prompt"] == "rendered vllm prompt"
    assert loaded_model.inputs[0]["multi_modal_data"]["image"][0].mode == "RGB"
    assert loaded_model.sampling_params.kwargs == {
        "max_tokens": 123,
        "temperature": 0.0,
    }

    model.stop()
    assert loaded_model.shutdown_called is True
    assert model._model is None
    assert model._processor is None
