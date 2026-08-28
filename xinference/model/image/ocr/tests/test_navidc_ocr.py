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

from ... import load_model_family_from_json
from .. import register_builtin_ocr_engines
from ..navidc_ocr import NaviDCOCRModel
from ..ocr_family import OCR_ENGINES, generate_engine_config_by_model_name


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
        'accelerate==1.12.0 ; #engine# == "transformers"' in spec.virtualenv.packages
        for spec in families
    )

    previous = OCR_ENGINES.pop("NaviDC-OCR", None)
    try:
        register_builtin_ocr_engines()
        generate_engine_config_by_model_name(specs["huggingface"])
        engine_params = OCR_ENGINES["NaviDC-OCR"]["transformers"]
        assert engine_params[0]["ocr_class"] is NaviDCOCRModel
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
