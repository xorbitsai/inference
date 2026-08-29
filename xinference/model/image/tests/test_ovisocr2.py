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

from types import SimpleNamespace
from unittest.mock import MagicMock

import PIL.Image
import torch


def test_ovisocr2_metadata_and_engine_registration():
    from .. import load_model_family_from_json
    from ..ocr import OvisOCR2Model, VLLMOvisOCR2Model, register_builtin_ocr_engines
    from ..ocr.ocr_family import (
        OCR_ENGINES,
        SUPPORTED_ENGINES,
        generate_engine_config_by_model_name,
    )

    models = {}
    load_model_family_from_json("model_spec.json", models)
    specs = models["OvisOCR2"]

    assert {spec.model_hub for spec in specs} == {"huggingface", "modelscope"}
    assert {spec.model_hub: spec.model_revision for spec in specs} == {
        "huggingface": "main",
        "modelscope": "master",
    }
    assert all(spec.model_family == "ocr" for spec in specs)
    assert all(spec.model_ability == ["ocr"] for spec in specs)

    old_engines = OCR_ENGINES.pop("OvisOCR2", None)
    old_supported = {
        engine: list(classes) for engine, classes in SUPPORTED_ENGINES.items()
    }
    try:
        register_builtin_ocr_engines()
        generate_engine_config_by_model_name(specs[0])

        engines = OCR_ENGINES["OvisOCR2"]
        assert set(engines) == {"transformers", "vllm"}
        assert engines["transformers"][0]["ocr_class"] is OvisOCR2Model
        assert engines["vllm"][0]["ocr_class"] is VLLMOvisOCR2Model
    finally:
        OCR_ENGINES.pop("OvisOCR2", None)
        if old_engines is not None:
            OCR_ENGINES["OvisOCR2"] = old_engines
        SUPPORTED_ENGINES.clear()
        SUPPORTED_ENGINES.update(old_supported)


def test_ovisocr2_filters_generated_image_tags():
    from ..ocr.ovisocr2 import OvisOCR2Model

    raw = (
        "first paragraph\n\n"
        '<img src="images/bbox_10_20_30_40.jpg" />\n\n'
        "last paragraph"
    )

    assert OvisOCR2Model._postprocess_output(raw, filter_imgtags=True) == (
        "first paragraph\n\nlast paragraph"
    )
    assert OvisOCR2Model._postprocess_output(raw, filter_imgtags=False) == raw


def test_ovisocr2_cleans_truncated_repeated_tail():
    from ..ocr.ovisocr2 import OvisOCR2Model

    prefix = "x" * 8000 + "\n"
    repeated_unit = "0123456789ABCDEF"
    raw = prefix + repeated_unit * 10

    assert OvisOCR2Model._clean_truncated_repeats(raw) == prefix + repeated_unit


class _FakeInputs(dict):
    def to(self, _device):
        return self


class _FakeProcessor:
    def __init__(self, raw_output):
        self._raw_output = raw_output

    def apply_chat_template(self, *_args, **_kwargs):
        return _FakeInputs(input_ids=torch.tensor([[1, 2]]))

    def batch_decode(self, *_args, **_kwargs):
        return [self._raw_output]


class _FakeTransformersModel:
    device = "cpu"

    def generate(self, **_kwargs):
        return torch.tensor([[1, 2, 3]])


def test_transformers_adapter_applies_shared_postprocessing(monkeypatch):
    from ..ocr.ovisocr2 import OvisOCR2Model

    raw = '<img src="images/bbox_1_2_3_4.jpg" />'
    calls = []

    def fake_postprocess(cls, text, filter_imgtags=True):
        calls.append((cls, text, filter_imgtags))
        return "processed"

    monkeypatch.setattr(
        OvisOCR2Model, "_postprocess_output", classmethod(fake_postprocess)
    )
    model = OvisOCR2Model(
        model_uid="ovis-test",
        model_path="/unused",
        model_spec=MagicMock(model_ability=["ocr"]),
    )
    model._model = _FakeTransformersModel()
    model._processor = _FakeProcessor(raw)

    image = PIL.Image.new("RGB", (8, 8), "white")
    assert model.ocr(image) == "processed"
    assert calls == [(OvisOCR2Model, raw, True)]


class _FakeTokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return "prompt"


class _FakeVLLMModel:
    def __init__(self, raw_output):
        self._raw_output = raw_output

    def generate(self, *_args, **_kwargs):
        return [SimpleNamespace(outputs=[SimpleNamespace(text=self._raw_output)])]


def test_vllm_adapter_applies_shared_postprocessing(monkeypatch):
    from ..ocr import vllm as vllm_module
    from ..ocr.ovisocr2 import OvisOCR2Model
    from ..ocr.vllm import VLLMOvisOCR2Model

    raw = '<img src="images/bbox_1_2_3_4.jpg" />'
    calls = []

    def fake_postprocess(cls, text, filter_imgtags=True):
        calls.append((cls, text, filter_imgtags))
        return "processed"

    monkeypatch.setattr(
        OvisOCR2Model, "_postprocess_output", classmethod(fake_postprocess)
    )
    monkeypatch.setattr(vllm_module, "_build_sampling_params", lambda _kwargs: object())

    model = VLLMOvisOCR2Model(
        model_uid="ovis-vllm-test",
        model_path="/unused",
        model_spec=MagicMock(model_ability=["ocr"]),
    )
    model._model = _FakeVLLMModel(raw)
    model._tokenizer = _FakeTokenizer()

    image = PIL.Image.new("RGB", (8, 8), "white")
    assert model.ocr(image) == "processed"
    assert calls == [(VLLMOvisOCR2Model, raw, True)]
