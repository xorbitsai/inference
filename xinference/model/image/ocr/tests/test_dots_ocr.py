# Copyright 2022-2026 XProbe Inc.
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
from types import ModuleType, SimpleNamespace

import PIL.Image
import torch

from ..dots_ocr import DEFAULT_OCR_PROMPT, DotsOCRModel


class _FakeInputs(dict):
    def __init__(self):
        super().__init__(input_ids=torch.tensor([[1, 2]]))
        self.input_ids = self["input_ids"]

    def to(self, _device):
        return self


class _FakeProcessor:
    def __init__(self, decoded_output):
        self.decoded_output = decoded_output
        self.messages = None

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return "rendered prompt"

    def __call__(self, **_kwargs):
        return _FakeInputs()

    def batch_decode(self, *_args, **_kwargs):
        return self.decoded_output


class _FakeModel:
    def generate(self, **_kwargs):
        return torch.tensor([[1, 2, 3]])


def _make_model(decoded_output):
    model = DotsOCRModel(
        model_uid="dots-ocr-test",
        model_spec=SimpleNamespace(model_ability=["ocr"]),
    )
    model._model = _FakeModel()
    model._processor = _FakeProcessor(decoded_output)
    return model


def test_dots_ocr_uses_upstream_prompt_by_default(monkeypatch):
    vision_utils = ModuleType("qwen_vl_utils")
    vision_utils.process_vision_info = lambda _messages: ([object()], None)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", vision_utils)
    model = _make_model(["recognized text"])

    result = model.ocr(PIL.Image.new("RGB", (8, 8), "white"))

    assert model._processor.messages[0]["content"][1] == {
        "type": "text",
        "text": DEFAULT_OCR_PROMPT,
    }
    assert result == "recognized text"
    assert isinstance(result, str)


def test_dots_ocr_returns_empty_string_for_empty_decoding(monkeypatch):
    vision_utils = ModuleType("qwen_vl_utils")
    vision_utils.process_vision_info = lambda _messages: ([object()], None)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", vision_utils)
    model = _make_model([])

    assert model.ocr(PIL.Image.new("RGB", (8, 8), "white")) == ""
