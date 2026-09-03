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
from types import ModuleType, SimpleNamespace

import pytest


def _model_spec(*, is_builtin=False):
    return SimpleNamespace(model_ability=["ocr"], is_builtin=is_builtin)


def test_monkeyocr_metadata_and_model_path_validation():
    from ..ocr.monkeyocr import MonkeyOCRModel

    assert MonkeyOCRModel.required_libs == ("transformers", "qwen_vl_utils")
    with pytest.raises(ValueError, match="model_path is required"):
        MonkeyOCRModel(model_uid="monkeyocr", model_path=None)


@pytest.mark.parametrize(
    ("is_builtin", "expected_trust_remote_code"), [(False, False), (True, True)]
)
def test_monkeyocr_preserves_remote_code_trust_boundary(
    monkeypatch, is_builtin, expected_trust_remote_code
):
    from .... import constants
    from ..ocr.monkeyocr import MonkeyOCRModel

    calls = []

    class FakeMonkeyChat:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    fake_module = ModuleType(
        "xinference.thirdparty.monkeyocr.magic_pdf.model.custom_model"
    )
    fake_module.MonkeyChat_transformers = FakeMonkeyChat
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    monkeypatch.setattr(constants, "XINFERENCE_TRUST_REMOTE_CODE", False)

    model = MonkeyOCRModel(
        model_uid="monkeyocr",
        model_path="/models/MonkeyOCR",
        model_spec=_model_spec(is_builtin=is_builtin),
    )
    model.load()

    assert calls == [
        (
            ("/models/MonkeyOCR/Recognition",),
            {"device": None, "trust_remote_code": expected_trust_remote_code},
        )
    ]


@pytest.mark.parametrize(("batch_result", "expected"), [(["text"], "text"), ([], "")])
def test_monkeyocr_returns_single_ocr_value(batch_result, expected):
    from ..ocr.monkeyocr import MonkeyOCRModel

    model = MonkeyOCRModel(
        model_uid="monkeyocr",
        model_path="/models/MonkeyOCR",
        model_spec=_model_spec(),
    )
    model._model = SimpleNamespace(batch_inference=lambda *_args: batch_result)

    assert model.ocr("image") == expected


def test_monkeyocr_cpu_uses_float32_and_forwards_trust(monkeypatch):
    import torch

    from ....thirdparty.monkeyocr.magic_pdf.model import custom_model

    calls = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["model"] = (args, kwargs)
            return cls()

        def eval(self):
            return None

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["processor"] = (args, kwargs)
            return cls()

        def __init__(self):
            self.tokenizer = SimpleNamespace(padding_side=None)

    fake_transformers = ModuleType("transformers")
    fake_transformers.Qwen2_5_VLForConditionalGeneration = FakeModel
    fake_transformers.AutoProcessor = FakeProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(custom_model.importlib.util, "find_spec", lambda _name: None)

    custom_model.MonkeyChat_transformers(
        "/models/MonkeyOCR/Recognition",
        device="cpu",
        trust_remote_code=False,
    )

    assert calls["model"][1]["torch_dtype"] is torch.float32
    assert calls["processor"][1]["trust_remote_code"] is False


def test_monkeyocr_oom_fallback_clears_cache_outside_exception(monkeypatch):
    from ....thirdparty.monkeyocr.magic_pdf.model import custom_model

    model = custom_model.MonkeyChat_transformers.__new__(
        custom_model.MonkeyChat_transformers
    )
    model.device = "cuda:0"
    model.max_batch_size = 1
    model._process_batch = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("CUDA out of memory")
    )
    model._process_single = lambda *_args: "fallback"
    cache_clear_exception_states = []

    def fake_empty_cache():
        cache_clear_exception_states.append(sys.exc_info()[0])

    monkeypatch.setattr(custom_model.torch.cuda, "empty_cache", fake_empty_cache)

    assert model.batch_inference(["image"], ["question"]) == ["fallback"]
    assert cache_clear_exception_states == [None, None]


def test_monkeyocr_single_inference_uses_configured_max_new_tokens(monkeypatch):
    from ....thirdparty.monkeyocr.magic_pdf.model import custom_model

    generated_kwargs = {}

    class FakeInputs(dict):
        input_ids = [[1, 2]]

        def to(self, _device):
            return self

    class FakeProcessor:
        tokenizer = SimpleNamespace(pad_token_id=0)

        def apply_chat_template(self, *_args, **_kwargs):
            return "prompt"

        def __call__(self, **_kwargs):
            return FakeInputs()

        def batch_decode(self, *_args, **_kwargs):
            return ["result"]

    class FakeModel:
        def generate(self, **kwargs):
            generated_kwargs.update(kwargs)
            return [[1, 2, 3]]

    model = custom_model.MonkeyChat_transformers.__new__(
        custom_model.MonkeyChat_transformers
    )
    model.device = "cpu"
    model.max_new_tokens = 321
    model.processor = FakeProcessor()
    model.model = FakeModel()
    monkeypatch.setattr(custom_model, "load_image", lambda image, max_size: image)
    monkeypatch.setattr(
        custom_model, "process_vision_info", lambda _messages: (["x"], [])
    )

    assert model._process_single("image", "question") == "result"
    assert generated_kwargs["max_new_tokens"] == 321
