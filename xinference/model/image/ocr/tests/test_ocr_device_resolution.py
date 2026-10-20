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

import torch

from .. import paddleocr_vl, unlimited_ocr
from ..paddleocr_vl import PaddleOCRVLModel
from ..unlimited_ocr import UnlimitedOCRModel


def _spec():
    return SimpleNamespace(model_ability=["ocr"])


class _RecordingModel:
    """Minimal stand-in that records the device it is moved to."""

    def __init__(self, record):
        self._record = record
        self.config = SimpleNamespace()

    def to(self, device):
        self._record["device"] = str(device)
        return self

    def eval(self):
        return self


def _install_fake_transformers(monkeypatch, record):
    module = ModuleType("transformers")

    class _AutoProcessor:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return SimpleNamespace()

    class _AutoTokenizer:
        eos_token_id = 0

        @staticmethod
        def from_pretrained(path, **kwargs):
            return _AutoTokenizer()

    class _AutoConfig:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return SimpleNamespace(language_config={"hidden_size": 8})

    class _Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            record.setdefault("dtypes", []).append(kwargs.get("torch_dtype"))
            return _RecordingModel(record)

    module.AutoProcessor = _AutoProcessor
    module.AutoTokenizer = _AutoTokenizer
    module.AutoConfig = _AutoConfig
    module.AutoModel = _Loader
    module.AutoModelForCausalLM = _Loader

    utils = ModuleType("transformers.utils")
    import_utils = ModuleType("transformers.utils.import_utils")
    import_utils.is_torch_fx_available = lambda: False
    utils.import_utils = import_utils
    module.utils = utils

    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setitem(sys.modules, "transformers.utils", utils)
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", import_utils)


def test_paddleocr_vl_defers_to_accelerator_registry(monkeypatch, tmp_path):
    """No device configured: the accelerator registry decides, not CUDA-or-CPU."""
    record = {}
    _install_fake_transformers(monkeypatch, record)
    monkeypatch.setattr(paddleocr_vl, "get_available_device", lambda: "npu")

    model = PaddleOCRVLModel("paddleocr-test", str(tmp_path), model_spec=_spec())
    model.load()

    assert record["device"] == "npu"
    assert record["dtypes"] == [torch.bfloat16]


def test_paddleocr_vl_honours_an_explicit_device(monkeypatch, tmp_path):
    """An explicitly requested device (with an index) must not be dropped."""
    record = {}
    _install_fake_transformers(monkeypatch, record)
    monkeypatch.setattr(paddleocr_vl, "get_available_device", lambda: "npu")

    model = PaddleOCRVLModel(
        "paddleocr-test", str(tmp_path), model_spec=_spec(), device="cuda:1"
    )
    model.load()

    assert record["device"] == "cuda:1"
    assert record["dtypes"] == [torch.bfloat16]


def test_paddleocr_vl_keeps_explicit_cpu_dtype(monkeypatch, tmp_path):
    record = {}
    _install_fake_transformers(monkeypatch, record)
    monkeypatch.setattr(paddleocr_vl, "get_available_device", lambda: "npu")

    model = PaddleOCRVLModel(
        "paddleocr-test", str(tmp_path), model_spec=_spec(), device="cpu"
    )
    model.load()

    assert record["device"] == "cpu"
    assert record["dtypes"] == [torch.float32]


def test_unlimited_ocr_defers_to_accelerator_registry(monkeypatch, tmp_path):
    """No device configured: the accelerator registry decides, not CUDA-or-CPU."""
    record = {}
    _install_fake_transformers(monkeypatch, record)
    monkeypatch.setattr(unlimited_ocr, "get_available_device", lambda: "npu")

    model = UnlimitedOCRModel("unlimited-test", str(tmp_path), model_spec=_spec())
    model.load()

    assert record["device"] == "npu"
    assert record["dtypes"] == [torch.bfloat16]


def test_unlimited_ocr_keeps_cpu_on_explicit_cpu(monkeypatch, tmp_path):
    record = {}
    _install_fake_transformers(monkeypatch, record)
    monkeypatch.setattr(unlimited_ocr, "get_available_device", lambda: "npu")

    model = UnlimitedOCRModel(
        "unlimited-test", str(tmp_path), model_spec=_spec(), device="cpu"
    )
    model.load()

    # The CPU branch loads without moving the model.
    assert "device" not in record
    assert record["dtypes"] == [torch.float32]
