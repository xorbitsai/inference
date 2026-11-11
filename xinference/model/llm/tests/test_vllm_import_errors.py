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

import builtins
import sys
from types import ModuleType

import pytest

from ..vllm.core import VLLMModel


@pytest.fixture
def fake_vllm(monkeypatch):
    vllm = ModuleType("vllm")
    vllm.__file__ = "/example/vllm/__init__.py"
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    for name, symbol in [
        ("vllm.engine.arg_utils", "AsyncEngineArgs"),
        ("vllm.engine.async_llm_engine", "AsyncLLMEngine"),
        ("vllm.lora.request", "LoRARequest"),
    ]:
        module = ModuleType(name)
        setattr(module, symbol, object())
        monkeypatch.setitem(sys.modules, name, module)
    return vllm


@pytest.mark.parametrize(
    "import_name, error, show_install_hint",
    [
        ("vllm", ModuleNotFoundError("No module named 'vllm'", name="vllm"), True),
        (
            "vllm.engine.arg_utils",
            ModuleNotFoundError("No module named 'triton'", name="triton"),
            False,
        ),
        (
            "vllm.engine.arg_utils",
            ModuleNotFoundError("No module named 'vllm._C'", name="vllm._C"),
            False,
        ),
        (
            "vllm.engine.async_llm_engine",
            ImportError(
                "cannot import name 'AsyncLLM' from 'vllm.v1.engine.async_llm'"
            ),
            False,
        ),
        (
            "vllm.engine.async_llm_engine",
            ImportError("libcudart.so.12: cannot open shared object file"),
            False,
        ),
    ],
)
def test_load_reports_vllm_import_failure(
    fake_vllm, monkeypatch, import_name, error, show_install_hint
):
    original_import = builtins.__import__

    def import_with_failure(name, *args, **kwargs):
        if name == import_name:
            raise error
        return original_import(name, *args, **kwargs)

    model = object.__new__(VLLMModel)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", import_with_failure)
        with pytest.raises(ImportError) as exc_info:
            model.load()

    message = str(exc_info.value)
    assert f"{type(error).__name__}: {error}" in message
    assert ("pip install vllm" in message) is show_install_hint
    assert exc_info.value.__cause__ is error


def test_load_reports_vllm_module_without_version(fake_vllm):
    model = object.__new__(VLLMModel)
    with pytest.raises(ImportError, match="has no __version__ value") as exc_info:
        model.load()

    assert fake_vllm.__file__ in str(exc_info.value)
