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

import importlib.util
import io

import pytest


def test_deepdoc_registration():
    from .. import BUILTIN_IMAGE_MODELS, register_builtin_model
    from ..ocr.deepdoc import DeepDocModel
    from ..ocr.ocr_family import OCR_ENGINES

    register_builtin_model()
    assert "DeepDoc" in BUILTIN_IMAGE_MODELS
    family = BUILTIN_IMAGE_MODELS["DeepDoc"][0]
    assert family.model_ability == ["ocr"]
    assert DeepDocModel.match(family)
    assert "deepdoc" in OCR_ENGINES["DeepDoc"]


def test_deepdoc_uses_standard_device_scheduling():
    """DeepDoc is CUDA-capable and must not bypass worker GPU allocation.

    On Linux x86_64, deepdoc-lib installs onnxruntime-gpu and selects
    CUDAExecutionProvider when CUDA is visible. The worker's standard device
    allocation restricts it to the assigned GPU through CUDA_VISIBLE_DEVICES.
    """
    from .. import register_builtin_model
    from ..ocr.deepdoc import DeepDocModel

    register_builtin_model()
    assert getattr(DeepDocModel, "cpu_only", False) is False


@pytest.mark.parametrize(
    ("has_cuda", "expected", "unexpected"),
    [
        (True, "deepdoc-lib[gpu]~=0.2.2", "deepdoc-lib~=0.2.2"),
        (False, "deepdoc-lib~=0.2.2", "deepdoc-lib[gpu]~=0.2.2"),
    ],
)
def test_deepdoc_virtualenv_selects_runtime_package(
    monkeypatch, has_cuda, expected, unexpected
):
    from xoscar.virtualenv import core as virtualenv_core

    from ....core.utils import filter_virtualenv_packages_by_markers
    from .. import BUILTIN_IMAGE_MODELS, register_builtin_model

    register_builtin_model()
    packages = BUILTIN_IMAGE_MODELS["DeepDoc"][0].virtualenv.packages
    # Xinference must preserve xoscar's extended has_cuda marker until the
    # virtual environment is created on the target worker.
    prepared = filter_virtualenv_packages_by_markers(packages, None, None)
    assert any("has_cuda" in package for package in prepared)

    env = virtualenv_core.get_env()
    env["has_cuda"] = has_cuda
    monkeypatch.setattr(virtualenv_core, "get_env", lambda: env)
    selected = virtualenv_core.filter_requirements(prepared)

    assert expected in selected
    assert unexpected not in selected
    assert "transformers<5,>=4.51.0" in selected


def test_deepdoc_gpu_repairs_onnxruntime_namespace(monkeypatch):
    from unittest.mock import MagicMock, call

    from ....core.worker import WorkerActor
    from ....model.core import VirtualEnvSettings

    manager = MagicMock(env_path="/tmp/deepdoc-test-venv")
    monkeypatch.setattr(WorkerActor, "_is_cuda_device_available", lambda: True)
    uninstall = MagicMock()
    monkeypatch.setattr(WorkerActor, "_uninstall_venv_package", uninstall)

    WorkerActor._prepare_virtual_env(
        manager,
        VirtualEnvSettings(
            packages=[
                "deepdoc-lib[gpu]~=0.2.2 ; has_cuda",
                "deepdoc-lib~=0.2.2 ; not has_cuda",
                "transformers<5",
            ],
            inherit_pip_config=False,
        ),
        None,
        "deepdoc",
        model_name="DeepDoc",
    )

    assert uninstall.call_args_list == [
        call(manager, "onnxruntime"),
        call(manager, "onnxruntime-gpu"),
    ]
    assert manager.install_packages.call_args_list[-1].args == (
        ["onnxruntime-gpu>=1.19.2"],
    )
    assert manager.install_packages.call_args_list[-1].kwargs["skip_installed"] is False


def _make_unloaded_model():
    from unittest.mock import MagicMock

    from ..ocr.deepdoc import DeepDocModel

    model = DeepDocModel(
        model_uid="test_uid",
        model_path="/tmp/unused",
        model_spec=MagicMock(model_ability=["ocr"]),
    )
    # bypass load(): tests below never touch the real onnx models
    model._ocr = MagicMock()
    return model


def test_deepdoc_input_validation():
    from PIL import Image

    model = _make_unloaded_model()

    with pytest.raises(ValueError, match="cannot be None"):
        model.ocr(None)

    with pytest.raises(ValueError, match="cannot contain None"):
        model.ocr([Image.new("RGB", (8, 8), "white"), None])


def test_deepdoc_threshold_parsing():
    from unittest.mock import MagicMock

    from PIL import Image

    model = _make_unloaded_model()
    layout = MagicMock()
    layout.forward.return_value = [[{"type": "text"}]]
    model._layout_recognizer = layout

    image = Image.new("RGB", (8, 8), "white")

    # an explicit JSON null falls back to the default threshold; structured
    # results come back as dicts (the REST layer serializes them exactly once)
    payload = model.ocr(image, task="layout", threshold=None)
    assert payload["task"] == "layout"
    assert payload["layouts"] == [{"type": "text"}]
    assert layout.forward.call_args.kwargs["thr"] == pytest.approx(0.2)

    # numeric strings are accepted
    model.ocr(image, task="layout", threshold="0.5")
    assert layout.forward.call_args.kwargs["thr"] == pytest.approx(0.5)

    # invalid values raise a clear error instead of a bare TypeError
    with pytest.raises(ValueError, match="Invalid threshold"):
        model.ocr(image, task="layout", threshold="abc")

    # empty recognizer output degrades to an empty layouts list
    layout.forward.return_value = []
    payload = model.ocr(image, task="layout")
    assert payload["layouts"] == []

    table = MagicMock()
    table.return_value = [[{"label": "table column"}]]
    model._table_recognizer = table
    payload = model.ocr(image, task="table", threshold=None)
    assert payload["structures"] == [{"label": "table column"}]
    assert table.call_args.kwargs["thr"] == pytest.approx(0.2)


@pytest.mark.skipif(
    importlib.util.find_spec("deepdoc") is None,
    reason="Skip because deepdoc-lib is not installed",
)
def test_deepdoc_ocr(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="deepdoc_test",
        model_name="DeepDoc",
        model_type="image",
        model_engine="deepdoc",
    )
    model = client.get_model(model_uid)

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (640, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 40), "Xinference DeepDoc", fill="black")
    draw.text((40, 90), "Hello World 2026", fill="black")
    bio = io.BytesIO()
    image.save(bio, format="PNG")

    # default task: plain text
    r = model.ocr(image=bio.getvalue())
    assert isinstance(r, str)

    # ocr with return_dict returns lines with boxes and scores; the client
    # parses the JSON body, so structured results arrive as dicts
    payload = model.ocr(image=bio.getvalue(), return_dict=True)
    assert payload["task"] == "ocr"
    assert isinstance(payload["lines"], list)

    # layout task returns layout blocks
    payload = model.ocr(image=bio.getvalue(), task="layout")
    assert payload["task"] == "layout"
    assert isinstance(payload["layouts"], list)

    # table task returns table structures
    payload = model.ocr(image=bio.getvalue(), task="table")
    assert payload["task"] == "table"
    assert isinstance(payload["structures"], list)
