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
import json

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

    # an explicit JSON null falls back to the default threshold
    payload = json.loads(model.ocr(image, task="layout", threshold=None))
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
    payload = json.loads(model.ocr(image, task="layout"))
    assert payload["layouts"] == []

    table = MagicMock()
    table.return_value = [[{"label": "table column"}]]
    model._table_recognizer = table
    payload = json.loads(model.ocr(image, task="table", threshold=None))
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

    # ocr with return_dict returns lines with boxes and scores
    r = model.ocr(image=bio.getvalue(), return_dict=True)
    payload = json.loads(r)
    assert payload["task"] == "ocr"
    assert isinstance(payload["lines"], list)

    # layout task returns JSON
    r = model.ocr(image=bio.getvalue(), task="layout")
    payload = json.loads(r)
    assert payload["task"] == "layout"
    assert isinstance(payload["layouts"], list)

    # table task returns JSON structures
    r = model.ocr(image=bio.getvalue(), task="table")
    payload = json.loads(r)
    assert payload["task"] == "table"
    assert isinstance(payload["structures"], list)
