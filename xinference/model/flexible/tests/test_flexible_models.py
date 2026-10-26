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

import json
import os
import shutil
import tempfile
from typing import Dict, Optional
from unittest.mock import Mock

import pytest


def test_register_flexible_model():
    from ..core import FlexibleModelSpec
    from ..custom import register_flexible_model, unregister_flexible_model

    tmp_dir = tempfile.mkdtemp()

    model_spec = FlexibleModelSpec(
        model_name="flexible_model",
        model_uri=os.path.abspath(tmp_dir),
        launcher="xinference.model.flexible.launchers.transformers",
    )

    register_flexible_model(model_spec, persist=False)

    unregister_flexible_model("flexible_model")

    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.parametrize(
    "spec_args, expected_kwargs",
    [
        pytest.param({}, {}, id="omitted"),
        pytest.param({"launcher_args": None}, {}, id="none"),
        pytest.param(
            {"launcher_args": '{"task": "mock", "device": "cpu"}'},
            {"task": "mock", "device": "cpu"},
            id="json",
        ),
    ],
)
def test_create_flexible_model_instance_launcher_args(
    monkeypatch: pytest.MonkeyPatch,
    spec_args: Dict[str, Optional[str]],
    expected_kwargs: Dict[str, str],
) -> None:
    from .. import core

    model_spec = core.FlexibleModelSpec(
        model_name="flexible_model",
        model_uri="mock",
        launcher="xinference.model.flexible.launchers.transformers",
        **spec_args,
    )
    launcher = Mock(return_value=object())
    get_launcher = Mock(return_value=launcher)
    monkeypatch.setattr(core, "match_flexible_model", lambda _: model_spec)
    monkeypatch.setattr(core, "get_launcher", get_launcher)

    model = core.create_flexible_model_instance(
        model_uid="flexible_model", model_name=model_spec.model_name
    )

    get_launcher.assert_called_once_with(model_spec.launcher)
    launcher.assert_called_once_with(
        model_uid="flexible_model", model_spec=model_spec, **expected_kwargs
    )
    assert model is launcher.return_value


def test_create_flexible_model_instance_invalid_launcher_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from .. import core

    model_spec = core.FlexibleModelSpec(
        model_name="flexible_model",
        model_uri="mock",
        launcher="xinference.model.flexible.launchers.transformers",
        launcher_args="{invalid json}",
    )
    launcher = Mock()
    get_launcher = Mock(return_value=launcher)
    monkeypatch.setattr(core, "match_flexible_model", lambda _: model_spec)
    monkeypatch.setattr(core, "get_launcher", get_launcher)

    with pytest.raises(
        json.JSONDecodeError, match="Expecting property name enclosed in double quotes"
    ):
        core.create_flexible_model_instance(
            model_uid="flexible_model", model_name=model_spec.model_name
        )

    get_launcher.assert_not_called()
    launcher.assert_not_called()


def test_model():
    from ..core import FlexibleModelSpec
    from ..utils import get_launcher

    launcher = get_launcher("xinference.model.flexible.launchers.transformers")
    model = launcher(
        model_uid="flexible_model",
        model_spec=FlexibleModelSpec(
            model_name="mock",
            model_uri="mock",
            launcher="xinference.model.flexible.launchers.transformers",
        ),
        task="mock",
    )

    model.load()

    result = model.infer(inputs="hello world")
    # assert result == {"inputs": "hello world"}
    assert result is not None
    assert "inputs" in result
    assert result["inputs"] == "hello world"
