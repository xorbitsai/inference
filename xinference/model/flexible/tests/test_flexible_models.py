# Copyright 2022-2024 XProbe Inc.
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

import os
import shutil
import tempfile


def test_register_flexible_model():
    from ..core import (
        FlexibleModelSpec,
        register_flexible_model,
        unregister_flexible_model,
    )

    tmp_dir = tempfile.mkdtemp()

    model_spec = FlexibleModelSpec(
        model_name="flexible_model",
        model_uri=os.path.abspath(tmp_dir),
        launcher="xinference.model.flexible.launchers.transformers",
    )

    register_flexible_model(model_spec, persist=False)

    unregister_flexible_model("flexible_model")

    shutil.rmtree(tmp_dir, ignore_errors=True)


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
