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

import json
import os

from ..core import BUILTIN_IMAGE_MODELS, IMAGE_MODEL_DESCRIPTIONS
from ..engine_family import IMAGE_ENGINES


def test_register_builtin_model_prunes_stale_derived_entries_on_catalog_removal(
    tmp_path, monkeypatch
):
    # A downloaded-only model still in IMAGE_ENGINES/IMAGE_MODEL_DESCRIPTIONS
    # after it drops out of a later catalog refresh keeps advertising a launch
    # config and a description, even though BUILTIN_IMAGE_MODELS (the table
    # both derive from) no longer has it.
    import xinference.model.image as image_module

    from .... import constants
    from .. import register_builtin_model

    # has_downloaded_models()/load_downloaded_models() read the module-level
    # XINFERENCE_MODEL_DIR bound at import time, not a fresh lookup, so both
    # bindings need patching (same shape as the rerank regression test).
    monkeypatch.setattr(image_module, "XINFERENCE_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(constants, "XINFERENCE_MODEL_DIR", str(tmp_path))

    spec_path = os.path.join(os.path.dirname(__file__), "..", "model_spec.json")
    with open(spec_path) as f:
        raw_entry = json.load(f)[0]
    downloaded_only = dict(raw_entry)
    downloaded_only["model_name"] = "downloaded-only-catalog-removal-test"

    builtin_dir = os.path.join(str(tmp_path), "v2", "builtin", "image")
    os.makedirs(builtin_dir, exist_ok=True)
    catalog_path = os.path.join(builtin_dir, "image_models.json")
    with open(catalog_path, "w") as f:
        json.dump([downloaded_only], f)

    register_builtin_model()
    assert "downloaded-only-catalog-removal-test" in BUILTIN_IMAGE_MODELS
    assert "downloaded-only-catalog-removal-test" in IMAGE_ENGINES
    assert "downloaded-only-catalog-removal-test" in IMAGE_MODEL_DESCRIPTIONS

    # A later refresh's catalog no longer lists the model (removed upstream).
    with open(catalog_path, "w") as f:
        json.dump([], f)

    register_builtin_model()
    assert "downloaded-only-catalog-removal-test" not in BUILTIN_IMAGE_MODELS
    assert "downloaded-only-catalog-removal-test" not in IMAGE_ENGINES
    assert "downloaded-only-catalog-removal-test" not in IMAGE_MODEL_DESCRIPTIONS
