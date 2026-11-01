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
import glob
import json
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..")


def test_builtin_repo_ids_are_owner_slash_name():
    # Hub repo ids have exactly one "/" (owner/name); anything else cannot be
    # downloaded, e.g. "mlx-community/Qwen/Qwen3-235B-A22B-{quantization}".
    bad = []
    for path in glob.glob(os.path.join(MODEL_DIR, "*", "models", "*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for family in data if isinstance(data, list) else [data]:
            for spec in family.get("model_specs") or [family]:
                for hub, src in (spec.get("model_src") or {}).items():
                    for key, value in src.items():
                        if key.endswith("model_id") and isinstance(value, str):
                            if value.count("/") != 1:
                                bad.append((os.path.basename(path), hub, key, value))
    assert bad == []
