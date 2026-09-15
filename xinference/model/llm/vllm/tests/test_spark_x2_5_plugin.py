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

from pathlib import Path

from ..patches.spark_x2_5_plugin import (
    _ORIGINAL,
    _PATCHED,
    _PLUGIN_RELATIVE_PATH,
    patch_spark_x2_5_plugin,
)


def test_patch_spark_x2_5_plugin(tmp_path: Path):
    target_file = tmp_path / "lib" / "python3.12" / "site-packages"
    target_file = target_file / _PLUGIN_RELATIVE_PATH
    target_file.parent.mkdir(parents=True)
    target_file.write_text(_ORIGINAL, encoding="utf-8")

    assert patch_spark_x2_5_plugin(str(tmp_path)) is True
    assert target_file.read_text(encoding="utf-8") == _PATCHED
    assert patch_spark_x2_5_plugin(str(tmp_path)) is False
