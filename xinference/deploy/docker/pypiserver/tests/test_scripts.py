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
from pathlib import Path
from types import ModuleType

SCRIPT_DIR = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_package_list_classification():
    generator = _load_script("generate_package_lists")

    assert generator.classify_spec("transformers>=4.53.3") == "pin"
    assert generator.classify_spec("https://example.com/pkg.whl") == "url"
    assert generator.classify_spec("pkg @ https://example.com/pkg.whl") == "url"
    assert generator.classify_spec("git+https://github.com/org/repo") == "git"
    assert generator.classify_spec("pkg @ git+https://github.com/org/repo") == "git"
    assert generator.system_placeholder_name("#system_torch#") == "torch"
    assert generator.system_placeholder_name("#system_numpy# ; marker") == "numpy"


def test_selfcheck_wheel_url_to_spec():
    selfcheck = _load_script("selfcheck")

    assert (
        selfcheck.wheel_url_to_spec(
            "https://example.com/sgl_kernel-0.3.21%2Bcu130-"
            "cp310-abi3-manylinux2014_x86_64.whl"
        )
        == "sgl_kernel==0.3.21+cu130"
    )
    assert selfcheck.wheel_url_to_spec("https://example.com/pkg.tar.gz") is None
    assert selfcheck.wheel_url_to_spec("https://example.com/invalid.whl") is None


def test_download_report_wheel_filename_parsing():
    downloader = _load_script("download_packages")

    assert downloader.wheel_name_version(
        "sentence_transformers-5.1.2-py3-none-any.whl"
    ) == ("sentence-transformers", "5.1.2")
    assert downloader.wheel_name_version("package.tar.gz") is None
    assert downloader.wheel_name_version("invalid.whl") is None
