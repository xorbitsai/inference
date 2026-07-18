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
import platform

from ..utils import (
    filter_virtualenv_packages_by_markers,
    find_direct_reference_packages,
    rewrite_direct_url_packages_for_index,
)
from ..virtual_env_manager import ENGINE_VIRTUALENV_PACKAGES

SGL_KERNEL_X86_URL = (
    "https://github.com/sgl-project/whl/releases/download/v0.3.21/"
    "sgl_kernel-0.3.21+cu130-cp310-abi3-manylinux2014_x86_64.whl"
)
SGL_KERNEL_AARCH64_URL = (
    "https://github.com/sgl-project/whl/releases/download/v0.3.21/"
    "sgl_kernel-0.3.21+cu130-cp310-abi3-manylinux2014_aarch64.whl"
)


def test_rewrite_bare_wheel_url():
    assert rewrite_direct_url_packages_for_index([SGL_KERNEL_X86_URL]) == [
        "sgl_kernel==0.3.21+cu130"
    ]
    assert rewrite_direct_url_packages_for_index([SGL_KERNEL_AARCH64_URL]) == [
        "sgl_kernel==0.3.21+cu130"
    ]


def test_rewrite_pep508_name_at_url():
    pkg = (
        "en_core_web_trf@https://github.com/explosion/spacy-models/releases/"
        "download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl"
    )
    assert rewrite_direct_url_packages_for_index([pkg]) == ["en_core_web_trf==3.8.0"]


def test_rewrite_preserves_marker():
    pkg = f'{SGL_KERNEL_X86_URL} ; platform_machine == "x86_64"'
    assert rewrite_direct_url_packages_for_index([pkg]) == [
        'sgl_kernel==0.3.21+cu130 ; platform_machine == "x86_64"'
    ]


def test_rewrite_unquotes_percent_encoding():
    pkg = (
        "https://example.com/wheels/"
        "sgl_kernel-0.3.21%2Bcu130-cp310-abi3-manylinux2014_x86_64.whl"
    )
    assert rewrite_direct_url_packages_for_index([pkg]) == ["sgl_kernel==0.3.21+cu130"]


def test_non_url_and_non_wheel_entries_unchanged():
    packages = [
        "transformers>=4.53.3",
        "numpy>=2.4.1",
        "#system_torch#",
        "git+https://github.com/huggingface/diffusers",
        "funasr @ git+https://github.com/modelscope/FunASR@b25472b",
        "https://example.com/archives/pkg-1.0.0.tar.gz",
        # not a URL despite the "http" prefix in the name
        "httpx==0.24.0",
    ]
    assert rewrite_direct_url_packages_for_index(packages) == packages


def test_find_direct_references_after_wheel_rewrite():
    packages = [
        SGL_KERNEL_X86_URL,
        "transformers @ git+https://github.com/huggingface/transformers.git",
        "git+https://github.com/huggingface/diffusers",
        "https://example.com/pkg-1.0.0.tar.gz",
        "transformers>=4.53.3",
    ]
    rewritten = rewrite_direct_url_packages_for_index(packages)

    assert find_direct_reference_packages(rewritten) == packages[1:4]


def test_malformed_wheel_filename_unchanged():
    packages = [
        # too few dash-separated fields to be a valid wheel filename
        "https://example.com/invalid.whl",
        "https://example.com/pkg-none-any.whl",
        # version field does not start with a digit
        "https://example.com/pkg-vNext-py3-none-any.whl",
    ]
    assert rewrite_direct_url_packages_for_index(packages) == packages


def test_filter_then_rewrite_sglang_cu130():
    """The real sglang engine list ends up fully index-resolvable."""
    machine = platform.machine().lower()
    if machine not in ("x86_64", "aarch64"):
        machine = None

    packages = filter_virtualenv_packages_by_markers(
        ENGINE_VIRTUALENV_PACKAGES["sglang"], "sglang", "13.0"
    )
    rewritten = rewrite_direct_url_packages_for_index(packages)

    assert not any(p.startswith(("http://", "https://")) for p in rewritten)
    assert "sglang>=0.5.6" in rewritten
    assert "numpy>=2.4.1" in rewritten
    if machine:
        # the arch-matching direct URL survives filtering and is rewritten
        assert "sgl_kernel==0.3.21+cu130" in rewritten


def test_filter_sglang_keeps_cuda_12_fallback():
    """The cu130 offline mirror must not remove online CUDA 12 support."""
    packages = filter_virtualenv_packages_by_markers(
        ENGINE_VIRTUALENV_PACKAGES["sglang"], "sglang", "12.8"
    )

    assert "sgl_kernel" in packages
    assert not any("+cu130" in package for package in packages)


def test_filter_combined_engine_and_linux_platform_markers(monkeypatch):
    package = (
        'eva-decord ; #engine# == "Transformers" and '
        'sys_platform == "linux" and platform_machine == "x86_64"'
    )
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")

    assert filter_virtualenv_packages_by_markers(
        [package], "Transformers", None, "linux"
    ) == ["eva-decord"]
    assert (
        filter_virtualenv_packages_by_markers([package], "Transformers", None, "darwin")
        == []
    )
    assert filter_virtualenv_packages_by_markers([package], "vllm", None, "linux") == []
