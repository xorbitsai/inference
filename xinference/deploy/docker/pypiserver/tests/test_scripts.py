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
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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


def test_generate_package_lists_main_orchestration(monkeypatch, tmp_path):
    generator = _load_script("generate_package_lists")
    src_root = tmp_path / "src"
    model_dir = src_root / "xinference" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "spec.json").write_text(
        json.dumps(
            {
                "model_name": "test-model",
                "virtualenv": {
                    "packages": [
                        "model-pin==1.0",
                        "https://example.invalid/direct-1.0-py3-none-any.whl",
                        "git-package @ git+https://example.invalid/repo.git@abc",
                    ]
                },
            }
        )
    )
    out = tmp_path / "manifest"
    loaded_roots = []

    def fake_load_xinference_modules(root):
        loaded_roots.append(root)
        core_utils = SimpleNamespace(
            filter_virtualenv_packages_by_markers=lambda packages, *_args: packages
        )
        venv_manager = SimpleNamespace(
            ENGINE_VIRTUALENV_PACKAGES={"test-engine": ["engine-pkg>=1.0"]},
            ENGINE_VIRTUALENV_EXTRA_INDEX_URLS={
                "test-engine": ["https://example.invalid/simple"]
            },
            ENGINE_VIRTUALENV_INDEX_STRATEGY={"test-engine": "unsafe-best-match"},
        )
        return core_utils, venv_manager

    monkeypatch.setattr(
        generator, "load_xinference_modules", fake_load_xinference_modules
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_package_lists.py",
            "--platform",
            "amd64",
            "--src-root",
            str(src_root),
            "--out",
            str(out),
        ],
    )
    original_machine = generator._platform.machine
    try:
        generator.main()
    finally:
        generator._platform.machine = original_machine

    assert loaded_roots == [src_root]
    assert (out / "engines" / "test_engine.in").read_text() == "engine-pkg>=1.0\n"
    assert json.loads((out / "pins.json").read_text()) == [
        {
            "spec": "model-pin==1.0",
            "sources": ["spec.json:test-model"],
        }
    ]
    assert "direct-1.0-py3-none-any.whl" in (out / "urls.txt").read_text()
    assert "git+https://example.invalid/repo.git@abc" in (out / "git.txt").read_text()
    assert json.loads((out / "manifest.json").read_text())["counts"] == {
        "engines": 1,
        "pins": 1,
        "urls": 1,
        "git": 1,
    }


def test_download_packages_main_orchestration_skips_git(monkeypatch, tmp_path):
    downloader = _load_script("download_packages")
    manifest = tmp_path / "manifest"
    engines = manifest / "engines"
    engines.mkdir(parents=True)
    (engines / "test.in").write_text("engine-pkg>=1.0\n")
    (engines / "test.meta.json").write_text("{}\n")
    (manifest / "pins.json").write_text(
        json.dumps([{"spec": "model-pin==1.0", "sources": ["test"]}])
    )
    (manifest / "urls.txt").write_text(
        "https://example.invalid/direct-1.0-py3-none-any.whl\n"
    )
    git_source = "git-package @ git+https://example.invalid/repo.git@abc"
    (manifest / "git.txt").write_text(git_source + "\n")
    dest = tmp_path / "packages"
    compile_calls = []
    download_calls = []
    run_calls = []

    def fake_uv_compile(in_file, out_file, **kwargs):
        compile_calls.append((in_file, out_file, kwargs))
        if out_file.name == "torch-family.lock":
            out_file.write_text("torch==2.11.0\n")
        else:
            out_file.write_text("engine-pkg==1.0\n")
        return subprocess.CompletedProcess([], 0)

    def fake_pip_download(args, target, **kwargs):
        download_calls.append((list(args), kwargs))
        if args == ["model-pin==1.0"]:
            (target / "source-1.0.tar.gz").write_text("sdist")
        return subprocess.CompletedProcess([], 0)

    def fake_run(cmd, **_kwargs):
        run_calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(downloader, "uv_compile", fake_uv_compile)
    monkeypatch.setattr(downloader, "pip_download", fake_pip_download)
    monkeypatch.setattr(downloader, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_packages.py",
            "--manifest-dir",
            str(manifest),
            "--dest",
            str(dest),
            "--platform",
            "amd64",
        ],
    )

    downloader.main()

    assert len(compile_calls) == 2
    assert ["-r", str(manifest / "locks" / "test.lock")] in [
        call[0] for call in download_calls
    ]
    assert ["model-pin==1.0"] in [call[0] for call in download_calls]
    assert ["https://example.invalid/direct-1.0-py3-none-any.whl"] in [
        call[0] for call in download_calls
    ]
    assert all(git_source not in " ".join(call[0]) for call in download_calls)
    assert all(git_source not in " ".join(call) for call in run_calls)
    assert not (dest / "source-1.0.tar.gz").exists()
    report = json.loads((manifest / "report.json").read_text())
    assert report["unconstrained_fallbacks"] == []
    assert report["sdist_left"] == []


def test_selfcheck_main_orchestration(monkeypatch, tmp_path, capsys):
    selfcheck = _load_script("selfcheck")
    manifest = tmp_path / "manifest"
    engines = manifest / "engines"
    engines.mkdir(parents=True)
    (engines / "test.in").write_text("engine-pkg>=1.0\n")
    (manifest / "pins.json").write_text(
        json.dumps([{"spec": "model-pin==1.0", "sources": ["test"]}])
    )
    (manifest / "urls.txt").write_text(
        "https://example.invalid/direct-1.0-py3-none-any.whl\n"
    )
    git_source = "git-package @ git+https://example.invalid/repo.git@abc"
    (manifest / "git.txt").write_text(git_source + "\n")
    compile_calls = []

    def fake_compile(requirements, index_url, python_version, python_platform):
        compile_calls.append((requirements, index_url, python_version, python_platform))
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(selfcheck, "wait_for_server", lambda *_args: None)
    monkeypatch.setattr(selfcheck, "compile_against_index", fake_compile)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selfcheck.py",
            "--index",
            "http://mirror.invalid/simple",
            "--manifest-dir",
            str(manifest),
            "--platform",
            "amd64",
        ],
    )

    selfcheck.main()

    assert [call[0] for call in compile_calls] == [
        ["engine-pkg>=1.0"],
        ["model-pin==1.0"],
        ["direct==1.0"],
    ]
    output = capsys.readouterr().out
    assert "UNSUPPORTED offline direct reference " + git_source in output
    assert "selfcheck passed" in output
