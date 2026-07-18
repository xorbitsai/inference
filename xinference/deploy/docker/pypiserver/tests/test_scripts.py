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
REPO_ROOT = SCRIPT_DIR.parents[3]


def _write_runtime_constraints(path: Path) -> None:
    path.write_text(
        "torch==2.11.0\n"
        "torchvision==0.26.0\n"
        "torchaudio==2.11.0\n"
        "torchcodec==0.14.0\n"
        "transformers==5.13.1\n"
        "accelerate==1.14.0\n"
        "numpy==2.3.5\n"
        "pandas==3.0.3\n"
    )


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


def test_load_xinference_modules_restores_sys_modules():
    generator = _load_script("generate_package_lists")
    original_xinference = sys.modules.get("xinference")
    sentinel = original_xinference or ModuleType("xinference")
    sys.modules["xinference"] = sentinel
    try:
        before = {
            name: module
            for name, module in sys.modules.items()
            if name == "xinference" or name.startswith("xinference.")
        }

        core_utils, venv_manager = generator.load_xinference_modules(REPO_ROOT)

        after = {
            name: module
            for name, module in sys.modules.items()
            if name == "xinference" or name.startswith("xinference.")
        }
        assert after == before
        assert sys.modules["xinference"] is sentinel
        assert callable(core_utils.filter_virtualenv_packages_by_markers)
        assert "transformers" in venv_manager.ENGINE_VIRTUALENV_PACKAGES
    finally:
        if original_xinference is None:
            sys.modules.pop("xinference", None)
        else:
            sys.modules["xinference"] = original_xinference


def test_iter_virtualenv_packages_reads_json_as_utf8(monkeypatch, tmp_path):
    generator = _load_script("generate_package_lists")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_json = model_dir / "model.json"
    model_json.write_text(
        json.dumps(
            {
                "model_name": "中文模型",
                "virtualenv": {"packages": ["model-package"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    original_read_text = Path.read_text
    encodings = []

    def tracked_read_text(path, *args, **kwargs):
        if path == model_json:
            encodings.append(kwargs.get("encoding"))
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked_read_text)

    assert list(generator.iter_virtualenv_packages(model_dir)) == [
        ("model.json", "中文模型", ["model-package"])
    ]
    assert encodings == ["utf-8"]


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


def test_pip_download_forwards_package_build_environment(monkeypatch, tmp_path):
    downloader = _load_script("download_packages")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(downloader, "run", fake_run)
    build_env = {"TORCH_VERSION": "2.12.3"}

    downloader.pip_download(
        ["gptqmodel==4.2.5"],
        tmp_path,
        index_url="https://example.invalid/simple",
        extra_index_urls=[],
        no_deps=True,
        env=build_env,
    )

    assert calls[0][1]["env"] is build_env


def test_runtime_constraints_require_exact_pins(tmp_path):
    downloader = _load_script("download_packages")
    constraints = tmp_path / "runtime.txt"
    _write_runtime_constraints(constraints)

    pins = downloader.load_runtime_constraints(constraints)
    assert pins["torch"] == "torch==2.11.0"
    assert pins["transformers"] == "transformers==5.13.1"

    constraints.write_text(constraints.read_text().replace("pandas==3.0.3", "pandas"))
    try:
        downloader.load_runtime_constraints(constraints)
    except ValueError as exc:
        assert "must be an exact pin" in str(exc)
    else:
        raise AssertionError("an unpinned runtime package must be rejected")


def test_runtime_and_mirror_share_the_same_constraints_file():
    downloader = _load_script("download_packages")
    constraints = REPO_ROOT / "xinference/deploy/docker/requirements-runtime.txt"
    pins = downloader.load_runtime_constraints(constraints)
    assert set(pins) == {
        "accelerate",
        "numpy",
        "pandas",
        "torch",
        "torchaudio",
        "torchcodec",
        "torchvision",
        "transformers",
    }

    runtime_dockerfile = (REPO_ROOT / "xinference/deploy/docker/Dockerfile").read_text()
    assert (
        "COPY xinference/deploy/docker/requirements-runtime.txt" in runtime_dockerfile
    )
    mirror_dockerfile = (
        REPO_ROOT / "xinference/deploy/docker/pypiserver/Dockerfile.pypiserver"
    ).read_text()
    assert "COPY xinference/deploy/docker/requirements-runtime.txt" in mirror_dockerfile
    assert "--runtime-constraints /build/requirements-runtime.txt" in mirror_dockerfile


def test_transformers_optional_dependencies_are_scoped_and_mirrored(
    monkeypatch, tmp_path
):
    dockerfile = (REPO_ROOT / "xinference/deploy/docker/Dockerfile").read_text()
    assert '".[otel]" transformers accelerate' in dockerfile
    assert '".[otel,transformers,transformers_quantization]"' not in dockerfile
    assert "--constraint /opt/requirements-runtime.txt" in dockerfile
    for package in ("qwen-vl-utils", "eva-decord"):
        assert package not in dockerfile

    generator = _load_script("generate_package_lists")
    _, venv_manager = generator.load_xinference_modules(REPO_ROOT)
    assert venv_manager.ENGINE_VIRTUALENV_PACKAGES["transformers"] == [
        "transformers>=4.53.3",
        "accelerate>=0.28.0",
    ]
    assert (
        venv_manager.get_engine_model_format_virtualenv_packages(
            "Transformers", "pytorch"
        )
        == []
    )
    assert any(
        package.startswith("gptqmodel")
        for package in venv_manager.get_engine_model_format_virtualenv_packages(
            "Transformers", "gptq"
        )
    )
    assert "optimum" in venv_manager.get_engine_model_format_virtualenv_packages(
        "Transformers", "gptq"
    )

    out = tmp_path / "manifest"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_package_lists.py",
            "--platform",
            "amd64",
            "--src-root",
            str(REPO_ROOT),
            "--out",
            str(out),
        ],
    )
    original_machine = generator._platform.machine
    try:
        generator.main()
    finally:
        generator._platform.machine = original_machine

    mirrored = (out / "engines" / "transformers.in").read_text().lower()
    for package in ("bitsandbytes", "gptqmodel", "optimum", "autoawq"):
        assert package in mirrored
    for package in (
        "qwen-vl-utils",
        "attrdict",
        "einops",
        "tiktoken",
        "sentencepiece",
        "transformers_stream_generator",
        "datamodel_code_generator",
        "jsonschema",
        "blobfile",
        "eva-decord",
    ):
        assert package not in mirrored

    pin_entries = json.loads((out / "pins.json").read_text())
    model_pins = {item["spec"].split(";", 1)[0].strip().lower() for item in pin_entries}
    for package in (
        "qwen-vl-utils!=0.0.9",
        "attrdict",
        "einops",
        "timm>=0.9.16",
        "tiktoken>=0.6.0",
        "sentencepiece",
        "transformers_stream_generator",
        "datamodel_code_generator",
        "jsonschema",
        "blobfile",
        "eva-decord",
    ):
        assert package in model_pins

    qwen_vl_sources = next(
        item["sources"]
        for item in pin_entries
        if item["spec"].split(";", 1)[0].strip().lower() == "qwen-vl-utils!=0.0.9"
    )
    for model_name in ("qwen3.5", "qwen3.6"):
        source = "llm/llm_family.json:" + model_name + " (Transformers)"
        assert source in qwen_vl_sources


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
    runtime_constraints = tmp_path / "runtime.txt"
    _write_runtime_constraints(runtime_constraints)
    runtime_constraints.write_text(
        runtime_constraints.read_text().replace("torch==2.11.0", "torch==2.12.3")
    )
    compile_calls = []
    download_calls = []
    run_calls = []

    def fake_uv_compile(in_file, out_file, **kwargs):
        compile_calls.append((in_file, out_file, kwargs))
        if out_file.name in ("runtime.lock", "torch-family.lock"):
            out_file.write_text(runtime_constraints.read_text())
        else:
            out_file.write_text("engine-pkg==1.0\n")
        return subprocess.CompletedProcess([], 0)

    def fake_pip_download(args, target, **kwargs):
        download_calls.append((list(args), kwargs))
        if args == ["model-pin==1.0"]:
            (target / "source-1.0.tar.gz").write_text("sdist")
        return subprocess.CompletedProcess([], 0)

    def fake_run(cmd, **kwargs):
        run_calls.append((list(cmd), kwargs))
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
            "--runtime-constraints",
            str(runtime_constraints),
        ],
    )

    downloader.main()

    assert len(compile_calls) == 3
    assert compile_calls[0][0] == runtime_constraints
    assert ["-r", str(manifest / "locks" / "test.lock")] in [
        call[0] for call in download_calls
    ]
    assert ["model-pin==1.0"] in [call[0] for call in download_calls]
    assert ["https://example.invalid/direct-1.0-py3-none-any.whl"] in [
        call[0] for call in download_calls
    ]
    assert all(git_source not in " ".join(call[0]) for call in download_calls)
    assert all(git_source not in " ".join(call[0]) for call in run_calls)
    assert all(call[1]["env"]["TORCH_VERSION"] == "2.12.3" for call in download_calls)
    assert all(call[1]["env"]["TORCH_VERSION"] == "2.12.3" for call in run_calls)
    assert not (dest / "source-1.0.tar.gz").exists()
    report = json.loads((manifest / "report.json").read_text())
    assert "transformers==5.13.1" in report["runtime_constraints"]
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
    runtime_constraints = tmp_path / "runtime.txt"
    _write_runtime_constraints(runtime_constraints)

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
            "--runtime-constraints",
            str(runtime_constraints),
        ],
    )

    selfcheck.main()

    assert [call[0] for call in compile_calls] == [
        runtime_constraints.read_text().splitlines(),
        ["engine-pkg>=1.0"],
        ["model-pin==1.0"],
        ["direct==1.0"],
    ]
    output = capsys.readouterr().out
    assert "UNSUPPORTED offline direct reference " + git_source in output
    assert "selfcheck passed" in output
