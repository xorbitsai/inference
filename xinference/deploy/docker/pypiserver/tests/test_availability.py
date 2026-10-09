# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from packaging.requirements import Requirement


@pytest.fixture
def checker():
    path = Path(__file__).resolve().parents[1] / "check_package_availability.py"
    spec = importlib.util.spec_from_file_location("availability", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "spec,filename,machine,requires_python,yanked,expected",
    [
        (
            "vllm>=0.32.0",
            "vllm-0.31.0-cp38-abi3-manylinux_2_28_x86_64.whl",
            "x86_64",
            ">=3.10",
            False,
            False,
        ),
        (
            "vllm>=0.31.0",
            "vllm-0.31.0-cp38-abi3-manylinux_2_28_x86_64.whl",
            "x86_64",
            ">=3.10",
            False,
            True,
        ),
        (
            "vllm>=0.31.0",
            "vllm-0.31.0-cp38-abi3-manylinux_2_28_x86_64.whl",
            "aarch64",
            ">=3.10",
            False,
            False,
        ),
        (
            "pkg",
            "pkg-1.0-cp312-cp312-manylinux2014_aarch64.whl",
            "aarch64",
            ">=3.12",
            False,
            True,
        ),
        (
            "pkg",
            "pkg-1.0-cp311-cp311-manylinux2014_aarch64.whl",
            "aarch64",
            None,
            False,
            False,
        ),
        (
            "pkg",
            "pkg-1.0-cp313-abi3-manylinux2014_aarch64.whl",
            "aarch64",
            None,
            False,
            False,
        ),
        ("pkg", "pkg-1.0-py3-none-any.whl", "aarch64", ">=3.13", False, False),
        ("pkg", "pkg-1.0-py3-none-any.whl", "aarch64", None, False, True),
        (
            "pkg",
            "pkg-1.0-cp312-cp312-macosx_11_0_arm64.whl",
            "aarch64",
            None,
            False,
            False,
        ),
        ("pkg", "pkg-1.0.tar.gz", "aarch64", None, False, True),
        ("pkg", "pkg-1.0.tar.gz", "x86_64", None, "", False),
        ("pkg==1.0", "pkg-1.0.tar.gz", "x86_64", None, "", True),
        ("pkg==1.*", "pkg-1.0.tar.gz", "x86_64", None, "", False),
        ("pkg>=1.0", "pkg-2.0rc1.tar.gz", "x86_64", None, False, False),
        ("pkg>=2.0rc1", "pkg-2.0rc1.tar.gz", "x86_64", None, False, True),
        ("pkg", "other-1.0.tar.gz", "x86_64", None, False, False),
    ],
)
def test_published_candidates(
    checker, spec, filename, machine, requires_python, yanked, expected
):
    assert (
        checker.matching_file(
            Requirement(spec),
            {
                "filename": filename,
                "requires-python": requires_python,
                "yanked": yanked,
            },
            machine,
            "3.12.0",
        )
        is expected
    )


def test_preflight_future_pin_fails_and_metadata_is_shared(checker, monkeypatch):
    calls = []

    def fetch(index, name):
        calls.append((index, name))
        return [{"filename": "vllm-0.31.0.tar.gz"}]

    monkeypatch.setattr(checker, "fetch_files", fetch)
    checks = [
        {
            "requirement": Requirement(spec),
            "machine": machine,
            "sources": ["embedding/models/new-model.json:new-model"],
            "indexes": ("https://index.example/simple",),
        }
        for machine in ("x86_64", "aarch64")
        for spec in ("vllm>=0.31.0", "vllm>=0.32.0")
    ]
    failures = checker.check_requirements(checks, "3.12.0")
    assert len(failures) == 2
    assert all(
        "vllm>=0.32.0" in failure and "new-model" in failure for failure in failures
    )
    assert calls == [("https://index.example/simple", "vllm")]


def test_index_failure_is_not_ignored(checker, monkeypatch):
    def fetch(index, name):
        if index == "https://broken.example":
            raise RuntimeError("index is unavailable")
        return [{"filename": "pkg-1.0.tar.gz"}]

    monkeypatch.setattr(checker, "fetch_files", fetch)
    failures = checker.check_requirements(
        [
            {
                "requirement": Requirement("pkg"),
                "machine": "x86_64",
                "sources": ["model"],
                "indexes": ("https://valid.example", "https://broken.example"),
            }
        ],
        "3.12.0",
    )
    assert len(failures) == 1
    assert "index is unavailable" in failures[0]


def test_collect_all_mirror_inputs_and_target_markers(checker, tmp_path):
    (tmp_path / "engines").mkdir()
    (tmp_path / "engines" / "vllm.in").write_text("vllm>=0.31\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "machine": "aarch64",
                "engines": {
                    "vllm": {
                        "engine": "vllm",
                        "extra_index_urls": ["https://custom.example/simple"],
                    }
                },
            }
        )
    )
    (tmp_path / "pins.json").write_text(
        json.dumps(
            [
                {
                    "spec": 'arm-package; platform_machine == "aarch64" and sys_platform == "linux" and python_version == "3.12"',
                    "sources": ["arm-model"],
                },
                {
                    "spec": 'x86-package; platform_machine == "x86_64"',
                    "sources": ["x86-model"],
                },
            ]
        )
    )
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("# shared runtime\ntorch==2.11.0\n")
    checks = checker.collect_requirements(tmp_path, constraints, "3.12.0")
    assert {check["requirement"].name for check in checks} == {
        "torch",
        "vllm",
        "arm-package",
    }
    engine = next(check for check in checks if check["requirement"].name == "vllm")
    assert "https://custom.example/simple" in engine["indexes"]
    assert checker.PYTORCH_INDEX in engine["indexes"]
    assert all(check["machine"] == "aarch64" for check in checks)


@pytest.mark.parametrize("format", ["json", "html"])
def test_simple_index_requests_never_download_artifacts(checker, format):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            assert self.path == "/simple/pkg/"
            self.send_response(200)
            if format == "json":
                self.send_header("Content-Type", "application/vnd.pypi.simple.v1+json")
                body = json.dumps(
                    {
                        "files": [
                            {
                                "filename": "pkg-1.0.tar.gz",
                                "url": "/large-artifact",
                                "requires-python": ">=3.12",
                                "yanked": False,
                            }
                        ]
                    }
                ).encode()
            else:
                self.send_header("Content-Type", "text/html")
                body = b'<a href="/large-artifact/pkg-1.0.tar.gz" data-requires-python="&gt;=3.12">pkg</a>'
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        files = checker.fetch_files(
            f"http://127.0.0.1:{server.server_port}/simple", "Pkg"
        )
        assert len(files) == 1
        assert checker.matching_file(
            Requirement("pkg==1.0"), files[0], "aarch64", "3.12.0"
        )
        assert requests == ["/simple/pkg/"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
