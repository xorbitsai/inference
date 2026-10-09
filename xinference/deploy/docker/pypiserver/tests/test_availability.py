# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import importlib.util
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

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
        ("pkg", "pkg-1.0-py3-none-any.whl", "aarch64", ">=3.6.*", False, True),
        (
            "pkg",
            "pkg-1.0-cp312-cp312-manylinux_2_34_aarch64.whl",
            "aarch64",
            None,
            False,
            True,
        ),
        (
            "pkg",
            "pkg-1.0-cp312-cp312-manylinux_2_35_aarch64.whl",
            "aarch64",
            None,
            False,
            False,
        ),
        (
            "pkg",
            "pkg-1.0-cp312-cp312-manylinux1_x86_64.whl",
            "x86_64",
            None,
            False,
            True,
        ),
        (
            "pkg",
            "pkg-1.0-cp312-cp312-manylinux2010_x86_64.whl",
            "x86_64",
            None,
            False,
            True,
        ),
        (
            "pkg",
            "pkg-1.0-cp312-cp312-manylinux2014_x86_64.whl",
            "x86_64",
            None,
            False,
            True,
        ),
        ("pkg", "pkg-1.0-cp312-cp312-linux_x86_64.whl", "x86_64", None, False, True),
        (
            "pkg",
            "pkg-1.0-cp312-cp312-musllinux_1_2_aarch64.whl",
            "aarch64",
            None,
            False,
            False,
        ),
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
                "cuda_version": "13.1",
                "deferred_model_pins": [
                    {
                        "spec": "vllm>=0.32.0",
                        "source": "future-model",
                        "reason": "unpublished backend",
                    }
                ],
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
    assert "https://download.pytorch.org/whl/cu131" in engine["indexes"]
    assert all(check["machine"] == "aarch64" for check in checks)
    deferred = next(check for check in checks if check["deferred"])
    assert deferred["sources"] == ["future-model"]
    assert str(deferred["requirement"]) == "vllm>=0.32.0"


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


def test_default_python_target_matches_dockerfile(checker):
    dockerfile = Path(__file__).resolve().parents[1] / "Dockerfile.pypiserver"
    target = re.search(
        r"^ARG PYTHON_VERSION=(\S+)$", dockerfile.read_text(), re.MULTILINE
    ).group(1)
    assert checker.DEFAULT_PYTHON_VERSION.split(".")[:2] == target.split(".")[:2]


def test_prerelease_only_candidates_are_accepted(checker, monkeypatch):
    monkeypatch.setattr(
        checker, "fetch_files", lambda *_: [{"filename": "pkg-2.0rc1.tar.gz"}]
    )
    calls = []
    original = checker.matching_file

    def match(*args, **kwargs):
        calls.append(kwargs["prereleases"])
        return original(*args, **kwargs)

    monkeypatch.setattr(checker, "matching_file", match)
    assert (
        checker.check_requirements(
            [
                {
                    "requirement": Requirement("pkg>=1.0"),
                    "machine": "x86_64",
                    "sources": ["model"],
                    "indexes": ("https://index.example",),
                }
            ],
            "3.12.0",
        )
        == []
    )
    assert calls == [False, True]


@pytest.mark.parametrize("published", [False, True])
def test_deferred_dependency_warns_only_when_published(
    checker, monkeypatch, capsys, published
):
    version = "0.32.0" if published else "0.31.0"
    requests = []

    def fetch(*args):
        requests.append(args)
        return [{"filename": f"vllm-{version}.tar.gz"}]

    monkeypatch.setattr(checker, "fetch_files", fetch)
    checks = [
        {
            "requirement": Requirement(spec),
            "machine": machine,
            "sources": ["embeddinggemma-2"],
            "indexes": ("https://index.example",),
            "deferred": "future backend" if deferred else "",
        }
        for machine in ("x86_64", "aarch64")
        for spec, deferred in [("vllm>=0.31.0", False), ("vllm>=0.32.0", True)]
    ]
    assert checker.check_requirements(checks, "3.12.0") == []
    assert capsys.readouterr().out.count("::warning::") == (2 if published else 0)
    assert requests == [("https://index.example", "vllm")]


@pytest.mark.parametrize(
    "status,pytorch,expected_requests",
    [
        (404, False, 1),
        (403, True, 1),
        (403, False, 1),
        (400, False, 1),
        (429, False, 3),
        (500, False, 3),
    ],
)
def test_index_http_status_classification(
    checker, monkeypatch, status, pytorch, expected_requests
):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(status)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    local_index = f"http://127.0.0.1:{server.server_port}/simple"

    def open_local(request, timeout):
        return urlopen(
            Request(local_index + "/pkg/", headers=request.headers), timeout=timeout
        )

    monkeypatch.setattr(checker, "urlopen", open_local)
    monkeypatch.setattr(checker.time, "sleep", lambda *_: None)
    index = "https://download.pytorch.org/whl/cu130" if pytorch else local_index
    try:
        if status == 404 or (status == 403 and pytorch):
            assert checker.fetch_files(index, "pkg") == []
        else:
            with pytest.raises(RuntimeError, match="Index request failed"):
                checker.fetch_files(index, "pkg")
        assert requests == ["/simple/pkg/"] * expected_requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "broken_response", ["missing-files", "invalid-json", "truncated"]
)
def test_broken_index_responses_are_reported(checker, monkeypatch, broken_response):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/vnd.pypi.simple.v1+json")
            if broken_response == "truncated":
                self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(
                b"not-json" if broken_response == "invalid-json" else b"{}"
            )

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(checker.time, "sleep", lambda *_: None)
    try:
        failures = checker.check_requirements(
            [
                {
                    "requirement": Requirement("pkg"),
                    "machine": "x86_64",
                    "sources": ["test-model"],
                    "indexes": (f"http://127.0.0.1:{server.server_port}/simple",),
                }
            ],
            "3.12.0",
        )
        assert len(failures) == 1
        assert "test-model" in failures[0]
        assert len(requests) == (3 if broken_response == "truncated" else 1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
