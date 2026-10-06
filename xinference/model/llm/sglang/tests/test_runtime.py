# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ..runtime import create_runtime


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("authenticated", [False, True])
def test_runtime_children_inherit_canonical_jit_cache(
    tmp_path, monkeypatch, relative, authenticated
):
    # SGLang records transient staging files as dependencies under symlinked roots.
    physical = tmp_path / "physical"
    physical.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(physical, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    monkeypatch.chdir(tmp_path)
    configured = Path("linked/jit") if relative else linked / "jit"
    monkeypatch.setenv("SGLANG_JIT_CACHE_DIR", str(configured))
    config = {"dtype": "float16"}
    if authenticated:
        config["api_key"] = "credential"
        monkeypatch.setitem(
            sys.modules,
            "sglang.lang.backend",
            SimpleNamespace(runtime_endpoint=SimpleNamespace(RuntimeEndpoint=object)),
        )

    def runtime(**kwargs):
        assert kwargs == config
        return subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import os; print(os.environ['SGLANG_JIT_CACHE_DIR'])",
            ],
            text=True,
            timeout=30,
        ).strip()

    assert create_runtime(runtime, **config) == str(physical / "jit")
    assert not (physical / "jit").exists()


@pytest.mark.parametrize("configured", [None, "", "~/.cache/sglang/jit"])
def test_runtime_default_jit_cache_is_expanded(monkeypatch, configured):
    if configured is None:
        monkeypatch.delenv("SGLANG_JIT_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("SGLANG_JIT_CACHE_DIR", configured)
    expected = str((Path.home() / ".cache" / "sglang" / "jit").resolve())
    assert create_runtime(lambda: os.environ.get("SGLANG_JIT_CACHE_DIR")) == expected
