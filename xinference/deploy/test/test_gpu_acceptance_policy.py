# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import json
import os
import subprocess
import sys

import pytest

from ...core.tests.test_model_generation_gpu import protected_gpu_environment

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX GPU acceptance")

CHECK_POLICY = """
import asyncio, json, os
from xinference.core import worker

async def check():
    assert await worker.WorkerActor._cleanup_gpu_orphans_on_startup(object()) == []
    assert await worker._kill_gpu_orphans_by_ppid([0, 1]) == []
    assert await worker._kill_orphan_gpu_pids([0, 1], set()) == []

asyncio.run(check())
print(json.dumps({"pid": os.getpid(), "path": os.environ.get("PYTHONPATH")}))
"""


def fake_worker(root, source):
    package = root / "gpu-test-policy" / "xinference"
    (package / "core").mkdir(parents=True)
    (package / "__init__.py").touch()
    (package / "core" / "__init__.py").touch()
    (package / "core" / "worker.py").write_text(source)


@pytest.mark.parametrize("explicit_target", [False, True])
def test_gpu_policy_survives_fresh_and_replacement_interpreters(
    tmp_path, explicit_target
):
    root = tmp_path / "test environment's files"
    root.mkdir()
    environment, replacement = protected_gpu_environment(
        root,
        {**os.environ, "PYTHONPATH": "stale-source-path"},
        sys.executable if explicit_target else None,
    )
    # A cleanup attempt is a test failure; no real GPU or foreign process is
    # touched. Each interpreter must replace all three cleanup entry points.
    fake_worker(
        root,
        "async def cleanup(*args, **kwargs):\n"
        "    raise AssertionError('must not reap unrelated GPU processes')\n"
        "_kill_gpu_orphans_by_ppid = _kill_orphan_gpu_pids = cleanup\n"
        "class WorkerActor:\n"
        "    _cleanup_gpu_orphans_on_startup = cleanup\n",
    )
    source = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", CHECK_POLICY], env=environment, text=True
        )
    )
    if explicit_target:
        environment.pop("PYTHONPATH")
    for _ in range(2):
        target = json.loads(
            subprocess.check_output(
                [replacement or sys.executable, "-c", CHECK_POLICY],
                env=environment,
                text=True,
            )
        )
        assert target["pid"] != source["pid"]
        assert target["path"] == (
            str(root / "gpu-test-policy")
            if explicit_target
            else environment["PYTHONPATH"]
        )


def test_gpu_policy_failure_stops_interpreter_before_startup(tmp_path):
    environment, replacement = protected_gpu_environment(
        tmp_path, os.environ, sys.executable
    )
    fake_worker(tmp_path, "raise ImportError('broken test environment')\n")
    environment.pop("PYTHONPATH")
    marker = tmp_path / "worker-started"
    result = subprocess.run(
        [replacement, "-c", f"open({str(marker)!r}, 'w').close()"],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 90
    assert "broken test environment" in result.stderr
    assert not marker.exists()
