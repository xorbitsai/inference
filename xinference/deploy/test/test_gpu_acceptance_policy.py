# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ...core.tests import test_model_generation_gpu as gpu_tests
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


@pytest.mark.asyncio
async def test_gpu_cleanup_timeout_preserves_failure_and_report(tmp_path, monkeypatch):
    # No GPU or real service is started. Force the harness to fail just after
    # opening its service log, then make waiting for that fake service time out.
    process = MagicMock(pid=2**30)
    process.poll.return_value = 0
    process.wait.side_effect = subprocess.TimeoutExpired("test service", 10)
    spawn = MagicMock(return_value=process)
    monkeypatch.setenv("XINFERENCE_TEST_GENERATION_MODEL_PATH", "unused")
    monkeypatch.setattr(gpu_tests.pytest, "importorskip", lambda engine: None)
    monkeypatch.setattr(gpu_tests.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        gpu_tests.subprocess, "check_output", lambda *args, **kwargs: "{}"
    )

    def fail_http(**kwargs):
        raise RuntimeError("original acceptance failure")

    monkeypatch.setattr(gpu_tests.httpx, "AsyncClient", fail_http)
    with pytest.raises(RuntimeError, match="original acceptance failure"):
        await gpu_tests.test_gpu_upgrade_keeps_streams_and_requests_serving(
            "vllm", "local", tmp_path
        )
    folder = (tmp_path / "vllm-latest").read_text()
    report = json.loads((Path(folder) / "result.json").read_text())
    assert report["cleanup_survivors"] == [process.pid]
    assert "original acceptance failure" in report["error"]
    assert spawn.call_args.kwargs["stdout"].closed
