# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import sys
import venv
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ..lifecycle import WorkerPhase
from ..worker_relaunch import WorkerRelaunchLauncher


@pytest.mark.asyncio
@pytest.mark.parametrize("environment", ["venv", "user-site"])
async def test_worker_relaunch_loads_selected_environment_from_checkout(
    tmp_path, monkeypatch, environment
):
    from ... import constants
    from ...deploy import utils
    from .. import worker_relaunch

    target = tmp_path / "target"
    if environment != "user-site":
        venv.EnvBuilder(with_pip=False).create(target)
    else:
        monkeypatch.setenv("PYTHONUSERBASE", str(target))
    site = target / (
        (
            f"Python{sys.version_info.major}{sys.version_info.minor}/site-packages"
            if environment == "user-site"
            else "Lib/site-packages"
        )
        if sys.platform == "win32"
        else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    checkout = tmp_path / "checkout"
    result = tmp_path / "loaded.txt"
    for root, identity in [(site, "target"), (checkout, "checkout")]:
        package = root / "xinference"
        deploy = package / "deploy"
        deploy.mkdir(parents=True)
        (package / "__init__.py").touch()
        (deploy / "__init__.py").touch()
        (deploy / "worker_relaunch.py").write_text(
            "from pathlib import Path\n"
            f"def main(): Path({str(result)!r}).write_text({identity!r})\n"
            "if __name__ == '__main__': main()\n"
        )
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(constants, "XINFERENCE_HOME", str(tmp_path))
    monkeypatch.setattr(utils, "get_log_file", lambda name: str(tmp_path / "child.log"))
    monkeypatch.setattr(worker_relaunch, "get_next_port", lambda: 12345)
    worker = SimpleNamespace(
        _lifecycle=SimpleNamespace(phase=WorkerPhase.ACTIVE),
        _services_started=True,
        check_worker_handoff_support=AsyncMock(
            return_value=SimpleNamespace(plan_worker_relaunch=AsyncMock())
        ),
        address="127.0.0.1:12344",
        _supervisor_address="127.0.0.1:12343",
        _supervisor_endpoint=None,
        _metrics_exporter_config=(None, None),
        _incarnation="source",
        _total_gpu_devices=[],
        _outgoing_handoff=None,
    )
    launcher = WorkerRelaunchLauncher(worker)
    python = target / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    if environment == "user-site":
        python = sys._base_executable
    await launcher.launch("one", str(python))
    try:
        await asyncio.wait_for(launcher._reaper, 5)
        assert launcher.process.returncode == 0, (tmp_path / "child.log").read_text()
        assert result.read_text() == "target"
    finally:
        if launcher.process.poll() is None:
            launcher.process.kill()
        launcher.process.wait()
