# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import os
import signal
import subprocess
import sys
import time

import psutil
import pytest

from ..weight_cache import WeightCacheDaemon, _process_group_alive, _watcher_command


def wait_until(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    pytest.fail("Timed out waiting for weight cache process cleanup")


def alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
@pytest.mark.parametrize("mode", ["graceful", "ignores_term", "parent_death"])
def test_watcher_waits_for_ranks_and_escalates(tmp_path, mode):
    rank_pid = tmp_path / "rank.pid"
    cleaned = tmp_path / "cleaned"
    watcher_pid = tmp_path / "watcher.pid"
    rank_code = (
        "import os, signal, time\nfrom pathlib import Path\n"
        "def stop(*args):\n"
        "    time.sleep(0.2)\n"
        f"    Path({str(cleaned)!r}).touch()\n"
        "    raise SystemExit(0)\n"
        + (
            "signal.signal(signal.SIGTERM, stop)\n"
            if mode == "graceful"
            else "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        )
        + f"Path({str(rank_pid)!r}).write_text(str(os.getpid()))\n"
        "while True: time.sleep(0.1)\n"
    )
    child_code = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {rank_code!r}])\n"
        "while True: time.sleep(0.1)\n"
    )
    command = _watcher_command(os.getpid(), ["-c", child_code])
    # A watcher must start without importing the model package or GPU libraries.
    # Block those imports in fresh interpreters rather than relaxing timeouts.
    guard = tmp_path / "import-guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import sys\n"
        "class Guard:\n"
        "    def find_spec(self, fullname, *args):\n"
        "        if fullname.split('.')[0] in {'xinference', 'torch', 'xoscar'}:\n"
        "            raise ImportError('Watcher imported a heavy package: ' + fullname)\n"
        "sys.meta_path.insert(0, Guard())\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(guard) + os.pathsep + env.get("PYTHONPATH", "")
    daemon = WeightCacheDaemon("sglang", "/unused", {})
    pgid = None
    parent = None
    try:
        if mode == "parent_death":
            parent_code = (
                "import os, subprocess, time\nfrom pathlib import Path\n"
                f"command = {command!r}\ncommand[2] = str(os.getpid())\n"
                "watcher = subprocess.Popen(command, start_new_session=True)\n"
                f"Path({str(watcher_pid)!r}).write_text(str(watcher.pid))\n"
                f"while not Path({str(rank_pid)!r}).exists(): time.sleep(0.05)\n"
            )
            parent = subprocess.Popen([sys.executable, "-c", parent_code], env=env)
            wait_until(watcher_pid.exists)
            pgid = int(watcher_pid.read_text())
            parent.wait(timeout=15)
        else:
            daemon.process = subprocess.Popen(command, start_new_session=True, env=env)
            pgid = daemon.process.pid
        wait_until(rank_pid.exists)
        rank = int(rank_pid.read_text())
        if mode != "parent_death":
            daemon.stop()
        wait_until(lambda: not alive(rank) and not alive(pgid))
        assert cleaned.exists() is (mode == "graceful")
    finally:
        if parent is not None and parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=5)
        if pgid is not None and _process_group_alive(pgid):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        daemon.stop()
