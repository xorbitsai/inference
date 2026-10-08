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
import io
import json
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "manage_install", Path(__file__).resolve().parents[3] / "scripts/manage_install.py"
)
assert _spec is not None and _spec.loader is not None
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)


@pytest.fixture
def installer(tmp_path, monkeypatch):
    root = tmp_path / "tools with spaces"
    bins = tmp_path / "bin"
    bins.mkdir()

    def run(args, env=None, check=True):
        if args[:3] == ["uv", "tool", "dir"]:
            output = str(bins if "--bin" in args else root)
            return subprocess.CompletedProcess(args, 0, output, "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(module, "run", run)
    result = module.Installer({"XINFERENCE_BACKEND": "cpu"})
    result.target = "2.0.0"
    result.fail_prepare = False
    result.fail_install = False
    result.fail_start = False
    result.calls = []
    result.running = True
    monkeypatch.setattr(result, "permission", lambda: None)

    def write_environment(environment, bin_dir, version):
        python = module.python_path(environment)
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text(version)
        bin_dir.mkdir(parents=True, exist_ok=True)
        (bin_dir / "xinference").write_text(version)
        (environment / "uv-receipt.toml").write_text(
            '[tool]\nrequirements = [{ name = "xinference", extras = ["embedding"] }]\n'
            f'entrypoints = [{{name = "xinference", install-path = {json.dumps(str(bin_dir / "xinference"))}}}]\n'
            '[tool.options]\ntorch-backend = "cpu"\n'
        )

    write_environment(result.environment, bins, "1.0.0")
    (result.environment / "preserve-dependency").write_text("old dependency")

    def info(python, service_pid=0):
        return {
            "version": python.read_text(),
            "python": "/original/python",
            "packages": [f"xinference=={python.read_text()}"],
            "busy": [],
        }

    monkeypatch.setattr(module, "runtime_info", info)

    def find_service():
        result.mode = "user"
        result.config = {
            "host": "127.0.0.1",
            "port": 19997,
            "home": str(tmp_path / "models"),
            "user": "original-account",
            "registered": True,
        }

    monkeypatch.setattr(result, "find_service", find_service)
    monkeypatch.setattr(result, "active", lambda: result.running)

    def install_tool(spec, python, backend, env, *extra):
        staged = "UV_TOOL_DIR" in env
        result.calls.append(("prepare" if staged else "install", spec, python, backend))
        if staged and result.fail_prepare:
            raise RuntimeError("dependency resolution failed")
        write_environment(
            Path(env["UV_TOOL_DIR"]) / "xinference" if staged else result.environment,
            Path(env["UV_TOOL_BIN_DIR"]) if staged else bins,
            result.target,
        )
        if not staged and result.fail_install:
            raise RuntimeError("installation interrupted")

    monkeypatch.setattr(result, "install_tool", install_tool)

    def service(action, *args, python=None):
        version = (python or result.python).read_text()
        result.calls.append((action, version))
        if action == "stop":
            result.running = False
        if action == "start":
            if result.fail_start and version != "1.0.0":
                raise RuntimeError("new service is not ready")
            result.running = True
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(result, "service", service)
    return result


def test_upgrade_prepares_then_stops_and_preserves_service_settings(installer):
    installer.update()
    assert [call[0] for call in installer.calls] == [
        "prepare",
        "stop",
        "install",
        "start",
    ]
    assert installer.calls[0][1:] == (
        "xinference[embedding]",
        "/original/python",
        "cpu",
    )
    assert installer.python.read_text() == "2.0.0"
    assert installer.config["port"] == 19997
    assert installer.config["user"] == "original-account"
    assert installer.running
    assert not installer.journal.exists()


def test_same_version_does_not_stop_or_rebuild(installer):
    installer.target = "1.0.0"
    installer.update()
    assert [call[0] for call in installer.calls] == ["prepare", "start"]
    assert (
        installer.environment / "preserve-dependency"
    ).read_text() == "old dependency"


def test_preparation_failure_keeps_running_service(installer):
    installer.fail_prepare = True
    with pytest.raises(RuntimeError, match="dependency resolution"):
        installer.update()
    assert [call[0] for call in installer.calls] == ["prepare"]
    assert installer.python.read_text() == "1.0.0"
    assert installer.running


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX directory permissions")
def test_unwritable_environment_fails_before_stopping_service(installer, monkeypatch):
    monkeypatch.setattr(module.os, "access", lambda path, flags: False)
    with pytest.raises(RuntimeError, match="Fix permissions"):
        installer.update()
    assert [call[0] for call in installer.calls] == ["prepare"]
    assert installer.running
    assert installer.python.read_text() == "1.0.0"


@pytest.mark.parametrize("failure", ["fail_install", "fail_start"])
@pytest.mark.parametrize("running", [False, True])
def test_failed_upgrade_restores_whole_environment_and_original_state(
    installer, failure, running
):
    setattr(installer, failure, True)
    installer.running = running
    with pytest.raises(RuntimeError):
        installer.update()
    assert installer.python.read_text() == "1.0.0"
    assert (
        installer.environment / "preserve-dependency"
    ).read_text() == "old dependency"
    assert (installer.bin_dir / "xinference").read_text() == "1.0.0"
    assert installer.running == running
    assert not installer.journal.exists()
    if running:
        assert installer.calls[-1] == ("start", "1.0.0")


def test_explicit_target_version_replaces_previous_pin(installer):
    installer.env["XINFERENCE_VERSION"] = "v2.0.0"
    installer.update()
    assert installer.calls[0][1] == "xinference[embedding]==2.0.0"


def test_foreground_process_blocks_in_place_upgrade(installer, monkeypatch):
    def no_service():
        installer.mode = "none"
        installer.config = None

    monkeypatch.setattr(installer, "find_service", no_service)
    original = module.runtime_info
    monkeypatch.setattr(
        module,
        "runtime_info",
        lambda python, service_pid=0: dict(original(python), busy=[123]),
    )
    with pytest.raises(RuntimeError, match="foreground"):
        installer.update()
    assert [call[0] for call in installer.calls] == ["prepare"]


def test_auto_detection_preserves_config_and_rejects_changed_port(
    installer, tmp_path, monkeypatch
):
    home = tmp_path / "account"
    path = home / ".xinference/service/config.json"
    path.parent.mkdir(parents=True)
    config = {
        "managed_by": "xinference",
        "platform": "Linux",
        "command": [str(installer.python)],
        "host": "127.0.0.1",
        "port": 12345,
        "home": "/original/models",
        "user": "original-account",
    }
    path.write_text(json.dumps(config))
    monkeypatch.setattr(module.Path, "home", lambda: home)
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    installer.mode = "auto"
    module.Installer.find_service(installer)
    assert installer.mode == "user"
    assert installer.config == config
    installer.env.update(XINFERENCE_HOME="/original/models/", XINFERENCE_PORT="012345")
    module.Installer.find_service(installer)
    installer.env["XINFERENCE_HOME"] = "/original/models"
    installer.env["XINFERENCE_PORT"] = "9997"
    with pytest.raises(RuntimeError, match="differs"):
        module.Installer.find_service(installer)


def test_managed_constraint_lock_can_be_upgraded_again(installer):
    (installer.environment / ".xinference-install-options.json").write_text(
        json.dumps({"backend": "cpu", "constraint_lock": True})
    )
    old = module.runtime_info(installer.python)
    tool = module.receipt(installer.environment)
    tool["constraints"] = [{"name": "xinference", "specifier": "==1.0.0"}]
    assert installer.settings(old, tool)[0] == "xinference[embedding]"


def test_interrupted_upgrade_recovers_before_preparing_again(installer, monkeypatch):
    restore = installer.restore

    def interrupted_restore(transaction):
        raise RuntimeError("interrupted recovery")

    monkeypatch.setattr(installer, "restore", interrupted_restore)
    installer.fail_install = True
    with pytest.raises(RuntimeError, match="Recovery files were preserved"):
        installer.update()
    transaction = json.loads(installer.journal.read_text())
    backup = Path(transaction["backup"])
    assert (backup / "environment/preserve-dependency").read_text() == "old dependency"
    assert not installer.running

    monkeypatch.setattr(installer, "restore", restore)
    installer.calls.clear()
    installer.fail_install = False
    installer.fail_prepare = True
    with pytest.raises(RuntimeError, match="dependency resolution"):
        installer.update()
    assert installer.calls == [
        ("stop", "1.0.0"),
        ("start", "1.0.0"),
        ("prepare", "xinference[embedding]", "/original/python", "cpu"),
    ]
    assert installer.python.read_text() == "1.0.0"
    assert installer.running
    assert not installer.journal.exists()
    assert not backup.exists()


def test_failed_old_service_restart_keeps_recoverable_backup(installer, monkeypatch):
    service = installer.service

    def fail_restart(action, *args, python=None):
        if action == "start":
            raise RuntimeError("service temporarily unavailable")
        return service(action, *args, python=python)

    monkeypatch.setattr(installer, "service", fail_restart)
    installer.fail_install = True
    with pytest.raises(RuntimeError, match="Rollback failed"):
        installer.update()
    transaction = json.loads(installer.journal.read_text())
    assert (Path(transaction["backup"]) / "environment").is_dir()
    assert installer.python.read_text() == "1.0.0"


def test_lock_rejects_concurrent_installer(tmp_path):
    with module.installation_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another"):
            with module.installation_lock(tmp_path):
                pass


def test_runtime_probe_excludes_its_windows_redirector_but_detects_server(monkeypatch):
    import psutil

    args = [str(Path(sys.prefix) / "python.exe"), "-c", "inspection"]
    redirector = SimpleNamespace(pid=101, cmdline=lambda: args)
    inspector = SimpleNamespace(
        pid=102, cmdline=lambda: args, parents=lambda: [redirector]
    )
    server = SimpleNamespace(
        pid=103,
        info={
            "cmdline": [str(Path(sys.prefix) / "xinference-local"), "--port", "9997"]
        },
    )
    processes = [
        SimpleNamespace(pid=101, info={"cmdline": args}),
        SimpleNamespace(pid=102, info={"cmdline": args}),
        server,
        SimpleNamespace(pid=104, info=server.info),
    ]
    managed = SimpleNamespace(pid=103, children=lambda recursive: [])
    monkeypatch.setattr(
        psutil, "Process", lambda pid=None: inspector if pid is None else managed
    )
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: processes)

    def inspect(args, env=None, check=True):
        output = io.StringIO()
        with monkeypatch.context() as context:
            context.setattr(sys, "argv", ["-c", args[-1]])
            with redirect_stdout(output):
                exec(args[-2], {})
        return subprocess.CompletedProcess(args, 0, output.getvalue(), "")

    monkeypatch.setattr(module, "run", inspect)
    assert module.runtime_info(Path(sys.executable))["busy"] == [103, 104]
    assert module.runtime_info(Path(sys.executable), 103)["busy"] == [104]


def test_noop_foreground_repeat_keeps_busy_environment(installer, monkeypatch):
    def no_service():
        installer.mode = "none"
        installer.config = None

    monkeypatch.setattr(installer, "find_service", no_service)
    original = module.runtime_info
    monkeypatch.setattr(
        module,
        "runtime_info",
        lambda python, service_pid=0: dict(original(python), busy=[123]),
    )
    installer.target = "1.0.0"
    installer.update()
    assert [call[0] for call in installer.calls] == ["prepare"]


def test_separate_foreground_process_blocks_service_environment_replacement(
    installer, monkeypatch
):
    original = module.runtime_info
    monkeypatch.setattr(
        module,
        "runtime_info",
        lambda python, service_pid=0: dict(original(python), busy=[123]),
    )
    with pytest.raises(RuntimeError, match="foreground"):
        installer.update()
    assert [call[0] for call in installer.calls] == ["prepare"]
    assert installer.running


def test_first_install_removes_new_service_when_startup_fails(installer, monkeypatch):
    shutil.rmtree(installer.environment)
    installer.running = False
    registered = [False]
    find = installer.find_service
    service = installer.service

    def find_service():
        find()
        if not registered[0]:
            installer.config = None

    def control(action, *args, python=None):
        if action in {"install", "uninstall"}:
            registered[0] = action == "install"
        return service(action, *args, python=python)

    monkeypatch.setattr(installer, "find_service", find_service)
    monkeypatch.setattr(installer, "service", control)
    installer.fail_start = True
    with pytest.raises(RuntimeError, match="not ready"):
        installer.update()
    assert not registered[0]
    assert installer.calls[-1][0] == "uninstall"
    assert installer.python.read_text() == "2.0.0"
    assert not installer.journal.exists()


def test_windows_foreground_waits_and_propagates_exit_code(installer, monkeypatch):
    installer.mode = "none"
    monkeypatch.setattr(installer, "update", lambda: None)
    monkeypatch.setattr(module.platform, "system", lambda: "Windows")
    calls = []

    def foreground(args, env=None):
        calls.append(args)
        return subprocess.CompletedProcess(args, 17)

    monkeypatch.setattr(module.subprocess, "run", foreground)
    with pytest.raises(SystemExit) as result:
        installer.execute()
    assert result.value.code == 17
    assert calls[0][:4] == [str(installer.python), "-I", "-u", "-c"]
