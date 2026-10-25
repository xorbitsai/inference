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

import hashlib
import io
import json
import os
import plistlib
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from .. import service as module


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    result = module.ServiceManager()
    monkeypatch.setitem(
        sys.modules,
        "pwd",
        SimpleNamespace(
            getpwnam=lambda name: SimpleNamespace(
                pw_dir=str(tmp_path), pw_uid=0, pw_gid=0
            )
        ),
    )
    result.directory = tmp_path / "service"
    result.definition = tmp_path / "units/xinference.service"
    result.config_path = result.directory / "config.json"
    monkeypatch.setattr(result, "_permission", lambda: None)
    monkeypatch.setattr(result, "_processes", lambda: [])
    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda command: None if command == "uv" else "/bin/systemctl",
    )
    return result


def _config(tmp_path):
    return {
        "managed_by": "xinference",
        "platform": "Linux",
        "host": "127.0.0.1",
        "port": 9997,
        "home": str(tmp_path / "models $literal%name"),
        "user": "runner",
        "command": ["/venv with spaces/bin/python", "-c", "print('hello $world%')"],
        "registered": True,
    }


def _save(manager, config):
    manager.directory.mkdir(parents=True, exist_ok=True)
    manager.config_path.write_text(json.dumps(config))


@pytest.mark.parametrize("system", [False, True])
def test_systemd_escaping_and_process_cleanup(manager, tmp_path, system):
    manager.system = system
    unit = manager._render(_config(tmp_path)).decode()
    assert '"/venv with spaces/bin/python"' in unit
    assert "hello $$world%%" in unit
    assert "models $literal%%name" in unit
    assert "models $$literal" not in unit
    assert f"WorkingDirectory={tmp_path / 'models $literal%%name'}\n" in unit
    assert "KillMode=control-group" in unit
    assert ("User=runner" in unit) == system
    assert ("multi-user.target" in unit) == system


def test_launchd_definition_preserves_arguments(manager, tmp_path):
    manager.platform = "Darwin"
    manager.system = True
    config = _config(tmp_path)
    definition = plistlib.loads(manager._render(config))
    assert definition["ProgramArguments"] == config["command"]
    assert definition["UserName"] == "runner"
    assert definition["EnvironmentVariables"]["XINFERENCE_HOME"] == config["home"]
    assert definition["KeepAlive"] == {"SuccessfulExit": False}


def test_windows_definition_escapes_arguments_and_restarts(manager, tmp_path):
    manager.platform = "Windows"
    config = _config(tmp_path)
    config["command"] = [
        r"C:\Program Files\Python\python.exe",
        "-c",
        'print("<hello>&")',
    ]
    root = ET.fromstring(manager._render(config))
    assert root.findtext("executable") == config["command"][0]
    assert root.findtext("arguments") == subprocess.list2cmdline(config["command"][1:])
    assert root.find("onfailure").attrib == {"action": "restart", "delay": "5sec"}
    assert root.findtext("startmode") == "Automatic"


def test_existing_unmanaged_definition_is_preserved(manager, tmp_path):
    manager.definition.parent.mkdir()
    manager.definition.write_text("existing service")
    with pytest.raises(click.ClickException, match="unmanaged"):
        manager.install("127.0.0.1", 9997, str(tmp_path / "data"), None)
    assert manager.definition.read_text() == "existing service"


def test_service_data_directory_rejects_line_breaks(manager, tmp_path):
    with pytest.raises(click.ClickException, match="line breaks"):
        manager.install("127.0.0.1", 9997, str(tmp_path / "data\ninvalid"), None)
    assert not manager.definition.exists()


def test_install_is_idempotent_and_uninstall_preserves_models(
    manager, tmp_path, monkeypatch
):
    calls = []

    def run(args, check=True):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(module, "_run", run)
    home = tmp_path / "data"
    manager.install("127.0.0.1", 9997, str(home), None)
    (home / "model.bin").write_bytes(b"model")
    installed_calls = list(calls)
    manager.install("127.0.0.1", 9997, str(home), None)
    assert calls == installed_calls
    with pytest.raises(click.ClickException, match="different settings"):
        manager.install("127.0.0.1", 9998, str(home), None)
    manager.uninstall()
    assert (home / "model.bin").read_bytes() == b"model"
    assert not manager.definition.exists()
    assert not manager.config_path.exists()
    assert ["systemctl", "--user", "disable", "xinference.service"] in calls


def test_failed_registration_can_be_retried(manager, tmp_path, monkeypatch):
    failed = [False]

    def run(args, check=True):
        if "enable" in args and not failed[0]:
            failed[0] = True
            raise click.ClickException("failed to enable")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(module, "_run", run)
    with pytest.raises(click.ClickException, match="failed to enable"):
        manager.install("127.0.0.1", 9997, str(tmp_path / "data"), None)
    assert not manager.load()["registered"]
    manager.install("127.0.0.1", 9997, str(tmp_path / "data"), None)
    assert manager.load()["registered"]


def test_interrupted_definition_write_can_be_retried(manager, tmp_path, monkeypatch):
    monkeypatch.setattr(
        module,
        "_run",
        lambda args, check=True: subprocess.CompletedProcess(args, 0, "", ""),
    )
    write_bytes = Path.write_bytes

    def interrupted_write(path, data):
        if path == manager.definition:
            assert not manager.load()["registered"]
            write_bytes(path, data[:10])
            raise OSError("definition write interrupted")
        return write_bytes(path, data)

    with monkeypatch.context() as context:
        context.setattr(Path, "write_bytes", interrupted_write)
        with pytest.raises(OSError, match="definition write interrupted"):
            manager.install("127.0.0.1", 9997, str(tmp_path / "data"), None)
    manager.install("127.0.0.1", 9997, str(tmp_path / "data"), None)
    assert manager.load()["registered"]
    assert manager.definition.read_bytes().startswith(b"[Unit]")


def test_system_launchd_logs_are_writable_by_the_service_account(
    manager, tmp_path, monkeypatch
):
    manager.platform = "Darwin"
    manager.system = True
    owners = []
    monkeypatch.setattr(
        module.os, "chown", lambda *args: owners.append(args), raising=False
    )
    monkeypatch.setattr(
        module,
        "_run",
        lambda args, check=True: subprocess.CompletedProcess(args, 1, "", ""),
    )
    manager.install("127.0.0.1", 9997, str(tmp_path / "data"), None)
    console_log = manager.directory / "logs/console.log"
    assert console_log.is_file()
    assert (console_log, 0, 0) in owners


def test_windows_registration_retry_reuses_owned_native_service(
    manager, tmp_path, monkeypatch
):
    manager.platform = "Windows"
    manager.system = True
    manager.definition = manager.directory / "xinference-service.xml"
    monkeypatch.setattr(module, "_WINSW_SHA256", hashlib.sha256(b"winsw").hexdigest())
    monkeypatch.setattr(
        manager, "_download_wrapper", lambda: manager.wrapper.write_bytes(b"winsw")
    )
    installed = [False]
    calls = []

    def run(args, check=True):
        calls.append(args)
        if args[0] == "sc.exe":
            return subprocess.CompletedProcess(args, 1060, "", "")
        if args[-1] == "status":
            return subprocess.CompletedProcess(
                args, 0, "Stopped" if installed[0] else "NonExistent", ""
            )
        if args[-1] == "install":
            installed[0] = True
            raise click.ClickException("registration interrupted")
        raise AssertionError(args)

    monkeypatch.setattr(module, "_run", run)
    with pytest.raises(click.ClickException, match="interrupted"):
        manager.install("127.0.0.1", 9997, str(tmp_path / "data"), None)
    assert not manager.load()["registered"]
    manager.install("127.0.0.1", 9997, str(tmp_path / "data"), None)
    assert manager.load()["registered"]
    assert sum(args[-1] == "install" for args in calls) == 1


def test_uvx_environment_cannot_be_registered(manager, tmp_path, monkeypatch):
    monkeypatch.setattr(module.shutil, "which", lambda command: "/bin/uv")
    monkeypatch.setattr(module.sys, "prefix", str(tmp_path / "cache/archive-v0/tool"))
    monkeypatch.setattr(
        module,
        "_run",
        lambda args: subprocess.CompletedProcess(args, 0, str(tmp_path / "cache"), ""),
    )
    with pytest.raises(click.ClickException, match="disposable uvx"):
        manager.install("127.0.0.1", 9997, None, None)
    assert not manager.definition.exists()


def test_failed_health_check_stops_new_service(manager, tmp_path, monkeypatch):
    _save(manager, _config(tmp_path))
    actions = []

    def control(action, check=True):
        actions.append(action)
        return subprocess.CompletedProcess(
            [], 3 if action == "is-active" else 0, "", ""
        )

    monkeypatch.setattr(manager, "_control", control)
    monkeypatch.setattr(module, "_check_port", lambda *args: None)

    def fail(*args):
        raise click.ClickException("not ready")

    monkeypatch.setattr(module, "_wait_ready", fail)
    with pytest.raises(click.ClickException, match="not ready"):
        manager.start(0.1)
    assert actions == ["is-active", "start", "status", "stop"]


def test_port_conflict_does_not_start_service(manager, tmp_path, monkeypatch):
    _save(manager, _config(tmp_path))
    actions = []
    monkeypatch.setattr(
        manager,
        "_control",
        lambda action, check=True: actions.append(action)
        or subprocess.CompletedProcess([], 3, "", ""),
    )

    def conflict(*args):
        raise click.ClickException("port occupied")

    monkeypatch.setattr(module, "_check_port", conflict)
    with pytest.raises(click.ClickException, match="port occupied"):
        manager.start(1)
    assert actions == ["is-active"]


def test_native_stop_failure_is_reported(manager, tmp_path, monkeypatch):
    _save(manager, _config(tmp_path))
    monkeypatch.setattr(
        manager,
        "_control",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 1, "", "permission denied"
        ),
    )
    with pytest.raises(click.ClickException, match="permission denied"):
        manager.stop()


def test_uninstall_cleans_up_an_inactive_invalid_unit(manager, tmp_path, monkeypatch):
    _save(manager, _config(tmp_path))
    manager.definition.parent.mkdir()
    manager.definition.write_text("invalid unit")

    def control(action, check=True):
        if action == "stop":
            return subprocess.CompletedProcess([], 1, "", "Unit not loaded.")
        if action == "is-active":
            return subprocess.CompletedProcess([], 3, "inactive\n", "")
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(manager, "_control", control)
    monkeypatch.setattr(
        module, "_run", lambda args: subprocess.CompletedProcess(args, 0, "", "")
    )
    manager.uninstall()
    assert not manager.definition.exists()
    assert not manager.config_path.exists()


def test_stop_waits_for_owned_processes(manager, tmp_path, monkeypatch):
    _save(manager, _config(tmp_path))
    alive = [True]
    process = SimpleNamespace(
        pid=123, is_running=lambda: alive[0], status=lambda: "running"
    )
    monkeypatch.setattr(manager, "_processes", lambda: [process])
    monkeypatch.setattr(
        manager,
        "_control",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )

    def release(interval):
        alive[0] = False

    monkeypatch.setattr(module.time, "sleep", release)
    manager.stop()
    assert not alive[0]


def test_stop_reports_live_processes_after_timeout(manager, tmp_path, monkeypatch):
    _save(manager, _config(tmp_path))
    process = SimpleNamespace(
        pid=123, is_running=lambda: True, status=lambda: "running"
    )
    monkeypatch.setattr(manager, "_processes", lambda: [process])
    monkeypatch.setattr(
        manager,
        "_control",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )
    clock = iter([0, 1, 36])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(module.time, "sleep", lambda *args: None)
    with pytest.raises(click.ClickException, match="processes did not stop"):
        manager.stop()


def test_checksum_failure_never_writes_executable(manager, monkeypatch):
    manager.platform = "Windows"
    manager.directory.mkdir()
    monkeypatch.setattr(module.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(b"corrupt download"),
    )
    with pytest.raises(click.ClickException, match="checksum"):
        manager._download_wrapper()
    assert not manager.wrapper.exists()


def test_wait_ready_checks_response_and_bypasses_proxy(monkeypatch):
    responses = [b'{"another": "server"}', b'{"uptime": 1, "workers": {}}']

    class Opener:
        def open(self, url, timeout):
            assert url == "http://127.0.0.1:9997/status"
            return io.BytesIO(responses.pop(0))

    def build(handler):
        assert handler.proxies == {}
        return Opener()

    monkeypatch.setattr(module.urllib.request, "build_opener", build)
    monkeypatch.setattr(module.time, "sleep", lambda *args: None)
    module._wait_ready("0.0.0.0", 9997, 1)
    assert not responses


def test_check_port_reports_occupied_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        with pytest.raises(click.ClickException, match="Cannot listen"):
            module._check_port("127.0.0.1", sock.getsockname()[1])


@pytest.mark.skipif(os.name == "nt", reason="POSIX TIME_WAIT reuse semantics")
def test_port_check_allows_restarting_a_closed_server():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen()
        with socket.create_connection(("127.0.0.1", port)) as client:
            connection, _ = listener.accept()
            connection.close()
            assert client.recv(1) == b""
    module._check_port("127.0.0.1", port)


def test_cli_uses_shared_manager(manager, monkeypatch):
    monkeypatch.setattr(module, "ServiceManager", lambda system: manager)
    calls = []
    monkeypatch.setattr(manager, "install", lambda *args: calls.append(args))
    monkeypatch.setattr(manager, "start", lambda timeout: calls.append(timeout))
    result = CliRunner().invoke(
        module.service, ["install", "--port", "12345", "--start"]
    )
    assert result.exit_code == 0, result.output
    assert calls == [("127.0.0.1", 12345, None, None), 120]
    from ..cmdline import cli

    assert CliRunner().invoke(cli, ["service", "--help"]).exit_code == 0
    result = CliRunner().invoke(module.service, ["install", "--port", "0"])
    assert result.exit_code != 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell installer")
def test_shell_installer_delegates_upgrade_flow(tmp_path):
    mode = "system"
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "calls.jsonl"
    uv = binaries / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        + "import json, os, sys\n"
        + f"with open({str(log)!r}, 'a') as f: f.write(json.dumps([sys.argv, os.environ.get('XINFERENCE_SERVICE')]) + '\\n')\n"
        + "if sys.argv[1:] == ['tool', 'install', '--help']: print('--torch-backend')\n"
    )
    uv.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
        XINFERENCE_SERVICE=mode,
    )
    script = Path(__file__).resolve().parents[3] / "scripts/install.sh"
    if not script.is_file():
        pytest.skip("Installer scripts are not included in wheels")
    result = subprocess.run(
        ["sh", str(script)], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[-1][0][1:7] == [
        "run",
        "--no-project",
        "--no-config",
        "--python",
        "3.12",
        "python",
    ]
    assert Path(calls[-1][0][-1]).resolve() == script.with_name("manage_install.py")
    assert calls[-1][0][-2] == "-I"
    assert calls[-1][1] == mode


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell installer")
def test_piped_installer_ignores_cwd_and_shared_temp_modules(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    uv = binaries / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "if sys.argv[1:] == ['tool', 'install', '--help']: print('--torch-backend')\n"
        "elif sys.argv[1] == 'run': os.execv(sys.executable, [sys.executable, *sys.argv[7:]])\n"
    )
    uv.chmod(0o755)
    (tmp_path / "manage_install.py").write_text(
        "raise RuntimeError('Executed a cwd driver')\n"
    )
    (tmp_path / "json.py").write_text(
        "raise RuntimeError('Imported a shared-temp module')\n"
    )
    driver = tmp_path / "trusted.py"
    driver.write_text("import json\nprint('Downloaded driver ran in isolation')\n")
    script = Path(__file__).resolve().parents[3] / "scripts/install.sh"
    if not script.is_file():
        pytest.skip("Installer scripts are not included in wheels")
    result = subprocess.run(
        ["sh"],
        input=script.read_text(),
        cwd=tmp_path,
        env=dict(
            os.environ,
            PATH=str(binaries) + os.pathsep + os.environ["PATH"],
            TMPDIR=str(tmp_path),
            XINFERENCE_INSTALLER_URL=driver.as_uri(),
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Downloaded driver ran in isolation" in result.stdout
    assert not list(
        tmp_path.glob("tmp.*")
    ), "Private download directory was not removed"


def test_windows_service_uses_protected_control_directory_and_isolated_python(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(module.platform, "system", lambda: "Windows")
    program_files = tmp_path / "Program Files"
    program_data = tmp_path / "ProgramData"
    monkeypatch.setenv("ProgramFiles", str(program_files))
    monkeypatch.setenv("PROGRAMDATA", str(program_data))
    manager = module.ServiceManager()
    monkeypatch.setattr(manager, "_permission", lambda: None)
    monkeypatch.setattr(manager, "_download_wrapper", lambda: None)
    monkeypatch.setattr(
        manager,
        "_control",
        lambda action, check=True: subprocess.CompletedProcess([], 0, "stopped", ""),
    )
    monkeypatch.setattr(module.shutil, "which", lambda command: None)
    monkeypatch.setattr(
        module,
        "_run",
        lambda args, check=True: subprocess.CompletedProcess(args, 1060, "", ""),
    )
    manager.install("127.0.0.1", 9997, None, None)
    config = manager.load()
    definition = ET.fromstring(manager.definition.read_bytes())
    assert manager.directory == program_files / "Xinference/service"
    assert config["home"] == str((program_data / "Xinference/data").absolute())
    assert definition.findtext("workingdirectory") == str(manager.directory)
    assert config["command"][1:4] == ["-I", "-u", "-c"]
    warning = capsys.readouterr().err
    assert sys.executable in warning
    assert "does not restrict" in warning
    assert "LocalSystem" in warning
