"""Exercise an installed candidate wheel, including native service cleanup."""

import getpass
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import psutil

HOST = "127.0.0.1"
PORT = 19997
TOOL_PYTHON = (
    Path(os.environ["UV_TOOL_DIR"])
    / "xinference"
    / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
)


def _wait_ready(host, port, timeout):
    deadline = time.monotonic() + timeout
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        try:
            with opener.open(f"http://{host}:{port}/status", timeout=2) as response:
                status = json.load(response)
                if "uptime" in status and isinstance(status.get("workers"), dict):
                    return
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    raise RuntimeError("The installed server did not become ready.")


def _check_port(host, port):
    with socket.socket() as sock:
        if os.name != "nt":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))


def command(*args):
    invocation = [
        str(TOOL_PYTHON),
        "-c",
        "from xinference.deploy.cmdline import cli; cli()",
        "service",
        "--system",
        *args,
    ]
    if os.name != "nt":
        invocation = ["sudo", "-n", *invocation]
    return invocation


def run(*args):
    result = subprocess.run(command(*args), capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def install_service(home=None, overrides=None, check=True):
    scripts = Path(os.environ["GITHUB_WORKSPACE"]) / "scripts"
    args = (
        ["pwsh", "-NoProfile", "-File", str(scripts / "install.ps1")]
        if os.name == "nt"
        else ["sh", str(scripts / "install.sh")]
    )
    env = dict(os.environ)
    if home is not None:
        env.update(
            XINFERENCE_SERVICE="system",
            XINFERENCE_START="1",
            XINFERENCE_HOME=str(home),
            XINFERENCE_HOST=HOST,
            XINFERENCE_PORT=str(PORT),
        )
    else:
        # Repeat the plain one-command installer: it must detect the service.
        for key in (
            "XINFERENCE_SERVICE",
            "XINFERENCE_HOME",
            "XINFERENCE_HOST",
            "XINFERENCE_PORT",
            "XINFERENCE_PACKAGE",
            "XINFERENCE_BACKEND",
        ):
            env.pop(key, None)
        env["XINFERENCE_START"] = "1"
    env.update(overrides or {})
    result = subprocess.run(
        args,
        env=env,
        capture_output=True,
        text=True,
    )
    if check and result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result


def version():
    return subprocess.check_output(
        [
            str(TOOL_PYTHON),
            "-c",
            "from importlib.metadata import version; print(version('xinference'))",
        ],
        text=True,
    ).strip()


def service_config():
    if os.name == "nt":
        return Path(os.environ["PROGRAMDATA"]) / "Xinference/service/config.json"
    if sys.platform == "darwin":
        return Path("/Library/Application Support/Xinference/service/config.json")
    return Path("/etc/xinference/config.json")


def candidate_wheel(source, directory, target, broken=False):
    # Keep actual runtime/dependencies, changing metadata and optionally startup.
    with zipfile.ZipFile(source) as original:
        metadata = next(
            name for name in original.namelist() if name.endswith(".dist-info/METADATA")
        )
        prefix = metadata.rsplit("/", 1)[0]
        replacement = f"xinference-{target}.dist-info"
        output = directory / f"xinference-{target}-py3-none-any.whl"
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
            for name in original.namelist():
                if name.endswith("/RECORD"):
                    continue
                data = original.read(name)
                if name == metadata:
                    data = re.sub(
                        rb"(?m)^Version: [^\r\n]+", f"Version: {target}".encode(), data
                    )
                if broken and name == "xinference/deploy/cmdline.py":
                    data += b"\ndef _installer_startup_failure(**kwargs):\n    raise RuntimeError('Intentional installer rollback smoke failure')\nlocal.callback = _installer_startup_failure\n"
                wheel.writestr(name.replace(prefix, replacement, 1), data)
            wheel.writestr(f"{replacement}/RECORD", "")
    return output


def check_upgrades(home, temporary):
    original_version = version()
    definition = service_config().read_bytes()
    source = Path(os.environ["XINFERENCE_PACKAGE"])
    first = descendants(native_pid())
    pid = native_pid()
    install_service(overrides={"XINFERENCE_PACKAGE": str(source)})
    assert native_pid() == pid, "Same-version repeat restarted the service"
    wheels = temporary / "upgrade-wheels"
    wheels.mkdir()
    candidate_wheel(source, wheels, "99.0.0")
    install_service(overrides={"UV_FIND_LINKS": str(wheels), "UV_OFFLINE": "1"})
    assert_stopped(first)
    assert version() == "99.0.0"
    assert service_config().read_bytes() == definition
    assert (home / "preserve-model-data").read_text() == "model data"
    pid = native_pid()
    install_service(overrides={"UV_FIND_LINKS": str(wheels), "UV_OFFLINE": "1"})
    assert native_pid() == pid
    bad = candidate_wheel(source, wheels, "99.0.1", broken=True)
    failed = install_service(
        overrides={"XINFERENCE_PACKAGE": str(bad), "XINFERENCE_TIMEOUT": "2"},
        check=False,
    )
    assert failed.returncode != 0, "Broken startup unexpectedly succeeded"
    assert "Restored the previous version" in failed.stdout, (
        failed.stdout + failed.stderr
    )
    assert version() == "99.0.0"
    _wait_ready(HOST, PORT, 30)
    assert service_config().read_bytes() == definition
    assert (home / "preserve-model-data").read_text() == "model data"
    install_service(overrides={"XINFERENCE_PACKAGE": str(source)})
    assert version() == original_version
    assert service_config().read_bytes() == definition
    print("Installer upgrade, no-op repeat, rollback, and downgrade passed.")


def native_pid():
    if sys.platform == "win32":
        args = ["sc.exe", "queryex", "Xinference"]
        pattern = r"PID\s+:\s+(\d+)"
    elif sys.platform == "darwin":
        args = ["sudo", "-n", "launchctl", "print", "system/io.xinference.local"]
        pattern = r"^\s*pid = (\d+)"
    else:
        args = [
            "systemctl",
            "show",
            "xinference.service",
            "--property=MainPID",
            "--value",
        ]
        pattern = r"^(\d+)"
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    match = re.search(pattern, result.stdout, re.MULTILINE)
    if not match or int(match.group(1)) == 0:
        raise RuntimeError(f"Could not find service PID: {result.stdout}")
    return int(match.group(1))


def descendants(pid):
    parent = psutil.Process(pid)
    return [parent, *parent.children(recursive=True)]


def assert_stopped(processes):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        live = []
        for process in processes:
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    live.append(process)
            except psutil.NoSuchProcess:
                pass
        if not live:
            return
        time.sleep(0.2)
    raise RuntimeError(f"Service left live processes: {[p.pid for p in live]}")


def main():
    with tempfile.TemporaryDirectory(prefix="xinference-install-smoke-") as temporary:
        home = Path(temporary) / "data with spaces"
        # Test the ordinary server entry point on POSIX. Windows lifecycle is
        # exercised through SCM below, where WinSW supplies console signals.
        if os.name != "nt":
            log = Path(temporary) / "foreground.log"
            with log.open("w") as stream:
                process = subprocess.Popen(
                    [
                        str(TOOL_PYTHON),
                        "-c",
                        "from xinference.deploy.cmdline import local; local()",
                        "--host",
                        HOST,
                        "--port",
                        str(PORT),
                    ],
                    env=dict(os.environ, XINFERENCE_HOME=str(home)),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                children = []
                try:
                    _wait_ready(HOST, PORT, 120)
                    children = descendants(process.pid)
                except BaseException:
                    print(log.read_text()[-6000:])
                    raise
                finally:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                            raise
                    assert_stopped(children)
            _check_port(HOST, PORT)
        arguments = [
            "install",
            "--host",
            HOST,
            "--port",
            str(PORT),
            "--home",
            str(home),
            "--start",
            "--timeout",
            "120",
        ]
        if os.name != "nt":
            arguments.extend(["--user", getpass.getuser()])
        try:
            install_service(home)
            run("status")
            old_processes = descendants(native_pid())
            run("restart", "--timeout", "120")
            assert_stopped(old_processes)
            _wait_ready(HOST, PORT, 10)
            processes = descendants(native_pid())
            run("stop")
            assert_stopped(processes)
            _check_port(HOST, PORT)
            run("start", "--timeout", "120")
            # Repeated installation must reuse the registered service.
            run(*arguments)
            run("logs", "--lines", "5")
            (home / "preserve-model-data").write_text("model data")
            check_upgrades(home, Path(temporary))
        except BaseException:
            result = subprocess.run(
                command("logs", "--lines", "100"), capture_output=True, text=True
            )
            print(result.stdout + result.stderr)
            raise
        finally:
            run("uninstall")
        assert (home / "preserve-model-data").read_text() == "model data"
        _check_port(HOST, PORT)
        print("Candidate installation and native service lifecycle passed.")


if __name__ == "__main__":
    main()
