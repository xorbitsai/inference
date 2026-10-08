"""Exercise an installed candidate wheel, including native service cleanup."""

import getpass
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

from xinference.deploy.service import _check_port, _wait_ready

HOST = "127.0.0.1"
PORT = 19997


def command(*args):
    invocation = [
        sys.executable,
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


def install_service(home):
    scripts = Path(os.environ["GITHUB_WORKSPACE"]) / "scripts"
    args = (
        ["pwsh", "-NoProfile", "-File", str(scripts / "install.ps1")]
        if os.name == "nt"
        else ["sh", str(scripts / "install.sh")]
    )
    result = subprocess.run(
        args,
        env=dict(
            os.environ,
            XINFERENCE_SERVICE="system",
            XINFERENCE_START="1",
            XINFERENCE_HOME=str(home),
            XINFERENCE_HOST=HOST,
            XINFERENCE_PORT=str(PORT),
        ),
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


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
                        sys.executable,
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
