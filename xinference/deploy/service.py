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

"""Native service management for a single local Xinference cluster."""

import getpass
import hashlib
import json
import os
import platform
import plistlib
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import psutil

_LABEL = "io.xinference.local"
_UNIT = "xinference.service"
_WINSW_URL = "https://github.com/winsw/winsw/releases/download/v2.12.0/WinSW-x64.exe"
_WINSW_SHA256 = "05b82d46ad331cc16bdc00de5c6332c1ef818df8ceefcd49c726553209b3a0da"
# Bump when installer-facing config paths, schema, unit names, or CLI flags change.
INSTALLER_API_VERSION = 1


def _run(args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise click.ClickException(str(exc)) from exc
    if check and result.returncode:
        raise click.ClickException(
            result.stderr.strip() or result.stdout.strip() or f"{args[0]} failed"
        )
    return result


def _unit_value(value: str, executable: bool = False) -> str:
    # systemd expands % specifiers even inside quotes; ExecStart also expands $.
    if executable:
        value = value.replace("$", "$$")
    return (
        '"'
        + value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("%", "%%")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        + '"'
    )


def _wait_ready(host: str, port: int, timeout: float) -> None:
    target = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    if ":" in target:
        target = f"[{target}]"
    url = f"http://{target}:{port}/status"
    # Do not send local health requests through a user's HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with opener.open(
                url, timeout=min(2, max(0.1, deadline - time.monotonic()))
            ) as response:
                status = json.load(response)
                if (
                    isinstance(status, dict)
                    and "uptime" in status
                    and isinstance(status.get("workers"), dict)
                ):
                    return
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(min(0.5, max(0, deadline - time.monotonic())))
    raise click.ClickException(
        f"Xinference did not become ready at {url} within {timeout:g}s. Run 'xinference service logs' for details."
    )


def _check_port(host: str, port: int) -> None:
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        family, socktype, proto, _, address = addresses[0]
        with socket.socket(family, socktype, proto) as sock:
            if os.name != "nt":
                # Allow a stopped server's TIME_WAIT sockets, but not a listener.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(address)
    except OSError as exc:
        raise click.ClickException(f"Cannot listen on {host}:{port}: {exc}") from exc


class ServiceManager:
    def __init__(self, system: bool = False):
        self.platform = platform.system()
        self.system = system or self.platform == "Windows"
        if self.platform == "Windows":
            self.directory = (
                Path(os.environ.get("ProgramFiles", "C:/Program Files"))
                / "Xinference"
                / "service"
            )
            self.definition = self.directory / "xinference-service.xml"
        elif self.platform == "Linux":
            self.directory = (
                Path("/etc/xinference")
                if system
                else Path.home() / ".xinference/service"
            )
            root = (
                Path("/etc/systemd/system")
                if system
                else Path(
                    os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
                )
                / "systemd/user"
            )
            self.definition = root / _UNIT
        elif self.platform == "Darwin":
            self.directory = (
                Path("/Library/Application Support/Xinference/service")
                if system
                else Path.home() / ".xinference/service"
            )
            root = (
                Path("/Library/LaunchDaemons")
                if system
                else Path.home() / "Library/LaunchAgents"
            )
            self.definition = root / f"{_LABEL}.plist"
        else:
            raise click.ClickException(
                f"Services are not supported on {self.platform}."
            )
        self.config_path = self.directory / "config.json"

    @property
    def domain(self) -> str:
        return "system" if self.system else f"gui/{os.getuid()}"

    @property
    def wrapper(self) -> Path:
        return self.directory / "xinference-service.exe"

    def _permission(self) -> None:
        if self.platform == "Windows":
            import ctypes

            if not ctypes.windll.shell32.IsUserAnAdmin():  # type: ignore[attr-defined]
                raise click.ClickException(
                    "Windows service management requires an Administrator PowerShell terminal."
                )
        elif self.system and os.geteuid() != 0:
            raise click.ClickException(
                "System service management requires root. Use sudo with the absolute path of this xinference command."
            )

    def load(self) -> Dict[str, Any]:
        try:
            config = json.loads(self.config_path.read_text())
        except (OSError, ValueError) as exc:
            raise click.ClickException(
                "No managed Xinference service was found. Run 'xinference service install' first (use --system for a system service)."
            ) from exc
        if (
            config.get("managed_by") != "xinference"
            or config.get("platform") != self.platform
        ):
            raise click.ClickException("Invalid Xinference service configuration.")
        return config

    def _save_config(self, config: Dict[str, Any]) -> None:
        temporary = self.config_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(config, indent=2) + "\n")
        temporary.replace(self.config_path)

    def _control(self, action: str, check: bool = True) -> subprocess.CompletedProcess:
        if self.platform == "Linux":
            return _run(
                ["systemctl"] + ([] if self.system else ["--user"]) + [action, _UNIT],
                check,
            )
        if self.platform == "Darwin":
            if action == "start":
                args = ["launchctl", "bootstrap", self.domain, str(self.definition)]
            elif action == "stop":
                args = ["launchctl", "bootout", f"{self.domain}/{_LABEL}"]
            elif action == "restart":
                args = ["launchctl", "kickstart", "-k", f"{self.domain}/{_LABEL}"]
            else:
                args = ["launchctl", "print", f"{self.domain}/{_LABEL}"]
            return _run(args, check)
        if (
            not self.wrapper.is_file()
            or hashlib.sha256(self.wrapper.read_bytes()).hexdigest() != _WINSW_SHA256
        ):
            raise click.ClickException(
                "The WinSW executable is missing or its checksum is invalid. Reinstall the service."
            )
        return _run([str(self.wrapper), action], check)

    def _download_wrapper(self) -> None:
        if platform.machine().lower() not in {"amd64", "x86_64"}:
            raise click.ClickException(
                "Windows services currently require x86-64 Windows."
            )
        if self.wrapper.exists():
            if hashlib.sha256(self.wrapper.read_bytes()).hexdigest() != _WINSW_SHA256:
                raise click.ClickException(
                    "Existing WinSW executable has an invalid checksum."
                )
            return
        try:
            with urllib.request.urlopen(_WINSW_URL, timeout=60) as response:
                data = response.read(20 * 1024 * 1024)
        except OSError as exc:
            raise click.ClickException(f"Could not download WinSW: {exc}") from exc
        if hashlib.sha256(data).hexdigest() != _WINSW_SHA256:
            raise click.ClickException("WinSW download checksum verification failed.")
        temporary = self.wrapper.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(self.wrapper)

    def _render(self, config: Dict[str, Any]) -> bytes:
        command = config["command"]
        home = config["home"]
        logs = str(self.directory / "logs")
        if self.platform == "Linux":
            user = f"User={config['user']}\n" if self.system else ""
            text = (
                "[Unit]\nDescription=Xinference local server\nAfter=network.target\n\n"
                f"[Service]\nType=simple\n{user}"
                f"ExecStart={' '.join(_unit_value(arg, executable=True) for arg in command)}\n"
                f"Environment={_unit_value('XINFERENCE_HOME=' + home)}\n"
                # WorkingDirectory takes a literal path, unlike ExecStart.
                f"WorkingDirectory={home.replace('%', '%%')}\n"
                "Restart=on-failure\nRestartSec=5\nKillMode=control-group\nTimeoutStopSec=30\n\n"
                f"[Install]\nWantedBy={'multi-user.target' if self.system else 'default.target'}\n"
            )
            return text.encode()
        if self.platform == "Darwin":
            values: Dict[str, Any] = {
                "Label": _LABEL,
                "ProgramArguments": command,
                "EnvironmentVariables": {"XINFERENCE_HOME": home},
                "WorkingDirectory": home,
                "RunAtLoad": True,
                "KeepAlive": {"SuccessfulExit": False},
                "ThrottleInterval": 5,
                "StandardOutPath": str(Path(logs) / "console.log"),
                "StandardErrorPath": str(Path(logs) / "console.log"),
            }
            if self.system:
                values["UserName"] = config["user"]
            return plistlib.dumps(values)
        root = ET.Element("service")
        for key, value in {
            "id": "Xinference",
            "name": "Xinference",
            "description": "Xinference local model server",
            "executable": command[0],
            "arguments": subprocess.list2cmdline(command[1:]),
            "workingdirectory": str(self.directory),
            "startmode": "Automatic",
            "stoptimeout": "30sec",
            "logpath": logs,
        }.items():
            ET.SubElement(root, key).text = value
        ET.SubElement(root, "env", name="XINFERENCE_HOME", value=home)
        ET.SubElement(root, "onfailure", action="restart", delay="5sec")
        log = ET.SubElement(root, "log", mode="roll-by-size")
        ET.SubElement(log, "sizeThreshold").text = "10240"
        ET.SubElement(log, "keepFiles").text = "4"
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)

    def install(
        self, host: str, port: int, home: Optional[str], user: Optional[str]
    ) -> None:
        self._permission()
        # A service must outlive uv cache cleanup.
        if shutil.which("uv"):
            cache = _run(["uv", "cache", "dir"]).stdout.strip()
            if cache and Path(sys.prefix).resolve().is_relative_to(
                Path(cache).resolve()
            ):
                raise click.ClickException(
                    "Cannot register a service from a disposable uvx environment. Run 'uv tool install xinference' first."
                )
        if self.platform == "Linux" and not shutil.which("systemctl"):
            raise click.ClickException(
                "systemd is required for Linux services. Run xinference-local in the foreground on this system."
            )
        if user and (not self.system or self.platform == "Windows"):
            raise click.ClickException(
                "--user is only supported for Linux/macOS system services."
            )
        account = user or os.environ.get("SUDO_USER") or getpass.getuser()
        if self.platform != "Windows":
            import pwd

            try:
                entry = pwd.getpwnam(account)
            except KeyError as exc:
                raise click.ClickException(f"Unknown service user: {account}") from exc
            default_home = str(Path(entry.pw_dir) / ".xinference")
        else:
            default_home = str(
                Path(os.environ.get("PROGRAMDATA", "C:/ProgramData"))
                / "Xinference/data"
            )
        data_home = str(
            Path(home or os.environ.get("XINFERENCE_HOME") or default_home)
            .expanduser()
            .absolute()
        )
        if any(character in data_home for character in ("\n", "\r")):
            raise click.ClickException("The data directory cannot contain line breaks.")
        command = [
            sys.executable,
            "-I",
            "-u",
            "-c",
            "from xinference.deploy.cmdline import local; local()",
            "--host",
            host,
            "--port",
            str(port),
        ]
        config = {
            "managed_by": "xinference",
            "platform": self.platform,
            "host": host,
            "port": port,
            "home": data_home,
            "user": account,
            "command": command,
        }
        if self.platform == "Windows":
            click.echo(
                f"Warning: this LocalSystem service executes {sys.executable}. "
                "The one-command installer keeps Python in the installing user's tool store "
                "and does not restrict its ACLs or the model data ACLs. "
                "Any account able to modify this Python environment or model code "
                "can run code as LocalSystem. Use administrator-controlled runtime "
                "and model data paths.",
                err=True,
            )
        if self.config_path.exists():
            previous = self.load()
            registered = previous.pop("registered", False)
            if previous != config:
                raise click.ClickException(
                    "A service with different settings is already installed. Stop and uninstall it before changing its settings; model data is preserved."
                )
            if registered:
                click.echo("Xinference service is already installed.")
                return
        elif self.definition.exists():
            raise click.ClickException(
                f"Refusing to overwrite an unmanaged service: {self.definition}"
            )
        if not self.config_path.exists():
            if self.platform == "Windows":
                existing = _run(["sc.exe", "query", "Xinference"], check=False)
                if existing.returncode == 0:
                    raise click.ClickException(
                        "Refusing to overwrite an unmanaged Windows service named Xinference."
                    )
            elif self.platform == "Linux":
                existing = _run(
                    ["systemctl"]
                    + ([] if self.system else ["--user"])
                    + ["show", _UNIT, "--property=FragmentPath", "--value"]
                )
                if existing.stdout.strip():
                    raise click.ClickException(
                        "Refusing to overwrite an unmanaged systemd Xinference service."
                    )
            elif self._control("status", check=False).returncode == 0:
                raise click.ClickException(
                    "Refusing to overwrite an unmanaged launchd Xinference service."
                )
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "logs").mkdir(exist_ok=True)
        home_exists = Path(data_home).exists()
        Path(data_home).mkdir(parents=True, exist_ok=True)
        if not home_exists and self.system and self.platform != "Windows":
            os.chown(data_home, entry.pw_uid, entry.pw_gid)
        if self.system and self.platform == "Darwin":
            # The service account must be able to open launchd's output files.
            console_log = self.directory / "logs/console.log"
            console_log.touch(exist_ok=True)
            os.chown(console_log, entry.pw_uid, entry.pw_gid)
        if self.platform == "Windows":
            self._download_wrapper()
        # Establish ownership before the native definition: an interrupted write
        # can then be retried instead of being mistaken for an unmanaged service.
        config["registered"] = False
        self._save_config(config)
        self.definition.parent.mkdir(parents=True, exist_ok=True)
        self.definition.write_bytes(self._render(config))
        if self.system and self.platform != "Windows":
            self.definition.chmod(0o644)
        if self.platform == "Linux":
            _run(
                ["systemctl"] + ([] if self.system else ["--user"]) + ["daemon-reload"]
            )
            self._control("enable")
        elif self.platform == "Windows":
            native = self._control("status")
            if "nonexistent" in native.stdout.lower():
                self._control("install")
        config["registered"] = True
        self._save_config(config)
        click.echo("Installed Xinference service.")

    def start(self, timeout: float) -> None:
        self._permission()
        config = self.load()
        running = self._control(
            "is-active" if self.platform == "Linux" else "status", check=False
        )
        active = running.returncode == 0
        if self.platform == "Windows":
            active = "started" in running.stdout.lower()
        elif self.platform == "Darwin":
            active = active and "state = running" in running.stdout
        if not active:
            _check_port(config["host"], config["port"])
            if self.platform == "Darwin" and running.returncode == 0:
                self._control("restart")
            else:
                self._control("start")
        try:
            _wait_ready(config["host"], config["port"], timeout)
        except BaseException:
            if not active:
                try:
                    status = self._control("status", check=False)
                    click.echo(status.stdout or status.stderr, err=True, nl=False)
                    self.stop()
                except click.ClickException as exc:
                    click.echo(f"Service cleanup failed: {exc}", err=True)
            raise
        click.echo(f"Xinference is ready on port {config['port']}.")

    def _processes(self) -> List[psutil.Process]:
        if self.platform == "Linux":
            args = (
                ["systemctl"]
                + ([] if self.system else ["--user"])
                + ["show", _UNIT, "--property=MainPID", "--value"]
            )
            output = _run(args, check=False).stdout
            pattern = r"^(\d+)"
        elif self.platform == "Darwin":
            output = self._control("status", check=False).stdout
            pattern = r"^\s*pid = (\d+)"
        else:
            output = _run(["sc.exe", "queryex", "Xinference"], check=False).stdout
            pattern = r"PID\s+:\s+(\d+)"
        match = re.search(pattern, output, re.MULTILINE)
        if not match or int(match.group(1)) == 0:
            return []
        try:
            process = psutil.Process(int(match.group(1)))
            return [process, *process.children(recursive=True)]
        except psutil.NoSuchProcess:
            return []

    def stop(self) -> None:
        self._permission()
        self.load()
        processes = self._processes()
        result = self._control("stop", check=False)
        if result.returncode:
            native = self._control("status", check=False)
            stopped = (
                self.platform == "Windows"
                and native.returncode == 0
                and any(
                    state in native.stdout.lower()
                    for state in ("stopped", "nonexistent")
                )
            )
            if self.platform == "Linux" and not processes:
                active = self._control("is-active", check=False)
                # A failed registration can leave a definition that systemd
                # never loaded. Allow uninstalling that inactive service.
                stopped = active.returncode in {3, 4} and active.stdout.strip() in {
                    "inactive",
                    "unknown",
                }
            missing = self.platform == "Darwin" and any(
                message in (native.stderr + result.stderr).lower()
                for message in ("could not find service", "no such process")
            )
            if not stopped and not missing:
                raise click.ClickException(
                    result.stderr.strip()
                    or result.stdout.strip()
                    or "Could not stop the service."
                )
        # launchctl bootout returns before a graceful shutdown finishes.
        # Wait for the captured tree so restart cannot race old actor processes.
        deadline = time.monotonic() + 35
        while processes and time.monotonic() < deadline:
            live = []
            for process in processes:
                try:
                    if (
                        process.is_running()
                        and process.status() != psutil.STATUS_ZOMBIE
                    ):
                        live.append(process)
                except psutil.NoSuchProcess:
                    pass
            processes = live
            if processes:
                time.sleep(0.2)
        if processes:
            raise click.ClickException(
                f"Service processes did not stop: {[p.pid for p in processes]}"
            )
        click.echo("Stopped Xinference service.")

    def uninstall(self) -> None:
        self._permission()
        self.load()
        self.stop()
        if self.platform == "Linux":
            self._control("disable")
        elif self.platform == "Windows":
            native = self._control("status", check=False)
            if "nonexistent" not in native.stdout.lower():
                self._control("uninstall")
        self.definition.unlink(missing_ok=True)
        self.config_path.unlink()
        if self.platform == "Linux":
            _run(
                ["systemctl"] + ([] if self.system else ["--user"]) + ["daemon-reload"]
            )
        elif self.platform == "Windows":
            self.wrapper.unlink(missing_ok=True)
        click.echo("Uninstalled the service. Model data and logs were preserved.")

    def logs(self, lines: int) -> None:
        self.load()
        if self.platform == "Linux":
            result = _run(
                ["journalctl"]
                + ([] if self.system else ["--user"])
                + ["-u", _UNIT, "-n", str(lines), "--no-pager"]
            )
            click.echo(result.stdout, nl=False)
            return
        for path in sorted((self.directory / "logs").glob("*.log")):
            click.echo(f"{path}:")
            from collections import deque

            with path.open(errors="replace") as stream:
                click.echo("".join(deque(stream, maxlen=lines)), nl=False)


@click.group(
    help="Manage one local Xinference service. Windows always uses a system service."
)
@click.option(
    "--system",
    is_flag=True,
    help="Manage a system service instead of a per-user service.",
)
@click.pass_context
def service(ctx: click.Context, system: bool) -> None:
    ctx.obj = ServiceManager(system)


@service.command("install")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=click.IntRange(1, 65535), default=9997, show_default=True)
@click.option(
    "--home",
    type=click.Path(file_okay=False),
    help="Persistent Xinference data directory.",
)
@click.option("--user", help="Account for a Linux/macOS system service.")
@click.option("--start", is_flag=True, help="Start the service and wait for readiness.")
@click.option(
    "--timeout",
    type=click.FloatRange(min=0, min_open=True),
    default=120,
    show_default=True,
)
@click.pass_obj
def install(
    manager: ServiceManager,
    host: str,
    port: int,
    home: Optional[str],
    user: Optional[str],
    start: bool,
    timeout: float,
) -> None:
    manager.install(host, port, home, user)
    if start:
        manager.start(timeout)


@service.command("start")
@click.option(
    "--timeout",
    type=click.FloatRange(min=0, min_open=True),
    default=120,
    show_default=True,
)
@click.pass_obj
def start(manager: ServiceManager, timeout: float) -> None:
    manager.start(timeout)


@service.command("stop")
@click.pass_obj
def stop(manager: ServiceManager) -> None:
    manager.stop()


@service.command("restart")
@click.option(
    "--timeout",
    type=click.FloatRange(min=0, min_open=True),
    default=120,
    show_default=True,
)
@click.pass_obj
def restart(manager: ServiceManager, timeout: float) -> None:
    manager.stop()
    manager.start(timeout)


@service.command("status")
@click.pass_obj
def status(manager: ServiceManager) -> None:
    manager.load()
    result = manager._control("status", check=False)
    click.echo(result.stdout or result.stderr, nl=False)
    if result.returncode:
        raise click.exceptions.Exit(result.returncode)


@service.command("logs")
@click.option("--lines", type=click.IntRange(1, 10000), default=50, show_default=True)
@click.pass_obj
def logs(manager: ServiceManager, lines: int) -> None:
    manager.logs(lines)


@service.command("uninstall")
@click.pass_obj
def uninstall(manager: ServiceManager) -> None:
    manager.uninstall()
