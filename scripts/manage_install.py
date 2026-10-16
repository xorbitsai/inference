"""Shared, standalone installation and upgrade transaction for both installers.

Run with uv's Python 3.12 bootstrap; the Xinference runtime may use another Python.
"""

import contextlib
import ctypes
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python 3.10 repository tests.
    import tomli as tomllib  # type: ignore[no-redef]


def run(args, env=None, check=True):
    result = subprocess.run(args, env=env, capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result


def python_path(environment):
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def runtime_info(python, service_pid=0):
    code = """
import importlib.metadata as metadata, json, os, pathlib, sys
import psutil
root = os.path.normcase(str(pathlib.Path(sys.prefix).absolute())) + os.sep
inspection = psutil.Process()
inspection_args = inspection.cmdline()[1:]
own = {inspection.pid}
if int(sys.argv[1]):
    try:
        service = psutil.Process(int(sys.argv[1]))
        own.update(p.pid for p in [service, *service.children(recursive=True)])
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
# Windows virtualenv Python can launch a child through its redirector. That
# waiting parent is part of this metadata probe, not a running server.
for parent in inspection.parents():
    try:
        if parent.cmdline()[1:] == inspection_args:
            own.add(parent.pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
busy = []
for process in psutil.process_iter(['cmdline']):
    try:
        args = process.info['cmdline'] or []
        if process.pid not in own and args and os.path.normcase(args[0]).startswith(root):
            busy.append(process.pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
print(json.dumps({
    'version': metadata.version('xinference'),
    'python': sys._base_executable,
    'packages': sorted(f'{d.metadata["Name"]}=={d.version}' for d in metadata.distributions()),
    'busy': busy,
}))
"""
    return json.loads(run([str(python), "-I", "-c", code, str(service_pid)]).stdout)


def receipt(environment):
    path = environment / "uv-receipt.toml"
    return tomllib.loads(path.read_text())["tool"] if path.exists() else {}


@contextlib.contextmanager
def installation_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".xinference-install.lock").open("a+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt

                stream.write(b"0")
                stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another Xinference installer is running.") from exc
        yield


class Installer:
    def __init__(self, env=None):
        self.env = dict(os.environ if env is None else env)
        if self.env.get("XINFERENCE_TOOL_DIR"):
            self.env["UV_TOOL_DIR"] = self.env["XINFERENCE_TOOL_DIR"]
        self.root = Path(run(["uv", "tool", "dir"], self.env).stdout.strip())
        self.bin_dir = Path(
            run(["uv", "tool", "dir", "--bin"], self.env).stdout.strip()
        )
        self.environment = self.root / "xinference"
        self.python = python_path(self.environment)
        self.journal = self.root / ".xinference-transaction.json"
        self.mode = self.env.get("XINFERENCE_SERVICE", "auto")
        self.start = self.env.get("XINFERENCE_START", "1") == "1"
        self.timeout = float(self.env.get("XINFERENCE_TIMEOUT", "120"))
        if self.mode not in {"auto", "none", "user", "system"}:
            raise RuntimeError(
                "XINFERENCE_SERVICE must be auto, none, user, or system."
            )
        if self.env.get("XINFERENCE_START", "1") not in {"0", "1"}:
            raise RuntimeError("XINFERENCE_START must be 0 or 1.")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise RuntimeError("XINFERENCE_TIMEOUT must be positive.")
        port = int(self.env.get("XINFERENCE_PORT", "9997"))
        if not 1 <= port <= 65535:
            raise RuntimeError("XINFERENCE_PORT must be between 1 and 65535.")
        self.config = None
        self.created_service = False

    def saved_settings(self):
        path = self.environment / ".xinference-install-options.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def find_service(self):
        system = platform.system()
        if system == "Windows":
            paths = {
                "system": Path(
                    self.env.get(
                        "PROGRAMFILES", self.env.get("ProgramFiles", "C:/Program Files")
                    )
                )
                / "Xinference/service/config.json"
            }
        else:
            paths = {
                "user": Path.home() / ".xinference/service/config.json",
                "system": Path("/etc/xinference/config.json")
                if system == "Linux"
                else Path(
                    "/Library/Application Support/Xinference/service/config.json"
                ),
            }
        found = []
        for mode, path in paths.items():
            if not path.exists():
                continue
            config = json.loads(path.read_text())
            command = config.get("command", [])
            owned = (
                config.get("managed_by") == "xinference"
                and config.get("platform") == system
            )
            matching = command and os.path.normcase(
                os.path.abspath(command[0])
            ) == os.path.normcase(str(self.python.absolute()))
            if owned and matching:
                found.append((mode, config))
            elif self.mode == mode:
                raise RuntimeError(
                    "The existing service uses another installation; use its original tool store."
                )
        if self.mode == "auto":
            if len(found) > 1:
                raise RuntimeError(
                    "Both user and system services exist. Set XINFERENCE_SERVICE explicitly."
                )
            self.mode, self.config = found[0] if found else ("none", None)
        else:
            self.config = next(
                (config for mode, config in found if mode == self.mode), None
            )
            if found and not self.config:
                raise RuntimeError(
                    "An existing service uses this tool. Select its service mode before updating."
                )
        if system == "Windows" and self.mode == "user":
            raise RuntimeError("Windows services require system mode.")
        if self.config:
            for key, variable in (
                ("host", "XINFERENCE_HOST"),
                ("port", "XINFERENCE_PORT"),
                ("home", "XINFERENCE_HOME"),
            ):
                if variable not in self.env or (
                    key == "home" and not self.env[variable]
                ):
                    continue
                expected, actual = self.config[key], self.env[variable]
                if key == "port":
                    expected, actual = int(expected), int(actual)
                elif key == "home":
                    expected, actual = (
                        os.path.normcase(
                            os.path.normpath(str(Path(value).expanduser().absolute()))
                        )
                        for value in (expected, actual)
                    )
                if expected != actual:
                    raise RuntimeError(
                        f"{variable} differs from the installed service. Uninstall before changing service settings."
                    )

    def permission(self):
        if self.mode != "system":
            return
        if os.name == "nt":
            if not ctypes.windll.shell32.IsUserAnAdmin():
                raise RuntimeError(
                    "Run the installer in an Administrator PowerShell terminal."
                )
        elif os.geteuid() != 0:
            if not shutil.which("sudo"):
                raise RuntimeError("System service installation requires sudo.")
            # Authenticate before downloading or stopping anything.
            subprocess.run(["sudo", "-v"], check=True)
        elif self.env.get("SUDO_USER"):
            raise RuntimeError(
                "Run the installer as your normal account; it uses sudo for the system service."
            )

    def service(self, *args, python=None):
        command = [
            str(python or self.python),
            "-I",
            "-B",
            "-c",
            "from xinference.deploy.cmdline import cli; cli()",
            "service",
        ]
        if self.mode == "system":
            command.append("--system")
        command.extend(args)
        if self.mode == "system" and os.name != "nt" and os.geteuid() != 0:
            if args and args[0] == "stop":
                subprocess.run(["sudo", "-v"], check=True)
            command = ["sudo", *command]
        return run(command, self.env)

    def active(self):
        if not self.config:
            return False
        if platform.system() == "Linux":
            args = [
                "systemctl",
                *([] if self.mode == "system" else ["--user"]),
                "is-active",
                "xinference.service",
            ]
            return run(args, self.env, check=False).returncode == 0
        if platform.system() == "Darwin":
            domain = "system" if self.mode == "system" else f"gui/{os.getuid()}"
            output = run(
                ["launchctl", "print", f"{domain}/io.xinference.local"],
                self.env,
                check=False,
            )
            return output.returncode == 0 and "state = running" in output.stdout
        output = run(["sc.exe", "query", "Xinference"], self.env, check=False)
        return output.returncode == 0 and bool(
            re.search(r"STATE\s*:\s*4\b", output.stdout)
        )

    def service_pid(self):
        if not self.config:
            return 0
        system = platform.system()
        if system == "Linux":
            args = [
                "systemctl",
                *([] if self.mode == "system" else ["--user"]),
                "show",
                "xinference.service",
                "--property=MainPID",
                "--value",
            ]
            pattern = r"^(\d+)"
        elif system == "Darwin":
            domain = "system" if self.mode == "system" else f"gui/{os.getuid()}"
            args = ["launchctl", "print", f"{domain}/io.xinference.local"]
            pattern = r"^\s*pid = (\d+)"
        else:
            args = ["sc.exe", "queryex", "Xinference"]
            pattern = r"PID\s+:\s+(\d+)"
        if self.mode == "system" and os.name != "nt" and os.geteuid() != 0:
            args = ["sudo", *args]
        output = run(args, self.env, check=False)
        match = re.search(pattern, output.stdout, re.MULTILINE)
        return int(match.group(1)) if output.returncode == 0 and match else 0

    def settings(self, old, tool):
        requirements = tool.get("requirements", [])
        saved = self.saved_settings()
        if (
            any(
                tool.get(key)
                for key in (
                    "overrides",
                    "excludes",
                    "build-constraint-dependencies",
                )
            )
            or len(requirements) > 1
            or (tool.get("constraints") and not saved.get("constraint_lock"))
        ):
            raise RuntimeError(
                "This tool has custom dependency requirements. Update it with uv tool upgrade instead."
            )
        requirement = requirements[0] if requirements else {}
        if isinstance(requirement, str) or requirement.get("editable"):
            raise RuntimeError(
                "This custom/editable tool must be updated with uv tool upgrade."
            )
        extras = self.env.get(
            "XINFERENCE_EXTRAS", ",".join(requirement.get("extras", []))
        )
        python = self.env.get("XINFERENCE_PYTHON") or (old["python"] if old else "3.12")
        backend = self.env.get(
            "XINFERENCE_BACKEND",
            saved.get("backend", tool.get("options", {}).get("torch-backend", "auto")),
        )
        spec = self.env.get("XINFERENCE_PACKAGE", "xinference")
        if not self.env.get("XINFERENCE_PACKAGE"):
            if extras:
                spec += f"[{extras}]"
            if self.env.get("XINFERENCE_VERSION"):
                version = self.env["XINFERENCE_VERSION"].removeprefix("v")
                if not version:
                    raise RuntimeError("XINFERENCE_VERSION must contain a version.")
                spec += f"=={version}"
        return spec, python, backend, extras

    def install_tool(self, spec, python, backend, env, *extra):
        command = ["uv", "tool", "install", "--python", python]
        if platform.system() != "Darwin":
            command.extend(["--torch-backend", backend])
        subprocess.run([*command, *extra, spec], env=env, check=True)

    def configure(self):
        if self.mode == "none":
            return
        if not self.config or not self.config.get("registered"):
            config = self.config or {}
            args = [
                "install",
                "--host",
                str(config.get("host", self.env.get("XINFERENCE_HOST", "127.0.0.1"))),
                "--port",
                str(config.get("port", self.env.get("XINFERENCE_PORT", "9997"))),
            ]
            home = config.get("home", self.env.get("XINFERENCE_HOME"))
            if home:
                args.extend(["--home", str(Path(home).expanduser().absolute())])
            if self.mode == "system" and os.name != "nt":
                import getpass

                args.extend(["--user", config.get("user", getpass.getuser())])
            self.created_service = not self.config
            result = self.service(*args)
            print(result.stderr, file=sys.stderr, end="", flush=True)
            print(result.stdout, end="", flush=True)
        if self.start:
            print(
                self.service("start", "--timeout", str(self.timeout)).stdout,
                end="",
                flush=True,
            )

    def restore(self, transaction):
        backup = Path(transaction["backup"])
        if backup.parent.parent != self.root or not (backup / "environment").is_dir():
            raise RuntimeError("Invalid upgrade recovery directory.")
        print("Restoring the previous Xinference environment...", flush=True)
        if transaction["service_before"]:
            self.find_service()
            if self.config:
                self.service("stop", python=python_path(backup / "environment"))
        elif self.created_service or self.config:
            self.find_service()
            if self.config:
                self.service("uninstall")
                self.config = None
        if self.environment.exists():
            shutil.rmtree(self.environment)
        shutil.copytree(backup / "environment", self.environment, symlinks=True)
        for name in transaction["new_names"]:
            (self.bin_dir / name).unlink(missing_ok=True)
        for entry in transaction["shims"]:
            destination = self.bin_dir / entry["name"]
            destination.unlink(missing_ok=True)
            source = backup / "shims" / entry["name"]
            shutil.copy2(source, destination, follow_symlinks=False)
        if transaction["active"] and self.config:
            print(
                self.service("start", "--timeout", str(max(120, self.timeout))).stdout,
                end="",
                flush=True,
            )
        self.journal.unlink()
        shutil.rmtree(backup.parent)
        if transaction["service_before"] and not self.config:
            print(
                "Restored the previous version; the removed service was not recreated.",
                flush=True,
            )
        else:
            print("Restored the previous version and service state.", flush=True)

    def remove_created_service(self, failure):
        if not self.created_service:
            return
        try:
            self.find_service()
            if self.config:
                self.service("uninstall")
                self.config = None
        except Exception as cleanup:
            raise RuntimeError(
                f"Installation failed: {failure}\nService cleanup failed: {cleanup}\n"
                "Run xinference service uninstall (with --system for a system service) before retrying."
            ) from failure

    def update(self):
        self.find_service()
        self.permission()
        if self.journal.exists():
            self.restore(json.loads(self.journal.read_text()))
        old = (
            runtime_info(self.python, self.service_pid())
            if self.python.exists()
            else None
        )
        tool = receipt(self.environment)
        spec, python, backend, extras = self.settings(old, tool)
        existing = shutil.which("xinference", path=self.env.get("PATH"))
        own_command = self.bin_dir / (
            "xinference.exe" if os.name == "nt" else "xinference"
        )
        if (
            existing
            and Path(existing).resolve().parent != self.python.parent
            and not (old and Path(existing).absolute() == own_command.absolute())
        ):
            print(
                f"Existing Xinference command: {existing}. This installer uses {self.environment}; commands are linked in {self.bin_dir}.",
                flush=True,
            )
        if platform.system() == "Darwin" and platform.machine().lower() == "x86_64":
            print(
                "Recent PyTorch releases do not provide Intel Mac wheels; resolution may select an older version or fail.",
                flush=True,
            )
        temporary = Path(tempfile.mkdtemp(prefix=".xinference-update-", dir=self.root))
        try:
            staged_env = dict(
                self.env,
                UV_TOOL_DIR=str(temporary / "tools"),
                UV_TOOL_BIN_DIR=str(temporary / "bin"),
            )
            print(f"Preparing {spec} before stopping any service...", flush=True)
            self.install_tool(spec, python, backend, staged_env)
            candidate_python = python_path(temporary / "tools/xinference")
            candidate = runtime_info(candidate_python)
            code = "from xinference.deploy.cmdline import cli, local"
            if self.mode != "none":
                code += (
                    "; assert 'service' in cli.commands, 'This release has no service CLI'"
                    "; from xinference.deploy import service"
                    "; assert getattr(service, 'INSTALLER_API_VERSION', None) == 1, 'Incompatible service installer API'"
                )
            run([str(candidate_python), "-I", "-c", code], self.env)
            unchanged = (
                old
                and old["version"] == candidate["version"]
                and old["python"] == candidate["python"]
                and extras
                == ",".join((tool.get("requirements") or [{}])[0].get("extras", []))
                and backend
                == self.saved_settings().get(
                    "backend", tool.get("options", {}).get("torch-backend", "auto")
                )
            )
            if unchanged:
                print(
                    f"Xinference {old['version']} is already installed; keeping its environment.",
                    flush=True,
                )
                try:
                    self.configure()
                except BaseException as failure:
                    self.remove_created_service(failure)
                    raise
                return
            transaction = None
            if old:
                if old["busy"]:
                    raise RuntimeError(
                        f"Stop the foreground Xinference processes before updating: {old['busy']}"
                    )
                if os.name != "nt":
                    for directory, _, _ in os.walk(self.environment):
                        if not os.access(directory, os.W_OK | os.X_OK):
                            raise RuntimeError(
                                f"The tool environment is not writable. Fix permissions before upgrading: {directory}"
                            )
                backup = temporary / "backup"
                backup.mkdir()
                shutil.copytree(self.environment, backup / "environment", symlinks=True)
                (backup / "shims").mkdir()
                shims = []
                for entry in tool.get("entrypoints", []):
                    source = Path(entry["install-path"])
                    if source.parent != self.bin_dir:
                        raise RuntimeError(
                            "Reuse the original UV_TOOL_BIN_DIR when updating this tool."
                        )
                    if source.exists() or source.is_symlink():
                        shutil.copy2(
                            source,
                            backup / "shims" / source.name,
                            follow_symlinks=False,
                        )
                        shims.append({"name": source.name})
                new_names = [
                    Path(entry["install-path"]).name
                    for entry in receipt(temporary / "tools/xinference").get(
                        "entrypoints", []
                    )
                ]
                transaction = {
                    "backup": str(backup),
                    "active": self.active(),
                    "service_before": bool(self.config),
                    "shims": shims,
                    "new_names": new_names,
                }
                journal = self.journal.with_suffix(".tmp")
                journal.write_text(json.dumps(transaction) + "\n")
                journal.replace(self.journal)
            try:
                if self.config:
                    print(
                        "Stopping the existing service for the version switch...",
                        flush=True,
                    )
                    self.service("stop")
                constraints = temporary / "constraints.txt"
                constraints.write_text("\n".join(candidate["packages"]) + "\n")
                self.install_tool(
                    spec,
                    python,
                    backend,
                    self.env,
                    "--force",
                    "--offline",
                    "--constraints",
                    str(constraints),
                )
                (self.environment / ".xinference-install-options.json").write_text(
                    json.dumps({"backend": backend, "constraint_lock": True}) + "\n"
                )
                self.configure()
            except BaseException as failure:
                if transaction:
                    try:
                        self.restore(transaction)
                    except Exception as rollback:
                        raise RuntimeError(
                            f"Upgrade failed: {failure}\nRollback failed: {rollback}\n"
                            f"Recovery files were preserved at {transaction['backup']}."
                        ) from failure
                elif self.created_service:
                    self.remove_created_service(failure)
                raise
            if transaction:
                self.journal.unlink()
            print(f"Installed Xinference {candidate['version']}.", flush=True)
        finally:
            if not self.journal.exists():
                shutil.rmtree(temporary, ignore_errors=True)

    def execute(self):
        with installation_lock(self.root):
            self.update()
        if self.mode == "none":
            server = self.environment / (
                "Scripts/xinference-local.exe"
                if os.name == "nt"
                else "bin/xinference-local"
            )
            host = self.env.get("XINFERENCE_HOST", "127.0.0.1")
            port = self.env.get("XINFERENCE_PORT", "9997")
            if self.start:
                print(
                    f"Starting Xinference on {host}:{port} (Ctrl+C to stop)...",
                    flush=True,
                )
                args = [
                    str(self.python),
                    "-I",
                    "-u",
                    "-c",
                    "from xinference.deploy.cmdline import local; local()",
                    "--host",
                    host,
                    "--port",
                    port,
                ]
                if platform.system() == "Windows":
                    raise SystemExit(subprocess.run(args, env=self.env).returncode)
                os.execve(str(self.python), args, self.env)
            print(f"Start the server: {server} --host {host} --port {port}", flush=True)


if __name__ == "__main__":
    try:
        Installer().execute()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
