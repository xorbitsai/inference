# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Prepare a package upgrade before the existing component handoff."""

import asyncio
import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlsplit, urlunsplit

from packaging.version import InvalidVersion, Version

from .process_host import start_process

UPGRADE_INDEX_ENV = "XINFERENCE_UPGRADE_INDEX_URL"
# Local service hosts also reap adopted children. These commands own their
# Popen exit status until their explicit watcher has consumed it.
active_command_pids: set[int] = set()


def validate_upgrade(
    version: Optional[str],
    executable: Optional[str] = None,
    current_version: Optional[str] = None,
) -> None:
    if version is None:
        return
    if executable is not None:
        raise ValueError("Specify either version or python_executable")
    try:
        if not isinstance(version, str) or len(version) > 128:
            raise InvalidVersion(str(version))
        parsed = Version(version)
        if parsed.is_prerelease or parsed.is_devrelease:
            raise InvalidVersion(version)
    except InvalidVersion as error:
        raise ValueError("version must be a stable package version") from error
    if current_version is not None and parsed <= Version(current_version):
        raise ValueError("The upgrade version must be newer than the running version")


def upgrade_index(environment: Dict[str, str]) -> str:
    return (
        environment.get(UPGRADE_INDEX_ENV)
        or environment.get("PIP_INDEX_URL")
        or "https://pypi.org/simple"
    )


def _environment(environment: Dict[str, str]) -> Dict[str, str]:
    # Use the same single source for discovery and installation. Do not inherit
    # project configuration, extra indexes or source-checkout import paths.
    excluded = {
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "__PYVENV_LAUNCHER__",
    }
    allowed_uv = {"UV_CACHE_DIR", "UV_HTTP_TIMEOUT", "UV_NATIVE_TLS", "UV_SYSTEM_CERTS"}
    return {
        **{
            key: value
            for key, value in environment.items()
            if key not in excluded and (not key.startswith("UV_") or key in allowed_uv)
        },
        # Keep index credentials out of the subprocess command line.
        "UV_INDEX_URL": upgrade_index(environment),
    }


def redact_output(text: str) -> str:
    """Keep rewritten project URLs useful without persisting their credentials."""

    def replace(match):
        url = urlsplit(match[0])
        return urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))

    return re.sub(r"https?://[^\s<>\"']+", replace, text)


async def _command(
    command: list[str],
    environment: Dict[str, str],
    directory: Path,
    report: Callable[..., None],
    timeout: float = 1800,
    capture_all: bool = False,
) -> str:
    # File output works on Windows' selector API loop too, without subprocess
    # transports or a process-lifetime wait on the serving executor.
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        process = await start_process(
            command,
            env=environment,
            cwd=directory,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=errors if capture_all else subprocess.STDOUT,
        )
        active_command_pids.add(process.pid)
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if time.monotonic() > deadline:
                    raise RuntimeError("Package upgrade command timed out")
                await asyncio.sleep(0.2)
            output.seek(0, os.SEEK_END)
            size = output.tell()
            output.seek(0 if capture_all else max(0, size - 8192))
            raw = output.read().decode(errors="replace")
            errors.seek(0, os.SEEK_END)
            errors.seek(max(0, errors.tell() - 8192))
            diagnostics = errors.read().decode(errors="replace")
            text = redact_output(raw + diagnostics)[-8192:]
            if text.strip():
                report(log=text)
            if process.returncode:
                raise RuntimeError("Package upgrade command failed: " + text[-2000:])
            return raw if capture_all else text
        finally:
            if process.poll() is None:
                import psutil

                with contextlib.suppress(psutil.Error):
                    children = psutil.Process(process.pid).children(recursive=True)
                    for child in reversed(children):
                        with contextlib.suppress(psutil.Error):
                            child.kill()
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            try:
                await asyncio.to_thread(process.wait)
            finally:
                active_command_pids.discard(process.pid)


def _uv() -> str:
    from uv import find_uv_bin

    return find_uv_bin()


async def check_update(
    executable: str,
    current_version: str,
    environment: Dict[str, str],
) -> Dict[str, Any]:
    env = _environment(environment)
    with tempfile.TemporaryDirectory(prefix="xinference-update-") as directory:
        root = Path(directory)
        requirements = root / "requirements.in"
        requirements.write_text("xinference\n")
        result = await _command(
            [
                _uv(),
                "--no-config",
                "pip",
                "compile",
                str(requirements),
                "--no-deps",
                "--no-header",
                "--no-annotate",
                "--prerelease",
                "disallow",
                "--python",
                executable,
                "--no-python-downloads",
                "--refresh",
            ],
            env,
            root,
            lambda **fields: None,
            timeout=60,
        )
    match = re.search(r"^xinference==([^\s;]+)", result, re.MULTILINE)
    if match is None:
        raise RuntimeError("Package source returned no stable Xinference version")
    latest = match[1]
    return {
        "current_version": current_version,
        "latest_version": latest,
        "update_available": Version(latest) > Version(current_version),
    }


MIGRATE = """
import asyncio, inspect, json, sys
from importlib.metadata import distribution
context = json.loads(sys.argv[1])
for entry in distribution('xinference').entry_points:
    if entry.group == 'xinference.migrations':
        result = entry.load()(context)
        if inspect.isawaitable(result):
            asyncio.run(result)
"""

VERIFY = """
import json, sys, xinference
from importlib.metadata import version
from pathlib import Path
from packaging.version import Version
assert Path(xinference.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), 'Xinference was imported outside the target environment'
assert Version(version('xinference')) == Version(sys.argv[1]), 'Installed package version differs from requested version'
assert Version(xinference.__version__) == Version(sys.argv[1]), 'Loaded code version differs from requested version'
"""

WORKER_BASELINE = """
import json
from importlib.metadata import distributions
from packaging.utils import canonicalize_name
packages = {}
for dist in distributions():
    name = canonicalize_name(dist.metadata['Name'])
    if name != 'xinference':
        packages.setdefault(name, dist.version)
print(json.dumps(packages))
"""

VERIFY_WORKER = """
import json, sys
from importlib.metadata import version
from packaging.version import Version
from pathlib import Path
for name, expected in json.loads(Path(sys.argv[1]).read_text()).items():
    if Version(version(name)) != Version(expected):
        raise RuntimeError('Worker dependency changed during upgrade: ' + name)
"""


async def prepare_upgrade(
    version: str,
    component: str,
    executable: str,
    current_version: str,
    environment: Dict[str, str],
    report: Callable[..., None],
) -> str:
    validate_upgrade(version, current_version=current_version)
    from ..constants import XINFERENCE_HOME

    home = Path(XINFERENCE_HOME).resolve()
    root = home / "upgrades" / f"{component}-{uuid.uuid4().hex}"
    root.parent.mkdir(parents=True, exist_ok=True)
    env = _environment(environment)
    uv = _uv()
    try:
        report(status="installing")
        baseline = None
        requirements = None
        constraints = None
        with tempfile.TemporaryDirectory(
            prefix=".upgrade-profile-", dir=root.parent
        ) as temporary:
            profile = Path(temporary)
            if component == "worker":
                # Freeze owned packages (including editable engines/direct URLs)
                # and constrain every visible distribution, including the torch
                # build inherited from the base Python. Xinference alone moves.
                frozen = await _command(
                    [
                        uv,
                        "--no-config",
                        "pip",
                        "freeze",
                        "--python",
                        executable,
                        "--exclude",
                        "xinference",
                    ],
                    env,
                    profile,
                    lambda **fields: None,
                    timeout=60,
                    capture_all=True,
                )
                raw = await _command(
                    [executable, "-I", "-c", WORKER_BASELINE],
                    env,
                    profile,
                    lambda **fields: None,
                    timeout=60,
                    capture_all=True,
                )
                baseline = json.loads(raw)
                requirements = profile / "requirements.txt"
                requirements.write_text(frozen)
                constraints = profile / "constraints.txt"
                constraints.write_text(
                    "".join(
                        f"{name}=={version}\n" for name, version in baseline.items()
                    )
                )
            await _command(
                [
                    uv,
                    "--no-config",
                    "venv",
                    "--python",
                    executable,
                    "--no-python-downloads",
                    *(["--system-site-packages"] if component == "worker" else []),
                    str(root),
                ],
                env,
                root.parent,
                report,
            )
            python = str(
                root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            )
            await _command(
                [
                    uv,
                    "--no-config",
                    "pip",
                    "install",
                    "--python",
                    python,
                    "--prerelease",
                    "disallow",
                    "--refresh-package",
                    "xinference",
                    f"xinference=={Version(version)}",
                    *(
                        ["-r", str(requirements), "-c", str(constraints)]
                        if requirements is not None
                        else []
                    ),
                ],
                env,
                root.parent,
                report,
            )
            await _command(
                [python, "-I", "-c", VERIFY, str(Version(version))],
                env,
                root.parent,
                report,
                timeout=60,
            )
            if baseline is not None:
                baseline_file = profile / "baseline.json"
                baseline_file.write_text(json.dumps(baseline))
                await _command(
                    [python, "-I", "-c", VERIFY_WORKER, str(baseline_file)],
                    env,
                    root.parent,
                    report,
                    timeout=60,
                )
        report(status="migrating")
        context = json.dumps(
            {
                "component": component,
                "source_version": current_version,
                "target_version": str(Version(version)),
                "home": str(home),
            }
        )
        await _command([python, "-I", "-c", MIGRATE, context], env, root.parent, report)
        report(status="starting", python_executable=python)
        return python
    except BaseException:
        # This environment has never served requests. Retain successful ones,
        # including after a failed handoff, for diagnostics and manual reuse.
        await asyncio.to_thread(shutil.rmtree, root, True)
        raise
