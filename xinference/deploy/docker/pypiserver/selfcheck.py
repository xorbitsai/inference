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
"""
Offline-resolution quality gate for the xinference-pypiserver image.

Resolves every engine set, every per-model pin and every direct-URL wheel
against ONLY the local pypiserver (started with --disable-fallback), which
is exactly what an air-gapped deployment sees. Any unresolvable entry fails
the image build instead of failing the user at model-launch time.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import List, Optional
from urllib.parse import unquote, urlparse


def wait_for_server(index_url: str, timeout: int = 60) -> None:
    base = index_url.rstrip("/").removesuffix("/simple")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base}/health", timeout=3)
            return
        except Exception:
            time.sleep(1)
    sys.exit(f"FATAL: pypiserver at {index_url} did not become healthy")


def wheel_url_to_spec(url: str) -> Optional[str]:
    """https://…/name-1.2.3+local-…​.whl -> 'name==1.2.3+local'."""
    if "@" in url and not url.startswith(("http://", "https://")):
        url = url.partition("@")[2].strip()
    filename = unquote(posixpath.basename(urlparse(url).path))
    if not filename.endswith(".whl"):
        return None
    parts = filename[: -len(".whl")].split("-")
    if len(parts) < 5 or not parts[1][:1].isdigit():
        return None
    return f"{parts[0]}=={parts[1]}"


def compile_against_index(
    requirements: List[str],
    index_url: str,
    python_version: str,
    python_platform: str,
) -> subprocess.CompletedProcess:
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False) as f:
        f.write("".join(f"{r}\n" for r in requirements))
        in_file = f.name
    # uv writes -o atomically via a sibling temp file, so the target must
    # live in a writable directory (not /dev/null).
    out_file = in_file + ".lock"
    return subprocess.run(
        [
            "uv",
            "pip",
            "compile",
            in_file,
            "-o",
            out_file,
            "--no-header",
            "--quiet",
            "--python-version",
            python_version,
            "--python-platform",
            python_platform,
            "--index-url",
            index_url,
        ],
        capture_output=True,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="local index URL")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--platform", required=True, choices=("amd64", "arm64"))
    parser.add_argument("--python-version", default="3.12")
    args = parser.parse_args()

    machine = {"amd64": "x86_64", "arm64": "aarch64"}[args.platform]
    # Must match download_packages.py: the runtime image supports
    # manylinux_2_34 wheels (e.g. recent sglang releases).
    python_platform = f"{machine}-manylinux_2_34"

    wait_for_server(args.index)

    failures: List[str] = []

    def gate(label: str, requirements: List[str]) -> None:
        proc = compile_against_index(
            requirements, args.index, args.python_version, python_platform
        )
        if proc.returncode != 0:
            failures.append(label)
            print(f"FAIL {label}\n{proc.stderr.strip()}", flush=True)
        else:
            print(f"OK   {label}", flush=True)

    for in_file in sorted((args.manifest_dir / "engines").glob("*.in")):
        requirements = [
            line
            for line in in_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if requirements:
            gate(f"engine:{in_file.stem}", requirements)

    for entry in json.loads((args.manifest_dir / "pins.json").read_text()):
        gate(f"pin:{entry['spec']}", [entry["spec"]])

    for url in (args.manifest_dir / "urls.txt").read_text().splitlines():
        if not url.strip():
            continue
        spec = wheel_url_to_spec(url.strip())
        if spec is None:
            failures.append(f"url:{url} (unparsable wheel filename)")
            print(f"FAIL url:{url} (unparsable wheel filename)", flush=True)
            continue
        gate(f"url:{spec}", [spec])

    if failures:
        sys.exit(
            f"FATAL: {len(failures)} entr{'y' if len(failures) == 1 else 'ies'} "
            "cannot be resolved from the mirror alone:\n  " + "\n  ".join(failures)
        )
    print("selfcheck passed: mirror is self-sufficient", flush=True)


if __name__ == "__main__":
    main()
