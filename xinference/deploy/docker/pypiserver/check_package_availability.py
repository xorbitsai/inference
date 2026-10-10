# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Check mirror requirements using index metadata, without downloading packages.

This catches unpublished versions and missing Python/architecture candidates.
It is not a dependency resolver or a substitute for the image's offline selfcheck:
sdist buildability, transitive conflicts, and direct references remain build checks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from html.parser import HTMLParser
from http.client import HTTPException
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import Tag, compatible_tags, cpython_tags
from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)

PYPI_INDEX = "https://pypi.org/simple"
DEFAULT_PYTHON_VERSION = "3.12.0"
MAX_GLIBC_MINOR = 34  # Matches download_packages.py and selfcheck.py's uv target.


class SimpleIndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.files: List[Dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Any]]) -> None:
        attributes = dict(attrs)
        if tag == "a" and attributes.get("href"):
            self.files.append(
                {
                    "filename": unquote(
                        urlsplit(attributes["href"]).path.rsplit("/", 1)[-1]
                    ),
                    "requires-python": attributes.get("data-requires-python"),
                    "yanked": "data-yanked" in attributes,
                }
            )


def fetch_files(index: str, name: str) -> List[Dict[str, Any]]:
    """Fetch only the Simple API page. Artifact URLs are never followed."""
    url = f"{index.rstrip('/')}/{canonicalize_name(name)}/"
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.pypi.simple.v1+json, text/html;q=0.9",
            "User-Agent": "xinference-mirror-preflight",
        },
    )
    error: Exception = RuntimeError("Index request did not complete")
    for attempt in range(3):
        try:
            with urlopen(request, timeout=20) as response:
                body = response.read().decode("utf-8")
                if "json" in response.headers.get("Content-Type", ""):
                    payload = json.loads(body)
                    if not isinstance(payload, dict) or not isinstance(
                        payload.get("files"), list
                    ):
                        raise RuntimeError(
                            f"Invalid index response from {url}: missing files list"
                        )
                    return payload["files"]
                parser = SimpleIndexParser()
                parser.feed(body)
                return parser.files
        except HTTPError as exc:
            # PyTorch's public S3 index returns 403 for absent project pages.
            if exc.code == 404 or (
                exc.code == 403
                and index.startswith("https://download.pytorch.org/whl/")
            ):
                return []
            if exc.code < 500 and exc.code != 429:
                raise RuntimeError(f"Index request failed: {url}: {exc}") from exc
            error = exc
        except (HTTPException, URLError, TimeoutError, OSError) as exc:
            error = exc
        if attempt < 2:
            time.sleep(attempt + 1)
    raise RuntimeError(f"Index request failed: {url}: {error}")


@lru_cache(maxsize=8)
def supported_tags(machine: str, python_version: str) -> Set[Tag]:
    # uv's manylinux_2_34 target accepts legacy aliases and linux_* as well,
    # but not wheels requiring a newer glibc. Match its target, not the host.
    minimum = 5 if machine == "x86_64" else 17
    platforms = [
        f"manylinux_2_{minor}_{machine}"
        for minor in range(MAX_GLIBC_MINOR, minimum - 1, -1)
    ]
    platforms.extend([f"manylinux2014_{machine}", f"linux_{machine}"])
    if machine == "x86_64":
        platforms.extend([f"manylinux2010_{machine}", f"manylinux1_{machine}"])
    major, minor = (int(v) for v in python_version.split(".")[:2])
    target = (major, minor)
    return set(
        cpython_tags(target, abis=[f"cp{major}{minor}"], platforms=platforms)
    ) | set(
        compatible_tags(target, interpreter=f"cp{major}{minor}", platforms=platforms)
    )


def matching_file(
    requirement: Requirement,
    file: Dict[str, Any],
    machine: str,
    python_version: str,
    prereleases: Optional[bool] = None,
) -> bool:
    exact_pin = any(
        spec.operator in {"==", "==="} and "*" not in spec.version
        for spec in requirement.specifier
    )
    if file.get("yanked", False) is not False and not exact_pin:
        return False
    requires_python = file.get("requires-python")
    if requires_python:
        try:
            if not SpecifierSet(requires_python).contains(python_version):
                return False
        except InvalidSpecifier:
            # Like pip, ignore malformed upstream Requires-Python metadata.
            pass
    filename = file["filename"]
    try:
        if filename.endswith(".whl"):
            name, version, _, tags = parse_wheel_filename(filename)
            if not tags.intersection(supported_tags(machine, python_version)):
                return False
        else:
            # The image builder can build sdists; metadata alone cannot prove
            # that their native build dependencies work on the target machine.
            name, version = parse_sdist_filename(filename)
    except (InvalidWheelFilename, InvalidSdistFilename):
        return False
    return name == canonicalize_name(
        requirement.name
    ) and requirement.specifier.contains(
        version,
        prereleases=(
            bool(requirement.specifier.prereleases)
            if prereleases is None
            else prereleases
        ),
    )


def collect_requirements(
    manifest_dir: Path, runtime_constraints: Path, python_version: str
) -> List[Dict[str, Any]]:
    manifest = json.loads((manifest_dir / "manifest.json").read_text())
    pytorch_index = "https://download.pytorch.org/whl/cu" + manifest[
        "cuda_version"
    ].replace(".", "")
    environment = default_environment()
    environment.update(
        sys_platform="linux",
        platform_system="Linux",
        platform_machine=manifest["machine"],
        implementation_name="cpython",
        platform_python_implementation="CPython",
        python_version=".".join(python_version.split(".")[:2]),
        python_full_version=python_version,
        extra="",
    )
    checks = []

    def add(
        spec: str, sources: List[str], extra_indexes: List[str], deferred: str = ""
    ) -> None:
        requirement = Requirement(spec)
        if requirement.url:
            return
        if requirement.marker and not requirement.marker.evaluate(environment):
            return
        checks.append(
            {
                "requirement": requirement,
                "sources": sources,
                "machine": manifest["machine"],
                "deferred": deferred,
                "indexes": tuple(
                    dict.fromkeys([PYPI_INDEX, *extra_indexes, pytorch_index])
                ),
            }
        )

    for spec in runtime_constraints.read_text().splitlines():
        if spec.strip() and not spec.lstrip().startswith("#"):
            add(spec, [str(runtime_constraints)], [])
    for engine, meta in manifest["engines"].items():
        for spec in (
            (manifest_dir / "engines" / f"{engine}.in").read_text().splitlines()
        ):
            if spec.strip() and not spec.lstrip().startswith("#"):
                add(
                    spec,
                    [f"engine:{meta['engine']}"],
                    meta.get("extra_index_urls") or [],
                )
    for pin in json.loads((manifest_dir / "pins.json").read_text()):
        add(pin["spec"], pin["sources"], [])
    for pin in manifest.get("deferred_model_pins", []):
        add(pin["spec"], [pin["source"]], [], deferred=pin["reason"])
    return checks


def check_requirements(checks: List[Dict[str, Any]], python_version: str) -> List[str]:
    # Share index responses across engines, version ranges and both architectures.
    keys = sorted(
        {
            (index, canonicalize_name(check["requirement"].name))
            for check in checks
            for index in check["indexes"]
        }
    )
    print(f"Fetching {len(keys)} package index pages (no artifacts)", flush=True)

    def fetch(key: Tuple[str, str]) -> Any:
        try:
            return fetch_files(*key)
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=8) as pool:
        files = dict(zip(keys, pool.map(fetch, keys)))
    failures = set()
    for check in checks:
        requirement = check["requirement"]
        name = canonicalize_name(requirement.name)
        candidates = [files[index, name] for index in check["indexes"]]
        errors = [
            str(candidate)
            for candidate in candidates
            if isinstance(candidate, Exception)
        ]
        label = f"{check['machine']}: {requirement} ({', '.join(check['sources'])})"
        if errors:
            failures.add(f"{label}: {'; '.join(errors)}")
            continue
        # Prefer finals, then permit a prerelease-only solution as current
        # pip/uv do. This is still only a direct candidate availability check.
        available = any(
            matching_file(
                requirement, file, check["machine"], python_version, prereleases=allow
            )
            for allow in (False, True)
            for candidate in candidates
            for file in candidate
        )
        if check.get("deferred"):
            if available:
                print(
                    f"::warning::{label}: deferred dependency is now available; remove its mirror exception"
                )
        elif not available:
            failures.add(f"{label}: no published candidate for Python {python_version}")
    return sorted(failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, action="append", required=True)
    parser.add_argument("--runtime-constraints", type=Path, required=True)
    parser.add_argument("--python-version", default=DEFAULT_PYTHON_VERSION)
    args = parser.parse_args()
    checks = [
        check
        for directory in args.manifest_dir
        for check in collect_requirements(
            directory, args.runtime_constraints, args.python_version
        )
    ]
    failures = check_requirements(checks, args.python_version)
    for failure in failures:
        print(f"ERROR: {failure}", file=sys.stderr)
    print(
        f"Checked {len(checks)} requirements using index metadata only; {len(failures)} failures"
    )
    sys.exit(bool(failures))


if __name__ == "__main__":
    main()
