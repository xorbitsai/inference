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
Download all wheels for the xinference-pypiserver image.

Consumes the output of ``generate_package_lists.py`` and fills a wheel
directory in four passes:

1. Lock the torch family to the runtime image's version, then lock each
   engine set with ``uv pip compile`` and fetch the fully-pinned locks with
   ``pip download --no-deps`` — one coherent resolution per engine, so
   unpinned specs cannot fan out into many versions of the same package.
2. Fetch per-model concrete pins with their transitive dependencies,
   constrained to the versions already locked (falling back to an
   unconstrained fetch when a pin genuinely conflicts — availability wins
   over minimization; every fallback is recorded in the report).
3. Fetch direct wheel URLs and build wheels for git sources.
4. Build wheels for any sdist-only downloads so the runtime never compiles.

Writes ``report.json`` with size/version statistics next to the manifest.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import (
    InvalidWheelFilename,
    canonicalize_name,
    parse_wheel_filename,
)

TORCH_FAMILY = ("torch", "torchvision", "torchaudio", "torchcodec")

# Engines whose releases hard-pin their own torch stack.
SELF_PINNING_ENGINES = ("vllm", "sglang")


def run(cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kwargs)


def check(cmd: List[str]) -> None:
    proc = run(cmd)
    if proc.returncode != 0:
        sys.exit(f"FATAL: command failed with {proc.returncode}: {' '.join(cmd)}")


def spec_name(spec: str) -> Optional[str]:
    try:
        return canonicalize_name(Requirement(spec.split(";", 1)[0].strip()).name)
    except InvalidRequirement:
        return None


def wheel_name_version(filename: str) -> Optional[Tuple[str, str]]:
    """Return a normalized ``(name, version)`` pair for a wheel filename."""
    if not filename.endswith(".whl"):
        return None
    try:
        name, version, _, _ = parse_wheel_filename(filename)
    except InvalidWheelFilename:
        return None
    return canonicalize_name(name), str(version)


def lock_names(lock_file: Path) -> Dict[str, str]:
    """Map canonical package name -> lock line for a compiled lock file."""
    names: Dict[str, str] = {}
    for line in lock_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        name = spec_name(line)
        if name and name not in names:
            names[name] = line
    return names


def uv_compile(
    in_file: Path,
    out_file: Path,
    *,
    python_version: str,
    python_platform: str,
    index_url: str,
    extra_index_urls: List[str],
    index_strategy: Optional[str],
    constraints: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    cmd = [
        "uv",
        "pip",
        "compile",
        str(in_file),
        "-o",
        str(out_file),
        "--no-header",
        "--no-annotate",
        "--python-version",
        python_version,
        "--python-platform",
        python_platform,
        "--index-url",
        index_url,
    ]
    for url in extra_index_urls:
        cmd += ["--extra-index-url", url]
    if index_strategy:
        cmd += ["--index-strategy", index_strategy]
    if constraints is not None:
        cmd += ["-c", str(constraints)]
    return run(cmd)


def pip_download(
    args: List[str],
    dest: Path,
    *,
    index_url: str,
    extra_index_urls: List[str],
    no_deps: bool = False,
    constraints: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--quiet",
        "--dest",
        str(dest),
        "--prefer-binary",
        "--index-url",
        index_url,
    ]
    for url in extra_index_urls:
        cmd += ["--extra-index-url", url]
    if no_deps:
        cmd.append("--no-deps")
    if constraints is not None:
        cmd += ["-c", str(constraints)]
    return run(cmd + args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--platform", required=True, choices=("amd64", "arm64"))
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument(
        "--torch-version",
        default="2.11.*",
        help="torch version of the xinference runtime image "
        "(see xinference/deploy/docker/Dockerfile)",
    )
    parser.add_argument("--index-url", default="https://pypi.org/simple")
    parser.add_argument(
        "--pytorch-index", default="https://download.pytorch.org/whl/cu130"
    )
    args = parser.parse_args()

    manifest_dir: Path = args.manifest_dir
    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    work = manifest_dir / "locks"
    work.mkdir(exist_ok=True)

    machine = {"amd64": "x86_64", "arm64": "aarch64"}[args.platform]
    # The runtime image's glibc supports manylinux_2_34 wheels; the default
    # (manylinux_2_17) would hide e.g. recent sglang releases.
    python_platform = f"{machine}-manylinux_2_34"
    unconstrained_fallbacks: List[str] = []
    sdist_left: List[str] = []

    # ------------------------------------------------------------------
    # 1a. Lock the torch family to the runtime image's version.
    # ------------------------------------------------------------------
    big_in = work / "torch-family.in"
    big_in.write_text(
        f"torch=={args.torch_version}\n"
        + "".join(f"{p}\n" for p in TORCH_FAMILY if p != "torch")
    )
    torch_family_lock = work / "torch-family.lock"
    proc = uv_compile(
        big_in,
        torch_family_lock,
        python_version=args.python_version,
        python_platform=python_platform,
        index_url=args.index_url,
        extra_index_urls=[args.pytorch_index],
        index_strategy="unsafe-best-match",
    )
    if proc.returncode != 0:
        sys.exit("FATAL: failed to lock the torch family")
    # Constrain other resolutions with ONLY the torch-family pins, not the
    # lock's transitive deps (which would over-constrain e.g. setuptools).
    constraints_big = work / "constraints-big.txt"
    constraints_big.write_text(
        "".join(
            f"{line}\n"
            for name, line in lock_names(torch_family_lock).items()
            if name in TORCH_FAMILY
        )
    )

    # ------------------------------------------------------------------
    # 1b. Lock each engine set, then fetch the pinned locks.
    # ------------------------------------------------------------------
    engine_locks: Dict[str, Path] = {"torch-family": torch_family_lock}
    for in_file in sorted((manifest_dir / "engines").glob("*.in")):
        engine = in_file.stem
        meta = json.loads(in_file.with_suffix(".meta.json").read_text())
        lock = work / f"{engine}.lock"
        # vllm/sglang releases hard-pin their own torch stack; forcing the
        # runtime image's torch on them is unsolvable. Their venvs install
        # a self-consistent stack anyway.
        self_pinning = engine in SELF_PINNING_ENGINES
        proc = uv_compile(
            in_file,
            lock,
            python_version=args.python_version,
            python_platform=python_platform,
            index_url=args.index_url,
            # The pytorch index makes the +cu130 torch pins resolvable for
            # engines whose runtime config has no extra indexes; at runtime
            # everything is served by the single mirror index anyway.
            extra_index_urls=list(
                dict.fromkeys(
                    (meta.get("extra_index_urls") or []) + [args.pytorch_index]
                )
            ),
            index_strategy="unsafe-best-match",
            constraints=None if self_pinning else constraints_big,
        )
        if proc.returncode != 0:
            sys.exit(
                f"FATAL: engine set '{engine}' does not resolve with "
                f"torch=={args.torch_version}; align --torch-version with the "
                "runtime image or pin the engine set"
            )
        engine_locks[engine] = lock

    # Shared substrate to keep unpinned transitive deps from fanning out.
    master_constraints = work / "constraints-master.txt"
    merged: Dict[str, str] = lock_names(constraints_big)
    for engine in ("transformers", "vllm"):
        if engine in engine_locks:
            for name, line in lock_names(engine_locks[engine]).items():
                merged.setdefault(name, line)
    master_constraints.write_text("".join(f"{line}\n" for line in merged.values()))

    for engine, lock in engine_locks.items():
        meta_file = manifest_dir / "engines" / f"{engine}.meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        proc = pip_download(
            ["-r", str(lock)],
            dest,
            index_url=args.index_url,
            extra_index_urls=(meta.get("extra_index_urls") or [])
            + [args.pytorch_index],
            no_deps=True,
        )
        if proc.returncode != 0:
            sys.exit(f"FATAL: failed to fetch locked engine set '{engine}'")

    # ------------------------------------------------------------------
    # 2. Per-model pins, constrained to the shared substrate.
    # ------------------------------------------------------------------
    pins = json.loads((manifest_dir / "pins.json").read_text())
    for entry in pins:
        spec = entry["spec"]
        pin_name = spec_name(spec)
        pruned = work / "constraints-pruned.txt"
        pruned.write_text(
            "".join(f"{line}\n" for n, line in merged.items() if n != pin_name)
        )
        proc = pip_download(
            [spec],
            dest,
            index_url=args.index_url,
            extra_index_urls=[args.pytorch_index],
            constraints=pruned,
        )
        if proc.returncode != 0:
            print(f"WARN: retrying '{spec}' without constraints", flush=True)
            proc = pip_download(
                [spec],
                dest,
                index_url=args.index_url,
                extra_index_urls=[args.pytorch_index],
            )
            if proc.returncode != 0:
                sys.exit(f"FATAL: pin '{spec}' cannot be downloaded")
            unconstrained_fallbacks.append(spec)

    # ------------------------------------------------------------------
    # 3. Direct wheel URLs and git sources.
    # ------------------------------------------------------------------
    for url in (manifest_dir / "urls.txt").read_text().splitlines():
        if not url.strip():
            continue
        # WITH dependencies: these wheels declare their own requirements
        # (e.g. the spaCy models depend on spacy), which must also land in
        # the mirror for the offline install to succeed.
        proc = pip_download(
            [url.strip()],
            dest,
            index_url=args.index_url,
            extra_index_urls=[args.pytorch_index],
            constraints=master_constraints,
        )
        if proc.returncode != 0:
            print(f"WARN: retrying '{url}' without constraints", flush=True)
            proc = pip_download(
                [url.strip()],
                dest,
                index_url=args.index_url,
                extra_index_urls=[args.pytorch_index],
            )
            if proc.returncode != 0:
                sys.exit(f"FATAL: failed to download '{url}'")
            unconstrained_fallbacks.append(url.strip())

    def pip_wheel(src: str, constraints: Optional[Path]) -> subprocess.CompletedProcess:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--quiet",
            "--wheel-dir",
            str(dest),
            "--index-url",
            args.index_url,
        ]
        if constraints is not None:
            cmd += ["-c", str(constraints)]
        return run(cmd + [src])

    for src in (manifest_dir / "git.txt").read_text().splitlines():
        src = src.strip()
        if not src:
            continue
        # Constraining a package to a locked version while building that
        # same package from git is self-contradictory — prune its own name
        # (when stated in 'name @ git+…' form) from the constraints.
        src_name = spec_name(src)
        pruned = work / "constraints-pruned.txt"
        pruned.write_text(
            "".join(f"{line}\n" for n, line in merged.items() if n != src_name)
        )
        proc = pip_wheel(src, pruned)
        if proc.returncode != 0:
            print(f"WARN: retrying '{src}' without constraints", flush=True)
            proc = pip_wheel(src, None)
            if proc.returncode != 0:
                sys.exit(f"FATAL: failed to build wheel for '{src}'")
            unconstrained_fallbacks.append(src)

    # ------------------------------------------------------------------
    # 4. Build wheels for sdist-only downloads so the runtime never
    #    compiles. Failures keep the sdist (the runtime image has a
    #    toolchain) and are recorded in the report.
    # ------------------------------------------------------------------
    for sdist in sorted(dest.iterdir()):
        if sdist.suffix not in (".gz", ".zip", ".bz2") and not sdist.name.endswith(
            ".tar.gz"
        ):
            continue
        proc = run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--quiet",
                "--no-deps",
                "--wheel-dir",
                str(dest),
                str(sdist),
            ]
        )
        if proc.returncode == 0:
            sdist.unlink()
        else:
            print(f"WARN: keeping sdist {sdist.name}", flush=True)
            sdist_left.append(sdist.name)

    # ------------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------------
    versions: Dict[str, Set[str]] = defaultdict(set)
    total_size = 0
    files = sorted(p for p in dest.iterdir() if p.is_file())
    for f in files:
        total_size += f.stat().st_size
        name_version = wheel_name_version(f.name)
        if name_version is not None:
            name, version = name_version
            versions[name].add(version)
    report: Dict[str, object] = {
        "unconstrained_fallbacks": unconstrained_fallbacks,
        "sdist_left": sdist_left,
        "file_count": len(files),
        "total_size_bytes": total_size,
        "total_size_human": format(total_size / 1024**3, ".2f") + " GiB",
        "multi_version_packages": {
            name: sorted(vs) for name, vs in sorted(versions.items()) if len(vs) > 1
        },
    }
    (manifest_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("file_count", "total_size_human")}, indent=None
        )
    )


if __name__ == "__main__":
    main()
