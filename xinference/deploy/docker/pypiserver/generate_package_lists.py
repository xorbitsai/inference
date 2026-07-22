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
Generate the package lists used to build the xinference-pypiserver image.

This script loads the *real* engine-dependency definitions
(``ENGINE_VIRTUALENV_PACKAGES`` etc.) and the *real* marker-filtering logic
(``filter_virtualenv_packages_by_markers``) straight from the xinference
source tree — without installing xinference — so the generated lists can
never drift from what the runtime actually installs. Only ``pydantic``,
``packaging`` and ``orjson`` are needed.

Usage:
    python generate_package_lists.py --platform amd64 --out /tmp/out
    python generate_package_lists.py --platform arm64 --cuda-version 13.0 \
        --src-root /path/to/repo --out /tmp/out

Outputs (under --out):
    engines/<engine>.in         one resolvable requirement set per engine
    engines/<engine>.meta.json  extra index URLs / index strategy per engine
    pins.json                   concrete per-model specs (versions kept as-is)
    urls.txt                    direct wheel URLs (target arch only)
    git.txt                     git+ sources
    manifest.json               generation parameters and counts
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform as _platform
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

PLATFORM_MACHINE = {"amd64": "x86_64", "arm64": "aarch64"}

SYSTEM_PLACEHOLDER_RE = re.compile(r"^#system_([a-z0-9_]+)#$")
ENGINE_MARKER_RE = re.compile(r"#(?:model_)?engine#\s*==\s*['\"]([^'\"]+)['\"]")


def load_xinference_modules(src_root: Path) -> Tuple[Any, Any]:
    """
    Load ``xinference.core.utils`` and ``xinference.core.virtual_env_manager``
    from source files without installing the package.
    """

    def _load(name: str, path: Path, is_pkg: bool = False) -> types.ModuleType:
        if is_pkg:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]  # type: ignore[attr-defined]
            sys.modules[name] = module
            return module
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "xinference" or name.startswith("xinference.")
    }
    try:
        pkg_root = src_root / "xinference"
        _load("xinference", pkg_root, is_pkg=True)
        _load("xinference.constants", pkg_root / "constants.py")
        # Stub xinference._compat instead of loading the real module: it drags
        # in openai at import time, while core.utils only needs BaseModel.
        compat = types.ModuleType("xinference._compat")
        try:
            from pydantic.v1 import BaseModel  # pydantic v2, matches _compat
        except ImportError:
            from pydantic import BaseModel  # type: ignore[assignment]
        compat.BaseModel = BaseModel  # type: ignore[attr-defined]
        sys.modules["xinference._compat"] = compat
        _load("xinference.core", pkg_root / "core", is_pkg=True)
        core_utils = _load("xinference.core.utils", pkg_root / "core" / "utils.py")
        venv_manager = _load(
            "xinference.core.virtual_env_manager",
            pkg_root / "core" / "virtual_env_manager.py",
        )
        return core_utils, venv_manager
    finally:
        for name in list(sys.modules):
            if name == "xinference" or name.startswith("xinference."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def iter_virtualenv_packages(
    model_dir: Path,
) -> Iterator[Tuple[str, str, List[str]]]:
    """Yield (json_relpath, model_name, packages) for every virtualenv block."""

    def _walk(obj: Any, model_name: str) -> Iterator[Tuple[str, List[str]]]:
        if isinstance(obj, dict):
            model_name = obj.get("model_name", model_name)
            virtualenv = obj.get("virtualenv")
            if isinstance(virtualenv, dict) and isinstance(
                virtualenv.get("packages"), list
            ):
                yield model_name, list(virtualenv["packages"])
            for value in obj.values():
                yield from _walk(value, model_name)
        elif isinstance(obj, list):
            for item in obj:
                yield from _walk(item, model_name)

    for json_path in sorted(model_dir.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        # Use forward slashes so the emitted source labels are identical across
        # platforms (Path str-ifies with backslashes on Windows), keeping the
        # generated package lists and their tests portable.
        rel = json_path.relative_to(model_dir).as_posix()
        for model_name, packages in _walk(data, "<unknown>"):
            yield rel, model_name, packages


def classify_spec(spec: str) -> str:
    """Classify a marker-free requirement spec: 'url', 'git' or 'pin'."""
    candidate = spec.strip()
    if candidate.startswith(("http://", "https://")):
        return "url"
    if candidate.startswith("git+"):
        return "git"
    if "@" in candidate:
        target = candidate.partition("@")[2].strip()
        if target.startswith("git+"):
            return "git"
        if target.startswith(("http://", "https://")):
            return "url"
    return "pin"


def is_dependency_macro(spec: str) -> bool:
    """True for '#xxx_dependencies#' / 'xxx_dependencies' engine macros."""
    name = spec.split(";", 1)[0].strip().lower()
    if name.startswith("#") and name.endswith("#"):
        name = name[1:-1]
    return name.endswith("_dependencies")


def system_placeholder_name(spec: str) -> Optional[str]:
    """Return the bare package name for a '#system_xxx#' placeholder."""
    name = spec.split(";", 1)[0].strip().lower()
    match = SYSTEM_PLACEHOLDER_RE.match(name)
    return match.group(1) if match else None


def engine_file_name(engine: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", engine.lower())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=("amd64", "arm64"))
    parser.add_argument("--cuda-version", default="13.0")
    parser.add_argument(
        "--exclude-engines",
        default="mlx",
        help="comma-separated engine buckets to skip (default: mlx, which is "
        "not usable in the Linux runtime image)",
    )
    parser.add_argument(
        "--exclude-packages",
        default="mlx",
        help="comma-separated case-insensitive substrings; per-model pins "
        "whose package name matches are skipped (default: mlx — MLX-only "
        "models cannot run in the Linux runtime image and their packages "
        "do not resolve for a Linux platform)",
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=None,
        help="repository root containing the xinference package "
        "(default: derived from this script's in-repo location)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # Resolve lazily: inside the image build this script does not live in
    # the repository tree and --src-root is always passed.
    src_root = args.src_root or Path(__file__).resolve().parents[4]

    machine = PLATFORM_MACHINE[args.platform]
    # filter_virtualenv_packages_by_markers consults platform.machine();
    # evaluate markers for the *target* architecture, not the build host.
    _platform.machine = lambda: machine  # type: ignore[assignment]

    core_utils, venv_manager = load_xinference_modules(src_root)
    filter_packages = core_utils.filter_virtualenv_packages_by_markers
    engine_packages: Dict[str, List[str]] = venv_manager.ENGINE_VIRTUALENV_PACKAGES
    engine_format_packages: Dict[str, Dict[str, List[str]]] = getattr(
        venv_manager, "ENGINE_VIRTUALENV_MODEL_FORMAT_PACKAGES", {}
    )
    engine_extra_indexes: Dict[str, List[str]] = (
        venv_manager.ENGINE_VIRTUALENV_EXTRA_INDEX_URLS
    )
    engine_index_strategy: Dict[str, str] = (
        venv_manager.ENGINE_VIRTUALENV_INDEX_STRATEGY
    )

    excluded_engines = {
        e.strip().lower() for e in args.exclude_engines.split(",") if e.strip()
    }
    excluded_packages = [
        p.strip().lower() for p in args.exclude_packages.split(",") if p.strip()
    ]

    out = args.out
    (out / "engines").mkdir(parents=True, exist_ok=True)

    urls: Set[str] = set()
    git_sources: Set[str] = set()
    pins: Dict[str, Set[str]] = {}  # spec -> sources
    excluded_pins: Set[str] = set()

    def _add(spec: str, source: str) -> None:
        kind = classify_spec(spec)
        if kind == "url":
            urls.add(spec)
        elif kind == "git":
            git_sources.add(spec)
        else:
            name = spec.split(";", 1)[0].strip().lower()
            if any(pattern in name for pattern in excluded_packages):
                excluded_pins.add(spec)
                return
            pins.setdefault(spec, set()).add(source)

    # ------------------------------------------------------------------
    # Engine buckets: straight from ENGINE_VIRTUALENV_PACKAGES.
    # ------------------------------------------------------------------
    engines_meta: Dict[str, Dict[str, Any]] = {}
    for engine, packages in sorted(engine_packages.items()):
        if engine.lower() in excluded_engines:
            continue
        combined_packages = list(packages)
        for format_packages in engine_format_packages.get(engine, {}).values():
            for package in format_packages:
                if package not in combined_packages:
                    combined_packages.append(package)
        filtered = filter_packages(
            combined_packages, engine, args.cuda_version, "linux"
        )
        specs: List[str] = []
        for spec in filtered:
            if classify_spec(spec) == "pin":
                if spec not in specs:
                    specs.append(spec)
            else:
                _add(spec, "engine:" + engine)
        fname = engine_file_name(engine)
        (out / "engines" / f"{fname}.in").write_text("".join(f"{s}\n" for s in specs))
        meta = {
            "engine": engine,
            "extra_index_urls": engine_extra_indexes.get(engine, []),
            "index_strategy": engine_index_strategy.get(engine),
        }
        (out / "engines" / f"{fname}.meta.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )
        engines_meta[fname] = meta

    # ------------------------------------------------------------------
    # Per-model concrete specs from the model JSONs.
    # ------------------------------------------------------------------
    model_dir = src_root / "xinference" / "model"
    for rel, model_name, packages in iter_virtualenv_packages(model_dir):
        concrete = [p for p in packages if not is_dependency_macro(p)]
        candidate_engines: Set[Optional[str]] = {None}
        for pkg in concrete:
            for engine in ENGINE_MARKER_RE.findall(pkg):
                if engine.lower() not in excluded_engines:
                    candidate_engines.add(engine)
        for cand in candidate_engines:
            for spec in filter_packages(
                list(concrete), cand, args.cuda_version, "linux"
            ):
                sysname = system_placeholder_name(spec)
                if sysname is not None:
                    spec = sysname
                source = rel + ":" + model_name + (f" ({cand})" if cand else "")
                _add(spec, source)

    (out / "urls.txt").write_text("".join(f"{u}\n" for u in sorted(urls)))
    (out / "git.txt").write_text("".join(f"{g}\n" for g in sorted(git_sources)))
    (out / "pins.json").write_text(
        json.dumps(
            [
                {"spec": spec, "sources": sorted(sources)}
                for spec, sources in sorted(pins.items())
            ],
            indent=2,
        )
        + "\n"
    )
    manifest = {
        "platform": args.platform,
        "machine": machine,
        "cuda_version": args.cuda_version,
        "excluded_engines": sorted(excluded_engines),
        "excluded_pins": sorted(excluded_pins),
        "engines": engines_meta,
        "counts": {
            "engines": len(engines_meta),
            "pins": len(pins),
            "urls": len(urls),
            "git": len(git_sources),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"]))


if __name__ == "__main__":
    main()
