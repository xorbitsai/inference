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

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..constants import (
    XINFERENCE_VIRTUAL_ENV_DIR,
    XINFERENCE_VIRTUAL_ENV_FIND_LINKS_ALLOWED_ROOTS,
)

logger = logging.getLogger(__name__)


class VirtualEnvConflictError(ValueError):
    """Raised when an active model prevents virtual environment mutation."""


ENGINE_VIRTUALENV_PACKAGES: Dict[str, List[str]] = {
    "sglang": [
        "pybase64",
        "zmq",
        "partial_json_parser",
        "sentencepiece",
        "dill",
        "ninja",
        "numpy<2.3",
        "pandas<3",
        "sglang>=0.5.6",
        'nvidia-cusparselt-cu13==0.8.0 ; cuda_version == "13.0" and sys_platform == "linux"',
        'https://github.com/sgl-project/whl/releases/download/v0.3.21/sgl_kernel-0.3.21+cu130-cp310-abi3-manylinux2014_x86_64.whl ; cuda_version == "13.0" and platform_machine == "x86_64"',
        'https://github.com/sgl-project/whl/releases/download/v0.3.21/sgl_kernel-0.3.21+cu130-cp310-abi3-manylinux2014_aarch64.whl ; cuda_version == "13.0" and platform_machine == "aarch64"',
        'sgl_kernel ; cuda_version < "13.0"',
    ],
    "vllm": [
        "vllm>=0.11.2",
    ],
    "transformers": [
        "transformers>=4.53.3",
        "accelerate>=0.28.0",
    ],
    "sentence_transformers": [
        "sentence_transformers",
        "einops",
        "transformers>=4.53.3",
        "accelerate>=0.28.0",
        "peft>0.17.1",  # dependence on transformers
        # Omit bare torchvision: use #system_torchvision# from model_spec to avoid mixed-source torch/torchvision.
    ],
    "diffusers": [
        "diffusers>=0.32.0",
        "huggingface-hub<1.0",
    ],
    "mlx": [
        "mlx-lm>=0.24.0",
    ],
    "llama.cpp": [
        "xllamacpp>=0.2.6",
    ],
}

# Optional engine packages selected by model format. Unlike
# ENGINE_VIRTUALENV_PACKAGES, these are not installed for every model using an
# engine: a plain-text pytorch Transformers launch remains limited to the two
# core packages above. The pypiserver generator includes the union in its
# wheel inventory so every format remains available offline.
ENGINE_VIRTUALENV_MODEL_FORMAT_PACKAGES: Dict[str, Dict[str, List[str]]] = {
    "transformers": {
        "gptq": [
            "gptqmodel",
            "optimum",
            "datasets>=3.4.0",
        ],
        "awq": [
            "autoawq!=0.2.6 ; sys_platform=='linux'",
            "datasets>=3.4.0",
        ],
        "bnb": [
            "bitsandbytes ; sys_platform=='linux'",
        ],
    }
}

# Critical dependencies of engine packages that may be inherited from the
# parent environment instead of installed into the venv: with skip_installed
# enabled (the default), an engine spec satisfied by the parent copy is not
# installed, so nothing pulls the engine's own dependency requirements into
# the venv. If the parent copy of such a dependency violates the engine's
# declared requirement (e.g. sglang declares transformers==4.57.1 while the
# Docker image ships transformers 5.x, which breaks sglang.srt at import),
# the engine's declared spec is added to the venv install list so the venv
# gets a compatible copy that shadows the parent's. The specs are read from
# the installed engine's metadata at launch time, so they follow whatever
# the installed engine version declares. Structure: engine name ->
# {distribution name -> [critical dependency names]}.
ENGINE_CRITICAL_DEPENDENCIES: Dict[str, Dict[str, List[str]]] = {
    "sglang": {"sglang": ["transformers"]},
}

ENGINE_VIRTUALENV_EXTRA_INDEX_URLS: Dict[str, List[str]] = {
    "vllm": [
        "https://wheels.vllm.ai/0.19.0/cu130",
        "https://download.pytorch.org/whl/cu130",
    ],
    "sglang": [
        "https://download.pytorch.org/whl/cu130",
    ],
}

ENGINE_VIRTUALENV_INDEX_STRATEGY: Dict[str, str] = {
    "vllm": "unsafe-best-match",
    "sglang": "unsafe-best-match",
}

# Mapping from CUDA version suffix to PyTorch wheel URL
# e.g., cu128 -> https://download.pytorch.org/whl/cu128
# The prebuilt offline mirror targets cu130, but online/non-Docker installs
# keep the CUDA wheel mappings they supported before that mirror was added.
PYTORCH_CUDA_WHEEL_URLS: Dict[str, str] = {
    "cu130": "https://download.pytorch.org/whl/cu130",
    "cu129": "https://download.pytorch.org/whl/cu129",
    "cu128": "https://download.pytorch.org/whl/cu128",
}

# Packages that use PyTorch CUDA wheels
PYTORCH_PACKAGES = {"torch", "torchaudio", "torchvision", "torchcodec"}

# xllamacpp (llama.cpp engine) ships GPU wheels on a self-hosted index, one per
# CUDA major line. Only these CUDA lines have prebuilt GPU wheels; any other
# environment (no GPU, or an unsupported CUDA line) falls back to the default
# PyPI index, which serves the CPU build.
# See https://github.com/xorbitsai/xllamacpp for the official install commands.
XLLAMACPP_CUDA_INDEX_URLS: Dict[str, str] = {
    "cu132": "https://xorbitsai.github.io/xllamacpp/whl/cu132",
    "cu128": "https://xorbitsai.github.io/xllamacpp/whl/cu128",
}


def get_xllamacpp_cuda_index_url(
    system_cuda_version: Optional[str],
) -> Optional[str]:
    """
    Pick the xllamacpp GPU wheel index URL matching the detected system CUDA
    version.

    ``system_cuda_version`` is the dotted version reported by the platform
    (e.g. ``"13.2"`` or ``"12.6"``). CUDA is backward compatible within a major
    line, so we map by major version to the highest available minor wheel index
    that does not exceed the system version:

    - CUDA 13.x  -> cu132
    - CUDA 12.8+ -> cu128

    Returns ``None`` when no GPU is detected or the CUDA line has no prebuilt
    wheel, in which case the caller should leave the index untouched so the CPU
    build is installed from PyPI.
    """
    if not system_cuda_version:
        return None
    # Minor version is optional so a major-only report (e.g. "13") still maps.
    match = re.match(r"^(\d+)(?:\.(\d+))?", system_cuda_version.strip())
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2)) if match.group(2) is not None else 0
    if major >= 13:
        return XLLAMACPP_CUDA_INDEX_URLS["cu132"]
    if major == 12 and minor >= 8:
        return XLLAMACPP_CUDA_INDEX_URLS["cu128"]
    return None


# torch companion packages whose binaries are compiled against a specific torch
# ABI. When any of these is pinned to the system version (via #system_<pkg>#) the
# environment MUST also keep torch at the system version, otherwise the resolver
# is free to upgrade torch (e.g. sentence_transformers pulling the latest wheel)
# while these stay at the system version, producing a broken pair that fails at
# import time with errors like "operator torchvision::nms does not exist".
TORCH_COMPANION_PACKAGES = {"torchvision", "torchaudio", "torchcodec"}

# Packages with compiled NumPy extensions that are commonly present in the
# parent environment used by sentence-transformers.  The child venv is created
# with --system-site-packages, so upgrading only part of this stack can leave an
# inherited binary linked against a different NumPy ABI.
SENTENCE_TRANSFORMERS_NUMPY_ABI_PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
)


def ensure_system_torch_pin(packages: List[str]) -> List[str]:
    """
    Pin torch to the system version whenever a torch companion package
    (torchvision/torchaudio/torchcodec) is pinned to the system version but torch
    itself is left unpinned.

    Model specs pin the companion via ``#system_torchvision#`` yet sometimes omit a
    torch pin. Without it the resolver is free to upgrade torch in the child venv
    (e.g. ``sentence_transformers`` pulling the latest wheel) while the companion
    stays at the older system version inherited through ``--system-site-packages``.
    The resulting ABI mismatch crashes the model subprocess at import time with
    errors like ``operator torchvision::nms does not exist`` — reproducibly on the
    second launch of a model, once the venv has been populated (issue #5156).

    A ``#system_torch#`` marker is appended (matching the convention of the specs
    that already carry it); xoscar's ``process_packages`` later resolves it to the
    installed ``torch==<version>``. Each companion's environment marker (e.g. the
    engine guard) is preserved so the injected torch pin applies under exactly the
    same conditions. When several companions carry different markers (e.g.
    ``#system_torchvision# ; #engine# == "sentence_transformers"`` alongside
    ``#system_torchaudio# ; #engine# == "audio"``), a matching torch pin is added
    for each distinct marker that does not already have one. This covers built-in
    specs and user-registered models alike, and is a no-op when torch is already
    pinned under the relevant condition or no companion is pinned.
    """
    if not packages:
        return packages

    def _marker_name(pkg: str) -> Optional[str]:
        # Split off any PEP 508 environment marker (";" onwards) before matching
        # the "#system_<name>#" placeholder.
        head = pkg.split(";", 1)[0].strip()
        if head.startswith("#system_") and head.endswith("#"):
            return head[len("#system_") : -1].lower()
        return None

    def _requirement_name(pkg: str) -> Optional[str]:
        # Best-effort package name for a plain requirement string like
        # "torch==2.11.0" or "torch ; marker".
        head = pkg.split(";", 1)[0].strip()
        if not head or head.startswith("#"):
            return None
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", "[", " "):
            if sep in head:
                head = head.split(sep, 1)[0]
                break
        return head.strip().lower() or None

    def _env_marker(pkg: str) -> Optional[str]:
        # The PEP 508 environment marker (text after ";"), or None if unconditional.
        return pkg.split(";", 1)[1].strip() if ";" in pkg else None

    # Collect the conditions under which torch is already pinned. ``None`` means an
    # unconditional pin (which satisfies every companion), so we can stop early.
    existing_torch_markers: set = set()
    for pkg in packages:
        if _marker_name(pkg) == "torch" or _requirement_name(pkg) == "torch":
            marker = _env_marker(pkg)
            if marker is None:
                return packages
            existing_torch_markers.add(marker)

    # Inject one torch pin per distinct companion condition that lacks one,
    # preserving that companion's environment marker so the pin applies under
    # exactly the same conditions.
    to_inject: List[str] = []
    seen_markers = set(existing_torch_markers)
    for pkg in packages:
        if _marker_name(pkg) not in TORCH_COMPANION_PACKAGES:
            continue
        marker = _env_marker(pkg)
        if marker in seen_markers:
            continue
        seen_markers.add(marker)
        to_inject.append(
            f"#system_torch# ; {marker}" if marker is not None else "#system_torch#"
        )

    if not to_inject:
        return packages

    for torch_entry in to_inject:
        logger.info(
            "Pinning torch to the system version (%s) to match the system torch "
            "companion package pinned in the virtual env; avoids a torch/torchvision "
            "ABI mismatch on relaunch (issue #5156).",
            torch_entry,
        )
    return packages + to_inject


def pin_sentence_transformers_numpy_abi(
    packages: List[str], model_engine: Optional[str]
) -> List[str]:
    """
    Keep an inherited sentence-transformers scientific stack ABI-compatible.

    Virtual environments use ``--system-site-packages``.  When uv resolves a
    new sentence-transformers installation, it may install a newer NumPy,
    SciPy, or scikit-learn into the child while continuing to inherit pandas
    (or another compiled extension) from the parent.  The mixed stack then
    fails at import time with errors such as ``numpy.core.multiarray failed to
    import``.

    Pin only packages that are already installed in the parent, and never
    replace an explicit model requirement.  Missing packages remain free to be
    resolved and installed normally, which is important for slim runtimes.
    """
    if not model_engine or model_engine.lower() != "sentence_transformers":
        return packages

    from importlib import metadata

    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    explicitly_requested = set()
    for package in packages:
        requirement = package.split(";", 1)[0].strip()
        if not requirement or requirement.startswith("#"):
            continue
        try:
            explicitly_requested.add(canonicalize_name(Requirement(requirement).name))
        except InvalidRequirement:
            continue

    pins: List[str] = []
    for distribution_name in SENTENCE_TRANSFORMERS_NUMPY_ABI_PACKAGES:
        canonical_name = canonicalize_name(distribution_name)
        if canonical_name in explicitly_requested:
            continue
        try:
            version = metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            continue
        pins.append(f"{distribution_name}=={version}")

    if pins:
        logger.info(
            "Pinning inherited sentence-transformers NumPy ABI packages: %s", pins
        )
    return packages + pins


def extract_cuda_version_from_url(url: str) -> Optional[str]:
    """Extract CUDA version suffix (e.g. 'cu130') from a wheel index URL."""
    if not url:
        return None

    # Handles both:
    #   - https://download.pytorch.org/whl/cu130
    #   - https://wheels.vllm.ai/0.19.0/cu130
    match = re.search(r"/(cu\d+)/?", url)
    return match.group(1) if match else None


def is_cuda_compatible(
    extra_index_url: Optional[Union[str, List[str]]],
    system_cuda_version: Optional[str],
) -> bool:
    """
    Check whether all CUDA-indexed URLs in extra_index_url are compatible with
    the system CUDA version.

    Returns True if there is no mismatch. If any URL has a known CUDA version that
    differs from the system, returns False. If the system version cannot be
    detected, returns False to be safe.
    """
    if not extra_index_url:
        return True
    if not system_cuda_version:
        return False

    urls = extra_index_url if isinstance(extra_index_url, list) else [extra_index_url]
    system_cuda_suffix = f"cu{system_cuda_version.replace('.', '')}"

    for url in urls:
        url_cuda_suffix = extract_cuda_version_from_url(url)
        if url_cuda_suffix and url_cuda_suffix != system_cuda_suffix:
            return False
    return True


def get_engine_virtualenv_packages(model_engine: Optional[str]) -> List[str]:
    if not model_engine:
        return []
    return ENGINE_VIRTUALENV_PACKAGES.get(model_engine.lower(), []).copy()


def get_engine_model_format_virtualenv_packages(
    model_engine: Optional[str], model_format: Optional[str]
) -> List[str]:
    if not model_engine or not model_format:
        return []
    engine_formats = ENGINE_VIRTUALENV_MODEL_FORMAT_PACKAGES.get(
        model_engine.lower(), {}
    )
    return engine_formats.get(model_format.lower(), []).copy()


def get_engine_critical_dependency_specs(
    model_engine: Optional[str], packages: Optional[List[str]] = None
) -> List[str]:
    """
    Return dependency specs that must be installed into the venv because the
    engine package will be inherited from the parent environment while the
    parent copies of its critical dependencies (see
    ``ENGINE_CRITICAL_DEPENDENCIES``) do not satisfy the engine's own declared
    requirements.

    Returns an empty list when the engine is absent from the parent
    environment or the requested spec forces a fresh engine install — in both
    cases the resolver installs the engine together with its dependencies and
    there is nothing to compensate for. Dependencies explicitly listed in
    ``packages`` are also left untouched so user-provided specs win.
    """
    if not model_engine:
        return []
    critical = ENGINE_CRITICAL_DEPENDENCIES.get(model_engine.lower())
    if not critical:
        return []

    import importlib.metadata

    from packaging.requirements import Requirement

    requested: Dict[str, Requirement] = {}
    for pkg in packages or []:
        try:
            req = Requirement(pkg.split(";", 1)[0].strip())
        except Exception:
            # placeholders (#system_torch#) and direct wheel URLs
            continue
        requested[req.name.lower()] = req

    specs: List[str] = []
    for dist_name, dep_names in critical.items():
        try:
            parent_version = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            continue

        engine_req = requested.get(dist_name.lower())
        if engine_req is not None and engine_req.specifier:
            try:
                if not engine_req.specifier.contains(parent_version, prereleases=True):
                    # parent copy doesn't satisfy the requested spec, the venv
                    # installs its own engine with full dependency resolution
                    continue
            except Exception:
                continue

        try:
            declared = importlib.metadata.requires(dist_name) or []
        except importlib.metadata.PackageNotFoundError:
            continue

        wanted = {name.lower() for name in dep_names}
        for req_str in declared:
            try:
                req = Requirement(req_str)
            except Exception:
                continue
            name = req.name.lower()
            if name not in wanted or name in requested:
                continue
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            try:
                dep_version: Optional[str] = importlib.metadata.version(req.name)
            except importlib.metadata.PackageNotFoundError:
                dep_version = None
            if dep_version is not None and req.specifier.contains(
                dep_version, prereleases=True
            ):
                continue
            specs.append(f"{req.name}{req.specifier}" if req.specifier else req.name)
    return specs


def get_engine_virtualenv_extra_index_urls(
    model_engine: Optional[str],
) -> Optional[List[str]]:
    if not model_engine:
        return None
    urls = ENGINE_VIRTUALENV_EXTRA_INDEX_URLS.get(model_engine.lower())
    result = urls.copy() if urls else None
    logger.debug(
        f"[DEBUG] get_engine_virtualenv_extra_index_urls: model_engine={model_engine}, urls={urls}, result={result}"
    )
    return result


def get_engine_virtualenv_index_strategy(model_engine: Optional[str]) -> Optional[str]:
    if not model_engine:
        return None
    return ENGINE_VIRTUALENV_INDEX_STRATEGY.get(model_engine.lower())


def resolve_virtualenv_python_path(virtual_env_manager: Any) -> Optional[str]:
    """
    Resolve a usable Python executable path for a virtual environment.

    This prefers the manager's reported path when it exists, otherwise falls
    back to OS-specific defaults under the env_path.
    """
    if virtual_env_manager is None:
        return None
    venv_python = virtual_env_manager.get_python_path()
    if venv_python and os.path.exists(venv_python):
        return venv_python
    env_path = getattr(virtual_env_manager, "env_path", None)
    if env_path is None:
        return venv_python
    candidates: List[str] = []
    if os.name == "nt":
        candidates.append(os.path.join(str(env_path), "Scripts", "python.exe"))
    candidates.append(os.path.join(str(env_path), "bin", "python"))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return venv_python


def expand_engine_dependency_placeholders(
    packages: List[str], model_engine: Optional[str]
) -> List[str]:
    if not packages:
        return []
    engine_name = model_engine.lower() if model_engine else None
    expanded: List[str] = []
    # Mapping for dependency names that differ from engine names
    dependency_to_engine_map: Dict[str, str] = {
        "llama_cpp": "llama.cpp",
    }
    for pkg in packages:
        name = pkg.split(";", 1)[0].strip().lower()
        if name.startswith("#") and name.endswith("#"):
            name = name[1:-1]
        if name.endswith("_dependencies") and engine_name:
            target_engine = name[: -len("_dependencies")]
            # Map dependency name to actual engine name if needed
            actual_engine = dependency_to_engine_map.get(target_engine, target_engine)
            if actual_engine == engine_name:
                expanded.extend(
                    ENGINE_VIRTUALENV_PACKAGES.get(actual_engine, []).copy()
                )
            continue
        expanded.append(pkg)
    return expanded


class VirtualEnvManager:
    """
    Manager class for handling virtual environments.
    Extracted from worker.py to improve code organization and maintainability.
    Supports multiple Python versions per model and engine with v4 structure:
    .xinference/virtualenv/v4/{model_name}/{model_engine}/{python_version}/
    """

    def __init__(self, worker_address: str):
        """
        Initialize VirtualEnvManager.

        Args:
            worker_address: The address of the worker using this manager
        """
        self.worker_address = worker_address

    def list_virtual_envs(
        self, model_name: Optional[str] = None, model_engine: Optional[str] = None
    ) -> List[Dict[Any, Any]]:
        """
        List all virtual environments or filter by model name.

        Args:
            model_name: Optional model name to filter results
            model_engine: Optional model engine to filter results

        Returns:
            List of virtual environment information dictionaries
        """

        virtual_envs: List[Dict[str, Any]] = []
        if model_engine is not None:
            model_engine = model_engine.lower()
        v4_env_dir = os.path.join(XINFERENCE_VIRTUAL_ENV_DIR, "v4")

        if os.path.exists(v4_env_dir):
            for model_dir in os.listdir(v4_env_dir):
                model_path = os.path.join(v4_env_dir, model_dir)
                if not os.path.isdir(model_path):
                    continue

                # Apply filter if model_name is specified
                if model_name and model_dir != model_name:
                    continue

                for engine_dir in os.listdir(model_path):
                    engine_path = os.path.join(model_path, engine_dir)
                    if not os.path.isdir(engine_path):
                        continue

                    # Apply filter if model_engine is specified
                    if model_engine and engine_dir != model_engine:
                        continue

                    # Check for Python version directories
                    for python_version_dir in os.listdir(engine_path):
                        python_version_path = os.path.join(
                            engine_path, python_version_dir
                        )
                        if os.path.isdir(python_version_path):
                            # Validate Python version format (e.g., "3.10", "3.13")
                            if self._is_valid_python_version(python_version_dir):
                                env_info: Dict[str, Any] = {
                                    "model_name": model_dir,
                                    "model_engine": engine_dir,
                                    "python_version": python_version_dir,
                                    "path": python_version_path,
                                    "real_path": os.path.realpath(python_version_path),
                                }
                                virtual_envs.append(env_info)

        return virtual_envs

    def remove_virtual_env(
        self,
        model_name: str,
        model_engine: Optional[str] = None,
        python_version: Optional[str] = None,
        active_model_uids_by_path: Optional[Dict[str, List[str]]] = None,
    ) -> bool:
        """
        Remove a virtual environment for a specific model.

        Args:
            model_name: Name of the model whose virtual environment should be removed
            model_engine: Optional model engine to remove specific engine
            python_version: Optional Python version to remove specific version,
                          if None, removes all Python versions for the model

        Returns:
            True if removal was successful, False otherwise
        """
        if not model_name:
            raise ValueError("model_name is required")

        try:
            if model_engine is not None:
                model_engine = model_engine.lower()
            from ..constants import XINFERENCE_VIRTUAL_ENV_DIR
        except ImportError:
            # Fallback for testing or when run as standalone
            XINFERENCE_VIRTUAL_ENV_DIR = os.path.join(
                os.path.expanduser("~/.xinference"), "virtualenv"
            )

        v4_env_dir = os.path.join(XINFERENCE_VIRTUAL_ENV_DIR, "v4")

        model_root = os.path.realpath(os.path.join(v4_env_dir, model_name))
        for env_path, model_uids in (active_model_uids_by_path or {}).items():
            if not model_uids:
                continue
            real_env_path = os.path.realpath(env_path)
            try:
                if os.path.commonpath([model_root, real_env_path]) != model_root:
                    continue
            except ValueError:
                continue
            relative_parts = Path(real_env_path).relative_to(model_root).parts
            if model_engine and (
                not relative_parts or relative_parts[0] != model_engine
            ):
                continue
            if python_version and (
                not relative_parts or relative_parts[-1] != python_version
            ):
                continue
            raise VirtualEnvConflictError(
                "Virtual environment cannot be removed while models are using it: "
                f"path={real_env_path}, active_model_uids={sorted(model_uids)}"
            )

        try:
            if python_version and not self._is_valid_python_version(python_version):
                logger.warning(f"Invalid Python version format: {python_version}")
                return False

            if model_engine:
                model_path = os.path.join(v4_env_dir, model_name, model_engine)
                if not os.path.exists(model_path):
                    logger.warning(
                        f"Virtual environment path not found for model: {model_name}"
                    )
                    return True

                if python_version:
                    version_path = os.path.join(model_path, python_version)
                    if not os.path.exists(version_path):
                        logger.warning(
                            "Virtual environment for %s engine %s Python %s not found",
                            model_name,
                            model_engine,
                            python_version,
                        )
                        return True

                    if os.path.islink(version_path):
                        os.unlink(version_path)
                    elif os.path.isdir(version_path):
                        shutil.rmtree(version_path)
                    else:
                        logger.warning(
                            "Virtual environment path is not a directory: %s",
                            version_path,
                        )

                    logger.info(
                        "Successfully removed virtual environment: %s", version_path
                    )
                else:
                    if os.path.islink(model_path):
                        os.unlink(model_path)
                    elif os.path.isdir(model_path):
                        shutil.rmtree(model_path)
                    else:
                        logger.warning(
                            "Virtual environment path is not a directory: %s",
                            model_path,
                        )

                    logger.info(
                        "Successfully removed all virtual environments for model engine: %s",
                        model_path,
                    )

                # Cleanup empty model directory
                model_root = os.path.join(v4_env_dir, model_name)
                try:
                    if os.path.exists(model_root) and not os.listdir(model_root):
                        os.rmdir(model_root)
                        logger.info("Removed empty model directory: %s", model_root)
                except OSError:
                    pass

                return True

            # No model_engine specified: remove across all engines (v4)
            model_path_v4 = os.path.join(v4_env_dir, model_name)

            if python_version:
                # Remove specific Python version across v4 engines
                if os.path.exists(model_path_v4):
                    for engine_dir in os.listdir(model_path_v4):
                        engine_path = os.path.join(model_path_v4, engine_dir)
                        if not os.path.isdir(engine_path):
                            continue
                        version_path = os.path.join(engine_path, python_version)
                        if not os.path.exists(version_path):
                            continue
                        if os.path.islink(version_path):
                            os.unlink(version_path)
                        elif os.path.isdir(version_path):
                            shutil.rmtree(version_path)
                        else:
                            logger.warning(
                                "Virtual environment path is not a directory: %s",
                                version_path,
                            )
                        try:
                            if os.path.exists(engine_path) and not os.listdir(
                                engine_path
                            ):
                                os.rmdir(engine_path)
                        except OSError:
                            pass

                # Cleanup empty model directory
                try:
                    if os.path.exists(model_path_v4) and not os.listdir(model_path_v4):
                        os.rmdir(model_path_v4)
                        logger.info("Removed empty model directory: %s", model_path_v4)
                except OSError:
                    pass

                return True

            # Remove all Python versions for the model (v4)
            if os.path.exists(model_path_v4):
                if os.path.islink(model_path_v4):
                    os.unlink(model_path_v4)
                elif os.path.isdir(model_path_v4):
                    shutil.rmtree(model_path_v4)
                else:
                    logger.warning(
                        "Virtual environment path is not a directory: %s",
                        model_path_v4,
                    )

            logger.info(
                "Successfully removed all virtual environments for model: %s",
                model_name,
            )
            return True

        except Exception as e:
            logger.error(f"Failed to remove virtual environment: {e}")
            return False

    def check_virtual_env_exists(self, model_name: str) -> Dict[str, Any]:
        """
        Check if a virtual environment exists for a specific model.

        Args:
            model_name: Name of the model to check

        Returns:
            Dictionary with existence check result
        """
        virtual_envs = self.list_virtual_envs(model_name)
        return {"has_virtual_env": len(virtual_envs) > 0, "model_name": model_name}

    def list_virtual_env_packages(self, model_name: str) -> Dict[str, Any]:
        """
        List packages installed in a specific virtual environment.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with package information or error message
        """
        # This method is deprecated and no longer needed
        # Virtual environments are managed by direct directory scanning
        return {
            "model_name": model_name,
            "worker_ip": self.worker_address,
            "error": "Package listing functionality has been removed",
        }

    def _detect_python_version(self, env_path: str) -> str:
        """
        Detect Python version in a virtual environment.

        Args:
            env_path: Path to the virtual environment

        Returns:
            Python version string or "unknown" if detection fails
        """
        try:
            # Try to find Python executable in the virtual environment
            if os.name != "nt":  # Unix-like systems
                python_exe = os.path.join(env_path, "bin", "python")
                python3_exe = os.path.join(env_path, "bin", "python3")
            else:  # Windows
                python_exe = os.path.join(env_path, "Scripts", "python.exe")
                python3_exe = os.path.join(env_path, "Scripts", "python3.exe")

            # Try to execute Python to get version
            for exe_path in [python_exe, python3_exe]:
                if os.path.exists(exe_path):
                    try:
                        result = subprocess.run(
                            [exe_path, "--version"],
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        if result.returncode == 0:
                            # Parse version from output like "Python 3.9.7"
                            version_output = (
                                result.stdout.strip() or result.stderr.strip()
                            )
                            if version_output.startswith("Python "):
                                return version_output[7:]  # Remove "Python " prefix
                    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                        continue

            # Fallback: try to read from lib/pythonX.Y structure
            if os.path.exists(env_path):
                for item in os.listdir(env_path):
                    if item.startswith("lib") and "python" in item:
                        # Extract version from directory name like "lib/python3.9"
                        parts = item.split("python")
                        if len(parts) > 1:
                            version = parts[1].replace(".", "")
                            # Format version properly (e.g., "39" -> "3.9")
                            if len(version) >= 2:
                                return f"{version[0]}.{version[1:]}"

            return "unknown"
        except Exception:
            return "unknown"

    def _is_valid_python_version(self, python_version: str) -> bool:
        """
        Validate Python version format (e.g., "3.10", "3.10.18")

        Args:
            python_version: Python version string to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            parts = python_version.split(".")
            if len(parts) not in [2, 3]:
                return False
            major, minor = parts[:2]
            if not major.isdigit() or not minor.isdigit():
                return False
            if len(parts) == 3:
                micro = parts[2]
                if not micro.isdigit():
                    return False
            return True
        except Exception:
            return False


# flashinfer AOT workaround for sm_120 Blackwell consumer GPUs.
# See optimize/20260702/2026070209.md for full root cause analysis.
FLASHINFER_AOT_ARCHES = frozenset({"Qwen3_5MoeForConditionalGeneration"})
FLASHINFER_AOT_PACKAGES = [
    "flashinfer-python==0.6.11.post3",
    "flashinfer-cubin==0.6.11.post3",
    "flashinfer-jit-cache==0.6.11.post3+cu130",
]
FLASHINFER_CUBIN_WHEEL_URL = "https://flashinfer.ai/whl"
FLASHINFER_AOT_WHEEL_URL = "https://flashinfer.ai/whl/cu130"


def _get_virtualenv_distribution_version(
    virtual_env_manager: Any, distribution: str
) -> Optional[str]:
    try:
        python_path = resolve_virtualenv_python_path(virtual_env_manager)
    except (AttributeError, TypeError):
        return None
    if not python_path:
        return None
    try:
        result = subprocess.run(
            [
                python_path,
                "-c",
                (
                    "import importlib.metadata as metadata; "
                    f"print(metadata.version({distribution!r}))"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def ensure_sglang_inherited_packages_compatible_post_install(
    model_engine: Optional[str], virtual_env_manager: Any
) -> None:
    """Keep SGLang dependencies compatible with inherited numba/cudf packages.

    SGLang and datasets leave NumPy and pandas broadly constrained, while the
    actor subprocess inherits parent site-packages so xoscar can import its
    serializers.  New child versions can therefore break inherited numba/cudf
    before SGLang starts.  New environments carry explicit compatibility caps;
    when the parent already has a compatible version, repair a cached environment
    without downloading anything by removing only the shadowing child copy.
    """
    if not model_engine or model_engine.lower() != "sglang":
        return

    import importlib.metadata

    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    constraints = {"numpy": "<2.3", "pandas": "<3"}
    for distribution, specifier in constraints.items():
        compatible = SpecifierSet(specifier)
        try:
            system_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        virtualenv_version = _get_virtualenv_distribution_version(
            virtual_env_manager, distribution
        )
        if virtualenv_version is None or Version(virtualenv_version) in compatible:
            continue
        if Version(system_version) not in compatible:
            raise RuntimeError(
                f"SGLang requires {distribution}{specifier} for compatibility "
                f"with inherited numba/cudf packages, but virtualenv "
                f"{distribution} is {virtualenv_version} and system "
                f"{distribution} is {system_version}."
            )

        logger.warning(
            "Incompatible SGLang virtualenv %s %s shadows system %s %s in %s; "
            "removing the child copy",
            distribution,
            virtualenv_version,
            distribution,
            system_version,
            virtual_env_manager.env_path,
        )
        uv_path = None
        if hasattr(virtual_env_manager, "_get_uv_path"):
            try:
                uv_path = virtual_env_manager._get_uv_path()
            except Exception:
                pass
        if not uv_path:
            uv_path = shutil.which("uv") or "uv"
        python_path = resolve_virtualenv_python_path(virtual_env_manager)
        if not python_path:
            raise RuntimeError(
                f"Cannot repair the SGLang virtualenv {distribution} mismatch: "
                "virtualenv Python was not found."
            )

        result = subprocess.run(
            [
                uv_path,
                "pip",
                "uninstall",
                "--python",
                python_path,
                distribution,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        repaired_version = _get_virtualenv_distribution_version(
            virtual_env_manager, distribution
        )
        if (
            result.returncode != 0
            or repaired_version is None
            or Version(repaired_version) not in compatible
        ):
            detail = result.stderr[-500:] if result.stderr else "(empty)"
            raise RuntimeError(
                f"Failed to repair the incompatible SGLang virtualenv "
                f"{distribution} {virtualenv_version}; system {distribution} "
                f"is {system_version}: {detail}"
            )
        logger.info(
            "SGLang virtualenv now uses compatible %s %s",
            distribution,
            repaired_version,
        )


def _as_uv_option_values(value: Any) -> List[str]:
    if not value:
        return []
    return [value] if isinstance(value, str) else list(value)


def build_uv_source_options(
    conf: Dict[str, Any],
    *,
    public_index_urls: Optional[List[str]] = None,
    allow_public_install: bool = True,
) -> List[str]:
    """Build uv source options from the virtualenv install configuration.

    Post-install hooks invoke ``uv`` directly, so they must explicitly reuse the
    same package sources as the main virtualenv installation. Administrator-
    supplied public indexes are appended as fallbacks only when public installs
    are allowed.
    """
    options: List[str] = []
    index_url = conf.get("index_url")
    extra_index_urls = _as_uv_option_values(conf.get("extra_index_url"))

    # uv gives every --index / --extra-index-url entry higher priority than
    # --default-index / --index-url. Keep the configured sources in their
    # original effective order (extra indexes before index_url), then place the
    # hook-specific public indexes after all of them. The final public index is
    # the default index so it is a true lowest-priority fallback.
    configured_index_urls = [*extra_index_urls]
    if index_url:
        configured_index_urls.append(index_url)
    configured_url_set = set(configured_index_urls)
    public_fallback_urls = (
        [
            url
            for url in (public_index_urls or [])
            if url and url not in configured_url_set
        ]
        if allow_public_install
        else []
    )

    default_index_url: Optional[str]
    if public_fallback_urls:
        priority_index_urls = configured_index_urls + public_fallback_urls[:-1]
        default_index_url = public_fallback_urls[-1]
    elif not allow_public_install and configured_index_urls:
        # Avoid uv's implicit PyPI default during a configured-source-only
        # attempt. Keep the final configured URL as the explicit default while
        # preserving the effective priority of the preceding URLs.
        priority_index_urls = configured_index_urls[:-1]
        default_index_url = configured_index_urls[-1]
    else:
        # Preserve uv's existing configured-source semantics when no public
        # fallback is added: extra_index_url entries outrank index_url.
        priority_index_urls = extra_index_urls
        default_index_url = index_url

    for url in priority_index_urls:
        options += ["--index", url]
    if default_index_url:
        options += ["--default-index", default_index_url]

    find_links = _as_uv_option_values(conf.get("find_links"))
    for link in find_links:
        options += ["--find-links", link]
    # ``--find-links`` supplements registry indexes; it does not disable uv's
    # implicit default index. A configured-source-only attempt must therefore
    # opt out explicitly when find-links is the only configured source.
    if not allow_public_install and find_links and not configured_index_urls:
        options.append("--no-index")
    for host in _as_uv_option_values(conf.get("trusted_host")):
        options += ["--trusted-host", host]

    # ``unsafe-best-match`` queries every index even after a configured source
    # provides the pinned package. An unavailable hook-specific public fallback
    # would therefore still fail the whole install. Use first-index semantics
    # whenever a public fallback is appended so configured sources remain
    # authoritative; otherwise preserve the caller's existing strategy.
    index_strategy = (
        "first-index"
        if configured_index_urls and public_fallback_urls
        else conf.get("index_strategy")
    )
    if index_strategy:
        options += ["--index-strategy", index_strategy]
    return options


def is_flash_attn_requirement(spec: str) -> bool:
    """Return whether *spec* targets the flash-attn distribution."""
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    try:
        requirement = Requirement(spec)
    except InvalidRequirement:
        return False
    return canonicalize_name(requirement.name) == "flash-attn"


def is_model_find_links_only_requirement(model_name: Optional[str], spec: str) -> bool:
    """Return whether a model requirement is supplied only via Find Links.

    These requirements are intentionally absent from package indexes.  The
    runtime installs them from worker-local wheel directories when configured
    and otherwise uses the model's documented fallback path.
    """
    return model_name == "jina-embeddings-v3" and is_flash_attn_requirement(spec)


_SUBPROCESS_OUTPUT_LIMIT = 2048


def _format_subprocess_output(output: Any) -> str:
    """Prepare bounded subprocess output for logs and user-facing errors."""
    if not isinstance(output, str) or not output:
        return "<empty>"

    sanitized = output.strip()
    sanitized = re.sub(r"(?i)([a-z][a-z0-9+.-]*://)[^/@\s]+@", r"\1***@", sanitized)
    sanitized = re.sub(
        r"(?i)\b(password|passwd|token|api[-_]?key|access[-_]?key)=" r"([^&\s]+)",
        r"\1=***",
        sanitized,
    )
    sanitized = re.sub(r"(?i)(\bBearer\s+)[^\s,;]+", r"\1***", sanitized)
    sanitized = sanitized.replace("\r", "\\r").replace("\n", "\\n")
    if len(sanitized) > _SUBPROCESS_OUTPUT_LIMIT:
        sanitized = sanitized[:_SUBPROCESS_OUTPUT_LIMIT] + "...<truncated>"
    return sanitized or "<empty>"


def _validate_flash_attn_install(
    virtual_env_manager: Any, flash_attn_packages: List[str]
) -> bool:
    """Validate flash-attn metadata and CUDA imports in a fresh venv process."""
    from packaging.requirements import InvalidRequirement, Requirement

    python_path = resolve_virtualenv_python_path(virtual_env_manager)
    if not python_path:
        logger.warning("flash-attn validation has no virtualenv Python")
        return False

    validation_script = """
import importlib.metadata as metadata
import json
import pathlib
import sys

prefix = pathlib.Path(sys.prefix).resolve()
distribution = metadata.distribution("flash-attn")
distribution_path = pathlib.Path(distribution.locate_file("")).resolve()
if prefix != distribution_path and prefix not in distribution_path.parents:
    raise RuntimeError(
        f"flash-attn distribution resolved outside the model virtualenv: "
        f"{distribution_path}"
    )

import flash_attn
from flash_attn import flash_attn_func

module_path = pathlib.Path(flash_attn.__file__).resolve()
if prefix != module_path and prefix not in module_path.parents:
    raise RuntimeError(
        f"flash_attn module resolved outside the model virtualenv: {module_path}"
    )
if not callable(flash_attn_func):
    raise RuntimeError("flash_attn.flash_attn_func is not callable")

print(json.dumps({"version": distribution.version}, sort_keys=True))
"""
    env = os.environ.copy()
    for variable in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"):
        env.pop(variable, None)
    try:
        result = subprocess.run(
            [python_path, "-c", validation_script],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
    except OSError:
        logger.warning("flash-attn validation process could not be started")
        return False
    if result.returncode != 0:
        logger.warning(
            "flash-attn validation failed in %s with exit code %s. "
            "stdout=%r, stderr=%r",
            virtual_env_manager.env_path,
            result.returncode,
            _format_subprocess_output(result.stdout),
            _format_subprocess_output(result.stderr),
        )
        return False

    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        installed_version = str(payload["version"])
    except (IndexError, KeyError, TypeError, ValueError):
        logger.warning(
            "flash-attn validation returned invalid metadata in %s",
            virtual_env_manager.env_path,
        )
        return False

    for spec in flash_attn_packages:
        try:
            requirement = Requirement(spec)
        except InvalidRequirement:
            logger.warning("Invalid flash-attn requirement during validation: %s", spec)
            return False
        if requirement.specifier and not requirement.specifier.contains(
            installed_version, prereleases=True
        ):
            logger.warning(
                "Installed flash-attn %s does not satisfy %s",
                installed_version,
                spec,
            )
            return False

    logger.info(
        "flash-attn %s import and version validation succeeded in %s",
        installed_version,
        virtual_env_manager.env_path,
    )
    return True


def apply_flash_attn_wheel_post_install(
    model_name: Optional[str],
    flash_attn_packages: List[str],
    virtual_env_manager: Any,
    conf: Dict[str, Any],
    cuda_available: bool,
) -> None:
    """Install Jina's flash-attn requirement from configured Wheel sources."""
    if (
        model_name != "jina-embeddings-v3"
        or not cuda_available
        or not flash_attn_packages
    ):
        return

    python_path = resolve_virtualenv_python_path(virtual_env_manager)
    if not python_path:
        raise RuntimeError("flash-attn post-install has no virtualenv Python")

    if not conf.get("find_links"):
        if not _validate_flash_attn_install(virtual_env_manager, flash_attn_packages):
            logger.warning(
                "flash-attn is not installed or invalid for %s, and no Find "
                "Links source was configured. The model will use PyTorch "
                "native attention (higher GPU memory usage).",
                model_name,
            )
        return

    uv_path = None
    if hasattr(virtual_env_manager, "_get_uv_path"):
        try:
            uv_path = virtual_env_manager._get_uv_path()
        except Exception:
            pass
    if not uv_path:
        uv_path = shutil.which("uv") or "uv"

    command = [
        uv_path,
        "pip",
        "install",
        "--python",
        python_path,
        "--no-deps",
        "--reinstall",
        "--only-binary=:all:",
        *build_uv_source_options(conf),
        *flash_attn_packages,
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError(
            "flash-attn Wheel installation could not be started"
        ) from exc
    if result.returncode != 0:
        stdout = _format_subprocess_output(result.stdout)
        stderr = _format_subprocess_output(result.stderr)
        logger.error(
            "flash-attn Wheel installation failed for %s with exit code %s. "
            "stdout=%r, stderr=%r",
            flash_attn_packages,
            result.returncode,
            stdout,
            stderr,
        )
        raise RuntimeError(
            f"flash-attn Wheel installation failed for {flash_attn_packages} "
            f"with exit code {result.returncode}. stdout={stdout!r}, stderr={stderr!r}"
        )

    if not _validate_flash_attn_install(virtual_env_manager, flash_attn_packages):
        raise RuntimeError(
            "flash-attn Wheel installation completed, but fresh-process "
            "version/path/CUDA import validation failed."
        )


def validate_virtual_env_find_links(
    find_links: Optional[List[str]],
    allowed_roots: Optional[tuple[str, ...]] = None,
) -> List[str]:
    """Validate request-level local wheel directories on the worker."""
    if find_links is None:
        return []
    if not isinstance(find_links, list):
        raise ValueError("virtual_env_find_links must be a list of local paths")

    roots = (
        XINFERENCE_VIRTUAL_ENV_FIND_LINKS_ALLOWED_ROOTS
        if allowed_roots is None
        else tuple(os.path.realpath(root) for root in allowed_roots)
    )
    if not roots:
        raise ValueError(
            "Request-level virtualenv find-links are disabled on this worker"
        )

    result: List[str] = []
    seen = set()
    for raw_path in find_links:
        if not isinstance(raw_path, str):
            raise ValueError("virtual_env_find_links entries must be strings")
        path = raw_path.strip()
        if not path:
            continue
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", path):
            raise ValueError(
                "virtual_env_find_links only accepts absolute local directories"
            )
        if not os.path.isabs(path):
            raise ValueError(f"virtual_env_find_links path must be absolute: {path!r}")
        try:
            canonical_path = str(Path(path).resolve(strict=True))
        except (OSError, RuntimeError) as exc:
            raise ValueError(
                f"virtual_env_find_links path does not exist: {path!r}"
            ) from exc
        if not os.path.isdir(canonical_path):
            raise ValueError(
                f"virtual_env_find_links path is not a directory: {path!r}"
            )
        if not os.access(canonical_path, os.R_OK | os.X_OK):
            raise ValueError(f"virtual_env_find_links path is not readable: {path!r}")

        contained = False
        for root in roots:
            try:
                contained = os.path.normcase(
                    os.path.commonpath([canonical_path, root])
                ) == os.path.normcase(root)
            except ValueError:
                contained = False
            if contained:
                break
        if not contained:
            raise ValueError(
                f"virtual_env_find_links path is outside the configured allowed roots: {path!r}"
            )

        key = os.path.normcase(canonical_path)
        if key not in seen:
            seen.add(key)
            result.append(canonical_path)
    return result


def merge_virtual_env_find_links(
    configured: Optional[Union[str, List[str]]], requested: List[str]
) -> Optional[List[str]]:
    values = _as_uv_option_values(configured) + requested
    if not values:
        return None
    return list(dict.fromkeys(values))


def _has_configured_package_source(conf: Dict[str, Any]) -> bool:
    return any(conf.get(key) for key in ("index_url", "extra_index_url", "find_links"))


def _run_uv_install_with_source_fallback(
    base_cmd: List[str],
    packages: List[str],
    conf: Dict[str, Any],
    *,
    public_index_urls: Optional[List[str]] = None,
    allow_public_install: bool = True,
) -> subprocess.CompletedProcess:
    """Run uv with configured sources before consulting public fallbacks."""
    source_conf = conf
    if (
        allow_public_install
        and public_index_urls
        and _has_configured_package_source(conf)
    ):
        configured_cmd = base_cmd + build_uv_source_options(
            conf, allow_public_install=False
        )
        result = subprocess.run(
            configured_cmd + packages,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result
        logger.info(
            "Post-install requirements were not satisfied by configured package "
            "sources; retrying with the allowed public fallback"
        )
        # Retry against the hook-specific fallback independently. With uv's
        # first-index semantics, retaining a configured index that contains the
        # project at a different version would prevent the fallback from ever
        # being considered.
        source_conf = {}

    cmd = base_cmd + build_uv_source_options(
        source_conf,
        public_index_urls=public_index_urls,
        allow_public_install=allow_public_install,
    )
    return subprocess.run(
        cmd + packages,
        check=False,
        capture_output=True,
        text=True,
    )


def ensure_flashinfer_cubin_matches_post_install(
    model_engine: Optional[str],
    virtual_env_manager: Any,
    allow_public_install: bool = True,
    conf: Optional[Dict[str, Any]] = None,
) -> None:
    """Keep FlashInfer's Python package and cubin package on the same version.

    ``flashinfer-cubin`` lives on FlashInfer's own wheel index. A reused vLLM
    environment can therefore upgrade ``flashinfer-python`` from PyPI while
    retaining an older cubin from a previous engine wheel index. FlashInfer
    rejects that pair at import time before the model can load.

    Repair the cubin in place from the official index. In explicit offline mode
    (or if the repair fails), use FlashInfer's documented version-check bypass
    so a stale environment does not fail before engine initialization.
    """
    if not model_engine or model_engine.lower() != "vllm":
        return

    python_version = _get_virtualenv_distribution_version(
        virtual_env_manager, "flashinfer-python"
    )
    if python_version is None:
        return
    cubin_version = _get_virtualenv_distribution_version(
        virtual_env_manager, "flashinfer-cubin"
    )
    if cubin_version == python_version:
        return

    logger.warning(
        "FlashInfer package mismatch in %s: flashinfer-python=%s, "
        "flashinfer-cubin=%s",
        virtual_env_manager.env_path,
        python_version,
        cubin_version or "not installed",
    )

    conf = conf or {}
    if allow_public_install or _has_configured_package_source(conf):
        uv_path = None
        if hasattr(virtual_env_manager, "_get_uv_path"):
            try:
                uv_path = virtual_env_manager._get_uv_path()
            except Exception:
                pass
        if not uv_path:
            uv_path = shutil.which("uv") or "uv"

        cmd = [
            uv_path,
            "pip",
            "install",
            "-p",
            str(virtual_env_manager.env_path),
            "--no-deps",
            "--upgrade",
        ]
        try:
            result = _run_uv_install_with_source_fallback(
                cmd,
                [f"flashinfer-cubin=={python_version}"],
                conf,
                public_index_urls=[FLASHINFER_CUBIN_WHEEL_URL],
                allow_public_install=allow_public_install,
            )
            if result.returncode == 0:
                repaired_version = _get_virtualenv_distribution_version(
                    virtual_env_manager, "flashinfer-cubin"
                )
                if repaired_version == python_version:
                    logger.info(
                        "Synchronized flashinfer-cubin to flashinfer-python %s",
                        python_version,
                    )
                    return
            else:
                logger.warning(
                    "Failed to synchronize flashinfer-cubin (exit %d): %s",
                    result.returncode,
                    result.stderr[-500:] if result.stderr else "(empty)",
                )
        except Exception as e:
            logger.warning("Failed to synchronize flashinfer-cubin: %s", e)

    os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    logger.warning(
        "Set FLASHINFER_DISABLE_VERSION_CHECK=1 because flashinfer-cubin could "
        "not be synchronized to flashinfer-python %s",
        python_version,
    )


def needs_flashinfer_aot(
    model_engine: Optional[str],
    architectures: Optional[List[str]],
    cuda_version: Optional[str] = None,
) -> bool:
    """Check if this model needs flashinfer AOT wheel post-install.

    Gate narrowly: only vllm engine + Qwen3_5MoeForConditionalGeneration
    architecture (qwen3.5 / qwen3.6 / Ornith-1.0-35B) triggers the
    flashinfer JIT failure on sm_120 Blackwell consumer GPUs.

    Also gates on CUDA runtime version: FLASHINFER_AOT_PACKAGES ships
    ``+cu130`` variants only, so on non-CUDA-13.0 systems the install
    would fail and force the fallback. Skip those systems rather than
    trigger a noisy failed install.
    """
    if not model_engine or model_engine.lower() != "vllm":
        return False
    if cuda_version != "13.0":
        return False
    return any(a in FLASHINFER_AOT_ARCHES for a in (architectures or []))


def apply_flashinfer_aot_post_install(
    model_engine: Optional[str],
    architectures: Optional[List[str]],
    virtual_env_manager: Any,
    conf: Dict[str, Any],
    cuda_version: Optional[str] = None,
    allow_public_install: bool = True,
) -> None:
    """Post-install hook: force-upgrade flashinfer to AOT versions for sm_120.

    vllm 0.21.0 hard-pins flashinfer-cubin==0.6.8.post1, which has JIT
    compilation failure on sm_120 (ptxas segfault + namespace resolution
    bug). Force-upgrade to 0.6.11.post3 + AOT wheel to bypass JIT.

    Uses --no-deps --upgrade to bypass vllm's hard pin without triggering
    uv dependency resolution conflict.

    Fallback: if upgrade fails (e.g. wheel unavailable offline), set
    FLASHINFER_DISABLE_VERSION_CHECK=1 — model still works because
    flashinfer 0.6.8.post1 Python binding can load 0.6.11.post3 AOT .so
    (verified ABI compatible for fused_moe path in production).

    See optimize/20260702/2026070209.md for root cause analysis.
    """
    if not needs_flashinfer_aot(model_engine, architectures, cuda_version):
        return

    logger.info(
        "Post-install: force-upgrading flashinfer to AOT versions for %s "
        "(sm_120 Blackwell workaround)",
        list(architectures or []),
    )

    if not allow_public_install and not _has_configured_package_source(conf):
        logger.info(
            "Skipping the FlashInfer AOT post-install because public installs "
            "are disabled and no package source is configured"
        )
        return

    # Resolve uv path with a fallback. ``_get_uv_path`` is a private method
    # on xoscar's VirtualEnvManager and could be renamed/removed in future
    # releases; fall back to PATH lookup so the hook degrades gracefully.
    uv_path = None
    if hasattr(virtual_env_manager, "_get_uv_path"):
        try:
            uv_path = virtual_env_manager._get_uv_path()
        except Exception:
            pass
    if not uv_path:
        uv_path = shutil.which("uv") or "uv"

    cmd = [
        uv_path,
        "pip",
        "install",
        "-p",
        str(virtual_env_manager.env_path),
        "--no-deps",
        "--upgrade",
        "--color=always",
    ]
    try:
        result = _run_uv_install_with_source_fallback(
            cmd,
            FLASHINFER_AOT_PACKAGES,
            conf,
            public_index_urls=[FLASHINFER_AOT_WHEEL_URL],
            allow_public_install=allow_public_install,
        )
        if result.returncode == 0:
            logger.info(
                "Post-install: flashinfer AOT upgrade SUCCEEDED — "
                "fused_moe JIT bypassed"
            )
        else:
            logger.warning(
                "Post-install: flashinfer AOT upgrade FAILED (exit %d). "
                "Falling back to JIT with FLASHINFER_DISABLE_VERSION_CHECK=1. "
                "stderr: %s",
                result.returncode,
                result.stderr[-500:] if result.stderr else "(empty)",
            )
            os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
            logger.warning(
                "Set FLASHINFER_DISABLE_VERSION_CHECK=1 — model may still "
                "work via AOT .so load (verified 0.6.8.post1 binding + "
                "0.6.11.post3 .so ABI compatible for fused_moe)"
            )
    except Exception as e:
        logger.warning(
            "Post-install: flashinfer AOT upgrade exception: %s. "
            "Falling back to JIT with FLASHINFER_DISABLE_VERSION_CHECK=1.",
            e,
        )
        os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
