# Copyright 2022-2026 XProbe Inc.
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

import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from ..constants import XINFERENCE_VIRTUAL_ENV_DIR

logger = logging.getLogger(__name__)

ENGINE_VIRTUALENV_PACKAGES: Dict[str, List[str]] = {
    "sglang": [
        "pybase64",
        "zmq",
        "partial_json_parser",
        "sentencepiece",
        "dill",
        "ninja",
        "numpy>=2.4.1",
        "sglang>=0.5.6",
        'https://github.com/sgl-project/whl/releases/download/v0.3.21/sgl_kernel-0.3.21+cu130-cp310-abi3-manylinux2014_x86_64.whl ; cuda_version == "13.0" and platform_machine == "x86_64"',
        'https://github.com/sgl-project/whl/releases/download/v0.3.21/sgl_kernel-0.3.21+cu130-cp310-abi3-manylinux2014_aarch64.whl ; cuda_version == "13.0" and platform_machine == "aarch64"',
        'sgl_kernel ; cuda_version < "13.0"',
    ],
    "vllm": [
        "vllm>=0.11.2,<0.15.0",
    ],
    "transformers": [
        "transformers>=4.46.0,<5.0.0",
        "accelerate>=0.28.0",
    ],
    "sentence_transformers": [
        "sentence_transformers",
        "einops",
        "transformers<5.0.0",
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

ENGINE_VIRTUALENV_EXTRA_INDEX_URLS: Dict[str, List[str]] = {
    "vllm": [
        "https://wheels.vllm.ai/0.14.1/cu130",
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


def get_engine_virtualenv_packages(model_engine: Optional[str]) -> List[str]:
    if not model_engine:
        return []
    return ENGINE_VIRTUALENV_PACKAGES.get(model_engine.lower(), []).copy()


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
