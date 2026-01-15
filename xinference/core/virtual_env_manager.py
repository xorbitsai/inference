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


class VirtualEnvManager:
    """
    Manager class for handling virtual environments.
    Extracted from worker.py to improve code organization and maintainability.
    Supports multiple Python versions per model with v3 structure:
    .xinference/virtualenv/v3/{model_name}/{python_version}/
    """

    def __init__(self, worker_address: str):
        """
        Initialize VirtualEnvManager.

        Args:
            worker_address: The address of the worker using this manager
        """
        self.worker_address = worker_address

    def list_virtual_envs(
        self, model_name: Optional[str] = None
    ) -> List[Dict[Any, Any]]:
        """
        List all virtual environments or filter by model name.

        Args:
            model_name: Optional model name to filter results

        Returns:
            List of virtual environment information dictionaries
        """

        v3_env_dir = os.path.join(XINFERENCE_VIRTUAL_ENV_DIR, "v3")
        virtual_envs = []

        if os.path.exists(v3_env_dir):
            for model_dir in os.listdir(v3_env_dir):
                model_path = os.path.join(v3_env_dir, model_dir)
                if not os.path.isdir(model_path):
                    continue

                # Apply filter if model_name is specified
                if model_name and model_dir != model_name:
                    continue

                # Check for Python version directories
                for python_version_dir in os.listdir(model_path):
                    python_version_path = os.path.join(model_path, python_version_dir)
                    if os.path.isdir(python_version_path):
                        # Validate Python version format (e.g., "3.10", "3.13")
                        if self._is_valid_python_version(python_version_dir):
                            virtual_envs.append(
                                {
                                    "model_name": model_dir,
                                    "python_version": python_version_dir,
                                    "path": python_version_path,
                                    "real_path": os.path.realpath(python_version_path),
                                }
                            )

        return virtual_envs

    def remove_virtual_env(
        self, model_name: str, python_version: Optional[str] = None
    ) -> bool:
        """
        Remove a virtual environment for a specific model.

        Args:
            model_name: Name of the model whose virtual environment should be removed
            python_version: Optional Python version to remove specific version,
                          if None, removes all Python versions for the model

        Returns:
            True if removal was successful, False otherwise
        """
        if not model_name:
            raise ValueError("model_name is required")

        try:
            from ..constants import XINFERENCE_VIRTUAL_ENV_DIR
        except ImportError:
            # Fallback for testing or when run as standalone
            XINFERENCE_VIRTUAL_ENV_DIR = os.path.join(
                os.path.expanduser("~/.xinference"), "virtualenv"
            )

        v3_env_dir = os.path.join(XINFERENCE_VIRTUAL_ENV_DIR, "v3")
        model_path = os.path.join(v3_env_dir, model_name)

        if not os.path.exists(model_path):
            logger.warning(
                f"Virtual environment path not found for model: {model_name}"
            )
            return True

        try:
            if python_version:
                # Remove specific Python version
                if not self._is_valid_python_version(python_version):
                    logger.warning(f"Invalid Python version format: {python_version}")
                    return False

                version_path = os.path.join(model_path, python_version)
                if not os.path.exists(version_path):
                    logger.warning(
                        f"Virtual environment for {model_name} Python {python_version} not found"
                    )
                    return True

                if os.path.islink(version_path):
                    os.unlink(version_path)
                elif os.path.isdir(version_path):
                    shutil.rmtree(version_path)
                else:
                    logger.warning(
                        f"Virtual environment path is not a directory: {version_path}"
                    )

                logger.info(f"Successfully removed virtual environment: {version_path}")

                # If model directory is empty, remove it too
                try:
                    if os.path.exists(model_path) and not os.listdir(model_path):
                        os.rmdir(model_path)
                        logger.info(f"Removed empty model directory: {model_path}")
                except OSError:
                    pass  # Directory not empty or other error, ignore

                return True
            else:
                # Remove all Python versions for the model
                if os.path.islink(model_path):
                    os.unlink(model_path)
                elif os.path.isdir(model_path):
                    shutil.rmtree(model_path)
                else:
                    logger.warning(
                        f"Virtual environment path is not a directory: {model_path}"
                    )

                logger.info(
                    f"Successfully removed all virtual environments for model: {model_path}"
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
