# Copyright 2022-2025 XProbe Inc.
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

logger = logging.getLogger(__name__)


class VirtualEnvManager:
    """
    Manager class for handling virtual environments.
    Extracted from worker.py to improve code organization and maintainability.
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
        try:
            from ..constants import XINFERENCE_VIRTUAL_ENV_DIR
        except ImportError:
            # Fallback for testing or when run as standalone
            XINFERENCE_VIRTUAL_ENV_DIR = os.path.join(
                os.path.expanduser("~/.xinference"), "virtualenv"
            )

        v2_env_dir = os.path.join(XINFERENCE_VIRTUAL_ENV_DIR, "v2")
        virtual_envs = []

        if os.path.exists(v2_env_dir):
            for env_name in os.listdir(v2_env_dir):
                env_path = os.path.join(v2_env_dir, env_name)
                if os.path.isdir(env_path):
                    # Apply filter if model_name is specified
                    if model_name and env_name != model_name:
                        continue

                    # Detect Python version
                    python_version = self._detect_python_version(env_path)

                    virtual_envs.append(
                        {
                            "model_name": env_name,
                            "path": env_path,
                            "real_path": os.path.realpath(env_path),
                            "python_version": python_version,
                        }
                    )

        return virtual_envs

    def remove_virtual_env(self, model_name: str) -> bool:
        """
        Remove a virtual environment for a specific model.

        Args:
            model_name: Name of the model whose virtual environment should be removed

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

        v2_env_dir = os.path.join(XINFERENCE_VIRTUAL_ENV_DIR, "v2")
        env_path = os.path.join(v2_env_dir, model_name)

        if not os.path.exists(env_path):
            logger.warning(
                f"Virtual environment path not found for model: {model_name}"
            )
            return True

        # Remove the virtual environment directory
        try:
            if os.path.islink(env_path):
                os.unlink(env_path)
            elif os.path.isdir(env_path):
                shutil.rmtree(env_path)
            else:
                logger.warning(
                    f"Virtual environment path is not a directory: {env_path}"
                )

            logger.info(f"Successfully removed virtual environment: {env_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove virtual environment {env_path}: {e}")
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
