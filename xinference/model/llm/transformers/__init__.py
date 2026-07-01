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

# Automatically scan and import all python scripts at the same level
import importlib
import os
import pkgutil
from typing import Dict


def import_submodules(package_path: str, package_name: str, globals_dict: Dict) -> None:
    """
    Recursively import all classes in submodules and subpackages
    """
    for _, module_name, is_pkg in pkgutil.iter_modules([package_path]):
        full_module_name = f"{package_name}.{module_name}"

        if module_name.startswith(
            ("_", "test_")
        ):  # Skip the modules which start with "_" or "test_"
            continue

        module = importlib.import_module(full_module_name)
        globals_dict[module_name] = module

        # If it's a pkg, recursive processing
        if is_pkg:
            subpackage_path = os.path.join(package_path, module_name)
            import_submodules(subpackage_path, full_module_name, globals_dict)


# Get the path and name of the current package
__path__ = [os.path.dirname(os.path.abspath(__file__))]
__package__ = __name__

# Automatic import of all sub-modules and sub-packages
import_submodules(__path__[0], __package__, globals())
