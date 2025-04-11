# Copyright 2022-2023 XProbe Inc.
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

# Get the path of the current package
__path__ = [os.path.dirname(os.path.abspath(__file__))]

# Automatically import all modules under the current package
for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if not module_name.startswith("_"):  # Skip modules starting with underscore
        module = importlib.import_module(f"{__name__}.{module_name}")
        globals()[module_name] = module
