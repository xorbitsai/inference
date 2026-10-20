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

import importlib
import os
from typing import Set

_LAUNCHER_ALLOWLIST_ENV = "XINFERENCE_FLEXIBLE_LAUNCHERS"
_BUILTIN_LAUNCHERS = {
    "xinference.model.flexible.launchers.transformers",
    "xinference.model.flexible.launchers.modelscope",
    "xinference.model.flexible.launchers.yolo",
    "xinference.model.flexible.launchers.image_process",
}


def _get_launcher_allowlist() -> Set[str]:
    raw = os.environ.get(_LAUNCHER_ALLOWLIST_ENV, "")
    return _BUILTIN_LAUNCHERS | {
        name.strip() for name in raw.split(",") if name.strip()
    }


def get_launcher(launcher_name: str):
    if launcher_name not in _get_launcher_allowlist():
        raise ValueError(
            f"Launcher {launcher_name!r} is not permitted. Add its exact dotted "
            f"name to the {_LAUNCHER_ALLOWLIST_ENV} environment variable "
            "(comma-separated) to allow it."
        )
    try:
        i = launcher_name.rfind(".")
        if i != -1:
            module = importlib.import_module(launcher_name[:i])
            fn = getattr(module, launcher_name[i + 1 :])
        else:
            importlib.import_module(launcher_name)
            fn = locals().get(launcher_name)

        if fn is None:
            raise ValueError(f"Launcher {launcher_name} not found.")

        return fn
    except ImportError as e:
        raise ImportError(f"Failed to import {launcher_name}: {e}")
