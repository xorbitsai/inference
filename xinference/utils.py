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

import os
import subprocess
import sys
from typing import Optional


def cuda_count():
    import torch

    # even if install torch cpu, this interface would return 0.
    return torch.cuda.device_count()


def get_real_path(path: str) -> Optional[str]:
    # parsing soft links
    if os.path.isdir(path):
        files = os.listdir(path)
        # dir has files
        if files:
            resolved_file = os.path.realpath(os.path.join(path, files[0]))
            if resolved_file:
                return os.path.dirname(resolved_file)
        return None
    else:
        return os.path.realpath(path)


def get_pip_config_args() -> dict[str, str]:
    """
    Parse pip config and return a dict with keys matching install_packages kwargs:
    index_url, extra_index_url, find_links, trusted_host.
    """
    key_map = {
        "global.index-url": "index_url",
        "global.extra-index-url": "extra_index_url",
        "global.trusted-host": "trusted_host",
        "global.find-links": "find_links",
    }

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "config", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        args: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" not in line:
                continue
            raw_key, raw_value = line.split("=", 1)
            key = raw_key.strip()
            value = raw_value.strip().strip("'\"")
            mapped_key = key_map.get(key)
            if mapped_key and value:
                args[mapped_key] = value

        return args
    except subprocess.CalledProcessError:
        return {}
