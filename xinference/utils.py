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
