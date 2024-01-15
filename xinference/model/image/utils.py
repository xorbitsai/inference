# Copyright 2022-2024 XProbe Inc.
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

from collections import defaultdict
from typing import Dict, List

from .core import ImageModelFamilyV1


def get_launch_version(image_model: ImageModelFamilyV1) -> Dict[str, List[str]]:
    res = defaultdict(list)
    res[image_model.model_name].append(image_model.model_name)
    if image_model.controlnet is not None:
        # TODO: currently, support one controlnet
        for cn in image_model.controlnet:
            res[image_model.model_name].append(
                f"{image_model.model_name}--{cn.model_name}"
            )
    return res
