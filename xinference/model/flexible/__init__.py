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

import codecs
import json
import logging
import os
import warnings

from ...constants import XINFERENCE_MODEL_DIR
from .core import (
    FLEXIBLE_MODEL_DESCRIPTIONS,
    FlexibleModel,
    FlexibleModelSpec,
    generate_flexible_model_description,
    get_flexible_model_descriptions,
    get_flexible_models,
    register_flexible_model,
    unregister_flexible_model,
)

logger = logging.getLogger(__name__)


def register_custom_model():
    model_dir = os.path.join(XINFERENCE_MODEL_DIR, "flexible")
    if os.path.isdir(model_dir):
        for f in os.listdir(model_dir):
            try:
                with codecs.open(os.path.join(model_dir, f), encoding="utf-8") as fd:
                    model_spec = FlexibleModelSpec.parse_obj(json.load(fd))
                    register_flexible_model(model_spec, persist=False)
            except Exception as e:
                warnings.warn(f"{model_dir}/{f} has error, {e}")


def _install():
    register_custom_model()

    # register model description
    for model in get_flexible_models():
        FLEXIBLE_MODEL_DESCRIPTIONS.update(generate_flexible_model_description(model))
