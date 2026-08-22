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
import codecs
import json
import os

from ..utils import flatten_model_src
from .core import (
    BUILTIN_WORLD_MODELS,
    WORLD_MODEL_DESCRIPTIONS,
    WorldModelFamilyV1,
    generate_world_description,
    get_world_model_descriptions,
)
from .engine import register_builtin_world_engines
from .engine_family import WORLD_ENGINES, generate_engine_config_by_model_name


class CustomWorldModelFamilyV1(WorldModelFamilyV1):
    """Compatibility placeholder for the custom registration interface."""


def register_world(model_family, persist=True):
    raise NotImplementedError("Custom world models are not supported yet")


def unregister_world(model_name, version=None):
    raise NotImplementedError("Custom world models are not supported yet")


def register_builtin_model():
    _install()


def _install():
    from ..utils import install_models_with_merge

    BUILTIN_WORLD_MODELS.clear()
    WORLD_MODEL_DESCRIPTIONS.clear()
    WORLD_ENGINES.clear()
    install_models_with_merge(
        BUILTIN_WORLD_MODELS,
        "model_spec.json",
        "world",
        "world_models.json",
        has_downloaded_models,
        load_model_family_from_json,
        model_identity_func=lambda model: (
            model.model_hub,
            model.model_name,
        ),
    )
    for model_name, model_specs in BUILTIN_WORLD_MODELS.items():
        WORLD_MODEL_DESCRIPTIONS[model_name] = [
            model_spec.to_version_info() for model_spec in model_specs
        ]

    register_builtin_world_engines()
    for model_specs in BUILTIN_WORLD_MODELS.values():
        for model_spec in model_specs:
            generate_engine_config_by_model_name(model_spec)


def has_downloaded_models():
    from ...constants import XINFERENCE_MODEL_DIR

    return os.path.exists(
        os.path.join(
            XINFERENCE_MODEL_DIR, "v2", "builtin", "world", "world_models.json"
        )
    )


def load_model_family_from_json(json_filename, target_families):
    if os.path.isabs(json_filename):
        json_path = json_filename
    else:
        json_path = os.path.join(os.path.dirname(__file__), json_filename)

    flattened_model_specs = []
    with codecs.open(json_path, "r", encoding="utf-8") as fd:
        for spec in json.load(fd):
            flattened_model_specs.extend(flatten_model_src(spec))

    for spec in flattened_model_specs:
        target_families.setdefault(spec["model_name"], []).append(
            WorldModelFamilyV1(**spec)
        )
