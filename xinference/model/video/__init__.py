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
import warnings

from ...constants import XINFERENCE_MODEL_DIR
from ..utils import flatten_model_src
from .core import (
    BUILTIN_VIDEO_MODELS,
    VIDEO_MODEL_DESCRIPTIONS,
    VIDEO_REGISTRY_LOCK,
    VideoModelFamilyV2,
    generate_video_description,
    get_video_model_descriptions,
)
from .engine import register_builtin_video_engines
from .engine_family import VIDEO_ENGINES, generate_engine_config_by_model_name


def register_custom_model():
    """Register custom video models."""
    # Video models don't support custom models yet
    pass


# For compatibility with worker's custom registration system
class CustomVideoModelFamilyV2(VideoModelFamilyV2):
    """Custom video model family, currently not supported."""

    pass


def register_video(model_family, persist=True):
    """Register a video model family. Currently not supported."""
    # Video models don't support custom registration yet
    pass


def unregister_video(model_name, version=None):
    """Unregister a video model family. Currently not supported."""
    # Video models don't support custom registration yet
    pass


def register_builtin_model():
    """Register built-in video models."""
    _install()


def _install():
    # Install models with intelligent merging based on timestamps
    from ..utils import install_models_with_merge

    # Build a complete replacement away from the live registries. A reload can
    # overlap model discovery/launch, and readers must never observe an empty or
    # partially rebuilt set (nor lose the previous set if rebuilding fails).
    new_builtin_video_models = {}
    new_video_model_descriptions = {}
    new_video_engines = {}

    install_models_with_merge(
        new_builtin_video_models,
        "model_spec.json",
        "video",
        "video_models.json",
        has_downloaded_models,
        load_model_family_from_json,
        model_identity_func=lambda model: (
            (model.engine or "diffusers").lower(),
            model.model_hub,
            model.cache_name or model.model_name,
        ),
    )

    # Register one cache/version entry per engine variant. Hugging Face is the
    # preferred representative when the same variant has multiple hubs.
    for model_name, model_specs in new_builtin_video_models.items():
        variants = {}
        for model_spec in model_specs:
            version = model_spec.cache_name or model_spec.model_name
            current = variants.get(version)
            if current is None or model_spec.model_hub == "huggingface":
                variants[version] = model_spec
        new_video_model_descriptions[model_name] = [
            model_spec.to_version_info() for model_spec in variants.values()
        ]

    register_builtin_video_engines()
    for model_specs in new_builtin_video_models.values():
        for model_spec in model_specs:
            generate_engine_config_by_model_name(
                model_spec, target_engines=new_video_engines
            )

    register_custom_model()

    # Keep the exported dictionaries stable for compatibility with existing
    # imports, but serialize the in-place publication with all registry readers.
    with VIDEO_REGISTRY_LOCK:
        BUILTIN_VIDEO_MODELS.clear()
        BUILTIN_VIDEO_MODELS.update(new_builtin_video_models)
        VIDEO_MODEL_DESCRIPTIONS.clear()
        VIDEO_MODEL_DESCRIPTIONS.update(new_video_model_descriptions)
        VIDEO_ENGINES.clear()
        VIDEO_ENGINES.update(new_video_engines)


def has_downloaded_models():
    """Check if downloaded JSON configurations exist."""
    builtin_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "builtin", "video")
    json_file_path = os.path.join(builtin_dir, "video_models.json")
    return os.path.exists(json_file_path)


def load_downloaded_models():
    """Load downloaded JSON configurations from the builtin directory."""
    builtin_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "builtin", "video")
    json_file_path = os.path.join(builtin_dir, "video_models.json")

    try:
        load_model_family_from_json(json_file_path, BUILTIN_VIDEO_MODELS)
    except Exception as e:
        warnings.warn(
            f"Failed to load downloaded video models from {json_file_path}: {e}"
        )
        # Fall back to built-in models if download fails
        load_model_family_from_json("model_spec.json", BUILTIN_VIDEO_MODELS)


def load_model_family_from_json(json_filename, target_families):
    # Handle both relative (module directory) and absolute paths
    if os.path.isabs(json_filename):
        json_path = json_filename
    else:
        json_path = os.path.join(os.path.dirname(__file__), json_filename)

    flattened_model_specs = []
    for spec in json.load(codecs.open(json_path, "r", encoding="utf-8")):
        flattened_model_specs.extend(flatten_model_src(spec))

    for spec in flattened_model_specs:
        if spec["model_name"] not in target_families:
            target_families[spec["model_name"]] = [VideoModelFamilyV2(**spec)]
        else:
            target_families[spec["model_name"]].append(VideoModelFamilyV2(**spec))

    del json_path
