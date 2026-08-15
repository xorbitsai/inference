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

    # Startup and tests may call the installer more than once in one process.
    # Rebuild the derived registries to avoid duplicate model and engine specs.
    BUILTIN_VIDEO_MODELS.clear()
    VIDEO_MODEL_DESCRIPTIONS.clear()
    VIDEO_ENGINES.clear()

    install_models_with_merge(
        BUILTIN_VIDEO_MODELS,
        "model_spec.json",
        "video",
        "video_models.json",
        has_downloaded_models,
        load_model_family_from_json,
    )

    # Register one cache/version entry per engine variant. Hugging Face is the
    # preferred representative when the same variant has multiple hubs.
    for model_name, model_specs in BUILTIN_VIDEO_MODELS.items():
        variants = {}
        for model_spec in model_specs:
            version = model_spec.cache_name or model_spec.model_name
            current = variants.get(version)
            if current is None or model_spec.model_hub == "huggingface":
                variants[version] = model_spec
        VIDEO_MODEL_DESCRIPTIONS[model_name] = [
            model_spec.to_version_info() for model_spec in variants.values()
        ]

    register_builtin_video_engines()
    for model_specs in BUILTIN_VIDEO_MODELS.values():
        for model_spec in model_specs:
            generate_engine_config_by_model_name(model_spec)

    register_custom_model()


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
