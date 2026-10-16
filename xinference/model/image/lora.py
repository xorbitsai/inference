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

import logging
import os
import re
from typing import List, Tuple

from ...constants import XINFERENCE_CACHE_DIR

XINFERENCE_IMG_LORA_DIR = os.path.join(XINFERENCE_CACHE_DIR, "image", "lora")
from ...types import LoRA

logger = logging.getLogger(__name__)


def process_prompt(prompt) -> Tuple[List[Tuple[str, float]], str]:
    pattern = r"<lora:([^:>]+)(?::([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)))?>,?\s*"
    matches = re.findall(pattern, prompt)
    lora_info = []
    for match in matches:
        lora_name = match[0]
        lora_weight = float(match[1]) if match[1] else 1.0  # default weight 1.0
        lora_info.append((lora_name, lora_weight))
    cleaned_prompt = re.sub(pattern, "", prompt)
    return lora_info, cleaned_prompt


def available_loras():
    from .core import BUILTIN_IMAGE_MODELS
    from .custom import get_user_defined_images

    candidates = [spec for specs in BUILTIN_IMAGE_MODELS.values() for spec in specs]
    candidates.extend(get_user_defined_images())
    return [spec for spec in candidates if spec.model_family == "lora"]


def process_loras(kwargs: dict, strict: bool = False, specs=None):
    from .cache_manager import ImageCacheManager
    from .utils import download_civitai_model

    names, cleaned_prompt = process_prompt(kwargs.get("prompt", ""))
    loras = []
    for name, weight in names:
        spec = next(
            (
                spec
                for spec in (available_loras() if specs is None else specs)
                if name
                in (
                    spec.model_name,
                    (getattr(spec, "metadata", None) or {}).get("ss_output_name"),
                )
            ),
            None,
        )
        if spec is None:
            if strict:
                raise ValueError(f"LoRA not found: {name}")
            logger.warning("LoRA not found: %s", name)
            continue
        uri = getattr(spec, "model_uri", None)
        if uri and uri.startswith("https://civitai.com/"):
            from .utils import make_valid_filename

            directory = os.path.join(
                XINFERENCE_IMG_LORA_DIR, make_valid_filename(spec.model_name)
            )
            os.makedirs(directory, exist_ok=True)
            path = download_civitai_model(
                uri, directory, os.environ.get("CIVITAI_API_TOKEN", "")
            )
        elif uri and os.path.exists(uri):
            path = uri
        else:
            path = ImageCacheManager(spec).cache()
        if os.path.isdir(path):
            candidates = list(__import__("pathlib").Path(path).glob("*.safetensors"))
            if len(candidates) != 1:
                raise ValueError(f"LoRA {name} needs an unambiguous safetensors file")
            path = str(candidates[0])
        loras.append(LoRA(name, path, lora_scale=weight))
    kwargs["prompt"] = cleaned_prompt
    if names:
        kwargs["loras"] = loras
