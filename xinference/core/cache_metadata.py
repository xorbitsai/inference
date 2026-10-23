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
from typing import Any, Dict, List, Literal, Optional, cast


def tag_model_type(
    version_info: Dict[str, List[Dict]], model_type: str
) -> Dict[str, List[Dict]]:
    for versions in version_info.values():
        for version in versions:
            version["model_type"] = model_type
    return version_info


def get_model_version_infos() -> Dict[str, List[Dict]]:
    from ..model.audio import get_audio_model_descriptions
    from ..model.embedding import get_embedding_model_descriptions
    from ..model.flexible import get_flexible_model_descriptions
    from ..model.image import get_image_model_descriptions
    from ..model.llm import get_llm_version_infos
    from ..model.rerank import get_rerank_model_descriptions
    from ..model.video import get_video_model_descriptions
    from ..model.world import get_world_model_descriptions

    result: Dict[str, List[Dict]] = {}
    for model_type, get_versions in (
        ("LLM", get_llm_version_infos),
        ("embedding", get_embedding_model_descriptions),
        ("rerank", get_rerank_model_descriptions),
        ("image", get_image_model_descriptions),
        ("audio", get_audio_model_descriptions),
        ("video", get_video_model_descriptions),
        ("world", get_world_model_descriptions),
        ("flexible", get_flexible_model_descriptions),
    ):
        result.update(tag_model_type(get_versions(), model_type))
    return result


def get_cached_model_engines(
    model_name: str,
    model_type: str,
    model_version: str,
    engine_params: Optional[Dict[str, Any]],
) -> Optional[List[str]]:
    """Match a cached artifact against discovery on the worker owning its files."""
    if engine_params is None:
        return None
    engines = []
    for engine, params in engine_params.items():
        # Unavailable engines contain an explanatory string instead of specs.
        if not isinstance(params, list):
            continue
        for param in params:
            if param.get("model_name") != model_name:
                continue
            if model_type == "LLM":
                from ..model.llm.utils import get_model_version

                versions = [
                    get_model_version(
                        model_name,
                        param["model_format"],
                        param["model_size_in_billions"],
                        quantization,
                    )
                    for quantization in param["quantizations"]
                ]
                matches = model_version in versions
            elif model_type == "embedding":
                # Engine entries identify format and quantization; token limit
                # and vector dimensions belong to the model, not the engine.
                suffix = f"--{param['model_format']}--{param['quantization']}"
                matches = model_version.startswith(
                    f"{model_name}--"
                ) and model_version.endswith(suffix)
            elif model_type == "audio":
                from ..model.audio import BUILTIN_AUDIO_MODELS
                from ..model.audio.core import match_audio
                from ..model.audio.custom import get_user_defined_audios

                families = list(BUILTIN_AUDIO_MODELS.get(model_name, []))
                families.extend(
                    family
                    for family in get_user_defined_audios()
                    if family.model_name == model_name
                )
                matches = False
                for candidate in families:
                    if model_version != (candidate.cache_name or candidate.model_name):
                        continue
                    try:
                        family = match_audio(
                            model_name,
                            download_hub=cast(
                                Literal[
                                    "huggingface",
                                    "modelscope",
                                    "openmind_hub",
                                    "csghub",
                                ],
                                candidate.model_hub,
                            ),
                            model_engine=engine,
                            quantization=candidate.quantization,
                        )
                    except ValueError:
                        # Discovery can offer an engine without this quantization.
                        continue
                    if model_version == (family.cache_name or family.model_name):
                        matches = True
                        break
            elif model_type == "video":
                from ..model.video import BUILTIN_VIDEO_MODELS
                from ..model.video.core import VIDEO_REGISTRY_LOCK

                # Cache metadata must not depend on the selected download hub.
                with VIDEO_REGISTRY_LOCK:
                    families = list(BUILTIN_VIDEO_MODELS.get(model_name, []))
                matches = any(
                    (family.engine or "").lower() == engine.lower()
                    and model_version == (family.cache_name or family.model_name)
                    for family in families
                )
            elif model_type == "image":
                # ControlNet versions share the base model's engine.
                matches = model_version == model_name or model_version.startswith(
                    f"{model_name}--"
                )
            else:
                matches = model_version == model_name
            if matches:
                engines.append(engine)
                break
    return engines
