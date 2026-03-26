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
import codecs
import json
import os
import warnings

from ..utils import flatten_quantizations
from .core import (
    LLM,
    LLM_VERSION_INFOS,
    generate_llm_version_info,
    get_llm_version_infos,
)
from .custom import get_user_defined_llm_families, register_llm, unregister_llm
from .llm_family import (
    BUILTIN_LLM_FAMILIES,
    BUILTIN_LLM_MODEL_CHAT_FAMILIES,
    BUILTIN_LLM_MODEL_GENERATE_FAMILIES,
    BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES,
    BUILTIN_LLM_PROMPT_STYLE,
    LLAMA_CLASSES,
    LLM_ENGINES,
    LMDEPLOY_CLASSES,
    MLX_CLASSES,
    SGLANG_CLASSES,
    SUPPORTED_ENGINES,
    TRANSFORMERS_CLASSES,
    VLLM_CLASSES,
    CustomLLMFamilyV2,
    LlamaCppLLMSpecV2,
    LLMFamilyV2,
    LLMSpecV1,
    MLXLLMSpecV2,
    PytorchLLMSpecV2,
    match_llm,
)


def register_builtin_model():
    """Register built-in LLM models."""
    _install()


def check_format_with_engine(model_format, engine):
    # only llama-cpp-python support and only support ggufv2
    if model_format in ["ggufv2"] and engine not in ["llama.cpp", "vLLM"]:
        return False
    if model_format not in ["ggufv2"] and engine == "llama.cpp":
        return False
    return True


def generate_engine_config_by_model_family(model_family: "LLMFamilyV2"):
    model_name = model_family.model_name
    specs = model_family.model_specs
    engines = LLM_ENGINES.get(model_name, {})  # structure for engine query
    for spec in specs:
        model_format = spec.model_format
        model_size_in_billions = spec.model_size_in_billions
        quantization = spec.quantization
        # traverse all supported engines to match the name, format, size in billions and quantization of model
        for engine in SUPPORTED_ENGINES:
            if not check_format_with_engine(
                model_format, engine
            ):  # match the format of model with engine
                continue
            CLASSES = SUPPORTED_ENGINES[engine]
            for cls in CLASSES:
                if cls.match(model_family, spec, quantization):
                    engine_params = engines.get(engine, [])
                    already_exists = False
                    # if the name, format and size in billions of model already exists in the structure, add the new quantization
                    for param in engine_params:
                        if (
                            model_name == param["model_name"]
                            and model_format == param["model_format"]
                            and model_size_in_billions
                            == param["model_size_in_billions"]
                        ):
                            if quantization not in param["quantizations"]:
                                param["quantizations"].append(quantization)
                            if "multimodal_projectors" not in param and hasattr(
                                spec, "multimodal_projectors"
                            ):
                                param["multimodal_projectors"] = (
                                    spec.multimodal_projectors
                                )
                            already_exists = True
                            break
                    # successfully match the params for the first time, add to the structure
                    if not already_exists:
                        engine_params.append(
                            {
                                "model_name": model_name,
                                "model_format": model_format,
                                "model_size_in_billions": model_size_in_billions,
                                "quantizations": [quantization],
                                "llm_class": cls,
                            }
                        )
                        if hasattr(spec, "multimodal_projectors"):
                            engine_params[-1][
                                "multimodal_projectors"
                            ] = spec.multimodal_projectors
                    engines[engine] = engine_params
                    break
    LLM_ENGINES[model_name] = engines


def register_custom_model():
    from ...constants import XINFERENCE_MODEL_DIR
    from ..custom import migrate_from_v1_to_v2

    # migrate from v1 to v2 first
    migrate_from_v1_to_v2("llm", CustomLLMFamilyV2)

    user_defined_llm_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "llm")
    if os.path.isdir(user_defined_llm_dir):
        for f in os.listdir(user_defined_llm_dir):
            try:
                with codecs.open(
                    os.path.join(user_defined_llm_dir, f), encoding="utf-8"
                ) as fd:
                    user_defined_llm_family = CustomLLMFamilyV2.parse_raw(fd.read())
                    register_llm(user_defined_llm_family, persist=False)
            except Exception as e:
                warnings.warn(f"{user_defined_llm_dir}/{f} has error, {e}")


def has_downloaded_models():
    """Check if downloaded JSON configurations exist."""
    from ...constants import XINFERENCE_MODEL_DIR

    builtin_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "builtin", "llm")
    json_file_path = os.path.join(builtin_dir, "llm_models.json")
    return os.path.exists(json_file_path)


def load_downloaded_models():
    """Load downloaded JSON configurations from the builtin directory."""
    from ...constants import XINFERENCE_MODEL_DIR

    builtin_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "builtin", "llm")
    json_file_path = os.path.join(builtin_dir, "llm_models.json")

    try:
        load_model_family_from_json(json_file_path, BUILTIN_LLM_FAMILIES)
    except Exception as e:
        warnings.warn(
            f"Failed to load downloaded llm models from {json_file_path}: {e}"
        )
        # Fall back to built-in models if download fails
        load_model_family_from_json("llm_family.json", BUILTIN_LLM_FAMILIES)


def load_model_family_from_json(json_filename, target_families):
    # Handle both relative (module directory) and absolute paths
    if os.path.isabs(json_filename):
        json_path = json_filename
    else:
        json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), json_filename
        )

    for json_obj in json.load(codecs.open(json_path, "r", encoding="utf-8")):
        flattened = []
        for spec in json_obj["model_specs"]:
            flattened.extend(flatten_quantizations(spec))
        json_obj["model_specs"] = flattened
        model_spec = LLMFamilyV2.parse_obj(json_obj)
        target_families.append(model_spec)

        # register chat_template
        if (
            "chat" in model_spec.model_ability
            and isinstance(model_spec.chat_template, str)
            and model_spec.model_name not in BUILTIN_LLM_PROMPT_STYLE
        ):
            # note that the key is the model name,
            # since there are multiple representations of the same prompt style name in json.
            if model_spec.model_name not in BUILTIN_LLM_PROMPT_STYLE:
                BUILTIN_LLM_PROMPT_STYLE[model_spec.model_name] = {
                    "chat_template": model_spec.chat_template,
                    "stop_token_ids": model_spec.stop_token_ids,
                    "stop": model_spec.stop,
                }
                if model_spec.reasoning_start_tag and model_spec.reasoning_end_tag:
                    BUILTIN_LLM_PROMPT_STYLE[model_spec.model_name][
                        "reasoning_start_tag"
                    ] = model_spec.reasoning_start_tag
                    BUILTIN_LLM_PROMPT_STYLE[model_spec.model_name][
                        "reasoning_end_tag"
                    ] = model_spec.reasoning_end_tag
                if model_spec.tool_parser:
                    BUILTIN_LLM_PROMPT_STYLE[model_spec.model_name][
                        "tool_parser"
                    ] = model_spec.tool_parser

        # register model family
        if "chat" in model_spec.model_ability:
            BUILTIN_LLM_MODEL_CHAT_FAMILIES.add(model_spec.model_name)
        else:
            BUILTIN_LLM_MODEL_GENERATE_FAMILIES.add(model_spec.model_name)
        if "tools" in model_spec.model_ability:
            BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES.add(model_spec.model_name)


def _install():
    from .llama_cpp.core import XllamaCppModel
    from .lmdeploy.core import LMDeployChatModel, LMDeployModel
    from .mlx.core import MLXChatModel, MLXModel, MLXVisionModel
    from .sglang.core import SGLANGChatModel, SGLANGModel, SGLANGVisionModel
    from .transformers.core import PytorchChatModel, PytorchModel
    from .vllm.core import VLLMChatModel, VLLMModel, VLLMMultiModel

    # register llm classes.
    LLAMA_CLASSES.extend([XllamaCppModel])
    SGLANG_CLASSES.extend([SGLANGModel, SGLANGChatModel, SGLANGVisionModel])
    VLLM_CLASSES.extend([VLLMModel, VLLMChatModel, VLLMMultiModel])
    MLX_CLASSES.extend([MLXModel, MLXChatModel, MLXVisionModel])
    LMDEPLOY_CLASSES.extend([LMDeployModel, LMDeployChatModel])
    TRANSFORMERS_CLASSES.extend([PytorchChatModel, PytorchModel])

    # support 4 engines for now
    SUPPORTED_ENGINES["vLLM"] = VLLM_CLASSES
    SUPPORTED_ENGINES["SGLang"] = SGLANG_CLASSES
    SUPPORTED_ENGINES["Transformers"] = TRANSFORMERS_CLASSES
    SUPPORTED_ENGINES["llama.cpp"] = LLAMA_CLASSES
    SUPPORTED_ENGINES["MLX"] = MLX_CLASSES
    SUPPORTED_ENGINES["LMDEPLOY"] = LMDEPLOY_CLASSES

    # Install models with intelligent merging based on timestamps
    # LLM models use a different structure (list instead of dict), so we need special handling

    # Always load built-in models first to ensure we have the latest models
    load_model_family_from_json("llm_family.json", BUILTIN_LLM_FAMILIES)

    # Then load user-defined models and merge with built-in models
    if has_downloaded_models():
        user_models = []
        from ..utils import load_downloaded_models_to_dict

        load_downloaded_models_to_dict(
            {"temp": user_models},
            "llm",
            "llm_models.json",
            lambda path, target: load_model_family_from_json(path, target["temp"]),
        )

        if user_models:
            # Create a copy of built-in models for merging
            built_in_models_copy = list(BUILTIN_LLM_FAMILIES)

            # Merge models, keeping the latest version based on updated_at
            all_models = built_in_models_copy + user_models

            # Sort by updated_at (newest first) and keep the latest for each model name
            all_models.sort(key=lambda x: x.updated_at, reverse=True)

            # Remove duplicates, keeping the first (newest) occurrence of each model name
            seen_models = set()
            merged_models = []

            for model in all_models:
                if model.model_name not in seen_models:
                    seen_models.add(model.model_name)
                    merged_models.append(model)

            # Update BUILTIN_LLM_FAMILIES with merged results
            BUILTIN_LLM_FAMILIES.clear()
            BUILTIN_LLM_FAMILIES.extend(merged_models)

    for family in BUILTIN_LLM_FAMILIES:
        if family.model_name not in LLM_VERSION_INFOS:
            LLM_VERSION_INFOS.update(generate_llm_version_info(family))

    # traverse all families and add engine parameters corresponding to the model name
    for family in BUILTIN_LLM_FAMILIES:
        generate_engine_config_by_model_family(family)

    register_custom_model()

    # register model description
    for ud_llm in get_user_defined_llm_families():
        LLM_VERSION_INFOS.update(generate_llm_version_info(ud_llm))
