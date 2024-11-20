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

import codecs
import json
import os
import warnings

from .core import (
    LLM,
    LLM_MODEL_DESCRIPTIONS,
    LLMDescription,
    generate_llm_description,
    get_llm_model_descriptions,
)
from .llm_family import (
    BUILTIN_CSGHUB_LLM_FAMILIES,
    BUILTIN_LLM_FAMILIES,
    BUILTIN_LLM_MODEL_CHAT_FAMILIES,
    BUILTIN_LLM_MODEL_GENERATE_FAMILIES,
    BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES,
    BUILTIN_LLM_PROMPT_STYLE,
    BUILTIN_MODELSCOPE_LLM_FAMILIES,
    BUILTIN_OPENMIND_HUB_LLM_FAMILIES,
    LLAMA_CLASSES,
    LLM_ENGINES,
    LMDEPLOY_CLASSES,
    MLX_CLASSES,
    SGLANG_CLASSES,
    SUPPORTED_ENGINES,
    TRANSFORMERS_CLASSES,
    VLLM_CLASSES,
    CustomLLMFamilyV1,
    LlamaCppLLMSpecV1,
    LLMFamilyV1,
    LLMSpecV1,
    MLXLLMSpecV1,
    PytorchLLMSpecV1,
    get_cache_status,
    get_user_defined_llm_families,
    match_llm,
    register_llm,
    unregister_llm,
)


def check_format_with_engine(model_format, engine):
    # only llama-cpp-python support and only support ggufv2
    if model_format in ["ggufv2"] and engine != "llama.cpp":
        return False
    if model_format not in ["ggufv2"] and engine == "llama.cpp":
        return False
    return True


def generate_engine_config_by_model_family(model_family):
    model_name = model_family.model_name
    specs = model_family.model_specs
    engines = LLM_ENGINES.get(model_name, {})  # structure for engine query
    for spec in specs:
        model_format = spec.model_format
        model_size_in_billions = spec.model_size_in_billions
        quantizations = spec.quantizations
        for quantization in quantizations:
            # traverse all supported engines to match the name, format, size in billions and quatization of model
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
                        engines[engine] = engine_params
                        break
    LLM_ENGINES[model_name] = engines


def register_custom_model():
    from ...constants import XINFERENCE_MODEL_DIR

    user_defined_llm_dir = os.path.join(XINFERENCE_MODEL_DIR, "llm")
    if os.path.isdir(user_defined_llm_dir):
        for f in os.listdir(user_defined_llm_dir):
            try:
                with codecs.open(
                    os.path.join(user_defined_llm_dir, f), encoding="utf-8"
                ) as fd:
                    user_defined_llm_family = CustomLLMFamilyV1.parse_raw(fd.read())
                    register_llm(user_defined_llm_family, persist=False)
            except Exception as e:
                warnings.warn(f"{user_defined_llm_dir}/{f} has error, {e}")


def _install():
    from .llama_cpp.core import LlamaCppChatModel, LlamaCppModel
    from .lmdeploy.core import LMDeployChatModel, LMDeployModel
    from .mlx.core import MLXChatModel, MLXModel
    from .sglang.core import SGLANGChatModel, SGLANGModel
    from .transformers.chatglm import ChatglmPytorchChatModel
    from .transformers.cogvlm2 import CogVLM2Model
    from .transformers.cogvlm2_video import CogVLM2VideoModel
    from .transformers.core import PytorchChatModel, PytorchModel
    from .transformers.deepseek_v2 import (
        DeepSeekV2PytorchChatModel,
        DeepSeekV2PytorchModel,
    )
    from .transformers.deepseek_vl import DeepSeekVLChatModel
    from .transformers.glm4v import Glm4VModel
    from .transformers.intern_vl import InternVLChatModel
    from .transformers.internlm2 import Internlm2PytorchChatModel
    from .transformers.minicpmv25 import MiniCPMV25Model
    from .transformers.minicpmv26 import MiniCPMV26Model
    from .transformers.opt import OptPytorchModel
    from .transformers.qwen2_audio import Qwen2AudioChatModel
    from .transformers.qwen2_vl import Qwen2VLChatModel
    from .transformers.qwen_vl import QwenVLChatModel
    from .transformers.yi_vl import YiVLChatModel
    from .vllm.core import VLLMChatModel, VLLMModel, VLLMVisionModel

    try:
        from .transformers.omnilmm import OmniLMMModel
    except ImportError as e:
        # For quite old transformers version,
        # import will generate error
        OmniLMMModel = None
        warnings.warn(f"Cannot import OmniLLMModel due to reason: {e}")

    # register llm classes.
    LLAMA_CLASSES.extend(
        [
            LlamaCppChatModel,
            LlamaCppModel,
        ]
    )
    SGLANG_CLASSES.extend([SGLANGModel, SGLANGChatModel])
    VLLM_CLASSES.extend([VLLMModel, VLLMChatModel, VLLMVisionModel])
    MLX_CLASSES.extend([MLXModel, MLXChatModel])
    LMDEPLOY_CLASSES.extend([LMDeployModel, LMDeployChatModel])
    TRANSFORMERS_CLASSES.extend(
        [
            ChatglmPytorchChatModel,
            PytorchChatModel,
            Internlm2PytorchChatModel,
            QwenVLChatModel,
            Qwen2VLChatModel,
            Qwen2AudioChatModel,
            YiVLChatModel,
            DeepSeekVLChatModel,
            InternVLChatModel,
            PytorchModel,
            CogVLM2Model,
            CogVLM2VideoModel,
            MiniCPMV25Model,
            MiniCPMV26Model,
            Glm4VModel,
            DeepSeekV2PytorchModel,
            DeepSeekV2PytorchChatModel,
            OptPytorchModel,
        ]
    )
    if OmniLMMModel:  # type: ignore
        TRANSFORMERS_CLASSES.append(OmniLMMModel)

    # support 4 engines for now
    SUPPORTED_ENGINES["vLLM"] = VLLM_CLASSES
    SUPPORTED_ENGINES["SGLang"] = SGLANG_CLASSES
    SUPPORTED_ENGINES["Transformers"] = TRANSFORMERS_CLASSES
    SUPPORTED_ENGINES["llama.cpp"] = LLAMA_CLASSES
    SUPPORTED_ENGINES["MLX"] = MLX_CLASSES
    SUPPORTED_ENGINES["LMDEPLOY"] = LMDEPLOY_CLASSES

    json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "llm_family.json"
    )
    for json_obj in json.load(codecs.open(json_path, "r", encoding="utf-8")):
        model_spec = LLMFamilyV1.parse_obj(json_obj)
        BUILTIN_LLM_FAMILIES.append(model_spec)

        # register chat_template
        if "chat" in model_spec.model_ability and isinstance(
            model_spec.chat_template, str
        ):
            # note that the key is the model name,
            # since there are multiple representations of the same prompt style name in json.
            BUILTIN_LLM_PROMPT_STYLE[model_spec.model_name] = {
                "chat_template": model_spec.chat_template,
                "stop_token_ids": model_spec.stop_token_ids,
                "stop": model_spec.stop,
            }
        # register model family
        if "chat" in model_spec.model_ability:
            BUILTIN_LLM_MODEL_CHAT_FAMILIES.add(model_spec.model_name)
        else:
            BUILTIN_LLM_MODEL_GENERATE_FAMILIES.add(model_spec.model_name)
        if "tools" in model_spec.model_ability:
            BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES.add(model_spec.model_name)

    modelscope_json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "llm_family_modelscope.json"
    )
    for json_obj in json.load(codecs.open(modelscope_json_path, "r", encoding="utf-8")):
        model_spec = LLMFamilyV1.parse_obj(json_obj)
        BUILTIN_MODELSCOPE_LLM_FAMILIES.append(model_spec)

        # register prompt style, in case that we have something missed
        # if duplicated with huggingface json, keep it as the huggingface style
        if (
            "chat" in model_spec.model_ability
            and isinstance(model_spec.chat_template, str)
            and model_spec.model_name not in BUILTIN_LLM_PROMPT_STYLE
        ):
            BUILTIN_LLM_PROMPT_STYLE[model_spec.model_name] = {
                "chat_template": model_spec.chat_template,
                "stop_token_ids": model_spec.stop_token_ids,
                "stop": model_spec.stop,
            }
        # register model family
        if "chat" in model_spec.model_ability:
            BUILTIN_LLM_MODEL_CHAT_FAMILIES.add(model_spec.model_name)
        else:
            BUILTIN_LLM_MODEL_GENERATE_FAMILIES.add(model_spec.model_name)
        if "tools" in model_spec.model_ability:
            BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES.add(model_spec.model_name)

    openmind_hub_json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "llm_family_openmind_hub.json"
    )
    for json_obj in json.load(
        codecs.open(openmind_hub_json_path, "r", encoding="utf-8")
    ):
        model_spec = LLMFamilyV1.parse_obj(json_obj)
        BUILTIN_OPENMIND_HUB_LLM_FAMILIES.append(model_spec)

        # register prompt style, in case that we have something missed
        # if duplicated with huggingface json, keep it as the huggingface style

        if (
            "chat" in model_spec.model_ability
            and isinstance(model_spec.chat_template, str)
            and model_spec.model_name not in BUILTIN_LLM_PROMPT_STYLE
        ):
            BUILTIN_LLM_PROMPT_STYLE[model_spec.model_name] = {
                "chat_template": model_spec.chat_template,
                "stop_token_ids": model_spec.stop_token_ids,
                "stop": model_spec.stop,
            }
        # register model family
        if "chat" in model_spec.model_ability:
            BUILTIN_LLM_MODEL_CHAT_FAMILIES.add(model_spec.model_name)
        else:
            BUILTIN_LLM_MODEL_GENERATE_FAMILIES.add(model_spec.model_name)
        if "tools" in model_spec.model_ability:
            BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES.add(model_spec.model_name)

    csghub_json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "llm_family_csghub.json"
    )
    for json_obj in json.load(codecs.open(csghub_json_path, "r", encoding="utf-8")):
        model_spec = LLMFamilyV1.parse_obj(json_obj)
        BUILTIN_CSGHUB_LLM_FAMILIES.append(model_spec)

        # register prompt style, in case that we have something missed
        # if duplicated with huggingface json, keep it as the huggingface style
        if (
            "chat" in model_spec.model_ability
            and isinstance(model_spec.chat_template, str)
            and model_spec.model_name not in BUILTIN_LLM_PROMPT_STYLE
        ):
            BUILTIN_LLM_PROMPT_STYLE[model_spec.model_name] = {
                "chat_template": model_spec.chat_template,
                "stop_token_ids": model_spec.stop_token_ids,
                "stop": model_spec.stop,
            }
        # register model family
        if "chat" in model_spec.model_ability:
            BUILTIN_LLM_MODEL_CHAT_FAMILIES.add(model_spec.model_name)
        else:
            BUILTIN_LLM_MODEL_GENERATE_FAMILIES.add(model_spec.model_name)
        if "tools" in model_spec.model_ability:
            BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES.add(model_spec.model_name)

    for llm_specs in [
        BUILTIN_LLM_FAMILIES,
        BUILTIN_MODELSCOPE_LLM_FAMILIES,
        BUILTIN_OPENMIND_HUB_LLM_FAMILIES,
        BUILTIN_CSGHUB_LLM_FAMILIES,
    ]:
        for llm_spec in llm_specs:
            if llm_spec.model_name not in LLM_MODEL_DESCRIPTIONS:
                LLM_MODEL_DESCRIPTIONS.update(generate_llm_description(llm_spec))

    # traverse all families and add engine parameters corresponding to the model name
    for families in [
        BUILTIN_LLM_FAMILIES,
        BUILTIN_MODELSCOPE_LLM_FAMILIES,
        BUILTIN_OPENMIND_HUB_LLM_FAMILIES,
        BUILTIN_CSGHUB_LLM_FAMILIES,
    ]:
        for family in families:
            generate_engine_config_by_model_family(family)

    register_custom_model()

    # register model description
    for ud_llm in get_user_defined_llm_families():
        LLM_MODEL_DESCRIPTIONS.update(generate_llm_description(ud_llm))
