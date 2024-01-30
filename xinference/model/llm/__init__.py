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

from .core import (
    LLM,
    LLM_MODEL_DESCRIPTIONS,
    LLMDescription,
    generate_llm_description,
    get_llm_model_descriptions,
)
from .llm_family import (
    BUILTIN_LLM_FAMILIES,
    BUILTIN_LLM_MODEL_CHAT_FAMILIES,
    BUILTIN_LLM_MODEL_GENERATE_FAMILIES,
    BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES,
    BUILTIN_LLM_PROMPT_STYLE,
    BUILTIN_MODELSCOPE_LLM_FAMILIES,
    LLM_CLASSES,
    CustomLLMFamilyV1,
    GgmlLLMSpecV1,
    LLMFamilyV1,
    LLMSpecV1,
    PromptStyleV1,
    PytorchLLMSpecV1,
    get_cache_status,
    get_user_defined_llm_families,
    match_llm,
    match_llm_cls,
    register_llm,
    unregister_llm,
)


def _install():
    from .ggml.chatglm import ChatglmCppChatModel
    from .ggml.ctransformers import CtransformersModel
    from .ggml.llamacpp import LlamaCppChatModel, LlamaCppModel
    from .pytorch.baichuan import BaichuanPytorchChatModel
    from .pytorch.chatglm import ChatglmPytorchChatModel
    from .pytorch.core import PytorchChatModel, PytorchModel
    from .pytorch.falcon import FalconPytorchChatModel, FalconPytorchModel
    from .pytorch.internlm2 import Internlm2PytorchChatModel
    from .pytorch.llama_2 import LlamaPytorchChatModel, LlamaPytorchModel
    from .pytorch.qwen_vl import QwenVLChatModel
    from .pytorch.vicuna import VicunaPytorchChatModel
    from .pytorch.yi_vl import YiVLChatModel
    from .vllm.core import VLLMChatModel, VLLMModel

    # register llm classes.
    LLM_CLASSES.extend(
        [
            LlamaCppChatModel,
            LlamaCppModel,
        ]
    )
    LLM_CLASSES.extend(
        [
            ChatglmCppChatModel,
        ]
    )
    LLM_CLASSES.extend(
        [
            CtransformersModel,
        ]
    )
    LLM_CLASSES.extend([VLLMModel, VLLMChatModel])
    LLM_CLASSES.extend(
        [
            BaichuanPytorchChatModel,
            VicunaPytorchChatModel,
            FalconPytorchChatModel,
            ChatglmPytorchChatModel,
            LlamaPytorchModel,
            LlamaPytorchChatModel,
            PytorchChatModel,
            FalconPytorchModel,
            Internlm2PytorchChatModel,
            QwenVLChatModel,
            YiVLChatModel,
            PytorchModel,
        ]
    )

    json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "llm_family.json"
    )
    for json_obj in json.load(codecs.open(json_path, "r", encoding="utf-8")):
        model_spec = LLMFamilyV1.parse_obj(json_obj)
        BUILTIN_LLM_FAMILIES.append(model_spec)

        # register prompt style
        if "chat" in model_spec.model_ability and isinstance(
            model_spec.prompt_style, PromptStyleV1
        ):
            # note that the key is the model name,
            # since there are multiple representations of the same prompt style name in json.
            BUILTIN_LLM_PROMPT_STYLE[model_spec.model_name] = model_spec.prompt_style
        # register model family
        if "chat" in model_spec.model_ability:
            BUILTIN_LLM_MODEL_CHAT_FAMILIES.add(model_spec.model_name)
        else:
            BUILTIN_LLM_MODEL_GENERATE_FAMILIES.add(model_spec.model_name)
        if "tool_call" in model_spec.model_ability:
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
            and isinstance(model_spec.prompt_style, PromptStyleV1)
            and model_spec.model_name not in BUILTIN_LLM_PROMPT_STYLE
        ):
            BUILTIN_LLM_PROMPT_STYLE[model_spec.model_name] = model_spec.prompt_style
        # register model family
        if "chat" in model_spec.model_ability:
            BUILTIN_LLM_MODEL_CHAT_FAMILIES.add(model_spec.model_name)
        else:
            BUILTIN_LLM_MODEL_GENERATE_FAMILIES.add(model_spec.model_name)
        if "tool_call" in model_spec.model_ability:
            BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES.add(model_spec.model_name)

    for llm_specs in [BUILTIN_LLM_FAMILIES, BUILTIN_MODELSCOPE_LLM_FAMILIES]:
        for llm_spec in llm_specs:
            if llm_spec.model_name not in LLM_MODEL_DESCRIPTIONS:
                LLM_MODEL_DESCRIPTIONS.update(generate_llm_description(llm_spec))

    from ...constants import XINFERENCE_MODEL_DIR

    user_defined_llm_dir = os.path.join(XINFERENCE_MODEL_DIR, "llm")
    if os.path.isdir(user_defined_llm_dir):
        for f in os.listdir(user_defined_llm_dir):
            with codecs.open(
                os.path.join(user_defined_llm_dir, f), encoding="utf-8"
            ) as fd:
                user_defined_llm_family = CustomLLMFamilyV1.parse_obj(json.load(fd))
                register_llm(user_defined_llm_family, persist=False)

    # register model description
    for ud_llm in get_user_defined_llm_families():
        LLM_MODEL_DESCRIPTIONS.update(generate_llm_description(ud_llm))
