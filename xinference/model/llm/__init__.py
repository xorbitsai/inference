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

from .core import LLM
from .llm_family import (
    BUILTIN_LLM_FAMILIES,
    LLM_CLASSES,
    GgmlLLMSpecV1,
    LLMFamilyV1,
    LLMSpecV1,
    PromptStyleV1,
    PytorchLLMSpecV1,
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
    from .pytorch.llama_2 import LlamaPytorchChatModel, LlamaPytorchModel
    from .pytorch.vicuna import VicunaPytorchChatModel

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
            PytorchModel,
        ]
    )

    json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "llm_family.json"
    )
    for json_obj in json.load(codecs.open(json_path, "r", encoding="utf-8")):
        BUILTIN_LLM_FAMILIES.append(LLMFamilyV1.parse_obj(json_obj))

    from ...constants import XINFERENCE_MODEL_DIR

    user_defined_llm_dir = os.path.join(XINFERENCE_MODEL_DIR, "llm")
    if os.path.isdir(user_defined_llm_dir):
        for f in os.listdir(user_defined_llm_dir):
            with codecs.open(
                os.path.join(user_defined_llm_dir, f), encoding="utf-8"
            ) as fd:
                user_defined_llm_family = LLMFamilyV1.parse_obj(json.load(fd))
                register_llm(user_defined_llm_family, persist=False)
