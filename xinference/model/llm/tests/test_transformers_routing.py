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
# CI: linux

import pytest

from ..llm_family import (
    BUILTIN_LLM_FAMILIES,
    check_engine_by_spec_parameters,
    check_engine_by_spec_parameters_with_virtual_env,
)
from ..transformers.deepseek_v2 import DeepSeekV2PytorchChatModel
from ..transformers.multimodal.deepseek_vl2 import DeepSeekVL2ChatModel
from ..transformers.multimodal.glm4_1v import Glm4_1VModel
from ..transformers.multimodal.glm4v import Glm4VModel

# DeepSeek-V2 and DeepSeek-VL2 share DeepseekV2ForCausalLM, so the text class must
# leave the vision family to its own class. glm-4v declares ChatGLMModel, which
# Glm4VModel serves; the newer GLM-4.1V/4.5V families have their own class.
ROUTES = [
    ("deepseek-vl2", DeepSeekVL2ChatModel),
    ("glm-4v", Glm4VModel),
    # Families that were already routed to these classes.
    ("deepseek-v2-chat", DeepSeekV2PytorchChatModel),
    ("deepseek-v2-chat-0628", DeepSeekV2PytorchChatModel),
    ("deepseek-v2.5", DeepSeekV2PytorchChatModel),
    ("glm-4.1v-thinking", Glm4_1VModel),
    ("glm-4.5v", Glm4_1VModel),
]


def _family(name):
    return next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == name)


def _pytorch_specs(family):
    return [spec for spec in family.model_specs if spec.model_format == "pytorch"]


@pytest.mark.parametrize("virtual_env", [False, True], ids=["plain", "virtualenv"])
@pytest.mark.parametrize("name,expected", ROUTES)
def test_transformers_launch_selects_dedicated_class(name, expected, virtual_env):
    family = _family(name)
    selected = set()
    for spec in _pytorch_specs(family):
        args = (
            "Transformers",
            name,
            spec.model_format,
            spec.model_size_in_billions,
            spec.quantization,
        )
        if virtual_env:
            selected.add(
                check_engine_by_spec_parameters_with_virtual_env(
                    *args, llm_family=family
                )
            )
        else:
            selected.add(check_engine_by_spec_parameters(*args))
    assert selected == {expected}


@pytest.mark.parametrize(
    "model_cls,name,accepted",
    [
        (DeepSeekV2PytorchChatModel, "deepseek-vl2", False),
        (DeepSeekVL2ChatModel, "deepseek-vl2", True),
        (DeepSeekV2PytorchChatModel, "deepseek-v2.5", True),
        (DeepSeekVL2ChatModel, "deepseek-v2.5", False),
        (Glm4VModel, "glm-4v", True),
        (Glm4VModel, "glm-4.1v-thinking", False),
        (Glm4VModel, "glm-4.5v", False),
        # A text-only ChatGLMModel family is not a GLM-4V family.
        (Glm4VModel, "codegeex4", False),
        (Glm4_1VModel, "glm-4v", False),
    ],
)
def test_match_json_keeps_shared_architectures_apart(model_cls, name, accepted):
    family = _family(name)
    specs = _pytorch_specs(family)
    assert specs
    for spec in specs:
        result = model_cls.match_json(family, spec, spec.quantization)
        assert (result is True) is accepted
