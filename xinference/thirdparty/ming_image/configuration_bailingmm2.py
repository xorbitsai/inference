# coding=utf-8
# Copyright 2024 ANT Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import PretrainedConfig
from .qwen2_5_vit import Qwen2_5_VLVisionConfig
from .configuration_bailing_moe_v2 import BailingMoeV2Config


class BailingMM2Config(PretrainedConfig):
    model_type = "bailingmm_moe_v2_lite"

    def __init__(
        self,
        mlp_depth=1,
        llm_config: BailingMoeV2Config = None,
        vision_config: Qwen2_5_VLVisionConfig = None,
        audio_config=None,
        **kwargs
    ):
        if audio_config is not None:
            raise ValueError("audio_config is not supported by Ming Image inference")
        self.audio_config = None
        self.vision_config = Qwen2_5_VLVisionConfig(**vision_config) if isinstance(vision_config, dict) else vision_config
        self.llm_config = BailingMoeV2Config(**llm_config) if isinstance(llm_config, dict) else llm_config
        self.mlp_depth = mlp_depth
        super().__init__(**kwargs)
