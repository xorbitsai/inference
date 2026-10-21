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

import importlib.util
import logging

from transformers import PretrainedConfig
from .audio_tokenizer.configuration_audio_vae import AudioVAEconfig

logger = logging.getLogger(__name__)

_ATTN_KEYS = ("_attn_implementation", "attn_implementation")


def _relax_flash_attn_pins(config, _relaxed=None):
    """Downgrade ``flash_attention_2`` pins to SDPA when flash-attn is absent.

    The released checkpoints pin the audio tokenizer backbones to
    flash_attention_2. Transformers treats that as a hard requirement and
    refuses to build the module at all, so a checkpoint published this way
    cannot be loaded on a machine without flash-attn even though every
    operation it needs has an SDPA path.
    """
    if importlib.util.find_spec("flash_attn") is not None:
        return config

    relaxed = [] if _relaxed is None else _relaxed
    if isinstance(config, dict):
        for key, value in config.items():
            if key in _ATTN_KEYS and value == "flash_attention_2":
                config[key] = "sdpa"
                relaxed.append(key)
            else:
                _relax_flash_attn_pins(value, relaxed)
    elif isinstance(config, (list, tuple)):
        for item in config:
            _relax_flash_attn_pins(item, relaxed)

    if _relaxed is None and relaxed:
        logger.warning(
            "flash-attn is not installed; %d attention pin(s) in this "
            "checkpoint were switched to SDPA. Results are unchanged; "
            "install flash-attn for throughput.",
            len(relaxed),
        )
    return config


class BailingMMConfig(PretrainedConfig):
    model_type = "bailingmm"

    def __init__(
        self,
        llm_config: dict = None,
        audio_tokenizer_config: AudioVAEconfig = None,
        ditar_config: dict = None,
        aggregator_config: dict = None,
        model_type: str = None,
        **kwargs
    ):
        _relax_flash_attn_pins(llm_config)
        _relax_flash_attn_pins(audio_tokenizer_config)
        self.model_type = model_type
        self.llm_config = llm_config
        self.audio_tokenizer_config = AudioVAEconfig(**audio_tokenizer_config) if isinstance(audio_tokenizer_config, dict) else audio_tokenizer_config
        self.ditar_config = ditar_config
        self.aggregator_config = aggregator_config
        super().__init__(**kwargs)

    def to_dict(self):
        """Serialise, keeping ``model_type`` as the checkpoint's own value.

        ``PretrainedConfig.to_dict`` replaces ``model_type`` with the class
        attribute, so a dense checkpoint saved through ``save_pretrained``
        would come back as ``"bailingmm"`` and be rebuilt on the MoE code
        path: a different LLM class against dense weights.
        """
        output = super().to_dict()
        if self.model_type:
            output["model_type"] = self.model_type
        return output
