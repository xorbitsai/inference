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

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.deepseek_v3 import DeepseekV3Model as _DeepseekV3Model
from mlx_lm.models.deepseek_v3 import Model as _Model
from mlx_lm.models.deepseek_v3 import ModelArgs

from .core import DistributedModelMixin


class DeepseekV3Model(_DeepseekV3Model, DistributedModelMixin):
    def __init__(self, *args, **kwargs):
        _DeepseekV3Model.__init__(self, *args, **kwargs)
        DistributedModelMixin.__init__(self)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        pipeline_rank = self.rank
        pipeline_size = self.world_size
        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * self.num_layers

        # Receive from the previous process in the pipeline

        if pipeline_rank < pipeline_size - 1:
            # wait for previous result
            h = self._wait_prev_stage_result()

        for i in range(self.num_layers):
            h = self.layers[self.start_idx + i](h, mask, cache[i])
        mx.eval(h)

        if pipeline_rank != 0:
            # Send to the next process in the pipeline
            self._send_stage_result(h)
            # wait for the final result
            h = self._get_result()
        else:
            self._set_result(h)

        return self.norm(h)


class Model(_Model):
    def __init__(self, config: ModelArgs):
        nn.Module.__init__(self)
        self.args = config
        self.model_type = config.model_type
        self.model = DeepseekV3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
