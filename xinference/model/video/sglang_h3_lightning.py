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

"""Bridge SGLang H3 requests to its internal dynamic LoRA scale."""

import os
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from typing import Any, Callable

from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.registry import register_pipeline
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context_or_none,
    set_forward_context,
)
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
    MiniMaxH3Pipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    MiniMaxH3DenoisingStage,
)


@dataclass
class RequestLightningSamplingParams(MiniMaxH3SamplingParams):
    # Keep this in the sampling dataclass so SGLang's batch signature separates
    # modes, even when both requests happen to use the same denoising schedule.
    use_lightning: bool = True

    def apply_request_extra(self, req: Req) -> None:
        super().apply_request_extra(req)
        # Dynamic LoRA keeps the adapter loaded and skips its forward at zero.
        # The request's scale travels to every GPU rank with the scheduler batch.
        req.runtime_lora_scale = float(self.use_lightning)


def _forward_with_lora_scale(
    forward: Callable[..., Req], batch: Req, server_args: Any
) -> Req:
    if batch.runtime_lora_scale != 0.0:
        return forward(batch, server_args)
    # H3 refines the prompt before installing the per-step forward context.
    # Lightning also adapts that refiner, so the zero scale must cover both.
    with set_forward_context(
        current_timestep=0, attn_metadata=None, forward_batch=batch
    ):
        return forward(batch, server_args)


def _graph_runner(get_runner: Callable[..., Any], current_model: Any) -> Any:
    context = get_forward_context_or_none()
    if (
        context is not None
        and context.forward_batch is not None
        and context.forward_batch.runtime_lora_scale == 0.0
    ):
        # Warmup captures the Lightning branch. Base requests skip graph replay
        # so they cannot execute a captured graph with the adapter baked in.
        return None
    return get_runner(current_model)


class _RequestLightningPipeline(MiniMaxH3Pipeline):
    sampling_params_cls = RequestLightningSamplingParams

    def create_pipeline_stages(self, server_args):
        super().create_pipeline_stages(server_args)
        stage = self.get_stage(MiniMaxH3DenoisingStage.__name__)
        if not isinstance(stage, MiniMaxH3DenoisingStage):
            raise RuntimeError("SGLang's MiniMax-H3 denoising stage has changed")
        # Keep the existing stage and its initialized transformer and compiler.
        stage.forward = partial(_forward_with_lora_scale, stage.forward)
        stage._maybe_get_bcg_runner = partial(
            _graph_runner, stage._maybe_get_bcg_runner
        )


external_package = os.environ.get("XINFERENCE_SGLANG_H3_EXTERNAL_MODEL_PACKAGE")
if external_package and external_package != __name__:
    import_module(external_package)

# Preserve H3's registered name for SGLang's built-in model detection and warmup.
# Remove this bridge when SGLang exposes native H3 request-level LoRA control
# across prompt refinement, batching and graph replay.
register_pipeline(
    _RequestLightningPipeline,
    sampling_param_cls=RequestLightningSamplingParams,
    pipeline_config_cls=MiniMaxH3Pipeline.pipeline_config_cls,
    overwrite=True,
)
