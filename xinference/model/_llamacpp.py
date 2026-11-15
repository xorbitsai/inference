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
"""Shared GPU placement compatibility for xllamacpp model loaders."""

import logging
import pprint
from typing import Any, List

logger = logging.getLogger(__name__)


def configure_gpu_layers(params: Any, model_path: str, projectors: List[str]) -> None:
    """Use native auto-fit when available, retaining legacy GPU estimation."""
    if params.n_gpu_layers != -1:
        return

    import xllamacpp

    estimate_gpu_layers = getattr(xllamacpp, "estimate_gpu_layers", None)
    if estimate_gpu_layers is None:
        # xllamacpp v2026.11.12361 removed the Python estimator. Keep -1 and
        # the configured splits/margins so Server can fit and retry natively.
        # Probe the API rather than the version to support unreleased builds.
        return

    # Legacy releases use INT32_MAX for all layers if estimation is unavailable.
    params.n_gpu_layers = 0x7FFFFFFF
    try:
        gpus = [
            info
            for info in xllamacpp.get_device_info()
            if info["type"]
            == xllamacpp.ggml_backend_dev_type.GGML_BACKEND_DEVICE_TYPE_GPU
        ]
        if gpus:
            logger.info(
                "Try to estimate num gpu layers, n_ctx: %s, n_batch: %s, n_parallel: %s, gpus:\n%s",
                params.n_ctx,
                params.n_batch,
                params.n_parallel,
                pprint.pformat(gpus),
            )
            estimate = estimate_gpu_layers(
                gpus=gpus,
                model_path=model_path,
                projectors=projectors,
                context_length=params.n_ctx,
                batch_size=params.n_batch,
                num_parallel=params.n_parallel,
                kv_cache_type="",
            )
            logger.info("Estimate num gpu layers: %s", estimate)
            if estimate.tensor_split:
                for i in range(len(estimate.tensor_split)):
                    params.tensor_split[i] = estimate.tensor_split[i]
            else:
                params.n_gpu_layers = estimate.layers
    except Exception as e:
        logger.exception("Estimate num gpu layers for llama.cpp backend failed: %s", e)
