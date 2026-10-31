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

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .core import VideoModelFamilyV2


def apply_runtime_precision(
    model_spec: "VideoModelFamilyV2",
    model_engine: Optional[str],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Translate an explicit H3 precision choice; preserve legacy custom configs."""
    config = kwargs.copy()
    precision = config.pop("runtime_precision", None)
    if precision in (None, "auto"):
        return config
    engine = (model_engine or model_spec.engine or "").lower()
    if (
        model_spec.model_name != "MiniMax-H3"
        or precision not in (model_spec.runtime_precisions or [])
        or engine not in ("diffusers", "vllm", "sglang")
    ):
        raise ValueError(
            f"runtime_precision {precision!r} is not supported by "
            f"{model_spec.model_name} on {model_engine or model_spec.engine}"
        )

    # An explicit unified choice owns these settings, including when a launch
    # history from another engine contains incompatible precision parameters.
    for key in (
        "quantization_config",
        "torch_dtype",
        "dtype",
        "component_quantizations",
        "component_precisions",
    ):
        config.pop(key, None)
    if engine == "diffusers":
        config.update(quantization_config=precision, torch_dtype="bfloat16")
    elif engine == "vllm":
        config.update(
            quantization_config="fp8" if precision == "fp8" else None,
            dtype="bfloat16",
        )
    else:
        config.update(
            component_quantizations=(
                {"transformer": "fp8", "text_encoder": "fp8"}
                if precision == "fp8"
                else {}
            ),
            component_precisions={"transformer": "bf16", "text_encoder": "bf16"},
        )
    return config
