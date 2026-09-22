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

"""Small, validated config snapshots for offline memory estimation."""

from typing import Any, Dict, Optional

from ..._compat import BaseModel, Field


class ModelMemoryMetadata(BaseModel):
    # Attention heads drive activation estimates; KV heads drive cache estimates.
    vocab_size: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_key_value_heads: Optional[int] = Field(default=None, gt=0)
    head_dim: Optional[int] = Field(default=None, gt=0)
    config_source: Optional[str] = None
    config_sha256: Optional[str] = None
    unsupported_reason: Optional[str] = None
    hidden_size: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    num_hidden_layers: int = Field(gt=0)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelMemoryMetadata":
        """Extract dimensions without downloading files or loading model code.

        Missing dimensions are deliberately not inferred from parameter count.
        Unsupported architectures retain their dimensions for maintenance, but
        carry a reason that prevents the dense estimator from using them.
        """
        reason = None
        for nested in (
            "thinker_config",
            "language_config",
            "llm_config",
            "text_config",
        ):
            if not isinstance(config.get(nested), dict):
                continue
            text = dict(config[nested])
            if "vocab_size" not in text and "vocab_size" in config:
                text["vocab_size"] = config["vocab_size"]
            config = text
            reason = "multimodal"
        if config.get("kv_lora_rank") is not None:
            reason = "mla"
        elif any(
            config.get(k)
            for k in ("num_experts", "num_local_experts", "n_routed_experts")
        ):
            reason = "moe"
        if any(
            "linear" in str(t) or "mamba" in str(t)
            for t in (config.get("layer_types") or [])
        ) or any(
            config.get(k) for k in ("linear_num_key_heads", "mamba_d_state", "ssm_cfg")
        ):
            reason = "hybrid_attention"
        aliases = {
            "vocab_size": ("vocab_size", "padded_vocab_size"),
            "num_attention_heads": ("num_attention_heads", "num_heads", "n_head"),
            "num_key_value_heads": ("num_key_value_heads", "n_head_kv"),
            "head_dim": ("head_dim",),
            "hidden_size": ("hidden_size", "d_model", "n_embd"),
            "intermediate_size": (
                "intermediate_size",
                "n_inner",
                "d_ff",
                "ffn_hidden_size",
                "ffn_dim",
                "moe_intermediate_size",
            ),
            "num_hidden_layers": ("num_hidden_layers", "num_layers", "n_layer"),
        }
        values = {}
        for field, keys in aliases.items():
            for key in keys:
                if config.get(key) is not None:
                    values[field] = config[key]
                    break
        if config.get("multi_query_attention") and "multi_query_group_num" in config:
            values["num_key_value_heads"] = config["multi_query_group_num"]
        if reason:
            values["unsupported_reason"] = reason
        return cls.parse_obj(values)
