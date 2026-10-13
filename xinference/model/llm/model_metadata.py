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

"""Validated model architecture facts extracted from checkpoint configurations."""

from typing import Any, Dict, List, Optional

from ..._compat import BaseModel, Field


def _periodic_layers(n: int, interval: int) -> List[str]:
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n)
    ]


class VisionMetadata(BaseModel):
    hidden_size: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    spatial_merge_size: int = Field(default=1, gt=0)
    config_source: Optional[str] = None
    config_sha256: Optional[str] = None


class ModelMetadata(BaseModel):
    # Attention heads and KV heads describe distinct parts of the architecture.
    vocab_size: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_key_value_heads: Optional[int] = Field(default=None, gt=0)
    head_dim: Optional[int] = Field(default=None, gt=0)
    config_source: Optional[str] = None
    config_sha256: Optional[str] = None
    architecture_type: Optional[str] = None
    hidden_size: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    num_hidden_layers: int = Field(gt=0)
    layer_types: Optional[List[str]] = None
    full_attention_interval: Optional[int] = Field(default=None, gt=0)
    linear_num_key_heads: Optional[int] = Field(default=None, gt=0)
    linear_num_value_heads: Optional[int] = Field(default=None, gt=0)
    linear_key_head_dim: Optional[int] = Field(default=None, gt=0)
    linear_value_head_dim: Optional[int] = Field(default=None, gt=0)
    linear_conv_kernel_dim: Optional[int] = Field(default=None, gt=0)
    num_experts: Optional[int] = Field(default=None, gt=0)
    num_experts_per_tok: Optional[int] = Field(default=None, gt=0)
    shared_expert_intermediate_size: Optional[int] = Field(default=None, gt=0)
    vision: Optional[VisionMetadata] = None
    weights_bytes: Optional[int] = Field(default=None, gt=0)
    weights_source: Optional[str] = None
    weights_sha256: Optional[str] = None

    def attention_layers(self) -> Optional[List[str]]:
        if self.layer_types is not None:
            return self.layer_types
        if self.full_attention_interval is not None:
            return _periodic_layers(
                self.num_hidden_layers, self.full_attention_interval
            )
        return None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelMetadata":
        """Extract dimensions without downloading files or loading model code.

        Missing dimensions are deliberately not inferred from parameter count.
        Architecture categories describe the model, not a consumer's support
        policy. Consumers decide which categories they can handle.
        """
        architecture_type = None
        vision = config.get("vision_config")
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
            architecture_type = "multimodal"
        if config.get("kv_lora_rank") is not None:
            architecture_type = "mla"
        elif any(
            config.get(k)
            for k in ("num_experts", "num_local_experts", "n_routed_experts")
        ):
            architecture_type = "moe"
        if architecture_type != "mla" and (
            any(
                "linear" in str(t) or "mamba" in str(t)
                for t in (config.get("layer_types") or [])
            )
            or any(
                config.get(k)
                for k in ("linear_num_key_heads", "mamba_d_state", "ssm_cfg")
            )
        ):
            architecture_type = "hybrid_attention"
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
        if any(
            config.get(key)
            for key in ("num_experts", "num_local_experts", "n_routed_experts")
        ) and config.get("moe_intermediate_size"):
            values["intermediate_size"] = config["moe_intermediate_size"]
        if architecture_type:
            values["architecture_type"] = architecture_type
        for field in (
            "layer_types",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
        ):
            if config.get(field) is not None:
                values[field] = config[field]
        if config.get("full_attention_interval") is not None:
            interval = config["full_attention_interval"]
            if type(interval) is not int or interval <= 0:
                raise ValueError("full_attention_interval must be a positive integer")
            periodic_types = _periodic_layers(
                values.get("num_hidden_layers", 0), interval
            )
            values["full_attention_interval"] = interval
            if values.get("layer_types", periodic_types) == periodic_types:
                values.pop("layer_types", None)
        for field, field_aliases in {
            "num_experts": ("num_experts", "num_local_experts", "n_routed_experts"),
            "num_experts_per_tok": ("num_experts_per_tok", "num_selected_experts"),
            "shared_expert_intermediate_size": ("shared_expert_intermediate_size",),
        }.items():
            for alias in field_aliases:
                if config.get(alias):
                    values[field] = config[alias]
                    break
        # Older multimodal catalogs need explicit layer facts before consumers
        # can distinguish full attention from recurrent caches.
        if architecture_type == "multimodal" and "layer_types" not in values:
            values["layer_types"] = ["full_attention"] * values.get(
                "num_hidden_layers", 0
            )
        if isinstance(vision, dict):
            try:
                values["vision"] = VisionMetadata(
                    hidden_size=vision.get("hidden_size"),
                    intermediate_size=vision.get("intermediate_size"),
                    num_attention_heads=vision.get("num_heads")
                    or vision.get("num_attention_heads"),
                    spatial_merge_size=vision.get("spatial_merge_size", 1),
                )
            except ValueError:
                # Missing vision facts must not discard usable text dimensions.
                pass
        return cls.parse_obj(values)
