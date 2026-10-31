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

"""Bounded GGUF metadata reader, stopping before tensor data."""

import hashlib
import os
import struct
from typing import Any


def resolve_gguf_model_path(model_path: str, spec: Any) -> str:
    """Resolve the same first shard / legacy filename used by the runtime."""
    if os.path.isfile(model_path):
        return model_path
    if (
        spec.model_file_name_split_template
        and spec.quantization in spec.quantization_parts
    ):
        part = spec.quantization_parts[spec.quantization][0]
        return os.path.join(
            model_path,
            spec.model_file_name_split_template.format(
                quantization=spec.quantization, part=part
            ),
        )
    legacy = os.path.join(model_path, "model.bin")
    if os.path.exists(legacy):
        return legacy
    return os.path.join(
        model_path, spec.model_file_name_template.format(quantization=spec.quantization)
    )


def read_gguf_config(stream, limit: int = 16 * 1024**2) -> tuple:
    consumed = 0
    digest = hashlib.sha256()

    def read(n):
        nonlocal consumed
        if n < 0 or consumed + n > limit:
            raise ValueError("GGUF metadata exceeds collection limit")
        data = stream.read(n)
        if len(data) != n:
            raise ValueError("Truncated GGUF metadata")
        consumed += n
        digest.update(data)
        return data

    def number(fmt):
        return struct.unpack("<" + fmt, read(struct.calcsize(fmt)))[0]

    def string():
        return read(number("Q")).decode("utf-8", errors="replace")

    formats = {
        0: "B",
        1: "b",
        2: "H",
        3: "h",
        4: "I",
        5: "i",
        6: "f",
        7: "?",
        10: "Q",
        11: "q",
        12: "d",
    }

    def value(kind, keep=True):
        if kind in formats:
            return number(formats[kind])
        if kind == 8:
            return string()
        if kind == 9:
            element, count = number("I"), number("Q")
            if count > 2000000 or element == 9:
                raise ValueError("Unsupported GGUF metadata array")
            # Token lists/merges are not retained, but their length gives vocab size.
            if element in formats:
                data = read(struct.calcsize(formats[element]) * count)
                return (
                    list(struct.unpack("<" + formats[element] * count, data))
                    if keep
                    else count
                )
            if element != 8:
                raise ValueError("Unsupported GGUF array type")
            for _ in range(count):
                read(number("Q"))
            return count
        raise ValueError(f"Unknown GGUF metadata type: {kind}")

    if read(4) != b"GGUF" or number("I") not in (2, 3):
        raise ValueError("Expected little-endian GGUF v2/v3")
    number("Q")  # tensor count; never read tensors
    count = number("Q")
    if count > 10000:
        raise ValueError("Too many GGUF metadata keys")
    fields = {}
    for _ in range(count):
        key = string()
        fields[key] = value(number("I"), not key.startswith("tokenizer."))
    architecture = fields.get("general.architecture")
    prefix = str(architecture) + "."
    mapping = {
        "num_attention_heads": "attention.head_count",
        "num_key_value_heads": "attention.head_count_kv",
        "hidden_size": "embedding_length",
        "intermediate_size": "feed_forward_length",
        "num_hidden_layers": "block_count",
        "head_dim": "attention.key_length",
        "num_experts": "expert_count",
        "kv_lora_rank": "attention.kv_lora_rank",
        "num_experts_per_tok": "expert_used_count",
        "shared_expert_intermediate_size": "expert_shared_feed_forward_length",
    }
    config = {
        name: fields[prefix + key]
        for name, key in mapping.items()
        if prefix + key in fields
    }
    config["vocab_size"] = fields.get(
        prefix + "vocab_size", fields.get("tokenizer.ggml.tokens")
    )
    # Unknown/hybrid layouts must not silently look like ordinary dense attention.
    config["model_type"] = architecture
    rope = {
        name: fields[prefix + "rope.scaling." + key]
        for name, key in {
            "rope_type": "type",
            "factor": "factor",
            "original_max_position_embeddings": "original_context_length",
        }.items()
        if prefix + "rope.scaling." + key in fields
    }
    if rope:
        config["rope_parameters"] = rope
    if architecture == "clip":
        config["vision_config"] = {
            name: fields["clip.vision." + key]
            for name, key in {
                "hidden_size": "embedding_length",
                "intermediate_size": "feed_forward_length",
                "num_attention_heads": "attention.head_count",
                "spatial_merge_size": "spatial_merge_size",
            }.items()
            if "clip.vision." + key in fields
        }
    if (
        "intermediate_size" not in config or config.get("num_experts")
    ) and prefix + "expert_feed_forward_length" in fields:
        config["intermediate_size"] = fields[prefix + "expert_feed_forward_length"]
    if architecture in ("qwen35", "qwen35moe", "qwen3next"):
        for name, key in {
            "linear_conv_kernel_dim": "ssm.conv_kernel",
            "linear_key_head_dim": "ssm.state_size",
            "linear_num_key_heads": "ssm.group_count",
            "linear_num_value_heads": "ssm.time_step_rank",
            "full_attention_interval": "full_attention_interval",
        }.items():
            if prefix + key in fields:
                config[name] = fields[prefix + key]
        inner = fields.get(prefix + "ssm.inner_size")
        heads = config.get("linear_num_value_heads")
        if (
            type(inner) is int
            and type(heads) is int
            and heads > 0
            and inner % heads == 0
        ):
            config["linear_value_head_dim"] = inner // heads
    if any(key.startswith(prefix + "ssm.") for key in fields):
        config["ssm_cfg"] = True
    if fields.get(
        prefix + "attention.value_length", config.get("head_dim")
    ) != config.get("head_dim"):
        # Even a known recurrent layout cannot make unequal full-attention K/V
        # dimensions compatible with the homogeneous KV formula.
        config["layer_types"] = ["unsupported_attention"] * config.get(
            "num_hidden_layers", 1
        )
    return config, digest.hexdigest(), consumed
