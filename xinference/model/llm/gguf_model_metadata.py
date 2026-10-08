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

"""Bounded GGUF metadata reader for maintenance, stopping before tensor data."""

import hashlib
import struct


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
    if any(key.startswith(prefix + "ssm.") for key in fields):
        config["ssm_cfg"] = True
    if fields.get(
        prefix + "attention.value_length", config.get("head_dim")
    ) != config.get("head_dim"):
        config["ssm_cfg"] = True  # Different K/V dimensions are not modeled.
    return config, digest.hexdigest(), consumed
