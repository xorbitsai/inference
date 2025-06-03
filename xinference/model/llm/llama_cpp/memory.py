# Copyright 2022-2023 XProbe Inc.
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

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

try:
    from gguf import GGUFReader, GGUFValueType  # noqa: E402
except ImportError:
    GGUFReader = GGUFValueType = None
logger = logging.getLogger(__name__)


def get_file_host_endian(reader: GGUFReader) -> tuple[str, str]:
    file_endian = reader.endianess.name  # codespell:ignore
    if reader.byte_order == "S":
        host_endian = "BIG" if file_endian == "LITTLE" else "LITTLE"
    else:
        host_endian = file_endian
    return (host_endian, file_endian)


def dump_metadata_json(reader: GGUFReader, model_path: str) -> dict:
    host_endian, file_endian = get_file_host_endian(reader)
    metadata: dict[str, Any] = {}
    tensors: dict[str, Any] = {}
    result = {
        "filename": model_path,
        "endian": file_endian,
        "metadata": metadata,
        "tensors": tensors,
    }
    for idx, field in enumerate(reader.fields.values()):
        curr: dict[str, Any] = {
            "index": idx,
            "type": field.types[0].name if field.types else "UNKNOWN",
            "offset": field.offset,
        }
        metadata[field.name] = curr
        if field.types[:1] == [GGUFValueType.ARRAY]:
            curr["array_types"] = [t.name for t in field.types][1:]
            curr["value"] = field.contents()
        else:
            curr["value"] = field.contents()
        for i, tensor in enumerate(reader.tensors):
            tensors[tensor.name] = {
                "index": i,
                "shape": tensor.shape.tolist(),
                "type": tensor.tensor_type.name,
                "offset": tensor.field.offset,
                "n_bytes": tensor.n_bytes,
            }
    return result


@dataclass
class MemoryEstimate:
    # How many layers we predict we can load
    layers: int
    # The size of the graph which occupies the main GPU
    graph: int
    # How much VRAM will be allocated given the number of layers we predict
    vram_size: int
    # The total size of the model if loaded into VRAM.  If all layers are loaded, vram_size == total_size
    total_size: int
    # For multi-GPU scenarios, this provides the tensor split parameter
    tensor_split: str
    # For multi-GPU scenarios, this is the size in bytes per GPU
    gpu_sizes: list[int]


def _get_max_min(value):
    if isinstance(value, Sequence):
        return max(value), min(value)
    else:
        return value, value


def graph_size(
    data: dict,
    context_length: int,
    batch_size: int,
    num_parallel: int,
    kv_cache_type: str,
):
    """
    Most of the logic comes from `GraphSize` in https://github.com/ollama/ollama/blob/main/fs/ggml/ggml.go
    """
    if context_length < batch_size:
        batch_size = context_length

    metadata = data["metadata"]
    architecture = metadata["general.architecture"]["value"]
    embedding_length = metadata[f"{architecture}.embedding_length"]["value"]
    block_count = metadata[f"{architecture}.block_count"]["value"]
    head_count_max, head_count_min = _get_max_min(
        metadata[f"{architecture}.attention.head_count"]["value"]
    )
    head_count_kv_max, head_count_kv_min = _get_max_min(
        metadata[f"{architecture}.attention.head_count_kv"]["value"]
    )
    vocab = len(metadata["tokenizer.ggml.tokens"]["value"])
    embedding_head_count_max = (
        (embedding_length // head_count_min) if head_count_min > 0 else 0
    )
    embedding_head_count_k = metadata.get(
        f"{architecture}.attention.key_length", {}
    ).get("value", embedding_head_count_max)
    embedding_head_count_v = metadata.get(
        f"{architecture}.attention.value_length", {}
    ).get("value", embedding_head_count_max)

    # f16(default)
    bytes_per_kv_element = {
        "q8_0": 1,  # 1/2 of fp16
        "q4_0": 0.5,  # 1/4 of fp16
    }.get(kv_cache_type, 2)

    kv = [0] * block_count
    for i in range(block_count):
        kv[i] = (
            context_length
            * (embedding_head_count_k + embedding_head_count_v)
            * head_count_kv_max
            * bytes_per_kv_element
        )

    full_offload = 0
    partial_offload = 0
    if architecture in ["llama", "llama4"]:
        full_offload = max(
            4
            * batch_size
            * (1 + 4 * embedding_length + context_length * (1 + head_count_max)),
            4 * batch_size * (embedding_length + vocab),
        )
        partial_offload = 4 * batch_size * embedding_length
        partial_offload += max(
            4
            * batch_size
            * (1 + embedding_length + max(context_length, embedding_length))
            + embedding_length * embedding_length * 9 / 16
            + 4
            * context_length
            * (
                batch_size * head_count_max
                + embedding_head_count_max * head_count_kv_max
            ),
            4 * batch_size * (embedding_length + vocab)
            + embedding_length * vocab * 105 / 128,
        )
    elif architecture in ["gemma", "gemma2", "gemma3"]:
        full_offload = max(
            4 * batch_size * (embedding_length + vocab),
            4
            * batch_size
            * (
                2
                + context_length
                + context_length * head_count_max
                + 2 * embedding_length
                + 2 * embedding_head_count_k * head_count_max
            ),
        )
        partial_offload = max(
            4 * embedding_length * batch_size
            + embedding_length * vocab * 105 / 128
            + 4 * vocab * batch_size,
            4
            * batch_size
            * (
                2 * embedding_length
                + 1
                + 2 * embedding_head_count_k * head_count_max
                + context_length
                + context_length * head_count_max
            )
            + 4 * embedding_head_count_k * context_length * 8
            + embedding_length * embedding_head_count_k * head_count_max * 9 / 16,
        )
        if architecture == "gemma3":
            gemma3_global_cache_count = 6
            sliding_window = (
                num_parallel
                * metadata[f"{architecture}.attention.sliding_window"]["value"]
                + batch_size
            )
            for i in range(block_count):
                if (i + 1) % gemma3_global_cache_count != 0:
                    kv[i] = (
                        sliding_window
                        * (embedding_head_count_k + embedding_head_count_v)
                        * head_count_kv_max
                        * bytes_per_kv_element
                    )
    elif architecture == "qwen2":
        full_offload = max(
            4 * batch_size * (embedding_length + vocab),
            4
            * batch_size
            * (
                1
                + 2 * embedding_length
                + context_length
                + context_length * head_count_max
            ),
        )

        partial_offload = max(
            4 * batch_size * (embedding_length + vocab)
            + embedding_length * vocab * 105 / 128,
            4
            * (
                batch_size
                * (1 + 2 * embedding_length + context_length * (1 + head_count_max))
                + embedding_length * (1 + context_length)
            ),
        )
    elif architecture == "stablelm":
        full_offload = (
            4
            * batch_size
            * (context_length * (1 + head_count_max) + 3 * embedding_length + 2)
        )
        partial_offload = max(
            4 * batch_size * (vocab + 2 * embedding_length), full_offload
        )
    elif architecture == "deepseek2":
        full_offload = max(
            4 * batch_size * (3 * embedding_length + vocab),
            4
            * batch_size
            * (
                3 * embedding_length
                + 2
                + context_length * (1 + head_count_kv_max)
                + 2 * embedding_head_count_k * head_count_kv_max
            ),
        )

        partial_offload = max(
            4 * batch_size * (3 * embedding_length + vocab)
            + embedding_length * vocab * 105 / 128,
            4
            * batch_size
            * (
                2 * embedding_length
                + 1
                + 2 * embedding_head_count_k * head_count_kv_max
                + context_length
                + context_length * head_count_kv_max
            )
            + 4 * embedding_head_count_k * context_length * head_count_kv_max
            + embedding_length * embedding_head_count_k * head_count_kv_max * 9 / 16,
        )

    kv_total = sum(kv)
    if partial_offload == 0:
        partial_offload = (
            head_count_max
            / (1 if head_count_kv_min <= 0 else head_count_kv_min)
            * kv_total
            / 6
        )
    if full_offload == 0:
        full_offload = partial_offload

    return kv, partial_offload, full_offload


def projector_memory_requirements(projector: str):
    reader = GGUFReader(projector, "r")
    data = dump_metadata_json(reader, projector)
    return sum(t["n_bytes"] for t in data["tensors"].values())


def estimate_gpu_layers(
    gpus: list[dict],
    model_path: str,
    projectors: list[str],
    context_length: int,
    batch_size: int,
    num_parallel: int,
    kv_cache_type: str,
):
    """
    Most of the logic comes from `EstimateGPULayers` in https://github.com/ollama/ollama/blob/main/llm/memory.go
    """
    # Projectors loaded into GPU0 only
    projector_weights = sum(map(projector_memory_requirements, projectors))
    if projector_weights > 0:
        # Multimodal models require at least 2048 context
        context_length = max(context_length, 2048)
    reader = GGUFReader(model_path, "r")
    data = dump_metadata_json(reader, model_path)
    kv, graph_partial_offload, graph_full_offload = graph_size(
        data,
        context_length=context_length,
        batch_size=batch_size,
        num_parallel=num_parallel,
        kv_cache_type=kv_cache_type,
    )
    # Get all layer sizes
    metadata = data["metadata"]
    architecture = metadata["general.architecture"]["value"]
    block_count = metadata[f"{architecture}.block_count"]["value"]
    layer_sizes = [0] * block_count
    for name, layer in data["tensors"].items():
        if name.startswith("blk."):
            index = int(name[len("blk.") :].split(".")[0])
            layer_sizes[index] += layer["n_bytes"]
    layer_size = layer_sizes[0] if layer_sizes else 0

    if len(kv) > 0:
        layer_size += kv[0]
    # On metal there's no partial offload overhead
    if gpus[0]["name"] == "Metal":
        graph_partial_offload = graph_full_offload
    elif len(gpus) > 1:
        # Multi gpu should always use the partial graph size
        graph_full_offload = graph_partial_offload

    # Get output layer size
    memory_layer_output = 0
    # Output layer handled at the end if we have space
    for name, layer in data["tensors"].items():
        if any(
            name.startswith(prefix)
            for prefix in ["output_norm", "output", "token_embd"]
        ):
            memory_layer_output += layer["n_bytes"]

    # Reduce set of GPUs to only those that have sufficient space to fit overhead and at least one layer
    default_memory_min = 512 * 1024**2
    gpu_allocations = [0] * len(gpus)
    gpus_with_space: list[int] = []
    for i in range(len(gpus)):
        gpu0_overhead = projector_weights if len(gpus_with_space) == 0 else 0
        minimum_memory = gpus[i].get("memory_min", default_memory_min)
        if (
            gpus[i]["memory_free"]
            < gpu0_overhead
            + max(graph_partial_offload, graph_full_offload)
            + minimum_memory
            + 2 * layer_size
        ):
            continue
        gpus_with_space.append(i)
        gpu_allocations[i] += gpu0_overhead + minimum_memory + layer_size

    overflow = 0
    if len(gpus_with_space) == 0:
        overflow = projector_weights

    # For all the layers, find where they can fit on the GPU(s)
    layer_count = 0
    layer_counts = [0] * len(gpus)
    for i in range(block_count - 1, -1, -1):
        layer_size = layer_sizes[i]
        layer_size += kv[i]

        # Distribute the layers across the GPU(s) that have space
        for j in range(len(gpus_with_space), 0, -1):
            g = gpus_with_space[i % j]
            used = gpu_allocations[g] + max(graph_partial_offload, graph_full_offload)
            if gpus[g]["memory_free"] > used + layer_size:
                gpu_allocations[g] += layer_size
                layer_counts[g] += 1
                layer_count += 1
                break
            else:
                gpus_with_space = (
                    gpus_with_space[: i % j] + gpus_with_space[i % j + 1 :]
                )

        if len(gpus_with_space) == 0:
            overflow += layer_size

    fully_loaded = False
    if layer_count >= block_count:
        fully_loaded = True

    # Determine if we need to consider output then find where it fits
    if memory_layer_output > 0:
        for j in range(len(gpus_with_space), 0, -1):
            g = gpus_with_space[layer_count % j]
            used = gpu_allocations[g] + max(graph_partial_offload, graph_full_offload)
            if gpus[g]["memory_free"] > used + memory_layer_output:
                gpu_allocations[g] += memory_layer_output
                layer_counts[g] += 1
                layer_count += 1
                break
            else:
                gpus_with_space = (
                    gpus_with_space[: layer_count % j]
                    + gpus_with_space[layer_count % j + 1 :]
                )

        if layer_count < block_count + 1:
            fully_loaded = False
            overflow += memory_layer_output

    # Add the applicable (full or partial) graph allocations
    for i in range(len(gpus)):
        if layer_counts[i] <= 0:
            continue
        if fully_loaded:
            gpu_allocations[i] += graph_full_offload
        else:
            gpu_allocations[i] += graph_partial_offload

    if fully_loaded:
        graph_offload = graph_full_offload
    else:
        graph_offload = graph_partial_offload

    # Summaries
    memory_required_partial = sum(gpu_allocations)
    memory_required_total = memory_required_partial + overflow

    tensor_split = ""
    if len(gpus) > 1:
        tensor_split = ",".join(str(c) for c in layer_counts)

    estimate = MemoryEstimate(
        layers=0,
        graph=0,
        vram_size=0,
        total_size=int(memory_required_total),
        tensor_split="",
        gpu_sizes=[],
    )
    if gpus[0]["name"] == "CPU":
        return estimate
    if layer_count == 0:
        return estimate

    estimate.layers = layer_count
    estimate.graph = int(graph_offload)
    estimate.vram_size = int(memory_required_partial)
    estimate.total_size = int(memory_required_total)
    estimate.tensor_split = tensor_split
    estimate.gpu_sizes = [int(i) for i in gpu_allocations]
    return estimate
