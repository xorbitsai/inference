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
import pytest

from ..utils import (
    SUPPORTED_QUANTIZATIONS,
    build_replica_model_uid,
    get_llama_cpp_quantization_info,
    get_match_quantization_filenames,
    get_model_size_from_model_id,
    get_prefix_suffix,
    iter_replica_model_uid,
    parse_replica_model_uid,
)


def test_replica_model_uid():
    all_gen_ids = []
    for replica_model_uid in iter_replica_model_uid("abc", 5):
        rebuild_replica_model_uid = build_replica_model_uid(
            *parse_replica_model_uid(replica_model_uid)
        )
        assert rebuild_replica_model_uid == replica_model_uid
        all_gen_ids.append(replica_model_uid)
    assert len(all_gen_ids) == 5
    assert len(set(all_gen_ids)) == 5


def test_get_model_size_from_model_id():
    model_id = "froggeric/WestLake-10.7B-v2-GGUF"
    model_size = get_model_size_from_model_id(model_id)
    assert model_size == "10_7"

    model_id = "m-a-p/OpenCodeInterpreter-DS-33B"
    model_size = get_model_size_from_model_id(model_id)
    assert model_size == 33

    model_id = "MBZUAI/MobiLlama-05B"
    model_size = get_model_size_from_model_id(model_id)
    assert model_size == "0_5"

    model_id = "ibivibiv/alpaca-dragon-72b-v1"
    model_size = get_model_size_from_model_id(model_id)
    assert model_size == 72

    model_id = "ISTA-DASLab/Mixtral-8x7B-Instruct-v0_1-AQLM-2Bit-1x16-hf"
    model_size = get_model_size_from_model_id(model_id)
    assert model_size == 7

    model_id = "internlm/internlm-xcomposer2-vl-7b-4bit"
    model_size = get_model_size_from_model_id(model_id)
    assert model_size == 7

    model_id = "ahxt/LiteLlama-460M-1T"
    model_size = get_model_size_from_model_id(model_id)
    assert model_size == "0_46"

    model_id = "Dracones/Midnight-Miqu-70B-v1.0_exl2_2.24bpw"
    model_size = get_model_size_from_model_id(model_id)
    assert model_size == 70

    model_id = "MaziyarPanahi/MixTAO-7Bx2-MoE-v8.1-GGUF"
    model_size = get_model_size_from_model_id(model_id)
    assert model_size == 7

    model_id = "ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf"
    model_size = get_model_size_from_model_id(model_id)
    assert model_size == 7

    model_id = "stabilityai/stablelm-2-zephyr-1_6b"
    model_size = get_model_size_from_model_id(model_id)
    assert model_size == "1_6"

    model_id = "Qwen/Qwen1.5-Chat-4bit-GPTQ-72B"
    model_size = get_model_size_from_model_id(model_id)
    assert model_size == 72

    model_id = "m-a-p/OpenCodeInterpreter-3Bee-DS-33B"
    model_size = get_model_size_from_model_id(model_id)
    assert model_size == 33

    model_id = "qwen/Qwen1.5-0.5B-Chat"
    model_size = get_model_size_from_model_id(model_id)
    assert model_size == "0_5"

    model_id = "mlx-community/c4ai-command-r-v01-4bit"
    model_size = get_model_size_from_model_id(model_id)
    assert model_size == 0

    model_id = "lemonilia/ShoriRP-v0.75d"
    model_size = get_model_size_from_model_id(model_id)
    assert model_size == 0

    model_id = "abc"
    with pytest.raises(ValueError, match=r"Cannot parse model_id: .+"):
        get_model_size_from_model_id(model_id)


def test_get_match_quantization_filenames():
    filenames = [
        "kafkalm-70b-german-v0.1.Q2_K.gguf",
        "kafkalm-70b-german-v0.1.Q3_K_L.gguf",
        "kafkalm-70b-german-v0.1.Q3_K_M.gguf",
        "kafkalm-70b-german-v0.1.Q3_K_S.gguf",
        "kafkalm-70b-german-v0.1.Q4_0.gguf",
        "kafkalm-70b-german-v0.1.Q4_K_M.gguf",
        "kafkalm-70b-german-v0.1.Q4_K_S.gguf",
        "kafkalm-70b-german-v0.1.Q5_K_M.gguf",
        "kafkalm-70b-german-v0.1.Q5_K_S.gguf",
        "kafkalm-70b-german-v0.1.Q6_K.gguf-split-a",
        "kafkalm-70b-german-v0.1.Q6_K.gguf-split-b",
        "kafkalm-70b-german-v0.1.Q8_0.gguf-split-a",
        "kafkalm-70b-german-v0.1.Q8_0.gguf-split-b",
    ]

    results = get_match_quantization_filenames(filenames)
    assert len(results) == 13
    assert all(x[0][: x[2]] == "kafkalm-70b-german-v0.1." for x in results)
    assert all(x[1].upper() in SUPPORTED_QUANTIZATIONS for x in results)
    assert results[0][0][results[0][2] + len(results[0][1]) :] == ".gguf"
    assert results[-1][0][results[-1][2] + len(results[-1][1]) :] == ".gguf-split-b"


def test_get_prefix_suffix():
    names = [
        ".gguf-split-a",
        ".gguf-split-b",
        ".gguf-split-a",
        ".gguf-split-b",
        ".gguf-split-c",
    ]
    prefix, suffix = get_prefix_suffix(names)
    assert prefix == ".gguf-split-"
    assert suffix == ""

    names = ["-part-a.gguf", "-part-b.gguf", "-part-c.gguf", "-part-a.gguf"]

    prefix, suffix = get_prefix_suffix(names)
    assert prefix == "-part-"
    assert suffix == ".gguf"

    names = ["-part-1.gguf", "-part-2.gguf", "-part-12.gguf", "-part-2.gguf"]

    prefix, suffix = get_prefix_suffix(names)
    assert prefix == "-part-"
    assert suffix == ".gguf"

    names = [".gguf", "-part-1.gguf", "-part-2.gguf", "-part-12.gguf", "-part-2.gguf"]
    prefix, suffix = get_prefix_suffix(names)
    assert prefix == ""
    assert suffix == ".gguf"

    names = [
        "-test.gguf",
        "-test-part-1.gguf",
        "-test-part-2.gguf",
        "-test-part-12.gguf",
        "-test-part-2.gguf",
    ]
    prefix, suffix = get_prefix_suffix(names)
    assert prefix == "-test"
    assert suffix == ".gguf"

    names = ["-part-1.gguf", "-part-1.gguf", "-part-1.gguf"]
    prefix, suffix = get_prefix_suffix(names)
    assert prefix == "-part-1.gguf"
    assert suffix == ""

    prefix, suffix = get_prefix_suffix([])
    assert prefix == ""
    assert suffix == ""

    names = ["-only-1.gguf"]
    prefix, suffix = get_prefix_suffix(names)
    assert prefix == "-only-1.gguf"
    assert suffix == ""


def test_get_llama_cpp_quantization_info():
    filenames = [
        "kafkalm-70b-german-v0.1.Q2_K.gguf",
        "kafkalm-70b-german-v0.1.Q3_K_L.gguf",
        "kafkalm-70b-german-v0.1.Q3_K_M.gguf",
        "kafkalm-70b-german-v0.1.Q3_K_S.gguf",
        "kafkalm-70b-german-v0.1.Q4_0.gguf",
        "kafkalm-70b-german-v0.1.Q4_K_M.gguf",
        "kafkalm-70b-german-v0.1.Q4_K_S.gguf",
        "kafkalm-70b-german-v0.1.Q5_K_M.gguf",
        "kafkalm-70b-german-v0.1.Q5_K_S.gguf",
        "kafkalm-70b-german-v0.1.Q6_K.gguf-split-a",
        "kafkalm-70b-german-v0.1.Q6_K.gguf-split-b",
        "kafkalm-70b-german-v0.1.Q8_0.gguf-split-a",
        "kafkalm-70b-german-v0.1.Q8_0.gguf-split-b",
    ]

    tpl1, tpl2, qs, parts = get_llama_cpp_quantization_info(filenames[:-4], "ggufv2")
    assert tpl1 == "kafkalm-70b-german-v0.1.{quantization}.gguf"
    assert tpl2 is None
    assert len(qs) == 9
    assert {
        "Q2_K",
        "Q3_K_L",
        "Q3_K_M",
        "Q3_K_S",
        "Q4_0",
        "Q4_K_M",
        "Q4_K_S",
        "Q5_K_M",
        "Q5_K_S",
    }.intersection(set(qs)) == set(qs)
    assert parts is None

    tpl1, tpl2, qs, parts = get_llama_cpp_quantization_info(filenames, "ggufv2")
    assert tpl1 == "kafkalm-70b-german-v0.1.{quantization}.gguf"
    assert tpl2 == "kafkalm-70b-german-v0.1.{quantization}.gguf-split-{part}"
    assert len(qs) == 11
    assert {
        "Q2_K",
        "Q3_K_L",
        "Q3_K_M",
        "Q3_K_S",
        "Q4_0",
        "Q4_K_M",
        "Q4_K_S",
        "Q5_K_M",
        "Q5_K_S",
        "Q6_K",
        "Q8_0",
    }.intersection(set(qs)) == set(qs)
    assert len(parts) == 2
    assert len(parts["Q6_K"]) == 2
    assert len(parts["Q8_0"]) == 2
    assert parts["Q6_K"][0] == "a"
    assert parts["Q8_0"][1] == "b"

    filenames = [
        "kafkalm-70b-german-v0.1.Q2_K.test.gguf",
        "kafkalm-70b-german-v0.1.Q3_K_L.test.gguf",
        "kafkalm-70b-german-v0.1.Q3_K_M.test.gguf",
        "kafkalm-70b-german-v0.1.Q3_K_S.test.gguf",
        "kafkalm-70b-german-v0.1.Q4_0.test.gguf",
        "kafkalm-70b-german-v0.1.Q4_K_M.test.gguf",
        "kafkalm-70b-german-v0.1.Q4_K_S.test.gguf",
        "kafkalm-70b-german-v0.1.Q5_K_M.test.gguf",
        "kafkalm-70b-german-v0.1.Q5_K_S.test.gguf",
        "kafkalm-70b-german-v0.1.Q6_K.test-split-a.gguf",
        "kafkalm-70b-german-v0.1.Q6_K.test-split-b.gguf",
        "kafkalm-70b-german-v0.1.Q8_0.test-split-a.gguf",
        "kafkalm-70b-german-v0.1.Q8_0.test-split-b.gguf",
    ]

    tpl1, tpl2, qs, parts = get_llama_cpp_quantization_info(filenames, "ggufv2")
    assert tpl1 == "kafkalm-70b-german-v0.1.{quantization}.test.gguf"
    assert tpl2 == "kafkalm-70b-german-v0.1.{quantization}.test-split-{part}.gguf"
    assert len(qs) == 11
    assert len(parts) == 2
    assert {
        "Q2_K",
        "Q3_K_L",
        "Q3_K_M",
        "Q3_K_S",
        "Q4_0",
        "Q4_K_M",
        "Q4_K_S",
        "Q5_K_M",
        "Q5_K_S",
        "Q6_K",
        "Q8_0",
    }.intersection(set(qs)) == set(qs)
    assert parts["Q8_0"][1] == "b"

    filenames = [
        "kafkalm-70b-german-v0.1.Q2_K.test.gguf",
        "kafkalm-70b-german-v0.1.Q3_K_L.test.gguf",
        "kafkalm-70b-german-v0.1.Q3_K_M.test.gguf",
        "kafkalm-70b-german-v0.1.Q3_K_S.test.gguf",
        "kafkalm-70b-german-v0.1.Q4_0.test.gguf",
        "kafkalm-70b-german-v0.1.Q4_K_M.test.gguf",
        "kafkalm-70b-german-v0.1.Q4_K_S.test.gguf",
        "kafkalm-70b-german-v0.1.Q5_K_M.test.gguf",
        "kafkalm-70b-german-v0.1.Q5_K_S.test.gguf",
        "kafkalm-70b-german-v0.1.Q6_K.test.gguf-part1of2",
        "kafkalm-70b-german-v0.1.Q6_K.test.gguf-part2of2",
        "kafkalm-70b-german-v0.1.Q8_0.test.gguf-part1of3",
        "kafkalm-70b-german-v0.1.Q8_0.test.gguf-part2of3",
        "kafkalm-70b-german-v0.1.Q8_0.test.gguf-part3of3",
    ]

    tpl1, tpl2, qs, parts = get_llama_cpp_quantization_info(filenames, "ggufv2")
    assert tpl1 == "kafkalm-70b-german-v0.1.{quantization}.test.gguf"
    assert tpl2 == "kafkalm-70b-german-v0.1.{quantization}.test.gguf-part{part}"
    assert len(qs) == 11
    assert {
        "Q2_K",
        "Q3_K_L",
        "Q3_K_M",
        "Q3_K_S",
        "Q4_0",
        "Q4_K_M",
        "Q4_K_S",
        "Q5_K_M",
        "Q5_K_S",
        "Q6_K",
        "Q8_0",
    }.intersection(set(qs)) == set(qs)
    assert len(parts) == 2
    assert len(parts["Q8_0"]) == 3
    assert parts["Q8_0"][2] == "3of3"

    filenames = [
        "llama-2-7b-chat.ggmlv3.q2_K.bin",
        "llama-2-7b-chat.ggmlv3.q3_K_L.bin",
        "llama-2-7b-chat.ggmlv3.q3_K_M.bin",
        "llama-2-7b-chat.ggmlv3.q3_K_S.bin",
        "llama-2-7b-chat.ggmlv3.q4_0.bin",
        "llama-2-7b-chat.ggmlv3.q4_K_M.bin",
        "llama-2-7b-chat.ggmlv3.q4_K_S.bin",
        "llama-2-7b-chat.ggmlv3.q5_K_M.bin",
        "llama-2-7b-chat.ggmlv3.q5_K_S.bin",
    ]

    tpl1, tpl2, qs, parts = get_llama_cpp_quantization_info(filenames, "ggmlv3")

    assert tpl1 == "llama-2-7b-chat.ggmlv3.{quantization}.bin"
    assert tpl2 is None
    assert len(qs) == 9
    assert parts is None
