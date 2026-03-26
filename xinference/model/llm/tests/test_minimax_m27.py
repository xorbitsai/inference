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

import json
import os

import pytest


@pytest.fixture
def minimax_m27_entry():
    """Load MiniMax-M2.7 entry from llm_family.json."""
    json_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm_family.json"
    )
    with open(json_path) as f:
        data = json.load(f)
    for entry in data:
        if entry.get("model_name") == "MiniMax-M2.7":
            return entry
    pytest.fail("MiniMax-M2.7 not found in llm_family.json")


def test_minimax_m27_exists_in_family(minimax_m27_entry):
    """MiniMax-M2.7 entry should exist in llm_family.json."""
    assert minimax_m27_entry is not None


def test_minimax_m27_context_length(minimax_m27_entry):
    """MiniMax-M2.7 should have 204800 context length."""
    assert minimax_m27_entry["context_length"] == 204800


def test_minimax_m27_languages(minimax_m27_entry):
    """MiniMax-M2.7 should support English and Chinese."""
    assert minimax_m27_entry["model_lang"] == ["en", "zh"]


def test_minimax_m27_abilities(minimax_m27_entry):
    """MiniMax-M2.7 should support chat, tools, and reasoning."""
    abilities = minimax_m27_entry["model_ability"]
    assert "chat" in abilities
    assert "tools" in abilities
    assert "reasoning" in abilities


def test_minimax_m27_pytorch_spec(minimax_m27_entry):
    """MiniMax-M2.7 pytorch spec should have correct model IDs and size."""
    specs = minimax_m27_entry["model_specs"]
    assert len(specs) >= 1

    pytorch_spec = specs[0]
    assert pytorch_spec["model_format"] == "pytorch"
    assert pytorch_spec["model_size_in_billions"] == 230
    assert pytorch_spec["activated_size_in_billions"] == 10
    assert (
        pytorch_spec["model_src"]["huggingface"]["model_id"] == "MiniMaxAI/MiniMax-M2.7"
    )
    assert pytorch_spec["model_src"]["modelscope"]["model_id"] == "MiniMax/MiniMax-M2.7"
    assert "none" in pytorch_spec["model_src"]["huggingface"]["quantizations"]


def test_minimax_m27_architecture(minimax_m27_entry):
    """MiniMax-M2.7 should use MiniMaxM2ForCausalLM architecture."""
    assert minimax_m27_entry["architectures"] == ["MiniMaxM2ForCausalLM"]


def test_minimax_m27_tool_parser(minimax_m27_entry):
    """MiniMax-M2.7 should use the minimax tool parser."""
    assert minimax_m27_entry["tool_parser"] == "minimax"


def test_minimax_m27_reasoning_tags(minimax_m27_entry):
    """MiniMax-M2.7 should have correct reasoning tags."""
    assert minimax_m27_entry["reasoning_start_tag"] == "<think>"
    assert minimax_m27_entry["reasoning_end_tag"] == "</think>"


def test_minimax_m27_stop_tokens(minimax_m27_entry):
    """MiniMax-M2.7 should have correct stop tokens."""
    assert minimax_m27_entry["stop_token_ids"] == [200020]
    assert minimax_m27_entry["stop"] == ["[e~["]


def test_minimax_m27_chat_template(minimax_m27_entry):
    """MiniMax-M2.7 chat template should reference correct model identity."""
    assert "MiniMax-M2.7" in minimax_m27_entry["chat_template"]
    # Should not reference M2.5 identity
    assert (
        "MiniMax-M2.5 and is built by MiniMax" not in minimax_m27_entry["chat_template"]
    )


def test_minimax_m27_version(minimax_m27_entry):
    """MiniMax-M2.7 should use version 2 format."""
    assert minimax_m27_entry["version"] == 2
