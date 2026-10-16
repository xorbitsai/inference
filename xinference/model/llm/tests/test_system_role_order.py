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

from types import SimpleNamespace

import pytest

from ..llm_family import LLMFamilyV2, PytorchLLMSpecV2, is_strict_system_first_template
from ..utils import MessageRoleOrderError, check_system_role_order

# A chat template carrying the Qwen3-family system-first guard.
STRICT_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message.role == 'system' and not loop.first %}"
    "{{- raise_exception('System message must be at the beginning.') }}"
    "{%- endif %}"
    "{%- endfor %}"
)
# A lenient template with no system-first guard.
LENIENT_TEMPLATE = "{%- for message in messages %}{{ message.role }}{%- endfor %}"


def _spec() -> PytorchLLMSpecV2:
    return PytorchLLMSpecV2(
        model_format="pytorch",
        model_size_in_billions=1,
        quantization="none",
    )


def _family(chat_template):
    return LLMFamilyV2(
        version=2,
        model_name="test-strict",
        model_lang=["en"],
        model_ability=["chat"],
        model_specs=[_spec()],
        chat_template=chat_template,
    )


# --- is_strict_system_first_template ----------------------------------------


def test_is_strict_true_for_guarded_template():
    assert is_strict_system_first_template(STRICT_TEMPLATE) is True


def test_is_strict_true_case_insensitive():
    # Lowercase variant of the guard must still be detected.
    lower_template = STRICT_TEMPLATE.lower()
    assert is_strict_system_first_template(lower_template) is True

    # Mixed-case variant.
    mixed_template = STRICT_TEMPLATE.replace(
        "System message must be at the beginning",
        "System Message Must Be At The Beginning",
    )
    assert is_strict_system_first_template(mixed_template) is True


def test_is_strict_false_for_lenient_template():
    assert is_strict_system_first_template(LENIENT_TEMPLATE) is False


def test_is_strict_false_for_none_and_empty():
    assert is_strict_system_first_template(None) is False
    assert is_strict_system_first_template("") is False


# --- to_description exposes strict_system_first -----------------------------


def test_to_description_strict_flag():
    assert _family(STRICT_TEMPLATE).to_description()["strict_system_first"] is True
    assert _family(LENIENT_TEMPLATE).to_description()["strict_system_first"] is False
    assert _family(None).to_description()["strict_system_first"] is False


# --- check_system_role_order ------------------------------------------------


def test_check_raises_on_non_first_system():
    messages = [
        {"role": "system", "content": "a"},
        {"role": "user", "content": "u"},
        {"role": "system", "content": "b"},  # non-first system
    ]
    with pytest.raises(MessageRoleOrderError) as exc:
        check_system_role_order(messages)
    msg = str(exc.value)
    assert "position 2" in msg
    assert "messages[0]" in msg


def test_check_passes_leading_system_only():
    # system only at index 0 must be allowed.
    check_system_role_order(
        [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "ok"},
        ]
    )


def test_check_passes_no_system():
    check_system_role_order(
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "ok"},
        ]
    )


def test_check_passes_normal_tool_sequence():
    # The legitimate multi-turn tool-calling shape from the bug report.
    check_system_role_order(
        [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "{}"},
            {"role": "user", "content": "go on"},
        ]
    )


def test_check_passes_empty_messages():
    # Empty and None messages lists must not raise.
    check_system_role_order([])
    check_system_role_order(None)  # type: ignore[arg-type]


def test_check_handles_non_dict_messages():
    # Messages parsed into objects (e.g. pydantic models) must still be
    # detected via the attribute path.
    messages = [
        SimpleNamespace(role="system", content="a"),
        SimpleNamespace(role="user", content="u"),
        SimpleNamespace(role="system", content="b"),
    ]
    with pytest.raises(MessageRoleOrderError):
        check_system_role_order(messages)
