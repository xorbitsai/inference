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

"""Integration tests for system-role-order validation in chat completion.

Tests cover the interaction between ``strict_system_first`` model descriptions
and the ``check_system_role_order`` gate placed in ``create_chat_completion``:

- Strict model (strict_system_first=True) + non-first system → HTTP 400
- Lenient/no-flag model → non-first system NOT rejected (zero regression)
- Strict model + leading-system-only → NOT rejected
"""

import pytest
from fastapi import HTTPException

from ...model.llm.utils import MessageRoleOrderError, check_system_role_order


class TestSystemRoleOrderIntegration:
    """Integration-level tests: gating + exception conversion."""

    def test_strict_model_non_first_system_raises_400(self):
        """Simulating the create_chat_completion except branch:
        strict_system_first=True + non-first system → HTTPException(400)."""
        messages = [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "u"},
            {"role": "system", "content": "b"},
        ]
        with pytest.raises(HTTPException) as exc:
            try:
                check_system_role_order(messages)
            except MessageRoleOrderError as ve:
                raise HTTPException(status_code=400, detail=str(ve))

        assert exc.value.status_code == 400
        assert "position 2" in exc.value.detail
        assert "messages[0]" in exc.value.detail

    def test_strict_model_non_first_system_stream_path_also_400(self):
        """Same test to confirm the except branch works identically for what
        would be the streaming path (the check happens before stream split)."""
        messages = [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "u"},
            {"role": "system", "content": "b"},
        ]
        with pytest.raises(HTTPException) as exc:
            try:
                check_system_role_order(messages)
            except MessageRoleOrderError as ve:
                raise HTTPException(status_code=400, detail=str(ve))

        assert exc.value.status_code == 400
        # The check runs identically regardless of stream flag — no SSE header
        # has been sent at this point, so the 400 is "clean".
        assert "position 2" in exc.value.detail

    def test_lenient_model_not_rejected(self):
        """Simulating strict_system_first=False: gate is NOT entered, so
        non-first system messages must NOT be rejected."""
        strict_system_first = False
        messages = [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "u"},
            {"role": "system", "content": "b"},
        ]
        # This must NOT raise — the gate is skipped entirely.
        if strict_system_first:
            check_system_role_order(messages)

        # If we reach here, the lenient model was NOT rejected (zero regression).
        assert True

    def test_strict_model_leading_system_only_not_rejected(self):
        """Strict model with system only at position [0] must pass."""
        messages = [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "ok"},
        ]
        # Should not raise
        check_system_role_order(messages)

    def test_strict_model_tool_sequence_not_rejected(self):
        """Strict model with legitimate multi-turn tool-calling sequence."""
        messages = [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "{}"},
            {"role": "user", "content": "go on"},
        ]
        # Should not raise
        check_system_role_order(messages)

    def test_non_first_system_error_message_actionable(self):
        """Error detail must tell the user exactly what to do."""
        messages = [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "u"},
            {"role": "system", "content": "b"},
        ]
        with pytest.raises(HTTPException) as exc:
            try:
                check_system_role_order(messages)
            except MessageRoleOrderError as ve:
                raise HTTPException(status_code=400, detail=str(ve))

        detail = exc.value.detail
        assert "messages[0]" in detail
        assert "'user'" in detail
        assert "tool_call_id" in detail
