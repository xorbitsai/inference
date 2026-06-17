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

"""Unit tests for B1 v3.2: _strip_test_envs function.

Tests cover:
- XINFERENCE_TEST_* prefix stripping
- Return value semantics (cleaned, stripped_keys)
- No-strip returns empty set
- All-test-envs pops the envs key
- No-envs field
- Partial prefix not stripped
- envs reference isolation (no reverse pollution)
- Non-dict envs -> error log + skip
- None envs
- Exception safety (WeirdDict fallback)
"""

from xinference.core.worker import _strip_test_envs


def test_strip_test_envs_removes_prefix():
    """XINFERENCE_TEST_* prefix keys are all stripped."""
    args = {
        "model_uid": "uid",
        "envs": {
            "XINFERENCE_TEST_OUT_OF_MEMORY_ERROR": "1",
            "XINFERENCE_TEST_ENGINE_DEAD": "1",
            "XINFERENCE_TEST_FUTURE_NEW": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "VLLM_USE_V1": "0",
        },
    }
    cleaned, stripped = _strip_test_envs(args)
    assert stripped == {
        "XINFERENCE_TEST_OUT_OF_MEMORY_ERROR",
        "XINFERENCE_TEST_ENGINE_DEAD",
        "XINFERENCE_TEST_FUTURE_NEW",
    }
    assert cleaned["envs"] == {
        "CUDA_VISIBLE_DEVICES": "0",
        "VLLM_USE_V1": "0",
    }
    # Input not mutated
    assert "XINFERENCE_TEST_OUT_OF_MEMORY_ERROR" in args["envs"]


def test_strip_test_envs_returns_stripped_keys():
    """Returns stripped_keys set for precise detection."""
    args = {"envs": {"XINFERENCE_TEST_FOO": "1", "CUDA_VISIBLE_DEVICES": "0"}}
    cleaned, stripped = _strip_test_envs(args)
    assert stripped == {"XINFERENCE_TEST_FOO"}
    assert cleaned["envs"] == {"CUDA_VISIBLE_DEVICES": "0"}


def test_strip_test_envs_no_strip_returns_empty_set():
    """When nothing is stripped, stripped_keys is empty."""
    args = {"envs": {"CUDA_VISIBLE_DEVICES": "0"}}
    cleaned, stripped = _strip_test_envs(args)
    assert stripped == set()


def test_strip_test_envs_all_test_envs_pops_key():
    """When all envs are test envs, the envs key is popped."""
    args = {"envs": {"XINFERENCE_TEST_OUT_OF_MEMORY_ERROR": "1"}}
    cleaned, stripped = _strip_test_envs(args)
    assert "envs" not in cleaned
    assert stripped == {"XINFERENCE_TEST_OUT_OF_MEMORY_ERROR"}


def test_strip_test_envs_no_envs():
    """No envs field: no error, no strip."""
    args = {"model_uid": "uid"}
    cleaned, stripped = _strip_test_envs(args)
    assert "envs" not in cleaned
    assert stripped == set()


def test_strip_test_envs_partial_prefix_not_stripped():
    """Non XINFERENCE_TEST_ prefixed envs are NOT stripped."""
    args = {
        "envs": {
            "XINFERENCE_SOMETHING": "1",  # Not TEST_ prefix
            "XINFERENCE_TEST_FOO": "1",
        },
    }
    cleaned, stripped = _strip_test_envs(args)
    assert cleaned["envs"] == {"XINFERENCE_SOMETHING": "1"}
    assert stripped == {"XINFERENCE_TEST_FOO"}


def test_strip_test_envs_envs_not_shared():
    """cleaned envs dict is not shared with input (no reverse pollution)."""
    original_envs = {"CUDA_VISIBLE_DEVICES": "0"}
    args = {"envs": original_envs}
    cleaned, _ = _strip_test_envs(args)
    # Mutating cleaned["envs"] must not affect original args["envs"]
    cleaned["envs"]["NEW_KEY"] = "1"
    assert "NEW_KEY" not in args["envs"]
    assert "NEW_KEY" not in original_envs


def test_strip_test_envs_non_dict_envs_logs_error(caplog):
    """Non-dict envs triggers error log with diagnostic info (v3.2)."""
    args = {"model_uid": "test-uid", "envs": "not_a_dict"}
    with caplog.at_level("ERROR"):
        cleaned, stripped = _strip_test_envs(args)
    assert cleaned["envs"] == "not_a_dict"
    assert stripped == set()
    # v3.2: error log includes diagnostic info; impl logs the type name of envs
    assert any(
        "not dict" in record.message
        and "test-uid" in record.message
        and "str" in record.message
        for record in caplog.records
        if record.levelname == "ERROR"
    )


def test_strip_test_envs_none_envs():
    """envs is None: no error, no strip."""
    args = {"envs": None}
    cleaned, stripped = _strip_test_envs(args)
    assert cleaned.get("envs") is None
    assert stripped == set()


def test_strip_test_envs_exception_safety():
    """Overridden dict methods are handled gracefully.

    WeirdDict overrides get() but dict() constructor converts it to a normal
    dict first, so the function continues normally. This demonstrates robustness
    against unexpected dict subclasses.
    """

    class WeirdDict(dict):
        def get(self, *args, **kwargs):
            raise RuntimeError("weird")

    args = WeirdDict({"envs": {"XINFERENCE_TEST_FOO": "1"}})
    cleaned, stripped = _strip_test_envs(args)
    # Function converts WeirdDict to normal dict, then strips normally
    assert isinstance(cleaned, dict)
    # The strip still works because dict(args) bypasses the overridden get()
    assert stripped == {"XINFERENCE_TEST_FOO"}


def test_strip_test_envs_empty_envs_dict():
    """Empty envs dict: treated as empty, no strip."""
    args = {"envs": {}}
    cleaned, stripped = _strip_test_envs(args)
    # Empty dict evaluates to False, so the function returns early
    assert stripped == set()


def test_strip_test_envs_input_not_mutated():
    """Input launch_args dict is never mutated."""
    original_envs = {
        "XINFERENCE_TEST_FOO": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    args = {"envs": original_envs, "model_uid": "uid"}
    original_args_copy = {
        "envs": dict(original_envs),
        "model_uid": "uid",
    }
    _strip_test_envs(args)
    assert args == original_args_copy
    assert args["envs"] is original_envs  # same reference, not mutated
