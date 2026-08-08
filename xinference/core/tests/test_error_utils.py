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
from ...constants import XINFERENCE_ENV_DISABLE_ERROR_TRACEBACK
from ..error_utils import (
    MAX_SUMMARY_LENGTH,
    MAX_TRACEBACK_LENGTH,
    AggregatedLaunchError,
    format_error_summary,
    format_error_traceback,
    format_replica_errors,
    root_cause,
    strip_actor_prefix,
)


def _raise_chain():
    """Raise a 3-deep cause chain the way a real launch failure produces one."""

    def deepest():
        raise FileNotFoundError("no such file: /nonexistent/model")

    def middle():
        try:
            deepest()
        except Exception as e:
            raise RuntimeError("engine init failed") from e

    try:
        middle()
    except Exception as e:
        raise RuntimeError("[address=127.0.0.1:1234, pid=99] launch failed") from e


def test_strip_actor_prefix():
    assert strip_actor_prefix("[address=127.0.0.1:1234, pid=99] boom") == "boom"
    # Negative pid (xoscar uses -1 when the pool is unknown).
    assert strip_actor_prefix("[address=host:80, pid=-1] boom") == "boom"
    # Untouched when there is no prefix.
    assert strip_actor_prefix("plain message") == "plain message"
    # Only a leading prefix is stripped, not one embedded in the text.
    assert (
        strip_actor_prefix("boom [address=h:1, pid=2] tail")
        == "boom [address=h:1, pid=2] tail"
    )


def test_root_cause_walks_to_deepest():
    try:
        _raise_chain()
    except Exception as e:
        root = root_cause(e)
    assert isinstance(root, FileNotFoundError)
    assert "nonexistent/model" in str(root)


def test_root_cause_without_chain_returns_input():
    exc = ValueError("solo")
    assert root_cause(exc) is exc


def test_root_cause_respects_suppressed_context():
    try:
        try:
            raise ValueError("inner")
        except ValueError:
            raise RuntimeError("outer") from None
    except RuntimeError as e:
        # `from None` sets __suppress_context__, so the inner error is not the
        # root cause the user should be shown.
        assert root_cause(e) is e


def test_root_cause_terminates_on_cycle():
    a = ValueError("a")
    b = ValueError("b")
    a.__cause__ = b
    b.__cause__ = a
    # Must terminate rather than loop forever.
    assert root_cause(a) is b


def test_format_error_summary_reports_root_with_type():
    try:
        _raise_chain()
    except Exception as e:
        assert (
            format_error_summary(e)
            == "FileNotFoundError: no such file: /nonexistent/model"
        )


def test_format_error_summary_strips_actor_prefix():
    exc = RuntimeError("[address=127.0.0.1:1234, pid=99] boom")
    assert format_error_summary(exc) == "RuntimeError: boom"


def test_format_error_summary_falls_back_when_root_message_empty():
    try:
        raise RuntimeError("outer message") from ValueError()
    except RuntimeError as e:
        # The root has no message of its own, so the outer one is used rather
        # than emitting a bare type name.
        assert format_error_summary(e) == "ValueError: outer message"


def test_format_error_summary_handles_fully_empty_message():
    assert format_error_summary(RuntimeError()) == "RuntimeError"


def test_format_error_summary_truncates():
    summary = format_error_summary(RuntimeError("x" * (MAX_SUMMARY_LENGTH * 2)))
    assert len(summary) <= MAX_SUMMARY_LENGTH + len("... [truncated]")
    assert summary.endswith("... [truncated]")


def test_format_error_traceback_contains_frames():
    def raising_helper():
        raise ValueError("kaboom")

    try:
        raising_helper()
    except ValueError as e:
        tb = format_error_traceback(e)
    assert tb is not None
    assert "raising_helper" in tb
    assert "ValueError: kaboom" in tb


def test_format_error_traceback_includes_cause_chain():
    try:
        _raise_chain()
    except Exception as e:
        tb = format_error_traceback(e)
    assert tb is not None
    # Every layer of the chain must be visible, not just the outermost.
    assert "FileNotFoundError" in tb
    assert "engine init failed" in tb


def test_format_error_traceback_disabled_by_env(monkeypatch):
    monkeypatch.setenv(XINFERENCE_ENV_DISABLE_ERROR_TRACEBACK, "1")
    assert format_error_traceback(ValueError("boom")) is None


def test_format_error_traceback_enabled_when_env_zero(monkeypatch):
    monkeypatch.setenv(XINFERENCE_ENV_DISABLE_ERROR_TRACEBACK, "0")
    assert format_error_traceback(ValueError("boom")) is not None


def test_format_error_traceback_truncates():
    exc = ValueError("y" * (MAX_TRACEBACK_LENGTH * 2))
    tb = format_error_traceback(exc)
    assert tb is not None
    assert len(tb) <= MAX_TRACEBACK_LENGTH + len("... [truncated]")
    assert tb.endswith("... [truncated]")


def test_format_replica_errors_one_line_per_failure():
    text = format_replica_errors(
        [
            ("model-1-0", ValueError("bad path")),
            ("model-1-1", RuntimeError("[address=h:1, pid=2] cuda oom")),
        ]
    )
    assert text.splitlines() == [
        "replica model-1-0: ValueError: bad path",
        "replica model-1-1: RuntimeError: cuda oom",
    ]


def test_format_replica_errors_empty():
    assert format_replica_errors([]) == ""


def test_aggregate_is_not_unwrapped_to_first_failure():
    """The aggregate summarises every replica; unwrapping would hide the rest."""
    failures = [
        ("m-rep0", ValueError("bad path")),
        ("m-rep1", RuntimeError("cuda oom")),
    ]
    try:
        try:
            raise failures[0][1]
        except Exception as first:
            raise AggregatedLaunchError(format_replica_errors(failures)) from first
    except AggregatedLaunchError as agg:
        summary = format_error_summary(agg)
        # Both replicas must survive into the reported message.
        assert "m-rep0" in summary and "bad path" in summary
        assert "m-rep1" in summary and "cuda oom" in summary
        # The per-replica lines already name each type; no wrapper prefix.
        assert not summary.startswith("AggregatedLaunchError")
        # The first failure stays reachable for the traceback.
        assert isinstance(agg.__cause__, ValueError)


def test_aggregate_maps_to_service_unavailable():
    """Subclassing RuntimeError keeps the REST layer's 503 mapping intact."""
    assert issubclass(AggregatedLaunchError, RuntimeError)
