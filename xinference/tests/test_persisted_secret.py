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
"""Regression tests for ``_get_or_create_persisted_secret``.

Covers the two issues raised in Gemini review on PR #5144:

* The wait loop must be bounded by a wall-clock deadline that comfortably
  exceeds the stale-file grace period, not a fixed iteration count that can
  expire before a file is old enough to be considered abandoned.
* A freshly created (not yet stale) empty file must not be deleted or
  mistaken for a failure -- the waiting process should keep polling until
  either the writer finishes or the file crosses the stale threshold.

    pytest xinference/tests/test_persisted_secret.py -v
"""

import os
import time

import pytest

from xinference import constants as xconst


@pytest.fixture(autouse=True)
def _isolated_auth_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(xconst, "XINFERENCE_AUTH_DIR", str(tmp_path))
    yield


def test_generates_and_persists_on_first_call():
    value = xconst._get_or_create_persisted_secret("XINFERENCE_TEST_SECRET", "secret")
    path = os.path.join(xconst.XINFERENCE_AUTH_DIR, "secret")
    assert os.path.exists(path)
    with open(path) as f:
        assert f.read().strip() == value


def test_second_call_reuses_persisted_value():
    first = xconst._get_or_create_persisted_secret("XINFERENCE_TEST_SECRET", "secret")
    second = xconst._get_or_create_persisted_secret("XINFERENCE_TEST_SECRET", "secret")
    assert first == second


def test_env_var_short_circuits_file_generation(monkeypatch):
    monkeypatch.setenv("XINFERENCE_TEST_SECRET", "from-env")
    value = xconst._get_or_create_persisted_secret("XINFERENCE_TEST_SECRET", "secret")
    assert value == "from-env"
    assert not os.path.exists(os.path.join(xconst.XINFERENCE_AUTH_DIR, "secret"))


def test_fresh_empty_file_is_not_treated_as_stale(monkeypatch):
    """A just-created empty file (writer hasn't flushed yet) must not be
    deleted before the stale grace period elapses, and the waiter must not
    time out before that -- reproduces the bug where a fixed iteration
    budget could expire before the stale threshold could ever be reached.
    """
    monkeypatch.setattr(xconst, "_STALE_SECRET_GRACE_SECONDS", 0.3)
    monkeypatch.setattr(xconst, "_SECRET_WAIT_DEADLINE_SECONDS", 2)

    path = os.path.join(xconst.XINFERENCE_AUTH_DIR, "secret")
    os.makedirs(xconst.XINFERENCE_AUTH_DIR, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)  # file exists, empty, freshly created (mtime ~= now)

    write_after = {"done": False}
    real_sleep = time.sleep

    def fake_sleep(seconds):
        # Simulate another process finishing the write shortly after the
        # first poll -- well before the (shrunk) stale grace period, and
        # well within the wait deadline.
        if not write_after["done"]:
            with open(path, "w") as f:
                f.write("winner-value")
            write_after["done"] = True
        real_sleep(0.01)

    monkeypatch.setattr(time, "sleep", fake_sleep)

    value = xconst._get_or_create_persisted_secret("XINFERENCE_TEST_SECRET", "secret")
    assert value == "winner-value"


def test_stale_empty_file_is_removed_and_regenerated(monkeypatch):
    """An empty file older than the stale grace period is abandoned by a
    crashed writer and must be cleaned up so startup can recover instead of
    raising ``RuntimeError`` forever. Uses a shrunk grace period and wait
    deadline so the test doesn't need to sleep for the real 10s/30s.
    """
    monkeypatch.setattr(xconst, "_STALE_SECRET_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(xconst, "_SECRET_WAIT_DEADLINE_SECONDS", 5)

    path = os.path.join(xconst.XINFERENCE_AUTH_DIR, "secret")
    os.makedirs(xconst.XINFERENCE_AUTH_DIR, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)

    old_mtime = time.time() - 3600
    os.utime(path, (old_mtime, old_mtime))

    value = xconst._get_or_create_persisted_secret("XINFERENCE_TEST_SECRET", "secret")
    assert value
    with open(path) as f:
        assert f.read().strip() == value


def test_wait_deadline_exceeds_stale_grace_period():
    """Guards the exact bug Gemini reported: the overall wait budget must
    outlast the stale-file grace period, or a fresh empty file could never
    be reclaimed before the function gives up and raises ``RuntimeError``.
    """
    assert xconst._SECRET_WAIT_DEADLINE_SECONDS > xconst._STALE_SECRET_GRACE_SECONDS


def test_raises_if_never_resolved_within_deadline(monkeypatch):
    """If the file never becomes readable and never goes stale within the
    wait deadline, the function must fail loudly rather than hang forever.
    """
    monkeypatch.setattr(xconst, "_STALE_SECRET_GRACE_SECONDS", 999)
    monkeypatch.setattr(xconst, "_SECRET_WAIT_DEADLINE_SECONDS", 0.2)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    path = os.path.join(xconst.XINFERENCE_AUTH_DIR, "secret")
    os.makedirs(xconst.XINFERENCE_AUTH_DIR, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)  # left empty forever, and grace period never elapses

    with pytest.raises(RuntimeError):
        xconst._get_or_create_persisted_secret("XINFERENCE_TEST_SECRET", "secret")
