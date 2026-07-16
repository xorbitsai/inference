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
"""Tests for SafeTimedAndSizeRotatingFileHandler and daily+size rotation."""

import logging
import os
import sys
import time
from unittest import mock

import pytest

from ..utils import SafeTimedAndSizeRotatingFileHandler, get_config_dict

_skip_on_windows = pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "rotation tests simulate a foreign-process rename of a file the "
        "handler still holds open; Windows lacks FILE_SHARE_DELETE by default "
        "so os.rename fails with WinError 32, and the production rotation "
        "path is itself Unix-only (fcntl-based)"
    ),
)


@pytest.fixture
def log_file(tmp_path):
    return str(tmp_path / "test.log")


@pytest.fixture
def handler(log_file):
    h = SafeTimedAndSizeRotatingFileHandler(
        filename=log_file,
        when="midnight",
        backupCount=3,
        maxBytes=100,
        retention_days=30,
        encoding="utf8",
    )
    h.setFormatter(logging.Formatter("%(message)s"))
    yield h
    h.close()


def _make_record(msg="hello", name="test"):
    return logging.LogRecord(
        name=name,
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=msg,
        args=None,
        exc_info=None,
    )


class TestShouldRollover:
    def test_no_rollover_below_maxbytes(self, handler):
        handler.maxBytes = 1000
        record = _make_record("short message")
        assert handler.shouldRollover(record) is False

    def test_rollover_when_exceeds_maxbytes(self, handler):
        handler.maxBytes = 10
        record = _make_record("a" * 20)
        assert handler.shouldRollover(record) is True

    def test_maxbytes_zero_skips_size_check(self, handler):
        handler.maxBytes = 0
        record = _make_record("a" * 10000)
        assert handler.shouldRollover(record) is False

    def test_stream_none_guard(self, handler):
        handler.maxBytes = 10
        handler.stream = None
        record = _make_record("a" * 20)
        # Should not raise AttributeError; should reopen stream
        assert handler.shouldRollover(record) is True
        assert handler.stream is not None


class TestDoRollover:
    def test_size_triggered_naming_incremental(self, handler, tmp_path):
        handler.maxBytes = 5
        handler.retention_days = 0
        handler.backupCount = 0

        # Write enough to trigger size rotation
        for i in range(5):
            handler.emit(_make_record(f"msg{i}"))

        files = sorted(os.listdir(tmp_path))
        # Should have test.log plus at least one .N file
        assert "test.log" in files
        rotated = [f for f in files if f.startswith("test.log.") and f != "test.log"]
        assert len(rotated) >= 1

    def test_size_triggered_does_not_update_rolloverAt(self, handler):
        original_rollover_at = handler.rolloverAt
        handler.maxBytes = 5
        handler.retention_days = 0
        handler.backupCount = 0

        handler.emit(_make_record("a" * 20))

        assert handler.rolloverAt == original_rollover_at

    def test_size_triggered_uses_rotation_filename(self, handler):
        handler.maxBytes = 5
        handler.retention_days = 0
        handler.backupCount = 0

        with mock.patch.object(
            handler, "rotation_filename", wraps=handler.rotation_filename
        ) as spy:
            handler.emit(_make_record("a" * 20))
            assert spy.called

    def test_size_triggered_calls_rotate(self, handler):
        handler.maxBytes = 5
        handler.retention_days = 0
        handler.backupCount = 0

        with mock.patch.object(handler, "rotate", wraps=handler.rotate) as spy:
            handler.emit(_make_record("a" * 20))
            assert spy.called

    def test_time_triggered_calls_super_doRollover(self, handler):
        handler.rolloverAt = int(time.time()) - 1  # past midnight

        with mock.patch("logging.handlers.TimedRotatingFileHandler.doRollover") as spy:
            handler.doRollover()
            spy.assert_called_once()

    @_skip_on_windows
    def test_foreign_rotation_advances_rolloverAt(self, handler, tmp_path):
        """When another process performs the time-based rollover, this
        handler's _check_inode_and_reopen reopens the stream and must
        advance rolloverAt; otherwise the stale past-due time triggers a
        redundant doRollover on the next emit that clobbers the archive
        the other process just wrote (log loss)."""
        handler.maxBytes = 1000  # disable size trigger
        handler.retention_days = 0
        handler.backupCount = 0
        handler.emit(_make_record("first-record"))
        handler.rolloverAt = int(time.time()) - 1  # midnight has passed

        # Simulate another process performing the midnight rollover: rename
        # the base file (holding "first-record") to the dated archive and
        # create a fresh base file with a new inode.
        base = handler.baseFilename
        date_suffix = time.strftime("%Y-%m-%d", time.localtime(time.time()))
        archive = f"{base}.{date_suffix}"
        os.rename(base, archive)
        with open(base, "w"):
            pass  # fresh file, new inode

        handler.emit(_make_record("second-record"))

        # The archive written by the other process must survive intact.
        assert os.path.exists(archive)
        with open(archive) as f:
            assert "first-record" in f.read()
        # The new record lands in the fresh base file, no redundant rotation.
        with open(base) as f:
            assert "second-record" in f.read()
        # rolloverAt was advanced past now.
        assert handler.rolloverAt > time.time()


class TestGetFilesToDelete:
    def test_matches_date_and_numbered_files(self, handler, tmp_path):
        # Create files with various names
        for name in [
            "test.log.2026-06-20",
            "test.log.2026-06-20.1",
            "test.log.2026-06-20.2",
            "test.log.2026-06-21",
            "test.log.2026-06-21.1",
            "irrelevant.log.2026-06-20",
            "test.log",
        ]:
            (tmp_path / name).write_text("x")

        handler.retention_days = 0
        handler.backupCount = 0
        result = handler.getFilesToDelete()
        # With both retention_days=0 and backupCount=0, nothing to delete
        assert result == []

    def test_excludes_unsuffixed_base_file(self, handler, tmp_path):
        (tmp_path / "test.log").write_text("x")
        handler.retention_days = 0
        handler.backupCount = 0
        result = handler.getFilesToDelete()
        assert result == []

    def test_retention_days_deletes_old_files(self, handler, tmp_path):
        # Create a file dated 100 days ago
        old_date = time.strftime("%Y-%m-%d", time.localtime(time.time() - 100 * 86400))
        (tmp_path / f"test.log.{old_date}").write_text("old")
        (tmp_path / "test.log.2099-01-01").write_text("future")

        handler.retention_days = 30
        handler.backupCount = 0
        result = handler.getFilesToDelete()
        assert any(old_date in p for p in result)

    def test_retention_days_1_keeps_yesterday(self, handler, tmp_path):
        """Regression: retention_days=1 must keep a full day. The archive's
        date_epoch is parsed at midnight while cutoff uses the current
        wall-clock time; without the +1 correction, yesterday's archive is
        deleted the moment the current time passes midnight, keeping zero
        full days."""
        yesterday = time.time() - 86400
        yest_date = time.strftime("%Y-%m-%d", time.localtime(yesterday))
        (tmp_path / f"test.log.{yest_date}").write_text("yesterday")
        today_date = time.strftime("%Y-%m-%d", time.localtime(time.time()))
        (tmp_path / f"test.log.{today_date}").write_text("today")

        handler.retention_days = 1
        handler.backupCount = 0
        result = handler.getFilesToDelete()
        # Yesterday's archive must be preserved (retention_days=1 keeps a
        # full day); today's is also preserved.
        assert not any(yest_date in p for p in result)
        assert not any(today_date in p for p in result)

    def test_backup_count_cap(self, handler, tmp_path):
        # Create 5 files, backupCount=2 → delete 3 oldest
        for i in range(5):
            f = tmp_path / f"test.log.2026-06-2{i}"
            f.write_text("x")
            # Set increasing mtime
            os.utime(str(f), (time.time() + i, time.time() + i))

        handler.retention_days = 0
        handler.backupCount = 2
        result = handler.getFilesToDelete()
        assert len(result) == 3

    def test_hybrid_retention_then_cap(self, handler, tmp_path):
        # 5 old files (distinct dates, 100+ days ago) + 5 recent files (future dates)
        for i in range(5):
            old_date = time.strftime(
                "%Y-%m-%d", time.localtime(time.time() - (100 + i) * 86400)
            )
            (tmp_path / f"test.log.{old_date}").write_text("old")
        for i in range(1, 6):
            f = tmp_path / f"test.log.2099-01-{i:02d}"
            f.write_text("recent")
            os.utime(str(f), (time.time() + i, time.time() + i))

        handler.retention_days = 30
        handler.backupCount = 3
        result = handler.getFilesToDelete()

        # 5 old deleted by retention + 2 recent deleted by cap (5-3=2)
        assert len(result) == 7

    def test_dir_not_exists_returns_empty(self, handler):
        handler.baseFilename = "/nonexistent/path/test.log"
        handler.retention_days = 30
        handler.backupCount = 10
        assert handler.getFilesToDelete() == []

    def test_invalid_date_format_skipped(self, handler, tmp_path):
        (tmp_path / "test.log.not-a-date").write_text("x")
        handler.retention_days = 30
        handler.backupCount = 0
        result = handler.getFilesToDelete()
        assert result == []


class TestGetConfigDict:
    def test_daily_plus_size_config(self, tmp_path):
        log_path = str(tmp_path / "xinf.log")
        config = get_config_dict(
            "INFO",
            log_path,
            log_backup_count=300,
            log_max_bytes=100 * 1024 * 1024,
            role="worker",
            address="localhost:9997",
            rotation="daily+size",
            log_retention_days=30,
        )

        fh_config = config["handlers"]["file_handler"]
        assert (
            fh_config["class"]
            == "xinference.deploy.utils.SafeTimedAndSizeRotatingFileHandler"
        )
        assert fh_config["maxBytes"] == 100 * 1024 * 1024
        assert fh_config["backupCount"] == 300
        assert fh_config["retention_days"] == 30
        assert fh_config["when"] == "midnight"

    def test_daily_config_unchanged(self, tmp_path):
        log_path = str(tmp_path / "xinf.log")
        config = get_config_dict(
            "INFO",
            log_path,
            log_backup_count=30,
            log_max_bytes=100,
            role="local",
            address="localhost:9997",
            rotation="daily",
            log_retention_days=30,
        )
        fh_config = config["handlers"]["file_handler"]
        assert (
            fh_config["class"] == "xinference.deploy.utils.SafeTimedRotatingFileHandler"
        )
        assert "maxBytes" not in fh_config

    def test_size_config_unchanged(self, tmp_path):
        log_path = str(tmp_path / "xinf.log")
        config = get_config_dict(
            "INFO",
            log_path,
            log_backup_count=300,
            log_max_bytes=100,
            role="worker",
            address="localhost:9997",
            rotation="size",
            log_retention_days=30,
        )
        fh_config = config["handlers"]["file_handler"]
        assert fh_config["class"] == "xinference.deploy.utils.SafeRotatingFileHandler"
        assert fh_config["maxBytes"] == 100


class TestCreateRotatingHandler:
    def test_daily_plus_size_handler(self, tmp_path):
        from ...core.log import create_rotating_handler

        log_path = str(tmp_path / "audit.log")
        handler = create_rotating_handler(
            filename=log_path,
            retention_days=90,
            rotation="daily+size",
            max_bytes=100 * 1024 * 1024,
            backup_count=300,
        )
        assert isinstance(handler, SafeTimedAndSizeRotatingFileHandler)
        assert handler.retention_days == 90
        assert handler.maxBytes == 100 * 1024 * 1024
        assert handler.backupCount == 300
        handler.close()

    def test_daily_handler_unchanged(self, tmp_path):
        from logging.handlers import TimedRotatingFileHandler

        from ...core.log import create_rotating_handler

        log_path = str(tmp_path / "audit.log")
        handler = create_rotating_handler(
            filename=log_path,
            retention_days=90,
            rotation="daily",
        )
        assert isinstance(handler, TimedRotatingFileHandler)
        assert not isinstance(handler, SafeTimedAndSizeRotatingFileHandler)
        handler.close()

    def test_size_handler_uses_safe_rotating(self, tmp_path):
        from ...core.log import create_rotating_handler
        from ..utils import SafeRotatingFileHandler

        log_path = str(tmp_path / "audit.log")
        handler = create_rotating_handler(
            filename=log_path,
            retention_days=30,
            rotation="size",
            max_bytes=100 * 1024 * 1024,
        )
        assert isinstance(handler, SafeRotatingFileHandler)
        assert handler.maxBytes == 100 * 1024 * 1024
        handler.close()


class TestSafeRotatingFileHandler:
    """Multi-process safety tests for SafeRotatingFileHandler (size mode)."""

    @_skip_on_windows
    def test_inode_check_reopens_after_rename(self, tmp_path):
        from ..utils import SafeRotatingFileHandler

        log_path = str(tmp_path / "test.log")
        h = SafeRotatingFileHandler(
            filename=log_path,
            maxBytes=1000,
            backupCount=3,
            encoding="utf8",
        )
        h.setFormatter(logging.Formatter("%(message)s"))
        h.emit(_make_record("first"))
        old_stream = h.stream
        os.rename(log_path, str(tmp_path / "test.log.1"))
        # ShouldRollover triggers inode check; stream should be reopened
        h.shouldRollover(_make_record("second"))
        assert h.stream is not old_stream
        h.close()

    def test_lock_file_created(self, tmp_path):
        from ..utils import SafeRotatingFileHandler

        log_path = str(tmp_path / "test.log")
        h = SafeRotatingFileHandler(
            filename=log_path,
            maxBytes=1000,
            backupCount=3,
            encoding="utf8",
        )
        h.close()
        assert os.path.exists(os.path.join(tmp_path, "test.log.rotate.lock"))

    def test_rolling_rename_on_size_trigger(self, tmp_path):
        from ..utils import SafeRotatingFileHandler

        log_path = str(tmp_path / "test.log")
        h = SafeRotatingFileHandler(
            filename=log_path,
            maxBytes=50,
            backupCount=2,
            encoding="utf8",
        )
        h.setFormatter(logging.Formatter("%(message)s"))
        # Write enough to trigger multiple rotations
        for i in range(20):
            h.emit(_make_record(f"message-{i}-" + "x" * 10))
        h.close()
        files = sorted(os.listdir(tmp_path))
        # Should have test.log plus .1 and .2 (backupCount=2)
        assert "test.log" in files
        assert "test.log.1" in files
        assert "test.log.2" in files


def _mp_worker_size(proc_id, log_path, max_bytes, n_records):
    """Module-level worker for multiprocessing (size mode)."""
    from ..utils import SafeRotatingFileHandler

    h = SafeRotatingFileHandler(
        filename=log_path,
        maxBytes=max_bytes,
        backupCount=200,
        encoding="utf8",
    )
    h.setFormatter(logging.Formatter("%(message)s"))
    for i in range(n_records):
        h.emit(_make_record(f"p{proc_id}-r{i}"))
    h.close()


def _mp_worker_daily_size(proc_id, log_path, max_bytes, n_records):
    """Module-level worker for multiprocessing (daily+size mode)."""
    from ..utils import SafeTimedAndSizeRotatingFileHandler

    h = SafeTimedAndSizeRotatingFileHandler(
        filename=log_path,
        when="midnight",
        backupCount=200,
        maxBytes=max_bytes,
        retention_days=30,
        encoding="utf8",
    )
    h.setFormatter(logging.Formatter("%(message)s"))
    for i in range(n_records):
        h.emit(_make_record(f"p{proc_id}-r{i}"))
    h.close()


class TestMultiProcessConcurrency:
    """Multi-process concurrency tests for both size and daily+size modes."""

    @pytest.mark.parametrize("mode", ["size", "daily+size"])
    @_skip_on_windows
    def test_multiprocess_no_log_loss(self, tmp_path, mode):
        """4 processes write concurrently; verify no log loss and no archive loss."""
        import multiprocessing as mp

        log_path = str(tmp_path / "xinf.log")
        n_processes = 4
        n_records_per_proc = 500
        max_bytes = 500

        worker = _mp_worker_size if mode == "size" else _mp_worker_daily_size
        procs = [
            mp.Process(
                target=worker,
                args=(i, log_path, max_bytes, n_records_per_proc),
            )
            for i in range(n_processes)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
            assert p.exitcode == 0, f"Worker exited with {p.exitcode}"

        total_records = 0
        for fname in os.listdir(tmp_path):
            if not fname.startswith("xinf.log") or fname.endswith(".lock"):
                continue
            fpath = os.path.join(tmp_path, fname)
            if os.path.exists(fpath):
                with open(fpath, encoding="utf8") as f:
                    total_records += sum(1 for line in f if line.strip())
        expected = n_processes * n_records_per_proc
        assert total_records == expected, (
            f"Log loss detected: expected {expected}, got {total_records} "
            f"(mode={mode})"
        )

    @_skip_on_windows
    def test_multiprocess_no_cross_write(self, tmp_path):
        """Verify no cross-writing: each record should be in exactly one file."""
        import multiprocessing as mp

        log_path = str(tmp_path / "xinf.log")
        n_processes = 4
        n_records_per_proc = 200
        max_bytes = 800

        procs = [
            mp.Process(
                target=_mp_worker_daily_size,
                args=(i, log_path, max_bytes, n_records_per_proc),
            )
            for i in range(n_processes)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
            assert p.exitcode == 0

        all_records = []
        for fname in sorted(os.listdir(tmp_path)):
            if not fname.startswith("xinf.log") or fname.endswith(".lock"):
                continue
            fpath = os.path.join(tmp_path, fname)
            if os.path.exists(fpath):
                with open(fpath, encoding="utf8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_records.append(line)

        expected = n_processes * n_records_per_proc
        assert (
            len(all_records) == expected
        ), f"Total records mismatch: expected {expected}, got {len(all_records)}"
        assert len(set(all_records)) == expected, "Duplicate records found"

    def test_lock_files_coexist_for_multiple_logs(self, tmp_path):
        """xinference.log and audit.log in same dir should have separate lock files."""
        from ..utils import SafeTimedAndSizeRotatingFileHandler

        xinf_path = str(tmp_path / "xinference.log")
        audit_path = str(tmp_path / "audit.log")
        h1 = SafeTimedAndSizeRotatingFileHandler(
            filename=xinf_path,
            when="midnight",
            backupCount=3,
            maxBytes=1000,
            retention_days=30,
            encoding="utf8",
        )
        h2 = SafeTimedAndSizeRotatingFileHandler(
            filename=audit_path,
            when="midnight",
            backupCount=3,
            maxBytes=1000,
            retention_days=30,
            encoding="utf8",
        )
        h1.close()
        h2.close()
        assert os.path.exists(os.path.join(tmp_path, "xinference.log.rotate.lock"))
        assert os.path.exists(os.path.join(tmp_path, "audit.log.rotate.lock"))
