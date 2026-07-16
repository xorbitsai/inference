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

import json
import logging
import logging.handlers
import os
import re
import socket
import sys
import threading
import time
import typing
import weakref
from typing import TYPE_CHECKING, Any, Optional

if sys.platform == "win32":
    fcntl = None
else:
    import fcntl

import xoscar as xo

from ..constants import (
    XINFERENCE_DEFAULT_LOG_FILE_NAME,
    XINFERENCE_LOG_DIR,
    XINFERENCE_LOG_DOWNLOAD_PROGRESS,
)

if TYPE_CHECKING:
    from xoscar.backends.pool import MainActorPoolType

logger = logging.getLogger(__name__)


class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler that auto-creates parent directories.

    Python's standard RotatingFileHandler raises FileNotFoundError if
    the parent directory of *filename* does not exist at __init__ time.
    In Xinference's xoscar sub-pool processes the logging config is
    received via shared memory; the directory was created by the
    parent Worker process and may no longer exist when the sub-pool
    calls ``dictConfig``.  This subclass ensures the directory is
    (re-)created before the file is opened.

    Multi-process safety (2026-06-25):
    1. fcntl file lock serializes ``doRollover`` across processes.
    2. ``shouldRollover`` checks inode at entry; if another process
       renamed the file, reopen the stream.
    3. Size check uses ``os.fstat().st_size`` instead of
       ``stream.tell()`` so it reflects the true file size (including
       writes from other processes).

    Assumption: log directory is on a local filesystem (fcntl.flock
    is unreliable on NFS).

    Windows caveat: ``fcntl`` is unavailable on Windows, so the
    rotation lock is a no-op there. Single-process use is unaffected;
    multi-process use on Windows may regress to lost archives or
    cleanup of files still held by another process.
    """

    def __init__(self, filename, *args, **kwargs):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        super().__init__(filename, *args, **kwargs)
        self._lock_path = os.path.join(
            os.path.dirname(filename),
            os.path.basename(filename) + ".rotate.lock",
        )
        # Pre-create the lock file so it exists even if doRollover is never
        # called (e.g. short-lived processes). The file is empty and harmless.
        open(self._lock_path, "a").close()

    def _acquire_rotation_lock(self):
        if fcntl is None:
            return None
        lock_fd = open(self._lock_path, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return lock_fd

    def _release_rotation_lock(self, lock_fd):
        if lock_fd is None:
            return
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    def _check_inode_and_reopen(self):
        """Reopen stream if another process renamed the base file.

        Returns True if the stream was reopened (i.e. another process
        rotated the file), False otherwise. doRollover uses this to
        decide whether to skip a size-triggered rotation: if the file
        was just rotated by another process, the new file is small and
        does not need rotation.
        """
        if self.stream is None:
            return False
        try:
            current_inode = os.stat(self.baseFilename).st_ino
            stream_inode = os.fstat(self.stream.fileno()).st_ino
            if current_inode != stream_inode:
                self.stream.close()
                self.stream = self._open()
                return True
        except OSError:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        return False

    def shouldRollover(self, record):
        self._check_inode_and_reopen()
        if self.maxBytes > 0:
            if self.stream is None:
                self.stream = self._open()
            try:
                actual_size = os.fstat(self.stream.fileno()).st_size
                msg = "%s\n" % self.format(record)
                if actual_size + len(msg) >= self.maxBytes:
                    return True
            except OSError:
                pass
        return False

    def doRollover(self):
        lock_fd = self._acquire_rotation_lock()
        try:
            # If another process rotated the file while we were waiting
            # for the lock, the stream has been reopened to a fresh file
            # and there is nothing to do.
            if self._check_inode_and_reopen():
                return
            # Trust shouldRollover's judgment: if we got here, either time
            # or size trigger fired and no other process has rotated since.
            self._do_rolling_rollover()
        finally:
            self._release_rotation_lock(lock_fd)

    def _do_rolling_rollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename("%s.%d" % (self.baseFilename, i))
                dfn = self.rotation_filename("%s.%d" % (self.baseFilename, i + 1))
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    self.rotate(sfn, dfn)
            dfn = self.rotation_filename(self.baseFilename + ".1")
            if os.path.exists(dfn):
                os.remove(dfn)
            self.rotate(self.baseFilename, dfn)
        if not self.delay:
            self.stream = self._open()


class SafeTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """TimedRotatingFileHandler that auto-creates parent directories."""

    def __init__(self, filename, *args, **kwargs):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        super().__init__(filename, *args, **kwargs)


class SafeTimedAndSizeRotatingFileHandler(SafeTimedRotatingFileHandler):
    """TimedRotatingFileHandler that also rotates when the file exceeds maxBytes.

    Same-day size-triggered rotations append a numeric suffix
    (e.g. ``xinference.log.2026-06-24.1``, ``.2`` ...) to avoid
    clobbering the midnight-rotated file. The midnight-rotated file
    keeps the bare ``YYYY-MM-DD`` suffix and represents the last
    segment of the day.

    Cleanup is hybrid: (1) delete files whose filename date is older
    than ``retention_days``; (2) if remaining files still exceed
    ``backupCount``, delete oldest by mtime.

    Note: ``_rotated_re`` assumes ``when="midnight"`` (suffix
    ``%Y-%m-%d``). The handler is only used with midnight rotation
    in ``get_config_dict()``, so this assumption holds.

    Multi-process safety (2026-06-25):
    1. fcntl file lock serializes ``doRollover`` across processes.
    2. ``shouldRollover`` checks inode at entry; if another process
       renamed the file, reopen the stream.
    3. Size check uses ``os.fstat().st_size`` instead of
       ``stream.tell()`` so it reflects the true file size (including
       writes from other processes).

    Assumption: log directory is on a local filesystem (fcntl.flock
    is unreliable on NFS).

    Windows caveat: ``fcntl`` is unavailable on Windows, so the
    rotation lock is a no-op there. Single-process use is unaffected;
    multi-process use on Windows may regress to lost archives or
    cleanup of files still held by another process.
    """

    _rotated_re = re.compile(r"^\d{4}-\d{2}-\d{2}(\.\d+)?$")

    def __init__(
        self,
        filename,
        when="midnight",
        interval=1,
        backupCount=0,
        encoding=None,
        delay=False,
        utc=False,
        atTime=None,
        maxBytes=0,
        errors=None,
        retention_days=0,
    ):
        self.maxBytes = maxBytes
        self.retention_days = retention_days
        super().__init__(
            filename,
            when=when,
            interval=interval,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
            utc=utc,
            atTime=atTime,
            errors=errors,
        )
        self._lock_path = os.path.join(
            os.path.dirname(filename),
            os.path.basename(filename) + ".rotate.lock",
        )
        # Pre-create the lock file so it exists even if doRollover is never
        # called (e.g. short-lived processes). The file is empty and harmless.
        open(self._lock_path, "a").close()

    def _acquire_rotation_lock(self):
        if fcntl is None:
            return None
        lock_fd = open(self._lock_path, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return lock_fd

    def _release_rotation_lock(self, lock_fd):
        if lock_fd is None:
            return
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    def _check_inode_and_reopen(self):
        """Reopen stream if another process renamed the base file.

        Returns True if the stream was reopened (i.e. another process
        rotated the file), False otherwise.
        """
        if self.stream is None:
            return False
        try:
            current_inode = os.stat(self.baseFilename).st_ino
            stream_inode = os.fstat(self.stream.fileno()).st_ino
            if current_inode != stream_inode:
                self.stream.close()
                self.stream = self._open()
                # Another process performed the time-based rollover for us.
                # Advance rolloverAt so the next shouldRollover/doRollover
                # does not see a stale past-due time and perform a redundant,
                # destructive rotation (clobbers the archive just written by
                # the other process). Only advance when we are actually past
                # the scheduled time — a size-only foreign rotation leaves
                # the pending time rollover intact.
                if time.time() >= self.rolloverAt:
                    self.rolloverAt = self.computeRollover(time.time())
                return True
        except OSError:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        return False

    def shouldRollover(self, record):
        self._check_inode_and_reopen()
        if super().shouldRollover(record):
            return True
        if self.maxBytes > 0:
            if self.stream is None:
                self.stream = self._open()
            try:
                actual_size = os.fstat(self.stream.fileno()).st_size
                msg = "%s\n" % self.format(record)
                if actual_size + len(msg) >= self.maxBytes:
                    return True
            except OSError:
                pass
        return False

    def doRollover(self):
        lock_fd = self._acquire_rotation_lock()
        try:
            # If another process rotated the file while we were waiting
            # for the lock, the stream has been reopened to a fresh file
            # and there is nothing to do.
            if self._check_inode_and_reopen():
                return
            if time.time() >= self.rolloverAt:
                super().doRollover()
            else:
                self._do_size_rollover()
        finally:
            self._release_rotation_lock(lock_fd)

    def _do_size_rollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None

        current_time = time.time()
        if self.utc:
            time_tuple = time.gmtime(current_time)
        else:
            time_tuple = time.localtime(current_time)
        date_suffix = time.strftime(self.suffix, time_tuple)

        n = 1
        while True:
            dfn = self.rotation_filename(
                "%s.%s.%d" % (self.baseFilename, date_suffix, n)
            )
            if not os.path.exists(dfn):
                break
            n += 1
        self.rotate(self.baseFilename, dfn)
        if not self.delay:
            self.stream = self._open()

        if self.backupCount > 0 or self.retention_days > 0:
            for s in self.getFilesToDelete():
                try:
                    os.remove(s)
                except OSError:
                    pass

    def getFilesToDelete(self):
        """Hybrid cleanup: date-based + file-count cap."""
        dir_name, base_name = os.path.split(self.baseFilename)
        prefix = base_name + "."
        plen = len(prefix)
        files = []
        try:
            file_names = os.listdir(dir_name)
        except OSError:
            return []
        for file_name in file_names:
            if file_name[:plen] != prefix:
                continue
            suffix = file_name[plen:]
            if not self._rotated_re.match(suffix):
                continue
            date_str = suffix[:10]
            try:
                date_epoch = time.mktime(time.strptime(date_str, "%Y-%m-%d"))
            except ValueError:
                continue
            full_path = os.path.join(dir_name, file_name)
            try:
                mtime = os.path.getmtime(full_path)
            except OSError:
                continue
            files.append((date_epoch, mtime, full_path))

        to_delete = set()

        if self.retention_days > 0:
            # +1 because date_epoch is midnight of the archive's day while
            # cutoff is the current wall-clock time. Without the +1, a file
            # from exactly retention_days ago is deleted the moment the
            # current time passes midnight (e.g. retention_days=1 drops
            # yesterday's archive at 00:01 today, keeping zero full days).
            cutoff = time.time() - (self.retention_days + 1) * 86400
            for date_epoch, mtime, full_path in files:
                if date_epoch < cutoff:
                    to_delete.add(full_path)

        if self.backupCount > 0:
            remaining = [
                (mtime, full_path)
                for _, mtime, full_path in files
                if full_path not in to_delete
            ]
            remaining.sort(key=lambda x: x[0])
            excess = len(remaining) - self.backupCount
            if excess > 0:
                for _, full_path in remaining[:excess]:
                    to_delete.add(full_path)

        return list(to_delete)


# mainly for k8s
XINFERENCE_POD_NAME_ENV_KEY = "XINFERENCE_POD_NAME"


class LoggerNameFilter(logging.Filter):
    def filter(self, record):
        return record.name.startswith("xinference") or (
            record.name.startswith("uvicorn.error")
            and record.getMessage().startswith("Uvicorn running on")
        )


_PROGRESS_RE = re.compile(r"(\d+)%\|")
# Strip all CSI sequences (cursor moves, hide-cursor, SGR colors), not just the
# `m`-terminated color codes, so tqdm artifacts like `\x1b[A` / `\x1b[?25l` do
# not leak into the log as junk lines (e.g. lone `[A`).
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class StreamToLogger:
    """Redirect stdout/stderr to a logger with progress bar sampling.

    Args:
        progress_mode: Controls how progress bars are logged.
            - "sampled": log 25/50/75/100% per bar + terminal state on finalize
            - "full": log every tqdm frame
            - "off": no progress frames, only start/error lines
    """

    def __init__(
        self,
        logger_instance: logging.Logger,
        original_stream,
        stream_name: str = "stdout",
        progress_mode: str = "sampled",
    ):
        self._logger = logger_instance
        self._original = original_stream
        self._stream_name = stream_name
        self._progress_mode = progress_mode
        self._buffer = ""
        self._progress_thresholds = {25, 50, 75, 100}
        self._last_reported_pct: dict = {}
        # Track last seen frame per bar key for terminal state emission
        self._last_seen: dict = {}
        # Downloads run multi-threaded (HF_HUB_DOWNLOAD_WORKERS); concurrent
        # writes to the shared buffer would interleave without this lock.
        # Reentrant: while the lock is held we call logger.info(), and a logging
        # handler error path (Handler.handleError -> sys.stderr) can re-enter
        # write() on the same thread; RLock avoids a self-deadlock there.
        self._lock = threading.RLock()

    def write(self, message: str) -> int:
        if not message:
            return 0
        with self._lock:
            self._buffer += message
            while True:
                pos_n = self._buffer.find("\n")
                pos_r = self._buffer.find("\r")
                if pos_n == -1 and pos_r == -1:
                    break
                if pos_n != -1 and (pos_r == -1 or pos_n < pos_r):
                    line, self._buffer = self._buffer.split("\n", 1)
                else:
                    line, self._buffer = self._buffer.split("\r", 1)
                self._process_line(line)
        return len(message)

    def _process_line(self, raw: str) -> None:
        line = _ANSI_ESCAPE_RE.sub("", raw).strip("\r\n").strip()
        if not line or len(line) <= 2:
            return
        if _PROGRESS_RE.search(line):
            self._handle_progress(line)
        else:
            self._logger.info(line)

    def _handle_progress(self, line: str) -> None:
        match = _PROGRESS_RE.search(line)
        if not match:
            return
        pct = int(match.group(1))
        # Derive the sampling key from the stable description *before* the
        # percentage, so every frame of the same bar shares one key. Using
        # line[:30] folds the changing percent/bar text into the key for
        # short-description formats (e.g. "Downloading: 26%|"), making each
        # frame a distinct key and defeating threshold sampling.
        key = line[: match.start()].strip()

        # Track last seen frame for terminal state emission
        with self._lock:
            if pct >= 100:
                self._last_seen.pop(key, None)
            else:
                self._last_seen[key] = (pct, line)

        # Mode-based filtering
        if self._progress_mode == "off":
            return
        elif self._progress_mode == "full":
            self._logger.info(line)
            return

        # sampled mode: threshold-based sampling
        last = self._last_reported_pct.get(key, 0)
        for threshold in sorted(self._progress_thresholds):
            if last < threshold <= pct:
                self._logger.info(line)
                self._last_reported_pct[key] = pct
                break
        if pct >= 100:
            self._last_reported_pct.pop(key, None)

    def flush(self) -> None:
        # Route residual buffer through the sampling path instead of dumping it
        # raw. tqdm calls flush() after every frame with a `\r`-terminated bar
        # still in the buffer; a raw dump here bypasses _handle_progress and
        # defeats sampling, producing a full per-frame storm in the log.
        with self._lock:
            if self._buffer:
                residual = self._buffer
                self._buffer = ""
                self._process_line(residual)
        if self._original:
            try:
                self._original.flush()
            except (OSError, ValueError):
                pass

    def _finalize(self) -> None:
        """Emit terminal state for in-flight progress bars.

        Called once when the redirect context exits (ref_count -> 0).
        For sampled mode, this ensures interrupted downloads show their
        last known percentage instead of the last sampled threshold.
        """
        if self._progress_mode != "sampled":
            return
        try:
            with self._lock:
                for key, (pct, line) in self._last_seen.items():
                    last_reported = self._last_reported_pct.get(key, 0)
                    if pct != last_reported:
                        self._logger.info(line)
                self._last_seen.clear()
        except Exception:
            # Never let finalize errors mask the real download exception
            pass

    def fileno(self) -> int:
        return self._original.fileno()

    def isatty(self) -> bool:
        return False


def install_stream_redirect() -> None:
    """Replace sys.stdout/stderr with StreamToLogger after logging is configured.

    WARNING: Do not call at process startup — breaks xoscar sub-pool spawning.
    Use redirect_streams_to_logger() context manager for scoped redirection.
    """
    import sys

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    stdout_logger = logging.getLogger("xinference.stdout")
    stderr_logger = logging.getLogger("xinference.stderr")
    sys.stdout = StreamToLogger(stdout_logger, original_stdout, "stdout", XINFERENCE_LOG_DOWNLOAD_PROGRESS)  # type: ignore[assignment]
    sys.stderr = StreamToLogger(stderr_logger, original_stderr, "stderr", XINFERENCE_LOG_DOWNLOAD_PROGRESS)  # type: ignore[assignment]

    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.stream = original_stderr


class redirect_streams_to_logger:
    """Context manager to temporarily redirect stdout/stderr to logger during model loading.

    Args:
        progress_mode: Controls how progress bars are logged. Defaults to "sampled".
            - "sampled": log 25/50/75/100% per bar + terminal state on finalize
            - "full": log every tqdm frame
            - "off": no progress frames, only start/error lines
    """

    _lock = threading.Lock()
    _ref_count = 0
    _original_stdout = None
    _original_stderr = None
    _handler_streams: list = []

    def __init__(self, progress_mode: str = "sampled"):
        self._progress_mode = progress_mode

    def __enter__(self):
        import sys

        with redirect_streams_to_logger._lock:
            if redirect_streams_to_logger._ref_count == 0:
                redirect_streams_to_logger._original_stdout = sys.stdout
                redirect_streams_to_logger._original_stderr = sys.stderr

                stdout_logger = logging.getLogger("xinference.stdout")
                stderr_logger = logging.getLogger("xinference.stderr")
                sys.stdout = StreamToLogger(stdout_logger, redirect_streams_to_logger._original_stdout, "stdout", self._progress_mode)  # type: ignore[assignment]
                sys.stderr = StreamToLogger(stderr_logger, redirect_streams_to_logger._original_stderr, "stderr", self._progress_mode)  # type: ignore[assignment]

                redirect_streams_to_logger._handler_streams = []
                for handler in logging.root.handlers:
                    if isinstance(handler, logging.StreamHandler) and not isinstance(
                        handler, logging.FileHandler
                    ):
                        redirect_streams_to_logger._handler_streams.append(
                            (handler, handler.stream)
                        )
                        handler.stream = redirect_streams_to_logger._original_stderr
            redirect_streams_to_logger._ref_count += 1
        return self

    def __exit__(self, *exc):
        import sys

        with redirect_streams_to_logger._lock:
            redirect_streams_to_logger._ref_count -= 1
            if redirect_streams_to_logger._ref_count == 0:
                if isinstance(sys.stdout, StreamToLogger):
                    sys.stdout.flush()
                    sys.stdout._finalize()
                if isinstance(sys.stderr, StreamToLogger):
                    sys.stderr.flush()
                    sys.stderr._finalize()

                sys.stdout = redirect_streams_to_logger._original_stdout
                sys.stderr = redirect_streams_to_logger._original_stderr

                for (
                    handler,
                    original_stream,
                ) in redirect_streams_to_logger._handler_streams:
                    handler.stream = original_stream
                redirect_streams_to_logger._original_stdout = None
                redirect_streams_to_logger._original_stderr = None
                redirect_streams_to_logger._handler_streams = []
        return False


def get_log_file(sub_dir: str):
    """
    Return a fixed log file path under XINFERENCE_LOG_DIR.

    The sub_dir parameter is kept for backward compatibility but is no longer
    used to create timestamped subdirectories. Logs now write to a single
    fixed file that is rotated by the unified log handler.
    """
    pod_name = os.environ.get(XINFERENCE_POD_NAME_ENV_KEY, None)
    os.makedirs(XINFERENCE_LOG_DIR, exist_ok=True)
    if pod_name:
        filename = f"xinference-{pod_name}.log"
    else:
        filename = XINFERENCE_DEFAULT_LOG_FILE_NAME
    return os.path.join(XINFERENCE_LOG_DIR, filename)


class AddressFormatter(logging.Formatter):
    _instances: weakref.WeakSet = weakref.WeakSet()

    def __init__(self, fmt=None, datefmt=None, style="%", role="", address=""):
        super().__init__(fmt, datefmt, style)
        self.role = role
        self.address = address
        AddressFormatter._instances.add(self)

    def format(self, record):
        record.xinference_role = self.role
        record.xinference_address = self.address
        return super().format(record)

    @classmethod
    def update_address(cls, role, address):
        for inst in cls._instances:
            if inst.role == role:
                inst.address = address


class JsonFileFormatter(logging.Formatter):
    """JSON formatter for file output — one JSON object per line."""

    _instances: weakref.WeakSet = weakref.WeakSet()

    def __init__(self, role="", address="", **kwargs):
        super().__init__()
        self.role = role
        self.address = address
        self._hostname = socket.gethostname()
        JsonFileFormatter._instances.add(self)

    def format(self, record):
        from datetime import datetime, timezone

        entry = {
            "@timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            + "Z",
            "level": record.levelname,
            "module": record.name,
            "pid": record.process,
            "role": self.role,
            "address": self.address,
            "node": self._hostname,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)

    @classmethod
    def update_address(cls, role, address):
        for inst in cls._instances:
            if inst.role == role:
                inst.address = address


class TextFileFormatter(logging.Formatter):
    """Text formatter — same fields as JsonFileFormatter but human-readable."""

    _instances: weakref.WeakSet = weakref.WeakSet()

    def __init__(self, role="", address="", **kwargs):
        super().__init__()
        self.role = role
        self.address = address
        self._hostname = socket.gethostname()
        TextFileFormatter._instances.add(self)

    def format(self, record):
        from datetime import datetime, timezone

        ts = (
            datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )[:-3]
            + "Z"
        )
        msg = record.getMessage()
        line = f"{ts} {record.levelname} {record.name} pid:{record.process} role:{self.role} address:{self.address} node:{self._hostname} {msg}"
        if record.exc_info and record.exc_info[0] is not None:
            line += "\n" + self.formatException(record.exc_info)
        return line

    @classmethod
    def update_address(cls, role, address):
        for inst in cls._instances:
            if inst.role == role:
                inst.address = address


def update_all_formatter_addresses(role: str, address: str):
    """Update address on both text and JSON formatters."""
    AddressFormatter.update_address(role, address)
    JsonFileFormatter.update_address(role, address)
    TextFileFormatter.update_address(role, address)


def get_config_dict(
    log_level: str,
    log_file_path: str,
    log_backup_count: int,
    log_max_bytes: int,
    role: str = "",
    address: str = "",
    rotation: Optional[str] = None,
    log_retention_days: int = 0,
) -> dict:
    from ..constants import (
        XINFERENCE_LOG_CONSOLE,
        XINFERENCE_LOG_FORMAT,
        XINFERENCE_LOG_ROTATION,
    )

    rotation = rotation or XINFERENCE_LOG_ROTATION
    use_json = XINFERENCE_LOG_FORMAT == "json"
    use_console = XINFERENCE_LOG_CONSOLE

    # for windows, the path should be a raw string.
    log_file_path = (
        log_file_path.encode("unicode-escape").decode()
        if os.name == "nt"
        else log_file_path
    )
    log_level = log_level.upper()

    formatter_name = "json_formatter" if use_json else "text_formatter"
    formatter_class = (
        "xinference.deploy.utils.JsonFileFormatter"
        if use_json
        else "xinference.deploy.utils.TextFileFormatter"
    )

    if rotation == "daily":
        file_handler_config = {
            "class": "xinference.deploy.utils.SafeTimedRotatingFileHandler",
            "formatter": formatter_name,
            "level": log_level,
            "filename": log_file_path,
            "when": "midnight",
            "backupCount": log_backup_count,
            "encoding": "utf8",
        }
    elif rotation == "daily+size":
        file_handler_config = {
            "class": "xinference.deploy.utils.SafeTimedAndSizeRotatingFileHandler",
            "formatter": formatter_name,
            "level": log_level,
            "filename": log_file_path,
            "when": "midnight",
            "backupCount": log_backup_count,
            "maxBytes": log_max_bytes,
            "retention_days": log_retention_days,
            "encoding": "utf8",
        }
    else:
        file_handler_config = {
            "class": "xinference.deploy.utils.SafeRotatingFileHandler",
            "formatter": formatter_name,
            "level": log_level,
            "filename": log_file_path,
            "mode": "a",
            "maxBytes": log_max_bytes,
            "backupCount": log_backup_count,
            "encoding": "utf8",
        }

    handlers_list = ["file_handler"]
    if use_console:
        handlers_list.insert(0, "console_handler")

    root_handlers = ["file_handler"]
    if use_console:
        root_handlers.insert(0, "stream_handler")

    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "root": {
            "handlers": root_handlers,
            "level": log_level,
        },
        "formatters": {
            formatter_name: {
                "()": formatter_class,
                "role": role,
                "address": address,
            },
        },
        "filters": {
            "logger_name_filter": {
                "()": __name__ + ".LoggerNameFilter",
            },
        },
        "handlers": {
            "stream_handler": {
                "class": "logging.StreamHandler",
                "formatter": formatter_name,
                "level": log_level,
                "stream": "ext://sys.stderr",
                "filters": ["logger_name_filter"],
            },
            "console_handler": {
                "class": "logging.StreamHandler",
                "formatter": formatter_name,
                "level": log_level,
                "stream": "ext://sys.stderr",
            },
            "file_handler": file_handler_config,
        },
        "loggers": {
            "xinference": {
                "handlers": handlers_list,
                "level": log_level,
                "propagate": False,
            },
            "uvicorn": {
                "handlers": handlers_list,
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": handlers_list,
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": handlers_list,
                "level": log_level,
                "propagate": False,
            },
            "transformers": {
                "handlers": (
                    ["console_handler", "file_handler"]
                    if use_console
                    else ["file_handler"]
                ),
                "level": log_level,
                "propagate": False,
            },
            "vllm": {
                "handlers": (
                    ["console_handler", "file_handler"]
                    if use_console
                    else ["file_handler"]
                ),
                "level": log_level,
                "propagate": False,
            },
        },
    }
    return config_dict


async def create_worker_actor_pool(
    address: str, logging_conf: Optional[dict] = None
) -> "MainActorPoolType":
    return await xo.create_actor_pool(
        address=address,
        n_process=0,
        auto_recover="process",
        logging_conf={"dict": logging_conf},
    )


def health_check(address: str, max_attempts: int, sleep_interval: int = 3) -> bool:
    async def health_check_internal():
        import time

        attempts = 0
        while attempts < max_attempts:
            time.sleep(sleep_interval)
            try:
                from ..core.supervisor import SupervisorActor

                supervisor_ref: xo.ActorRefType[SupervisorActor] = await xo.actor_ref(  # type: ignore
                    address=address, uid=SupervisorActor.default_uid()
                )

                await supervisor_ref.get_status()
                return True
            except Exception as e:
                logger.debug(f"Error while checking cluster: {e}")

            attempts += 1
            if attempts < max_attempts:
                logger.debug(
                    f"Cluster not available, will try {max_attempts - attempts} more times"
                )

        return False

    import asyncio

    from ..isolation import Isolation

    isolation = Isolation(asyncio.new_event_loop(), threaded=True)
    isolation.start()
    available = isolation.call(health_check_internal())
    isolation.stop()
    return available


def get_timestamp_ms():
    t = time.time()
    return int(round(t * 1000))


@typing.no_type_check
def handle_click_args_type(arg: str) -> Any:
    """Convert CLI string arguments to appropriate Python types.

    Handles type conversion for Click command arguments with the following priority:
    1. Special string values: "None" → None, "true"/"True" → True, "false"/"False" → False
    2. Numeric values: "42" → int(42), "3.14" → float(3.14)
    3. JSON objects/arrays: '{"key": "value"}' → dict, '[1, 2, 3]' → list
    4. Default: return original string

    JSON parsing is attempted for arguments starting with '{' or '['.
    If JSON parsing fails, the original string is returned unchanged.

    Examples:
        >>> handle_click_args_type("None")
        None
        >>> handle_click_args_type("42")
        42
        >>> handle_click_args_type('{"mode": 3}')
        {'mode': 3}

    Args:
        arg: String argument from CLI command

    Returns:
        Converted value (None, bool, int, float, dict, list, or str)
    """
    if arg == "None":
        return None
    if arg in ("True", "true"):
        return True
    if arg in ("False", "false"):
        return False
    try:
        result = int(arg)
        return result
    except Exception:
        pass

    try:
        result = float(arg)
        return result
    except Exception:
        pass

    # Try to parse JSON objects and arrays.
    # This allows CLI users to pass structured parameters like:
    #   --compilation_config '{"mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}'
    #   --kv_transfer_config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'
    # See: https://github.com/xorbitsai/inference/issues/4760
    if arg.startswith(("{", "[")):
        try:
            result = json.loads(arg)
            return result
        except (json.JSONDecodeError, ValueError):
            pass

    return arg


def set_envs(key: str, value: str):
    """
    Environment variables are set by the parent process and inherited by child processes
    """
    os.environ[key] = value
