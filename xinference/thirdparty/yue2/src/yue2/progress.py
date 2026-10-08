"""Dependency-free progress output; never inspects tensors or changes RNG state.

Use ``with Progress().stage("Generating audio tokens", unit="tokens") as s``
and pass ``s.token`` as the output-token callback. Counts belong to the caller:
prefill, external ABC, and extra CFG branches must not be counted as outputs.
Rates are observed counts divided by elapsed stage wall time, including waits.
An unknown total stays unknown; a generation limit is not a progress target.
"""
from __future__ import annotations

import math
import operator
import os
import sys
import threading
import time

_clock = time.monotonic
_STATUSES = {
    "completed": "Completed",
    "failed": "Failed",
    "cancelled": "Cancelled",
    "truncated": "Finished (generation limit reached)",
}


def _count(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _ascii(text):
    return "".join(c if " " <= c <= "~" else " " for c in str(text)).strip()


def _terminal_columns(stream):
    # Unconfigured PTYs can report (0, 0), including those created by `script`.
    # Honor a positive COLUMNS override, then the device, then a usable fallback.
    try:
        columns = int(os.environ.get("COLUMNS", ""))
        if columns > 0:
            return columns
    except ValueError:
        pass
    try:
        columns = os.get_terminal_size(stream.fileno()).columns
        if columns > 0:
            return columns
    except (AttributeError, OSError, ValueError):
        pass
    return 80


def _exit_status(exc_type):
    if exc_type is None:
        return "completed"
    if issubclass(exc_type, (InterruptedError, KeyboardInterrupt)) or any(
        base.__name__ in {"CancelledError", "CanceledError"} for base in exc_type.__mro__
    ):
        return "cancelled"
    return "failed"


class Progress:
    """Write ASCII progress to stderr, with a heartbeat during blocking work.

    TTYs at least 60 columns wide refresh one line. Narrow terminals and other
    streams receive start/end lines and at most one intermediate line per five
    seconds, without clipping labels or counts. ``enabled=False`` starts no thread
    and performs no stream operations. A stage owns its heartbeat lifetime, so
    an outer ``with Progress()`` is optional. ``close()`` is also available.
    """

    def __init__(self, enabled=True, stream=None, refresh_interval=0.25):
        if not math.isfinite(refresh_interval) or refresh_interval <= 0:
            raise ValueError("refresh_interval must be positive and finite")
        self.enabled = bool(enabled)
        self.stream = sys.stderr if stream is None else stream
        self.refresh_interval = float(refresh_interval)
        self._lock = threading.RLock()
        self._active = []
        self._thread = None
        self._stop = None
        self._closed = False
        self._summary_written = False
        self._line_width = 0
        self._tty = False
        self._columns = 80
        if self.enabled:
            try:
                self._tty = bool(self.stream.isatty())
            except (AttributeError, OSError, ValueError):
                pass
            if self._tty:
                self._columns = _terminal_columns(self.stream)
                self._tty = self._columns >= 60
        self._interval = self.refresh_interval if self._tty else max(5.0, self.refresh_interval)

    def __enter__(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("Progress is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(status=_exit_status(exc_type))
        return False

    def stage(self, label, total=None, unit=None):
        """Return a context manager; entering it starts this stage at zero."""
        return _Stage(self, label, total=total, unit=unit)

    def _write(self, text, final=False):
        if not self.enabled:
            return
        try:
            if self._tty:
                text = text[:self._columns - 1]
                self.stream.write("\r" + text + " " * max(0, self._line_width - len(text)))
                self._line_width = len(text)
                if final:
                    self.stream.write("\n")
                    self._line_width = 0
            else:
                self.stream.write(text + "\n")
            self.stream.flush()
        except (OSError, ValueError):
            # A closed terminal must not turn successful inference into failure.
            self.enabled = False
            if self._stop is not None:
                self._stop.set()

    def _newline(self):
        if self.enabled and self._tty and self._line_width:
            self._write("", final=True)

    def _render(self, stage, now, status=None, force=False):
        if not self.enabled or (not force and now - stage._last_render < self._interval):
            return
        stage._last_render = now
        elapsed = max(0.0, now - stage._started)
        parts = []
        if stage.total is not None:
            # Keep an honest count even if a caller exceeds its advertised total.
            amount = f"{stage.completed}/{stage.total} {stage.unit or 'items'}"
            if stage.total > 0:
                amount += f" ({100 * stage.completed / stage.total:.0f}%)"
                if self._tty:
                    filled = min(8, int(8 * stage.completed / stage.total))
                    amount = "[" + "#" * filled + "-" * (8 - filled) + "] " + amount
            parts.append(amount)
        elif stage.unit is not None or stage.completed:
            parts.append(f"{stage.completed} {stage.unit or 'items'}")
        if stage.unit == "tokens":
            rate = stage.completed / elapsed if elapsed > 0 else 0.0
            parts.append(f"{rate:.1f} tokens/s")
        parts.append(f"{elapsed:.1f}s" if self._tty else f"elapsed {elapsed:.1f}s")
        suffix = " | ".join(parts)
        if status is not None:
            prefix = "Limit reached" if self._tty and status == "truncated" else _STATUSES[status]
        elif self._tty:
            prefix = "|/-\\"[int(elapsed / self.refresh_interval) % 4]
        else:
            prefix = "Starting" if force and stage.completed == 0 else "Running"
        label = stage.label
        if self._tty:
            available = self._columns - 1 - len(f"[YuE2] {prefix} : {suffix}")
            if len(label) > max(0, available):
                label = label[:max(0, available - 3)] + ("..." if available >= 3 else "")
        self._write(f"[YuE2] {prefix} {label}: {suffix}", final=status is not None)

    def _heartbeat(self, stop):
        while not stop.wait(self._interval):
            with self._lock:
                if stop.is_set() or not self.enabled:
                    return
                if self._active:
                    self._render(self._active[-1], _clock())

    def _start(self, stage):
        with self._lock:
            if self._closed:
                raise RuntimeError("Progress is closed")
            if stage._started is not None:
                raise RuntimeError("A progress stage can only be entered once")
            stage._started = _clock()
            self._newline()
            self._active.append(stage)
            self._render(stage, stage._started, force=True)
            if self.enabled and self._thread is None:
                self._stop = threading.Event()
                self._thread = threading.Thread(target=self._heartbeat, args=(self._stop,),
                                                name="yue2-progress", daemon=True)
                self._thread.start()

    def _finish(self, stage, status):
        if status not in _STATUSES:
            raise ValueError("Unknown progress status")
        thread = None
        with self._lock:
            if stage._finished:
                return
            if stage._started is None:
                raise RuntimeError("Enter a progress stage before finishing it")
            stage._finished = True
            stage.status = status
            if self._active[-1] is not stage:
                self._newline()
            self._active.remove(stage)
            self._render(stage, _clock(), status=status, force=True)
            if self._active:
                self._render(self._active[-1], _clock(), force=True)
            elif self._thread is not None:
                self._stop.set()
                thread, self._thread = self._thread, None
                self._stop = None
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def close(self, status="completed"):
        """Finish active stages and stop the heartbeat; safe to call repeatedly."""
        if status not in _STATUSES:
            raise ValueError("Unknown progress status")
        with self._lock:
            self._closed = True
            active = list(reversed(self._active))
        for stage in active:
            self._finish(stage, status)

    def complete(self, audio_seconds, elapsed, *, truncated=False):
        """Write one final summary without starting a heartbeat thread."""
        if not all(math.isfinite(x) and x >= 0 for x in (audio_seconds, elapsed)):
            raise ValueError("audio_seconds and elapsed must be nonnegative and finite")
        with self._lock:
            if self._active:
                raise RuntimeError("Finish active stages before the completion summary")
            if self._summary_written:
                return
            self._summary_written = True
            status = _STATUSES["truncated" if truncated else "completed"]
            self._write(f"[YuE2] {status}: {audio_seconds:.1f}s audio in {elapsed:.1f}s", final=True)


class _Stage:
    """Stage handle returned by :meth:`Progress.stage`; updates are thread-safe."""

    def __init__(self, owner, label, total=None, unit=None):
        self._owner = owner
        self.label = _ascii(label)
        self.unit = _ascii(unit) if unit is not None else None
        self.total = None if total is None else _count(total, "total")
        self.completed = 0
        self.status = None
        self._started = None
        self._last_render = -math.inf
        self._finished = False

    def __enter__(self):
        self._owner._start(self)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.finish(status=_exit_status(exc_type))
        return False

    def _check_active(self):
        if self._started is None:
            raise RuntimeError("Enter a progress stage before updating it")

    def update(self, completed, total=None):
        """Set an absolute count and optionally the total discovered at runtime."""
        completed = _count(completed, "completed")
        total = None if total is None else _count(total, "total")
        with self._owner._lock:
            self._check_active()
            if self._finished:
                return
            if completed < self.completed:
                raise ValueError("completed must not decrease")
            self.completed = completed
            if total is not None:
                self.total = total
            if self._owner._active[-1] is self:
                self._owner._render(self, _clock())

    def set_total(self, total):
        """Set the real work total; ``None`` restores an unknown total."""
        total = None if total is None else _count(total, "total")
        with self._owner._lock:
            self._check_active()
            if self._finished:
                return
            self.total = total
            if self._owner._active[-1] is self:
                self._owner._render(self, _clock())

    def advance(self, count=1):
        """Add completed work without a read/update race between callbacks."""
        count = _count(count, "count")
        with self._owner._lock:
            self.update(self.completed + count)

    def token(self, phase, token):
        """Count one actual emitted token; phase/token contents are not inspected."""
        self.advance()

    def finish(self, status="completed"):
        """Stop this stage. Supports completed, failed, cancelled, or truncated."""
        self._owner._finish(self, status)
