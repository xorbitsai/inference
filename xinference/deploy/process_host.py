# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""OS boundaries shared by the stable service hosts."""

import asyncio
import base64
import contextlib
import signal
import socket
import subprocess
import sys
from typing import Callable, List, Optional, cast


def start_process(
    command, *, listener: Optional[socket.socket] = None, **kwargs
) -> subprocess.Popen:
    if sys.platform != "win32":
        if listener is not None:
            kwargs["pass_fds"] = (listener.fileno(),)
        return subprocess.Popen(command, **kwargs)

    import psutil
    from xoscar.backends.indigen.fate_sharing import (
        detect_fate_sharing_support_win32,
        set_kill_child_on_death_win32,
    )

    if not detect_fate_sharing_support_win32():
        raise RuntimeError("Windows service hosting requires Job Object support")
    kwargs.pop("start_new_session", None)
    kwargs["creationflags"] = (
        kwargs.get("creationflags", 0)
        | 0x00000004  # CREATE_SUSPENDED: assign the job before descendants exist.
        | subprocess.CREATE_NEW_PROCESS_GROUP
    )
    if listener is not None:
        kwargs["stdin"] = subprocess.PIPE
    process = subprocess.Popen(command, **kwargs)
    try:
        # This job belongs to the stable host, not to one control generation.
        # Releasing a retired generation must not kill its replacements.
        set_kill_child_on_death_win32(process)
        if listener is not None:
            assert process.stdin is not None
            try:
                process.stdin.write(
                    base64.b64encode(listener.share(process.pid)) + b"\n"
                )
            finally:
                process.stdin.close()
        psutil.Process(process.pid).resume()
        return process
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        raise


def inherited_listener(fd: int) -> socket.socket:
    if sys.platform == "win32":
        return socket.fromshare(base64.b64decode(sys.stdin.buffer.readline()))
    return socket.socket(fileno=fd)


def pipe_bytes_available(source) -> int:
    """Poll a Windows output pipe without blocking behind a retained child."""
    if sys.platform != "win32":
        raise NotImplementedError("Windows pipe polling is unavailable")
    import msvcrt

    import _winapi

    try:
        return cast(
            int, _winapi.PeekNamedPipe(msvcrt.get_osfhandle(source.fileno()))[0]
        )
    except BrokenPipeError:
        return -1


@contextlib.contextmanager
def shutdown_signals(callback: Callable[[], None]):
    loop = asyncio.get_running_loop()
    signals: List[signal.Signals] = [signal.SIGTERM, signal.SIGINT]
    if sys.platform == "win32":
        signals.append(signal.SIGBREAK)
        previous = {}

        def handler(signum, frame):
            loop.call_soon_threadsafe(callback)

        try:
            for sig in signals:
                previous[sig] = signal.signal(sig, handler)
            yield
        finally:
            for sig, value in previous.items():
                signal.signal(sig, value)
    else:
        try:
            for sig in signals:
                loop.add_signal_handler(sig, callback)
            yield
        finally:
            for sig in signals:
                loop.remove_signal_handler(sig)
