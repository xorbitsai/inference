"""
Debug helpers (global).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Callable, Union

from acestep.constants import (
    TENSOR_DEBUG_MODE,
    DEBUG_API_SERVER,
    DEBUG_INFERENCE,
    DEBUG_TRAINING,
    DEBUG_DATASET,
    DEBUG_AUDIO,
    DEBUG_LLM,
    DEBUG_UI,
    DEBUG_MODEL_LOADING,
    DEBUG_GPU,
)

# ----------------------------------------------------------------------
# CPU thread configuration
# ----------------------------------------------------------------------
# When running on CPU we want to make use of most of the available cores
# but leave a couple free for the OS / other processes.  The logic is:
#   * If the system has ≤ 2 logical CPUs, use all of them.
#   * Otherwise, use (cpu_count - 2) threads.
# This mirrors the common “all‑but‑two” heuristic while guaranteeing at
#   least one thread.
# The function is executed at import time so that any subsequent
# torch operations respect the setting.
import os
import torch

def _configure_cpu_threads() -> None:
    """Set torch's intra-op and inter-op thread counts based on available CPUs.

    This function configures PyTorch to use most available CPU cores while
    leaving a couple free for the OS and other processes. The logic is:
      * If the system has ≤ 2 logical CPUs, use all of them.
      * Otherwise, use (cpu_count - 2) threads.

    This mirrors the common "all-but-two" heuristic while guaranteeing at
    least one thread.

    Raises:
        RuntimeError: If torch.set_num_threads or torch.set_num_interop_threads
            fails (e.g., if called after threads have already been used).
    """
    cpu_cnt = os.cpu_count() or 1
    # Ensure we never set a non-positive number of threads.
    threads = cpu_cnt - 2 if cpu_cnt > 2 else cpu_cnt
    threads = max(threads, 1)

    try:
        torch.set_num_threads(threads)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to set torch intra-op threads to {threads}: {exc}"
        ) from exc

    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to set torch inter-op threads to {threads}: {exc}"
        ) from exc

# Track whether CPU threads have been configured to avoid redundant calls.
_cpu_threads_configured = False


def configure_cpu_threads_if_needed() -> bool:
    """Configure CPU threads if enabled via environment variable.

    This function provides an opt-in mechanism for configuring PyTorch's
    thread counts. It only takes effect if the environment variable
    ``ACESTEP_CONFIGURE_THREADS`` is set to a truthy value (e.g., "1", "true", "yes").

    The configuration is applied at most once per process; subsequent calls
    are no-ops.

    Returns:
        True if configuration was applied, False if skipped (either because
        the environment variable is not set or configuration was already done).

    Raises:
        RuntimeError: If thread configuration fails (propagated from
            ``_configure_cpu_threads``).
    """
    global _cpu_threads_configured

    if _cpu_threads_configured:
        return False

    env_value = os.environ.get("ACESTEP_CONFIGURE_THREADS", "").strip().lower()
    if env_value not in ("1", "true", "yes", "on"):
        return False

    _configure_cpu_threads()
    _cpu_threads_configured = True
    return True


# Apply opt-in CPU thread configuration early so torch respects it.
configure_cpu_threads_if_needed()


def _normalize_mode(mode: str) -> str:
    return (mode or "").strip().upper()


def is_debug_enabled(mode: str) -> bool:
    return _normalize_mode(mode) != "OFF"


def is_debug_verbose(mode: str) -> bool:
    return _normalize_mode(mode) == "VERBOSE"


def debug_log(message: Union[str, Callable[[], str]], *, mode: str = TENSOR_DEBUG_MODE, prefix: str = "debug") -> None:
    """Emit a timestamped debug log line if the mode is enabled."""
    if not is_debug_enabled(mode):
        return
    if callable(message):
        message = message()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{prefix}] {ts} {message}", flush=True)


# Placeholder debug switches registry (for centralized access)
DEBUG_SWITCHES = {
    "tensor": TENSOR_DEBUG_MODE,
    "api_server": DEBUG_API_SERVER,
    "inference": DEBUG_INFERENCE,
    "training": DEBUG_TRAINING,
    "dataset": DEBUG_DATASET,
    "audio": DEBUG_AUDIO,
    "llm": DEBUG_LLM,
    "ui": DEBUG_UI,
    "model_loading": DEBUG_MODEL_LOADING,
    "gpu": DEBUG_GPU,
}


def get_debug_mode(name: str, default: str = "OFF") -> str:
    """Fetch a placeholder debug mode by name."""
    return DEBUG_SWITCHES.get((name or "").strip().lower(), default)


def debug_log_for(name: str, message: Union[str, Callable[[], str]], *, prefix: str | None = None) -> None:
    """Emit a timestamped debug log for a named subsystem."""
    mode = get_debug_mode(name)
    debug_log(message, mode=mode, prefix=prefix or name)


def debug_start_for(name: str, label: str) -> Optional[float]:
    """Start timing for a named subsystem."""
    mode = get_debug_mode(name)
    return debug_start(label, mode=mode, prefix=name)


def debug_end_for(name: str, label: str, start_ts: Optional[float]) -> None:
    """End timing for a named subsystem."""
    mode = get_debug_mode(name)
    debug_end(label, start_ts, mode=mode, prefix=name)


def debug_log_verbose_for(name: str, message: Union[str, Callable[[], str]], *, prefix: str | None = None) -> None:
    """Emit a timestamped debug log only in VERBOSE mode for a named subsystem."""
    mode = get_debug_mode(name)
    if not is_debug_verbose(mode):
        return
    debug_log(message, mode=mode, prefix=prefix or name)


def debug_start_verbose_for(name: str, label: str) -> Optional[float]:
    """Start timing only in VERBOSE mode for a named subsystem."""
    mode = get_debug_mode(name)
    if not is_debug_verbose(mode):
        return None
    return debug_start(label, mode=mode, prefix=name)


def debug_end_verbose_for(name: str, label: str, start_ts: Optional[float]) -> None:
    """End timing only in VERBOSE mode for a named subsystem."""
    mode = get_debug_mode(name)
    if not is_debug_verbose(mode):
        return
    debug_end(label, start_ts, mode=mode, prefix=name)


def debug_start(name: str, *, mode: str = TENSOR_DEBUG_MODE, prefix: str = "debug") -> Optional[float]:
    """Return a start timestamp (perf counter) if enabled, otherwise None."""
    if not is_debug_enabled(mode):
        return None
    debug_log(f"START {name}", mode=mode, prefix=prefix)
    from time import perf_counter
    return perf_counter()


def debug_end(name: str, start_ts: Optional[float], *, mode: str = TENSOR_DEBUG_MODE, prefix: str = "debug") -> None:
    """Emit an END log with elapsed ms if enabled and start_ts is present."""
    if start_ts is None or not is_debug_enabled(mode):
        return
    from time import perf_counter
    elapsed_ms = (perf_counter() - start_ts) * 1000.0
    debug_log(f"END {name} ({elapsed_ms:.1f} ms)", mode=mode, prefix=prefix)
