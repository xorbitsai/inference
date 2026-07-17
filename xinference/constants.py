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

import os
from pathlib import Path

XINFERENCE_ENV_ENDPOINT = "XINFERENCE_ENDPOINT"
XINFERENCE_ENV_MODEL_SRC = "XINFERENCE_MODEL_SRC"
XINFERENCE_ENV_CSG_TOKEN = "XINFERENCE_CSG_TOKEN"
XINFERENCE_ENV_CSG_ENDPOINT = "XINFERENCE_CSG_ENDPOINT"
XINFERENCE_ENV_HOME_PATH = "XINFERENCE_HOME"
XINFERENCE_ENV_HEALTH_CHECK_FAILURE_THRESHOLD = (
    "XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD"
)
XINFERENCE_ENV_HEALTH_CHECK_INTERVAL = "XINFERENCE_HEALTH_CHECK_INTERVAL"
XINFERENCE_ENV_HEALTH_CHECK_TIMEOUT = "XINFERENCE_HEALTH_CHECK_TIMEOUT"
XINFERENCE_ENV_TCP_REQUEST_TIMEOUT = "XINFERENCE_TCP_REQUEST_TIMEOUT"
# Supervisor -> Worker RPC bounds
XINFERENCE_ENV_LIST_MODELS_PER_WORKER_TIMEOUT = (
    "XINFERENCE_LIST_MODELS_PER_WORKER_TIMEOUT"
)
XINFERENCE_ENV_GET_MODEL_RPC_TIMEOUT = "XINFERENCE_GET_MODEL_RPC_TIMEOUT"
XINFERENCE_ENV_STATUS_GATHER_TIMEOUT = "XINFERENCE_STATUS_GATHER_TIMEOUT"
XINFERENCE_ENV_STATUS_REPORT_MULTIPLIER = "XINFERENCE_STATUS_REPORT_MULTIPLIER"
XINFERENCE_ENV_MAX_CONCURRENT_LAUNCHES = "XINFERENCE_MAX_CONCURRENT_LAUNCHES"
XINFERENCE_ENV_DISABLE_HEALTH_CHECK = "XINFERENCE_DISABLE_HEALTH_CHECK"
XINFERENCE_ENV_DISABLE_METRICS = "XINFERENCE_DISABLE_METRICS"
XINFERENCE_ENV_DOWNLOAD_MAX_ATTEMPTS = "XINFERENCE_DOWNLOAD_MAX_ATTEMPTS"
XINFERENCE_ENV_TEXT_TO_IMAGE_BATCHING_SIZE = "XINFERENCE_TEXT_TO_IMAGE_BATCHING_SIZE"
XINFERENCE_ENV_VIRTUAL_ENV = "XINFERENCE_ENABLE_VIRTUAL_ENV"
XINFERENCE_ENV_VIRTUAL_ENV_SKIP_INSTALLED = "XINFERENCE_VIRTUAL_ENV_SKIP_INSTALLED"
XINFERENCE_ENV_VIRTUAL_ENV_OFFLINE_INSTALL = "XINFERENCE_VIRTUAL_ENV_OFFLINE_INSTALL"
XINFERENCE_ENV_SSE_PING_ATTEMPTS_SECONDS = "XINFERENCE_SSE_PING_ATTEMPTS_SECONDS"
XINFERENCE_ENV_MAX_TOKENS = "XINFERENCE_MAX_TOKENS"
XINFERENCE_ENV_ALLOWED_IPS = "XINFERENCE_ALLOWED_IPS"
XINFERENCE_ENV_BATCH_SIZE = "XINFERENCE_BATCH_SIZE"
XINFERENCE_ENV_BATCH_INTERVAL = "XINFERENCE_BATCH_INTERVAL"
XINFERENCE_ENV_ALLOW_MULTI_REPLICA_PER_GPU = "XINFERENCE_ALLOW_MULTI_REPLICA_PER_GPU"
XINFERENCE_ENV_LAUNCH_STRATEGY = "XINFERENCE_LAUNCH_STRATEGY"

# OTEL environment variable names
XINFERENCE_ENV_ENABLE_OTEL = "XINFERENCE_ENABLE_OTEL"
XINFERENCE_ENV_OTLP_TRACE_ENDPOINT = "XINFERENCE_OTLP_TRACE_ENDPOINT"
XINFERENCE_ENV_OTLP_METRIC_ENDPOINT = "XINFERENCE_OTLP_METRIC_ENDPOINT"
XINFERENCE_ENV_OTLP_BASE_ENDPOINT = "XINFERENCE_OTLP_BASE_ENDPOINT"
XINFERENCE_ENV_OTLP_API_KEY = "XINFERENCE_OTLP_API_KEY"
XINFERENCE_ENV_OTEL_EXPORTER_OTLP_PROTOCOL = "XINFERENCE_OTEL_EXPORTER_OTLP_PROTOCOL"
XINFERENCE_ENV_OTEL_EXPORTER_TYPE = "XINFERENCE_OTEL_EXPORTER_TYPE"
XINFERENCE_ENV_OTEL_SAMPLING_RATE = "XINFERENCE_OTEL_SAMPLING_RATE"
XINFERENCE_ENV_OTEL_BATCH_EXPORT_SCHEDULE_DELAY = (
    "XINFERENCE_OTEL_BATCH_EXPORT_SCHEDULE_DELAY"
)
XINFERENCE_ENV_OTEL_MAX_QUEUE_SIZE = "XINFERENCE_OTEL_MAX_QUEUE_SIZE"
XINFERENCE_ENV_OTEL_MAX_EXPORT_BATCH_SIZE = "XINFERENCE_OTEL_MAX_EXPORT_BATCH_SIZE"
XINFERENCE_ENV_OTEL_METRIC_EXPORT_INTERVAL = "XINFERENCE_OTEL_METRIC_EXPORT_INTERVAL"
XINFERENCE_ENV_OTEL_BATCH_EXPORT_TIMEOUT = "XINFERENCE_OTEL_BATCH_EXPORT_TIMEOUT"
XINFERENCE_ENV_OTEL_METRIC_EXPORT_TIMEOUT = "XINFERENCE_OTEL_METRIC_EXPORT_TIMEOUT"


def get_xinference_home() -> str:
    home_path = os.environ.get(XINFERENCE_ENV_HOME_PATH)
    if home_path is None:
        home_path = str(Path.home() / ".xinference")
    # Always change huggingface, modelscope, and openmind_hub default download path
    # to ensure xinference process has write permissions for downloading dependencies
    # (e.g., Qwen3-ASR's forced aligner model downloaded from Hugging Face Hub)
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(home_path, "huggingface")
    os.environ["MODELSCOPE_CACHE"] = os.path.join(home_path, "modelscope")
    os.environ["XDG_CACHE_HOME"] = os.path.join(home_path, "openmind_hub")
    return home_path


XINFERENCE_HOME = get_xinference_home()
XINFERENCE_CACHE_DIR = os.path.join(XINFERENCE_HOME, "cache")
XINFERENCE_TENSORIZER_DIR = os.path.join(XINFERENCE_HOME, "tensorizer")
XINFERENCE_MODEL_DIR = os.path.join(XINFERENCE_HOME, "model")
XINFERENCE_LOG_DIR = os.environ.get(
    "XINFERENCE_LOG_DIR", os.path.join(XINFERENCE_HOME, "logs")
)
XINFERENCE_IMAGE_DIR = os.path.join(XINFERENCE_HOME, "image")
XINFERENCE_VIDEO_DIR = os.path.join(XINFERENCE_HOME, "video")
XINFERENCE_AUTH_DIR = os.path.join(XINFERENCE_HOME, "auth")


# Database-backed auth (user accounts, API keys) is on by default. Set
# XINFERENCE_AUTH_ADVANCED=0/false/no to run with no authentication at all.
#
# Read the environment at call time rather than caching a module-level
# constant: the server process is sometimes started in a subprocess created
# with the ``fork`` start method (the default on Linux), which inherits the
# parent's already-imported modules. If this were a constant frozen at import
# time, a forked child would keep the parent's value and ignore an
# XINFERENCE_AUTH_ADVANCED set after this module was first imported.
def is_auth_advanced() -> bool:
    return os.environ.get("XINFERENCE_AUTH_ADVANCED", "true").lower() not in (
        "0",
        "false",
        "no",
    )


# How long an empty secret file must sit untouched before it's considered
# abandoned by a crashed writer (rather than mid-write by a live one).
_STALE_SECRET_GRACE_SECONDS = 10
# Overall wait budget for a losing process to read the winner's file. Kept
# well above _STALE_SECRET_GRACE_SECONDS so the wait can actually reach and
# act on the staleness check instead of timing out first.
_SECRET_WAIT_DEADLINE_SECONDS = 30


def _get_or_create_persisted_secret(env_name: str, file_name: str) -> str:
    """Return a secret from the environment, or generate one on first run.

    Generated secrets are persisted under XINFERENCE_AUTH_DIR so that
    restarts (and multiple supervisor/worker processes sharing the same
    XINFERENCE_HOME) keep using the same key instead of invalidating
    existing JWTs / encrypted API keys.

    File creation uses O_EXCL so that concurrent first-time launches
    (e.g. supervisor and worker starting together) race safely: only one
    process wins the create, and the others fall back to reading the file
    it wrote instead of each keeping a different generated value in memory.
    A stale, empty file (left behind by a process that was killed between
    creating and writing to it) is treated as abandoned after a grace
    period and removed so startup can recover automatically instead of
    failing forever. The overall wait is bounded by a wall-clock deadline
    (not a fixed iteration count) so it comfortably outlasts the stale
    grace period even under scheduling jitter.
    """
    import time

    env_val = os.environ.get(env_name, "")
    if env_val:
        return env_val

    secret_path = os.path.join(XINFERENCE_AUTH_DIR, file_name)
    os.makedirs(XINFERENCE_AUTH_DIR, exist_ok=True)

    deadline = time.monotonic() + _SECRET_WAIT_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        try:
            stat_result = os.stat(secret_path)
        except OSError:
            stat_result = None

        if stat_result is not None:
            if stat_result.st_size > 0:
                try:
                    with open(secret_path, "r") as f:
                        existing = f.read().strip()
                except OSError:
                    existing = ""
                if existing:
                    return existing
            elif time.time() - stat_result.st_mtime > _STALE_SECRET_GRACE_SECONDS:
                # Empty and old: likely left behind by a process that was
                # killed after creating the file but before writing to it.
                try:
                    os.remove(secret_path)
                except OSError:
                    pass
                continue
            # Another process created the file but hasn't written to it yet.
            time.sleep(0.1)
            continue

        import secrets as _secrets

        generated = _secrets.token_hex(32)
        try:
            fd = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w") as f:
            f.write(generated)
        return generated

    raise RuntimeError(f"Failed to read or create secret file: {secret_path}")


def get_auth_jwt_secret_key() -> str:
    """Resolve the JWT secret at call time (see is_auth_advanced for why this
    is a function rather than a module-level constant). When advanced auth is
    on, generate/persist a key on first use; otherwise honor an explicitly set
    env value or return empty."""
    if is_auth_advanced():
        return _get_or_create_persisted_secret(
            "XINFERENCE_AUTH_JWT_SECRET_KEY", "jwt_secret_key"
        )
    return os.environ.get("XINFERENCE_AUTH_JWT_SECRET_KEY", "")


def get_auth_encryption_key() -> str:
    """Resolve the API-key encryption key at call time (see is_auth_advanced).
    Generated/persisted when advanced auth is on, else read from env."""
    if is_auth_advanced():
        return _get_or_create_persisted_secret(
            "XINFERENCE_AUTH_ENCRYPTION_KEY", "encryption_key"
        )
    return os.environ.get("XINFERENCE_AUTH_ENCRYPTION_KEY", "")


XINFERENCE_AUTH_DB_PATH = os.environ.get(
    "XINFERENCE_AUTH_DB_PATH", os.path.join(XINFERENCE_HOME, "auth", "auth.db")
)

XINFERENCE_LAUNCH_HISTORY_DB_PATH = os.environ.get(
    "XINFERENCE_LAUNCH_HISTORY_DB_PATH",
    os.path.join(XINFERENCE_HOME, "launch_history.db"),
)

XINFERENCE_MONITOR_CONFIG_DB_PATH = os.environ.get(
    "XINFERENCE_MONITOR_CONFIG_DB_PATH",
    os.path.join(XINFERENCE_HOME, "monitor_config.db"),
)

# Whether to allow models to run their own bundled code (transformers /
# sentence-transformers / FlagEmbedding / engine `trust_remote_code`). Off by
# default; set XINFERENCE_TRUST_REMOTE_CODE=1 to allow it.
XINFERENCE_TRUST_REMOTE_CODE = os.environ.get(
    "XINFERENCE_TRUST_REMOTE_CODE", ""
).lower() in (
    "1",
    "true",
    "yes",
)

# OIDC / SSO
XINFERENCE_OIDC_ENABLED = os.environ.get("XINFERENCE_OIDC_ENABLED", "").lower() in (
    "1",
    "true",
    "yes",
)
XINFERENCE_OIDC_ISSUER = os.environ.get("XINFERENCE_OIDC_ISSUER", "")
XINFERENCE_OIDC_CLIENT_ID = os.environ.get("XINFERENCE_OIDC_CLIENT_ID", "")
XINFERENCE_OIDC_CLIENT_SECRET = os.environ.get("XINFERENCE_OIDC_CLIENT_SECRET", "")
XINFERENCE_OIDC_REDIRECT_URI = os.environ.get("XINFERENCE_OIDC_REDIRECT_URI", "")

# Audit logging
XINFERENCE_AUDIT_LOG_RETENTION_DAYS = int(
    os.environ.get("XINFERENCE_AUDIT_LOG_RETENTION_DAYS", "90")
)
XINFERENCE_AUDIT_ES_INDEX = os.environ.get(
    "XINFERENCE_AUDIT_ES_INDEX", "xinference-audit-*"
)

XINFERENCE_VIRTUAL_ENV_DIR = os.path.join(XINFERENCE_HOME, "virtualenv")
XINFERENCE_CSG_ENDPOINT = str(
    os.environ.get(XINFERENCE_ENV_CSG_ENDPOINT, "https://hub-stg.opencsg.com/")
)

XINFERENCE_DEFAULT_LOCAL_HOST = "127.0.0.1"
XINFERENCE_DEFAULT_DISTRIBUTED_HOST = "0.0.0.0"
XINFERENCE_DEFAULT_ENDPOINT_PORT = 9997
XINFERENCE_DEFAULT_LOG_FILE_NAME = "xinference.log"
XINFERENCE_LOG_ROTATION = os.environ.get("XINFERENCE_LOG_ROTATION", "daily+size")
XINFERENCE_LOG_FORMAT = os.environ.get("XINFERENCE_LOG_FORMAT", "text").lower()
XINFERENCE_LOG_CONSOLE = (
    os.environ.get("XINFERENCE_LOG_CONSOLE", "true").lower() == "true"
)

# Download progress logging level (only effective when XINFERENCE_LOG_CONSOLE=false)
# - "sampled": log 25/50/75/100% per shard + terminal state on exit (default)
# - "full": log every tqdm frame
# - "off": no progress frames, only start/error lines
_XINFERENCE_LOG_DOWNLOAD_PROGRESS_RAW = os.environ.get(
    "XINFERENCE_LOG_DOWNLOAD_PROGRESS", "sampled"
).lower()
if _XINFERENCE_LOG_DOWNLOAD_PROGRESS_RAW not in ("sampled", "full", "off"):
    import sys

    print(
        f"WARNING: XINFERENCE_LOG_DOWNLOAD_PROGRESS={_XINFERENCE_LOG_DOWNLOAD_PROGRESS_RAW!r} "
        f"is invalid, falling back to 'sampled'",
        file=sys.stderr,
    )
    XINFERENCE_LOG_DOWNLOAD_PROGRESS = "sampled"
else:
    XINFERENCE_LOG_DOWNLOAD_PROGRESS = _XINFERENCE_LOG_DOWNLOAD_PROGRESS_RAW
XINFERENCE_LOG_RETENTION_DAYS = int(
    os.environ.get("XINFERENCE_LOG_RETENTION_DAYS", "30")
)
XINFERENCE_LOG_MAX_BYTES = int(
    os.environ.get("XINFERENCE_LOG_MAX_BYTES", str(100 * 1024 * 1024))
)
XINFERENCE_LOG_BACKUP_COUNT = int(os.environ.get("XINFERENCE_LOG_BACKUP_COUNT", "300"))
XINFERENCE_LOG_ARG_MAX_LENGTH = 100
XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD = int(
    os.environ.get(XINFERENCE_ENV_HEALTH_CHECK_FAILURE_THRESHOLD, 5)
)
XINFERENCE_HEALTH_CHECK_INTERVAL = int(
    os.environ.get(XINFERENCE_ENV_HEALTH_CHECK_INTERVAL, 5)
)
XINFERENCE_HEALTH_CHECK_TIMEOUT = int(
    os.environ.get(XINFERENCE_ENV_HEALTH_CHECK_TIMEOUT, 10)
)
XINFERENCE_TCP_REQUEST_TIMEOUT = int(
    os.environ.get(XINFERENCE_ENV_TCP_REQUEST_TIMEOUT, 5)
)
# Per-worker list_models RPC; default well below TCP read-timeout (~600s+) seen in production
XINFERENCE_LIST_MODELS_PER_WORKER_TIMEOUT = int(
    os.environ.get(XINFERENCE_ENV_LIST_MODELS_PER_WORKER_TIMEOUT, "60")
)
# get_model -> worker_ref.get_model RPC
XINFERENCE_GET_MODEL_RPC_TIMEOUT = int(
    os.environ.get(XINFERENCE_ENV_GET_MODEL_RPC_TIMEOUT, "30")
)
XINFERENCE_DISABLE_HEALTH_CHECK = bool(
    int(os.environ.get(XINFERENCE_ENV_DISABLE_HEALTH_CHECK, 0))
)

# Max concurrent download threads for huggingface_hub.snapshot_download.
# Default 8 in hf_hub causes GIL contention that starves the actor event loop.
XINFERENCE_ENV_MODEL_DOWNLOAD_WORKERS = "XINFERENCE_MODEL_DOWNLOAD_WORKERS"
XINFERENCE_MODEL_DOWNLOAD_WORKERS = int(
    os.environ.get(XINFERENCE_ENV_MODEL_DOWNLOAD_WORKERS, 2)
)


def is_metrics_disabled() -> bool:
    # Read at call time rather than freezing a module-level constant: the
    # supervisor/worker often run in a forked subprocess (the default start
    # method on Linux), which inherits the parent's already-imported modules.
    # A frozen constant would keep the parent's value and ignore a
    # XINFERENCE_DISABLE_METRICS set after this module was first imported.
    return bool(int(os.environ.get(XINFERENCE_ENV_DISABLE_METRICS, 0)))


XINFERENCE_DOWNLOAD_MAX_ATTEMPTS = int(
    os.environ.get(XINFERENCE_ENV_DOWNLOAD_MAX_ATTEMPTS, 3)
)
XINFERENCE_TEXT_TO_IMAGE_BATCHING_SIZE = os.environ.get(
    XINFERENCE_ENV_TEXT_TO_IMAGE_BATCHING_SIZE, None
)
XINFERENCE_SSE_PING_ATTEMPTS_SECONDS = int(
    os.environ.get(XINFERENCE_ENV_SSE_PING_ATTEMPTS_SECONDS, 600)
)
XINFERENCE_LAUNCH_MODEL_RETRY = 3
XINFERENCE_DEFAULT_CANCEL_BLOCK_DURATION = 30
XINFERENCE_ENABLE_VIRTUAL_ENV = bool(int(os.getenv(XINFERENCE_ENV_VIRTUAL_ENV, "1")))
XINFERENCE_VIRTUAL_ENV_SKIP_INSTALLED = bool(
    int(os.getenv(XINFERENCE_ENV_VIRTUAL_ENV_SKIP_INSTALLED, "1"))
)
XINFERENCE_VIRTUAL_ENV_OFFLINE_INSTALL = bool(
    int(os.getenv(XINFERENCE_ENV_VIRTUAL_ENV_OFFLINE_INSTALL, "0"))
)
XINFERENCE_MAX_TOKENS = os.getenv(XINFERENCE_ENV_MAX_TOKENS)
XINFERENCE_MAX_TOKENS = int(XINFERENCE_MAX_TOKENS) if XINFERENCE_MAX_TOKENS else None  # type: ignore
XINFERENCE_ALLOWED_IPS = os.getenv(XINFERENCE_ENV_ALLOWED_IPS)
XINFERENCE_BATCH_SIZE = int(os.getenv(XINFERENCE_ENV_BATCH_SIZE, "32"))
XINFERENCE_BATCH_INTERVAL = float(os.getenv(XINFERENCE_ENV_BATCH_INTERVAL, "0.003"))
XINFERENCE_ALLOW_MULTI_REPLICA_PER_GPU = bool(
    int(os.getenv(XINFERENCE_ENV_ALLOW_MULTI_REPLICA_PER_GPU, "1"))
)
XINFERENCE_LAUNCH_STRATEGY = os.getenv(
    XINFERENCE_ENV_LAUNCH_STRATEGY, "IDLE_FIRST_LAUNCH_STRATEGY"
)

# Status report multiplier: how many heartbeats before a full status report
# Default: 3 (report full status every 30 seconds with 10s heartbeat interval)
XINFERENCE_STATUS_REPORT_MULTIPLIER = int(
    os.environ.get(XINFERENCE_ENV_STATUS_REPORT_MULTIPLIER, 3)
)

XINFERENCE_MAX_CONCURRENT_LAUNCHES = max(
    1, int(os.environ.get(XINFERENCE_ENV_MAX_CONCURRENT_LAUNCHES, 5))
)

# Sub-pool creation timeout in seconds. Only covers fork+exec+import xoscar+
# bind port+write shm, NOT model weight download (which happens in
# create_model_instance after _create_subpool). Normal sub-pool creation < 1s;
# 60s is 60x+ headroom. On timeout: kill leftover subprocess + release GPU +
# raise TimeoutError so launch_builtin_model's try/finally runs the failure
# path and the supervisor's _workers_launching counter decrements. Tunable for
# extreme environments (slow NFS, vLLM version upgrade slowing import, etc.).
XINFERENCE_SUBPOOL_LAUNCH_TIMEOUT = int(
    os.environ.get("XINFERENCE_SUBPOOL_LAUNCH_TIMEOUT", "60")
)

# Status gather timeout in seconds (for collecting GPU info, etc.)
# Default: 10 seconds, increased from original 5 seconds
XINFERENCE_STATUS_GATHER_TIMEOUT = int(
    os.environ.get(XINFERENCE_ENV_STATUS_GATHER_TIMEOUT, 10)
)

# Model actor auto-recreate budget after a subpool death (e.g. CUDA OOM).
# None (default) = unbounded retry; int N = recreate up to N times, then evict
# the replica on the next death. Set via the env var of the same name.
_raw_recover_limit = os.getenv("XINFERENCE_MODEL_ACTOR_AUTO_RECOVER_LIMIT")
XINFERENCE_MODEL_ACTOR_AUTO_RECOVER_LIMIT = (
    int(_raw_recover_limit) if _raw_recover_limit is not None else None
)

# OTEL resolved values
XINFERENCE_ENABLE_OTEL = (
    os.getenv(XINFERENCE_ENV_ENABLE_OTEL, "false").strip().lower() == "true"
)
_XINFERENCE_OTLP_BASE_ENDPOINT = os.getenv(
    XINFERENCE_ENV_OTLP_BASE_ENDPOINT, "http://localhost:4318"
)
XINFERENCE_OTLP_TRACE_ENDPOINT = (
    os.getenv(XINFERENCE_ENV_OTLP_TRACE_ENDPOINT)
    or f"{_XINFERENCE_OTLP_BASE_ENDPOINT}/v1/traces"
)
XINFERENCE_OTLP_METRIC_ENDPOINT = (
    os.getenv(XINFERENCE_ENV_OTLP_METRIC_ENDPOINT)
    or f"{_XINFERENCE_OTLP_BASE_ENDPOINT}/v1/metrics"
)
XINFERENCE_OTLP_API_KEY = os.getenv(XINFERENCE_ENV_OTLP_API_KEY, "")
XINFERENCE_OTEL_EXPORTER_OTLP_PROTOCOL = os.getenv(
    XINFERENCE_ENV_OTEL_EXPORTER_OTLP_PROTOCOL, "http/protobuf"
)
XINFERENCE_OTEL_EXPORTER_TYPE = os.getenv(XINFERENCE_ENV_OTEL_EXPORTER_TYPE, "otlp")
XINFERENCE_OTEL_SAMPLING_RATE = float(
    os.getenv(XINFERENCE_ENV_OTEL_SAMPLING_RATE, "0.1")
)
XINFERENCE_OTEL_BATCH_EXPORT_SCHEDULE_DELAY = int(
    os.getenv(XINFERENCE_ENV_OTEL_BATCH_EXPORT_SCHEDULE_DELAY, "5000")
)
XINFERENCE_OTEL_MAX_QUEUE_SIZE = int(
    os.getenv(XINFERENCE_ENV_OTEL_MAX_QUEUE_SIZE, "2048")
)
XINFERENCE_OTEL_MAX_EXPORT_BATCH_SIZE = int(
    os.getenv(XINFERENCE_ENV_OTEL_MAX_EXPORT_BATCH_SIZE, "512")
)
XINFERENCE_OTEL_METRIC_EXPORT_INTERVAL = int(
    os.getenv(XINFERENCE_ENV_OTEL_METRIC_EXPORT_INTERVAL, "60000")
)
XINFERENCE_OTEL_BATCH_EXPORT_TIMEOUT = int(
    os.getenv(XINFERENCE_ENV_OTEL_BATCH_EXPORT_TIMEOUT, "10000")
)
XINFERENCE_OTEL_METRIC_EXPORT_TIMEOUT = int(
    os.getenv(XINFERENCE_ENV_OTEL_METRIC_EXPORT_TIMEOUT, "30000")
)

# Rate limiting defaults (overridable via environment variables)
XINFERENCE_RATE_LIMIT_IP_MAX_FAILURES = int(
    os.environ.get("XINFERENCE_RATE_LIMIT_IP_MAX_FAILURES", "10")
)
XINFERENCE_RATE_LIMIT_IP_WINDOW_SECONDS = int(
    os.environ.get("XINFERENCE_RATE_LIMIT_IP_WINDOW_SECONDS", "300")
)
XINFERENCE_RATE_LIMIT_IP_BAN_SECONDS = int(
    os.environ.get("XINFERENCE_RATE_LIMIT_IP_BAN_SECONDS", "3600")
)
XINFERENCE_RATE_LIMIT_KEY_MAX_FAILURES = int(
    os.environ.get("XINFERENCE_RATE_LIMIT_KEY_MAX_FAILURES", "5")
)
XINFERENCE_RATE_LIMIT_KEY_WINDOW_SECONDS = int(
    os.environ.get("XINFERENCE_RATE_LIMIT_KEY_WINDOW_SECONDS", "300")
)
XINFERENCE_RATE_LIMIT_KEY_BAN_SECONDS = int(
    os.environ.get("XINFERENCE_RATE_LIMIT_KEY_BAN_SECONDS", "3600")
)

# Trusted proxy IPs — only honor X-Forwarded-For/X-Real-IP from these peers
XINFERENCE_TRUSTED_PROXIES = os.environ.get("XINFERENCE_TRUSTED_PROXIES", "")
