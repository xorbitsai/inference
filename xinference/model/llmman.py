# Copyright 2022-2023 XProbe Inc.
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
"""Client for a running ``llmman serve`` daemon, used to acquire CNCF ModelPack
(https://github.com/modelpack/model-spec) OCI artifacts.

Daemon contract:
  - ``LLMMAN_HOST`` is ``[scheme://]host[:port][/path]``, default
    127.0.0.1:17434.
  - ``GET /api/version`` -> ``{"version": ..., "exe": ..., "pid": ...}``.
  - ``POST /api/pull`` ``{"model": ref}`` -> NDJSON stream of ``{"status": ...}``
    ending in ``{"status": "success"}`` or ``{"error": ...}``; an error can
    arrive in-band at HTTP 200.
  - ``POST /api/show`` ``{"model": ref}`` -> the manifest digest visible to
    the daemon.
  - The daemon and local CLI must expose the same ``LLMMAN_MODELS`` store before
    the local resolve step is allowed.
  - ``llmman resolve --no-pull <ref>`` -> one line of JSON carrying ``path``.
"""

import ipaddress
import json
import logging
import os
import shutil
import subprocess
import urllib.error
import urllib.request

from ..constants import XINFERENCE_ENV_LLMMAN_BIN

logger = logging.getLogger(__name__)

# llmman's own variable, honoured as llmman defines it
HOST_ENV = "LLMMAN_HOST"
MODELS_ENV = "LLMMAN_MODELS"
BIN_ENV = XINFERENCE_ENV_LLMMAN_BIN

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 17434
# An explicit scheme shifts the default port, as llmman does.
SCHEME_PORTS = {"http": 80, "https": 443}
MAX_PORT = 65535

PROBE_TIMEOUT_SECONDS = 5


def _connectable_host(host: str) -> str:
    """Rewrite a wildcard bind host (0.0.0.0, ::) to loopback: a client cannot
    connect to "every interface"."""
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return host
    if not ip.is_unspecified:
        return host
    return "127.0.0.1" if ip.version == 4 else "::1"


def _split_host_port(hostport: str) -> tuple[str, str]:
    """Split ``host``, ``host:port`` or a bracketed IPv6 literal. A bare
    multi-colon literal (``::1``) is kept whole rather than guessed at."""
    if hostport.startswith("["):
        host, bracket, rest = hostport.partition("]")
        if not bracket:  # malformed; keep as one host
            return hostport, ""
        return host[1:], rest[1:] if rest.startswith(":") else ""
    host, colon, port = hostport.rpartition(":")
    if colon and host and ":" not in host and port:
        return host, port
    return hostport, ""


def endpoint() -> str:
    """The http origin of the llmman daemon, honouring ``LLMMAN_HOST``.

    The origin is http regardless of the configured scheme, matching llmman's
    own daemon client; the scheme only selects the default port.
    """
    raw = os.getenv(HOST_ENV, "").strip().strip("\"'")

    default_port = DEFAULT_PORT
    if "://" in raw:
        scheme, raw = raw.split("://", 1)
        default_port = SCHEME_PORTS.get(scheme.lower(), DEFAULT_PORT)
    raw = raw.split("/", 1)[0]  # a trailing path is never used

    host, port_text = _split_host_port(raw)
    port = (
        int(port_text)
        if port_text.isdigit() and int(port_text) <= MAX_PORT
        else default_port
    )

    host = _connectable_host(host or DEFAULT_HOST)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def llmman_bin() -> str:
    """The llmman executable, overridable via ``XINFERENCE_LLMMAN_BIN``."""
    return os.getenv(BIN_ENV, "").strip() or "llmman"


def check_daemon(base: str) -> None:
    """Confirm an llmman daemon is listening and is actually llmman."""
    url = base + "/api/version"
    try:
        with urllib.request.urlopen(url, timeout=PROBE_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Something is listening, it just is not llmman -- worth distinguishing
        # from nothing listening at all.
        raise RuntimeError(
            f"the server at {base} is not an llmman daemon "
            f"(/api/version answered HTTP {exc.code})"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"no llmman daemon reachable at {base} ({exc.reason}). Start one with "
            f"`llmman serve`, or point {HOST_ENV} at an existing daemon."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"the server at {base} is not an llmman daemon (unparsable /api/version)"
        ) from exc

    if not isinstance(payload, dict) or not payload.get("version"):
        raise RuntimeError(
            f"the server at {base} is not an llmman daemon (no version in /api/version)"
        )


def pull(base: str, reference: str, progress=None) -> None:
    """Stream POST /api/pull until the daemon reports success.

    ``progress`` receives ``(status, completed, total)``.
    """
    body = json.dumps({"model": reference}).encode("utf-8")
    req = urllib.request.Request(
        base + "/api/pull",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    succeeded = False
    try:
        with urllib.request.urlopen(req) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"llmman pull of {reference!r} failed: HTTP {resp.status}"
                )
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # a non-JSON diagnostic must not abort a live pull
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("error"):
                    raise RuntimeError(
                        f"llmman pull of {reference!r} failed: {obj['error']}"
                    )
                status = obj.get("status")
                if status == "success":
                    succeeded = True
                    continue
                if progress is not None and status:
                    progress(status, obj.get("completed", 0), obj.get("total", 0))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"llmman pull of {reference!r} failed: HTTP {exc.code}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"llmman pull of {reference!r} failed: {exc.reason}"
        ) from exc

    if not succeeded:
        raise RuntimeError(
            f"llmman pull of {reference!r} ended without reporting success"
        )


def daemon_model_digest(base: str, reference: str) -> str:
    """Return the manifest digest visible to the selected daemon."""
    body = json.dumps({"model": reference}).encode("utf-8")
    req = urllib.request.Request(
        base + "/api/show",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"llmman show of {reference!r} failed: HTTP {exc.code}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"llmman show of {reference!r} failed: {exc.reason}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"llmman show of {reference!r} returned invalid JSON"
        ) from exc

    model_info = payload.get("model_info") if isinstance(payload, dict) else None
    digest = model_info.get("digest") if isinstance(model_info, dict) else None
    if not isinstance(digest, str) or not digest.strip():
        raise RuntimeError(
            f"llmman show of {reference!r} did not return a manifest digest"
        )
    return digest.strip()


def parse_resolve_output(stdout: str, reference: str) -> str:
    """Parse ``llmman resolve`` stdout into the resolved local path."""
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"llmman resolve {reference!r}: no output on stdout")

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"llmman resolve {reference!r}: could not parse output as JSON: {lines[-1]}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"llmman resolve {reference!r}: expected a JSON object, got {lines[-1]}"
        )

    path = payload.get("path")
    if not isinstance(path, str) or not path.strip():
        raise RuntimeError(f"llmman resolve {reference!r}: returned an empty path")
    if not os.path.exists(path):
        raise RuntimeError(
            f"llmman resolve {reference!r}: reported path {path!r} does not exist"
        )
    return path


def _require_llmman_bin() -> str:
    """Return the configured llmman executable after verifying it exists."""
    binary = llmman_bin()
    if shutil.which(binary) is None and not os.path.isfile(binary):
        raise RuntimeError(
            f"{binary!r} not found. Install llmman "
            "(https://github.com/llmmanorg/llmman) and put it on PATH, or set "
            f"{BIN_ENV} to its location."
        )
    return binary


def _reference_with_default_tag(reference: str) -> str:
    """Make llmman's implicit ``latest`` tag explicit for an exact list query."""
    if "@" in reference or ":" in reference.rsplit("/", 1)[-1]:
        return reference
    return f"{reference}:latest"


def local_model_digest(binary: str, reference: str) -> str:
    """Return the manifest digest visible to the local llmman CLI store."""
    lookup_reference = _reference_with_default_tag(reference)
    completed = subprocess.run(
        [binary, "list", lookup_reference, "--format={{.Digest}}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"`{binary} list {lookup_reference}` failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )

    digests = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(digests) != 1:
        raise RuntimeError(
            f"the local llmman store does not contain exactly one manifest for "
            f"{lookup_reference!r}; the daemon selected by {HOST_ENV} and the "
            f"local CLI must use the same {MODELS_ENV} store"
        )
    return digests[0]


def _resolve(binary: str, reference: str) -> str:
    """Resolve a model already known to exist in the local llmman store."""

    completed = subprocess.run(
        [binary, "resolve", "--no-pull", reference],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"`{binary} resolve --no-pull {reference}` failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    return parse_resolve_output(completed.stdout, reference)


def resolve(reference: str) -> str:
    """Ask the local CLI for a model path without performing another pull."""
    return _resolve(_require_llmman_bin(), reference)


def pull_and_resolve(reference: str, progress=None) -> str:
    """Pull through the daemon and resolve from the same underlying store."""
    binary = _require_llmman_bin()
    base = endpoint()
    check_daemon(base)
    logger.info("Pulling %s via llmman daemon at %s", reference, base)
    pull(base, reference, progress)
    daemon_digest = daemon_model_digest(base, reference)
    local_digest = local_model_digest(binary, reference)
    if daemon_digest != local_digest:
        raise RuntimeError(
            f"llmman stores disagree for {reference!r}: daemon has "
            f"{daemon_digest}, local CLI has {local_digest}. The daemon selected "
            f"by {HOST_ENV} and the local CLI must use the same {MODELS_ENV} store."
        )
    return _resolve(binary, reference)
