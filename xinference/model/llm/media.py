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
"""Validation and bounded fetching of client-supplied media URLs.

Chat requests may carry ``image_url`` / ``video_url`` / ``audio_url`` parts whose
URL is attacker controlled.  Downstream readers (``qwen_omni_utils``, ``PIL``,
``urlopen``) happily read local files and fetch arbitrary hosts without a
timeout, so every media URL has to pass through here first.

Environment variables:

``XINFERENCE_MEDIA_ALLOW_LOCAL_PATH``
    Allow ``file://`` URLs and bare filesystem paths.  Default ``false``.
``XINFERENCE_MEDIA_BLOCK_PRIVATE_ADDRESS``
    Refuse URLs resolving to loopback/private/link-local addresses.  Default
    ``true``; set to ``false`` only when the server is trusted to reach
    internal hosts (e.g. an internal object store) and is not publicly exposed.
``XINFERENCE_MEDIA_FETCH_TIMEOUT``
    Total wall-clock seconds allowed for one remote fetch.  Default ``20``.
``XINFERENCE_MEDIA_MAX_BYTES``
    Maximum size of one fetched media file.  Default ``67108864`` (64 MiB).

Two levels of protection
------------------------

:func:`validate_messages_media` only gates the URL string.  That is enough where
the fetch itself already goes through :func:`load_media_bytes` (``_decode_image``
and everything reaching it), but not where a vendor reader does its own
``requests.get``: it follows redirects, so a URL that validated as public can
still 302 into the private network.  Engines on that path call
:func:`materialize_messages_media` instead, which fetches the bytes here and
rewrites the message to a local copy, leaving the reader nothing to resolve.

Known limits
------------

Transformers-engine families that hand raw messages to
``processor.apply_chat_template`` (minicpm-v) still let HF's ``load_image`` fetch a
validated remote URL itself, so redirect-based SSRF survives there.  Those readers
are also untimed, and the engines run them through :func:`asyncio.to_thread`, i.e.
the event loop's *shared* default executor (``min(32, cpu + 4)`` threads).  Enough
concurrent slow-drip responses therefore exhaust that pool and stall every other
``to_thread`` caller in the model actor, ``Model.__pre_destroy__`` and ``Model.stop``
included, so ``terminate_model`` can still hang.  The bar moved from one request to
the pool size; it was not removed.
"""

import base64
import binascii
import contextlib
import os
import re
import time
from typing import Any, Iterator, List, Optional, Set, Tuple, cast
from urllib.parse import urlparse

from ...constants import parse_env_bool, parse_env_float

XINFERENCE_ENV_MEDIA_ALLOW_LOCAL_PATH = "XINFERENCE_MEDIA_ALLOW_LOCAL_PATH"
XINFERENCE_ENV_MEDIA_BLOCK_PRIVATE_ADDRESS = "XINFERENCE_MEDIA_BLOCK_PRIVATE_ADDRESS"
XINFERENCE_ENV_MEDIA_FETCH_TIMEOUT = "XINFERENCE_MEDIA_FETCH_TIMEOUT"
XINFERENCE_ENV_MEDIA_MAX_BYTES = "XINFERENCE_MEDIA_MAX_BYTES"

DEFAULT_FETCH_TIMEOUT = 20.0
# A request's whole media budget, as a multiple of the per-url one.
REQUEST_TIMEOUT_FACTOR = 3
DEFAULT_MAX_BYTES = 64 * 1024 * 1024

# RFC 2397: dataurl := "data:" [ mediatype ] [ ";base64" ] "," data
_TOKEN = r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+"
_DATA_URI = re.compile(
    rf'data:(?:{_TOKEN}/{_TOKEN})?(?:;{_TOKEN}=(?:{_TOKEN}|"[^"\\]*"))*(;base64)?,(.*)',
    re.IGNORECASE | re.DOTALL,
)
_UNRESERVED_OR_ESCAPED = re.compile(
    r"(?:[A-Za-z0-9\-_.!~*'()/:@&=+$,;?#\[\]]|%[0-9A-Fa-f]{2})*"
)

# Shape produced by ChatModelMixin._transform_messages.
_TRANSFORMED_MEDIA_KEYS = ("image", "video", "audio")
# Every key a downstream reader may resolve as a media reference.  HF's
# processing_utils pulls all of these out of a content part whatever its declared
# type, and load_image tries os.path.isfile() before base64, so the key a value
# arrived under says nothing about how it is read.  Enumerating keys beats
# matching one shape at a time.
_MEDIA_KEYS = (
    "image_url",
    "video_url",
    "audio_url",
    *_TRANSFORMED_MEDIA_KEYS,
    "url",
    "path",
    "base64",
)


def _allow_local_path() -> bool:
    return parse_env_bool(XINFERENCE_ENV_MEDIA_ALLOW_LOCAL_PATH, False)


def _block_private_address() -> bool:
    return parse_env_bool(XINFERENCE_ENV_MEDIA_BLOCK_PRIVATE_ADDRESS, True)


def _fetch_timeout() -> float:
    value = parse_env_float(XINFERENCE_ENV_MEDIA_FETCH_TIMEOUT, DEFAULT_FETCH_TIMEOUT)
    return value if value > 0 else DEFAULT_FETCH_TIMEOUT


def _max_bytes() -> int:
    raw = os.environ.get(XINFERENCE_ENV_MEDIA_MAX_BYTES)
    if not raw or not raw.strip():
        return DEFAULT_MAX_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_BYTES
    return value if value > 0 else DEFAULT_MAX_BYTES


def decode_data_uri(url: str) -> bytes:
    """Parse a ``data:`` URI against the RFC 2397 grammar and return its bytes."""
    match = _DATA_URI.fullmatch(url)
    if match is None:
        raise ValueError("Invalid data URI")
    is_base64, payload = match.groups()
    if is_base64:
        try:
            return base64.b64decode("".join(payload.split()), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Invalid base64 data URI") from exc
    if _UNRESERVED_OR_ESCAPED.fullmatch(payload) is None:
        raise ValueError("Invalid data URI")
    from urllib.parse import unquote_to_bytes

    return unquote_to_bytes(payload)


def validate_media_url(url: Any) -> str:
    """Raise ``ValueError`` unless ``url`` is a media source the server may read."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Media url must be a non-empty string")
    scheme = urlparse(url).scheme.lower()
    if scheme in ("http", "https"):
        if _block_private_address():
            from ..image.utils import _public_addresses

            try:
                _public_addresses(url)
            except OSError as exc:
                raise ValueError(f"Cannot resolve media url host: {exc}") from exc
        return scheme
    if scheme == "data":
        decode_data_uri(url)
        return scheme
    # A single-letter scheme is a Windows drive letter, not a URL scheme.
    is_local = scheme in ("", "file") or len(scheme) == 1
    if is_local:
        if _allow_local_path() or _is_materialized(local_path(url)):
            return "file"
        raise ValueError(
            "Local file media urls are rejected; set "
            f"{XINFERENCE_ENV_MEDIA_ALLOW_LOCAL_PATH}=true to allow them"
        )
    raise ValueError(f"Unsupported media url scheme: {scheme}")


# Paths the server itself wrote for this request.  Materialization rewrites
# messages to local files, and prompt builders re-validate what they are handed,
# so those paths have to pass without opening local reads to clients.
_ACTIVE_WORKSPACES: Set[str] = set()


def local_path(url: str) -> str:
    if not url.lower().startswith("file://"):
        return url
    # Stripping the prefix leaves "/C:/..." on Windows, which no open() accepts.
    from urllib.request import url2pathname

    try:
        return url2pathname(urlparse(url).path)
    except Exception:
        # 3.14 rejects a non-localhost authority with URLError, an OSError.
        raise ValueError(f"Invalid file media url: {url}") from None


def _is_materialized(path: str) -> bool:
    try:
        resolved = os.path.realpath(path)
    except (OSError, ValueError):  # embedded NUL and friends
        return False
    return any(
        resolved.startswith(workspace + os.sep)
        for workspace in tuple(_ACTIVE_WORKSPACES)
    )


@contextlib.contextmanager
def media_workspace(prefix: str = "xinference-media-") -> Iterator[str]:
    """A request-scoped directory whose files validate as server-owned."""
    import shutil
    import tempfile

    workspace = os.path.realpath(tempfile.mkdtemp(prefix=prefix))
    _ACTIVE_WORKSPACES.add(workspace)
    try:
        yield workspace
    finally:
        _ACTIVE_WORKSPACES.discard(workspace)
        shutil.rmtree(workspace, ignore_errors=True)


def _expand_slot(container: Any, key: Any, kind: str) -> Iterator[Tuple[Any, Any, str]]:
    """Walk one media value down to the slots holding an actual url."""
    value = container[key]
    if isinstance(value, dict):
        if "url" in value:
            yield from _expand_slot(value, "url", kind)
    elif isinstance(value, list):
        for index in range(len(value)):
            yield from _expand_slot(value, index, kind)
    else:
        # Not a string: yielded anyway so validation rejects it rather than
        # letting an unrecognised shape through untouched.
        yield container, key, kind


def _iter_media_slots(
    messages: Optional[List[Any]],
) -> Iterator[Tuple[Any, Any, str]]:
    """Yield ``(container, key, kind)`` for every client-supplied media url."""
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            for key in _MEDIA_KEYS:
                if key in item:
                    kind = (
                        key
                        if key in _TRANSFORMED_MEDIA_KEYS
                        else (
                            key[: -len("_url")]
                            if key.endswith("_url")
                            # "url"/"path" carry no kind of their own.
                            else str(item_type or "").replace("_url", "")
                        )
                    )
                    yield from _expand_slot(item, key, kind)


def _media_slots(messages: Optional[List[Any]]) -> List[Tuple[Any, Any, str]]:
    try:
        return list(_iter_media_slots(messages))
    except RecursionError:
        # json.loads accepts nesting far deeper than the walker can recurse; a
        # client error must not surface as a 500.
        raise ValueError("Media url nesting is too deep") from None


def validate_messages_media(messages: Optional[List[Any]]) -> None:
    """Validate every media url in OpenAI-formatted (or transformed) messages."""
    for container, key, _ in _media_slots(messages):
        validate_media_url(container[key])


def _read_bounded(reader: Any, deadline: float, limit: int) -> bytes:
    # read1 returns as soon as any data arrives, so a dribbling server cannot
    # stay under the per-socket timeout forever without hitting the deadline.
    read = getattr(reader, "read1", None) or reader.read
    chunks: List[bytes] = []
    total = 0
    while True:
        if time.monotonic() > deadline:
            raise ValueError("Media fetch exceeded its time budget")
        chunk = read(64 * 1024)
        if not chunk:
            # The watchdog shuts the socket down, which reads as a clean EOF.
            if time.monotonic() > deadline:
                raise ValueError("Media fetch exceeded its time budget")
            return b"".join(chunks)
        total += len(chunk)
        if total > limit:
            raise ValueError(f"Media exceeds {limit} bytes")
        chunks.append(chunk)


def fetch_media(url: str, deadline: Optional[float] = None) -> bytes:
    """Fetch a remote media url under a total wall-clock deadline and size cap."""
    from ..image.utils import _open_public_url

    if deadline is None:
        deadline = time.monotonic() + _fetch_timeout()
    limit = _max_bytes()
    try:
        # Same fetcher either way: it pins every hop to the address it validated,
        # so relaxing the public-address rule does not reopen DNS rebinding.
        with _open_public_url(
            url, deadline=deadline, require_public=_block_private_address()
        ) as (response, _):
            return _read_bounded(response, deadline, limit)
    except Exception:
        # Socket-level timeouts surface as library-specific errors; the budget
        # is what the caller was promised.
        # Windows' coarse clock can fire the socket timeout a tick early.
        if time.monotonic() > deadline - 0.1:
            raise ValueError("Media fetch exceeded its time budget") from None
        raise


def load_media_bytes(url: str, deadline: Optional[float] = None) -> bytes:
    """Validate a media url and return its content."""
    scheme = validate_media_url(url)
    if scheme == "data":
        return decode_data_uri(url)
    if scheme == "file":
        with open(local_path(url), "rb") as f:
            return f.read()
    return fetch_media(url, deadline)


_DEFAULT_SUFFIX = {"image": ".jpg", "video": ".mp4", "audio": ".wav"}
_SAFE_SUFFIX = re.compile(r"\.[A-Za-z0-9]{1,8}$")


def _suffix_for(url: str, kind: str) -> str:
    match = _SAFE_SUFFIX.search(urlparse(url).path)
    return match.group(0) if match else _DEFAULT_SUFFIX.get(kind, ".bin")


def materialize_messages_media(messages: Optional[List[Any]], temp_dir: str) -> None:
    """Validate every media url, fetching remote ones into ``temp_dir`` in place.

    Pre-flight validation alone cannot secure the engines whose vendor reader does
    its own fetching (``qwen_omni_utils``, ``qwen_vl_utils``): the reader follows
    redirects, so a validated public URL can still 302 into the private network.
    Fetching here and rewriting the message to the local copy is what closes it —
    the reader never sees a URL.  Mutates ``messages``; callers pass a copy.
    """
    import tempfile
    from pathlib import Path

    # Per-part budget, capped for the request as a whole: a single shared budget
    # starves a legitimate many-image request, while per-part budgets alone
    # multiply by part count and each one holds a shared executor thread.
    request_deadline = time.monotonic() + _fetch_timeout() * REQUEST_TIMEOUT_FACTOR
    for container, key, kind in _media_slots(messages):
        raw = container[key]
        if validate_media_url(raw) not in ("http", "https"):
            continue
        url = cast(str, raw)
        deadline = min(time.monotonic() + _fetch_timeout(), request_deadline)
        data = fetch_media(url, deadline)
        with tempfile.NamedTemporaryFile(
            dir=temp_dir, delete=False, suffix=_suffix_for(url, kind)
        ) as tmp:
            tmp.write(data)
        # Same convention as VLLMMultiModel._handle_base64_media: qwen's video
        # reader needs a URI, its image/audio readers take a bare path.
        container[key] = Path(tmp.name).as_uri() if kind == "video" else tmp.name
