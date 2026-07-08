# Copyright 2022-2026 XProbe Inc.
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

"""Serve the built frontend static export from the FastAPI backend.

In the single-process (pip) deployment the web UI is shipped as a Next.js
static export bundled into the ``xinference`` package and served by this
backend, so no Node runtime is needed. When the export directory is absent
(e.g. a source checkout without a frontend build), the backend stays API-only.

Next static export emits one HTML file per route. Dynamic routes
(``[modelType]``/``[modelUid]``/...) are emitted as a ``__shell__`` placeholder
file (see the server wrappers under ``frontend/src/app``); the matching client
page reads the real value from the URL. This module reconstructs the route
table from those files so an arbitrary URL resolves to the correct HTML shell.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

_SHELL_TOKEN = "__shell__"
_SPA_ROUTE_NAME = "xinference_spa_fallback"


def _collect_backend_prefixes(app: FastAPI) -> frozenset[str]:
    """Collect the literal first path segments owned by the backend.

    Derived from the registered route table rather than hand-maintained, so it
    can't silently drift from the real routes. Any request whose first segment
    is in this set but reached the SPA catch-all (i.e. matched no route) gets a
    JSON 404 instead of the SPA shell. Must be called after all routers/mounts
    are registered and before the catch-all is added (the catch-all's own
    ``{full_path:path}`` segment is dynamic and therefore skipped here).
    """
    prefixes: set[str] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        if not path or not path.startswith("/"):
            continue
        first = path[1:].split("/", 1)[0]
        # Skip empty (root) and dynamic segments like "{full_path:path}".
        if not first or first.startswith("{"):
            continue
        prefixes.add(first)
    return frozenset(prefixes)


def _build_shell_patterns(dist_dir: Path) -> list[tuple[re.Pattern[str], Path]]:
    """Turn every ``__shell__`` export file into a (URL regex -> file) rule.

    ``running-model/__shell__.html``           -> ``^running-model/[^/]+$``
    ``register-model/__shell__/__shell__.html``-> ``^register-model/[^/]+/[^/]+$``
    """
    patterns: list[tuple[re.Pattern[str], Path]] = []
    for html in sorted(dist_dir.rglob("*.html")):
        rel = html.relative_to(dist_dir).with_suffix("")  # drop .html
        segments = rel.as_posix().split("/")
        if _SHELL_TOKEN not in segments:  # placeholder may be a dir or the file
            continue
        regex = ["[^/]+" if seg == _SHELL_TOKEN else re.escape(seg) for seg in segments]
        patterns.append((re.compile("^" + "/".join(regex) + "$"), html))
    return patterns


def ensure_spa_fallback_last(app: FastAPI) -> None:
    """Move the SPA catch-all back to the end of the route table.

    Starlette matches routes in registration order, so anything mounted after
    startup (e.g. per-model Gradio apps at ``/{model_uid}``) would be shadowed
    by the catch-all. Call this after every runtime mount. No-op if the
    frontend is not mounted.
    """
    routes = app.router.routes
    for i, route in enumerate(routes):
        if getattr(route, "name", None) == _SPA_ROUTE_NAME:
            routes.append(routes.pop(i))
            return


def mount_frontend(app: FastAPI, dist_dir: Path) -> bool:
    """Serve the frontend static export from ``dist_dir`` on ``app``.

    Returns True if the frontend was mounted, False if the directory is absent
    (in which case the backend stays API-only). Must be called after all API
    routers are registered so the catch-all only receives unmatched paths.
    """
    if not dist_dir.is_dir() or not (dist_dir / "index.html").is_file():
        logger.info(
            "Frontend static export not found at %s; serving API only", dist_dir
        )
        return False

    # Hashed build assets: let StaticFiles handle them directly.
    next_assets = dist_dir / "_next"
    if next_assets.is_dir():
        app.mount("/_next", StaticFiles(directory=str(next_assets)), name="next-assets")

    shell_patterns = _build_shell_patterns(dist_dir)
    not_found = dist_dir / "404.html"
    resolved_root = dist_dir.resolve()
    backend_prefixes = _collect_backend_prefixes(app)

    def _resolve(rel_path: str) -> Path | None:
        # Reject path traversal: the request path must stay within dist_dir.
        try:
            (resolved_root / rel_path).resolve().relative_to(resolved_root)
        except (ValueError, RuntimeError):
            return None

        # Exact asset (favicon, images, and static routes' own .html/.txt files).
        candidate = dist_dir / rel_path
        if rel_path and candidate.is_file():
            return candidate

        # A trailing slash (e.g. "running-model/uid1/") would otherwise miss
        # every lookup below, since none of the emitted file/shell names end
        # in "/". Strip it before continuing; the root path is already
        # handled by the caller.
        rel_path = rel_path.rstrip("/")
        candidate = dist_dir / rel_path

        # Next prefetches a route's RSC flight payload at "<route>.txt". Match it
        # against the route table on the base path and serve the .txt variant, so
        # dynamic-route prefetches get the shell's flight payload (not the HTML,
        # which would break client-side soft navigation).
        is_rsc = rel_path.endswith(".txt")
        lookup = rel_path[:-4] if is_rsc else rel_path
        ext = ".txt" if is_rsc else ".html"

        # Static route emitted as "<name>.html" / "<name>.txt".
        static_file = dist_dir / f"{lookup}{ext}"
        if lookup and static_file.is_file():
            return static_file

        # Directory index (HTML navigations only).
        if not is_rsc:
            index = candidate / "index.html"
            if candidate.is_dir() and index.is_file():
                return index

        # Dynamic route -> shell, in the format the request asked for.
        for pattern, target in shell_patterns:
            if pattern.match(lookup):
                if is_rsc:
                    txt = target.with_suffix(".txt")
                    return txt if txt.is_file() else target
                return target
        return None

    @app.get("/ui", include_in_schema=False)
    @app.get("/ui/{legacy_path:path}", include_in_schema=False)
    def redirect_legacy_ui(legacy_path: str = "") -> Response:
        # The UI used to be served under /ui/; keep old bookmarks working.
        return RedirectResponse(url="/")

    @app.get("/{full_path:path}", include_in_schema=False, name=_SPA_ROUTE_NAME)
    async def serve_spa(full_path: str) -> Response:
        if full_path == "":
            return FileResponse(dist_dir / "index.html")

        first_segment = full_path.split("/", 1)[0]
        if first_segment in backend_prefixes:
            # Owned by the backend but unmatched above -> genuine 404. Mirror
            # Starlette's default 404 body so API clients see a consistent shape
            # whether or not the frontend is mounted.
            return JSONResponse({"detail": "Not Found"}, status_code=404)

        resolved = _resolve(full_path)
        if resolved is not None:
            return FileResponse(resolved)

        if not_found.is_file():
            return FileResponse(not_found, status_code=404)
        return JSONResponse({"detail": "Not Found"}, status_code=404)

    logger.info(
        "Serving frontend static export from %s (%d dynamic-route shells)",
        dist_dir,
        len(shell_patterns),
    )
    return True
