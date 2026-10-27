#!/usr/bin/env python3
"""Serve this checkout as a newer Xinference release for manual upgrade tests."""

import argparse
import base64
import hashlib
import html
import http.server
import importlib.metadata
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit


def build_wheel(
    source: Path, output: Path, version: str, metadata: str, entries: str
) -> Path:
    """Keep real dependencies and code; give the test artifact its own version."""
    info = f"xinference-{version}.dist-info"
    wheel_path = output / f"xinference-{version}-py3-none-any.whl"
    records = []
    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as wheel:

        def write(name: str, data: bytes) -> None:
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                .decode()
                .rstrip("=")
            )
            records.append(f"{name},sha256={digest},{len(data)}\n")
            wheel.writestr(name, data)

        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if not path.is_file() or {
                "tests",
                "test",
                "__pycache__",
                "node_modules",
                ".next",
            } & set(relative.parts):
                continue
            if relative.as_posix() == "_version.py" or path.suffix in {
                ".pyc",
                ".pyd",
                ".so",
            }:
                continue
            write("xinference/" + relative.as_posix(), path.read_bytes())
        write("xinference/_version.py", f"__version__ = {version!r}\n".encode())
        metadata = re.sub(
            r"^Version:.*$",
            f"Version: {version}",
            metadata,
            count=1,
            flags=re.MULTILINE,
        )
        write(f"{info}/METADATA", metadata.encode())
        write(
            f"{info}/WHEEL",
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        if entries:
            write(f"{info}/entry_points.txt", entries.encode())
        wheel.writestr(f"{info}/RECORD", "".join(records) + f"{info}/RECORD,,\n")
    return wheel_path


def handler_for(wheel: Path, upstream: str):
    class Index(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.serve(head=True)

        def do_GET(self):
            self.serve(head=False)

        def serve(self, *, head: bool):
            path = unquote(urlsplit(self.path).path)
            if path.rstrip("/") in {"/simple", "/simple/xinference"}:
                link = f"/packages/{quote(wheel.name)}"
                body = f'<!doctype html><a href="{html.escape(link)}">{html.escape(wheel.name)}</a>'.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if not head:
                    self.wfile.write(body)
            elif path == "/packages/" + wheel.name:
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(wheel.stat().st_size))
                self.end_headers()
                if not head:
                    with wheel.open("rb") as stream:
                        shutil.copyfileobj(stream, self.wfile)
            elif path.startswith("/simple/"):
                # Dependencies use the normal package source. Only Xinference
                # is mocked, so a normal environment can really install B.
                project = path[len("/simple/") :].strip("/")
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", project):
                    self.send_error(404)
                    return
                self.send_response(302)
                origin = urlsplit(upstream)
                self.send_header(
                    "Location",
                    urlunsplit(
                        (
                            origin.scheme,
                            origin.netloc,
                            origin.path.rstrip("/") + "/" + quote(project) + "/",
                            origin.query,
                            "",
                        )
                    ),
                )
                self.end_headers()
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            pass

    return Index


def main():
    import os

    from packaging.version import Version

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        default="9999.0.0",
        help="A stable version newer than the running service",
    )
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--upstream",
        default=os.environ.get("PIP_INDEX_URL") or "https://pypi.org/simple",
    )
    args = parser.parse_args()
    version = Version(args.version)
    if version.is_prerelease:
        parser.error("Use a stable test version so update checks offer it")
    if urlsplit(args.upstream).scheme not in {"http", "https"}:
        parser.error("--upstream must be an HTTP(S) package index")
    dist = importlib.metadata.distribution("xinference")
    metadata = dist.read_text("METADATA")
    if metadata is None:
        parser.error("Install Xinference in the current Python environment first")
    with tempfile.TemporaryDirectory(prefix="xinference-mock-index-") as temporary:
        source = Path(__file__).resolve().parents[1] / "xinference"
        wheel = build_wheel(
            source,
            Path(temporary),
            str(version),
            metadata,
            dist.read_text("entry_points.txt") or "",
        )
        with http.server.ThreadingHTTPServer(
            (args.host, args.port), handler_for(wheel, args.upstream)
        ) as server:
            host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
            print(f"Mock Xinference version: {version}", flush=True)
            print(
                f"XINFERENCE_UPGRADE_INDEX_URL=http://{host}:{server.server_port}/simple",
                flush=True,
            )
            print(
                "Set the variable before starting your normal Xinference service; use the version control in the sidebar to update.",
                flush=True,
            )
            print(
                "This starts only the test package source. Ctrl+C stops it.", flush=True
            )
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
