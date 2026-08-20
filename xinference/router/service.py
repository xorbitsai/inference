# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Token Router service bootstrap."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Optional

import uvicorn

from .app import create_app
from .config import config_from_control_plane, load_config
from .control_plane import RouterControlPlaneClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Xinference Token-aware Router")
    parser.add_argument("--config", default=None, help="Standalone YAML config")
    parser.add_argument("--supervisor-url", default=None)
    parser.add_argument("--router-uid", default=None)
    parser.add_argument("--internal-token", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10080)
    parser.add_argument("--public-endpoint", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


async def _load_control_plane_config(args: argparse.Namespace):
    token = args.internal_token or os.getenv(
        "XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN", ""
    )
    endpoint = args.public_endpoint or f"http://{args.host}:{args.port}"
    client = RouterControlPlaneClient(
        args.supervisor_url,
        args.router_uid,
        internal_token=token,
        endpoint=endpoint,
        listen_host=args.host,
        listen_port=args.port,
        log_level=args.log_level,
    )
    try:
        data = await client.get_config()
        if data is None:
            raise RuntimeError("Supervisor returned no Token Router configuration")
        config = config_from_control_plane(
            data,
            listen_host=args.host,
            listen_port=args.port,
            log_level=args.log_level,
        )
        # Register as not yet ready. The application lifespan ACKs the initial
        # revision only after RouterRuntime.start() has completed successfully.
        await client.register()
        return config, client
    except Exception:
        await client.aclose()
        raise


async def _serve(args: argparse.Namespace) -> None:
    control_plane = None
    if args.supervisor_url or args.router_uid:
        if not (args.supervisor_url and args.router_uid):
            raise SystemExit("--supervisor-url and --router-uid must be used together")
        config, control_plane = await _load_control_plane_config(args)
    else:
        config = load_config(args.config)

    try:
        app = create_app(config)
        app.state.control_plane = control_plane
        uvicorn_config = uvicorn.Config(
            app,
            host=config.listen_host,
            port=config.listen_port,
            log_level=config.log_level.lower(),
            proxy_headers=True,
        )
        server = uvicorn.Server(uvicorn_config)
    except BaseException:
        # The application lifespan owns unregistering once Server.serve() has
        # started. If bootstrapping fails before then, close the async client
        # here so it is not leaked on the current event loop.
        if control_plane is not None:
            await control_plane.aclose()
        raise

    # Keep control-plane configuration loading, registration, lifespan ACK,
    # heartbeat, hot reload, and shutdown on the same asyncio event loop.
    await server.serve()


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_serve(args))
