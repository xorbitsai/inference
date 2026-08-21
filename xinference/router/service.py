# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Token Router service bootstrap."""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Optional

import uvicorn

from .app import create_app
from .config import config_from_control_plane, load_config
from .control_plane import RouterControlPlaneClient
from .logging_config import (
    configure_router_logging,
    normalize_log_level,
    router_access_log_enabled,
    update_router_logging_address,
)


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
    assignment_id = os.getenv("XINFERENCE_TOKEN_ROUTER_ASSIGNMENT_ID") or None
    generation_value = os.getenv("XINFERENCE_TOKEN_ROUTER_ASSIGNMENT_GENERATION")
    assignment_generation = int(generation_value) if generation_value else None
    node_id = os.getenv("XINFERENCE_TOKEN_ROUTER_NODE_ID") or None
    assignment_fields = (assignment_id, assignment_generation, node_id)
    if any(value is not None for value in assignment_fields) and not all(
        value is not None for value in assignment_fields
    ):
        raise RuntimeError(
            "Managed Router Runtime requires assignment ID, generation, and node ID"
        )
    client = RouterControlPlaneClient(
        args.supervisor_url,
        args.router_uid,
        internal_token=token,
        endpoint=endpoint,
        listen_host=args.host,
        listen_port=args.port,
        log_level=args.log_level,
        assignment_id=assignment_id,
        assignment_generation=assignment_generation,
        node_id=node_id,
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


async def _serve(args: argparse.Namespace, logging_conf: dict) -> None:
    control_plane = None
    if args.supervisor_url or args.router_uid:
        if not (args.supervisor_url and args.router_uid):
            raise SystemExit("--supervisor-url and --router-uid must be used together")
        config, control_plane = await _load_control_plane_config(args)
    else:
        config = load_config(args.config)

    try:
        logging_conf = update_router_logging_address(
            logging_conf, config.listen_host, config.listen_port
        )
        app = create_app(config)
        app.state.control_plane = control_plane
        uvicorn_config = uvicorn.Config(
            app,
            host=config.listen_host,
            port=config.listen_port,
            log_level=normalize_log_level(config.log_level).lower(),
            log_config=logging_conf,
            access_log=router_access_log_enabled(),
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
    logging_conf = configure_router_logging(args.log_level, args.host, args.port)
    asyncio.run(_serve(args, logging_conf))
