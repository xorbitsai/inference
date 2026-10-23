# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""An API generation prepares its ASGI lifespan before accepting the listener."""

import asyncio
import contextlib
import json
import logging.config
import os
import signal
import socket
import sys
from pathlib import Path

import xoscar as xo
from uvicorn import Server
from uvicorn.lifespan.on import LifespanOn

from .. import __version__
from ..api.restful_api import RESTfulAPI


class HostedServer(Server):
    def __init__(self, config, api, configuration):
        super().__init__(config)
        self.api = api
        self.configuration = configuration
        self.activation = asyncio.Event()
        self.writer = None
        self.control_task = None
        self.draining = False
        config.load()
        application = config.loaded_app

        async def draining_application(scope, receive, send):
            async def send_response(message):
                if self.draining and message["type"] == "http.response.start":
                    message = {
                        **message,
                        "headers": [
                            (name, value)
                            for name, value in message.get("headers", [])
                            if name.lower() != b"connection"
                        ]
                        + [(b"connection", b"close")],
                    }
                await send(message)

            await application(scope, receive, send_response)

        config.loaded_app = draining_application
        owner = self

        class PreparedLifespan(LifespanOn):
            async def startup(self):
                await super().startup()
                if self.should_exit:
                    return
                supervisor = await owner.api._get_supervisor_ref()
                await xo.wait_for(supervisor.get_supervisor_info(), 10)
                await owner.connect()
                await owner.notify("ready", protocol=1, version=__version__)
                await owner.activation.wait()
                if owner.should_exit:
                    raise asyncio.CancelledError

        config.lifespan_class = PreparedLifespan

    async def connect(self):
        control = self.configuration["control"]
        reader, self.writer = await asyncio.open_connection(
            "127.0.0.1", control["port"]
        )
        self.writer.write(
            json.dumps(
                {
                    "token": control["token"],
                    "command": "register",
                    "identity": self.configuration["identity"],
                    "pid": os.getpid(),
                }
            ).encode()
            + b"\n"
        )
        await self.writer.drain()

        async def commands():
            try:
                while line := await reader.readline():
                    command = json.loads(line)["command"]
                    if command == "activate":
                        self.activation.set()
                    elif command == "drain":
                        self.begin_drain()
                        await self.notify("draining")
            finally:
                self.begin_drain()
                self.activation.set()

        self.control_task = asyncio.create_task(commands())

    def begin_drain(self):
        self.draining = True
        for listener in getattr(self, "servers", ()):
            listener.close()
        self.activation.set()
        if not self.started:
            self.should_exit = True

    async def on_tick(self, counter):
        # Keep draining separate from exiting: SSE libraries watch should_exit
        # and cancel streams. Idle connections retain their keep-alive timeout;
        # their next response advertises Connection: close before closing.
        if (
            self.draining
            and not self.server_state.connections
            and not self.server_state.tasks
        ):
            self.should_exit = True
        return await super().on_tick(counter)

    def handle_exit(self, sig, frame):
        # Older SSE versions patch Server.handle_exit to cancel every stream.
        # Handle this generation's signals without that global notification.
        if self.draining and sig == signal.SIGINT:
            self.force_exit = True
            self.should_exit = True
        self.begin_drain()

    async def notify(self, state, **fields):
        self.writer.write(json.dumps({"state": state, **fields}).encode() + b"\n")
        await self.writer.drain()

    async def startup(self, sockets=None):
        await super().startup(sockets=sockets)
        if self.started:
            await self.notify("serving")

    async def shutdown(self, sockets=None):
        try:
            await super().shutdown(sockets=sockets)
        finally:
            if self.control_task is not None:
                self.control_task.cancel()
                await asyncio.gather(self.control_task, return_exceptions=True)
            if self.writer is not None:
                self.writer.close()
                with contextlib.suppress(ConnectionError):
                    await self.writer.wait_closed()


def main():
    path = Path(sys.argv[1])
    configuration = json.loads(path.read_text())
    path.unlink()
    from .utils import update_all_formatter_addresses, update_logging_config_addresses

    logging_conf = configuration.get("logging_conf")
    update_all_formatter_addresses("supervisor", configuration["supervisor_address"])
    update_logging_config_addresses(
        logging_conf, "supervisor", configuration["supervisor_address"]
    )
    if logging_conf:
        logging.config.dictConfig(logging_conf)
    logging.getLogger(__name__).info(
        "Starting Xinference at endpoint: http://%s:%s",
        configuration["host"],
        configuration["port"],
    )
    from ..core.system_settings_store import SystemSettings, apply_system_settings
    from .api_host import host_request

    baseline = asyncio.run(host_request("settings", operation="baseline"))
    apply_system_settings(SystemSettings.from_dict(baseline))
    api = RESTfulAPI(
        configuration["supervisor_address"],
        configuration["host"],
        configuration["port"],
        # Multiple generations may authorize requests and commit revocations
        # concurrently. Use the auth store's existing authoritative read path.
        api_key_cache_enabled=False,
    )
    listener = socket.socket(fileno=configuration["listener_fd"])
    api.serve(
        logging_conf=configuration.get("logging_conf"),
        server_factory=lambda config: HostedServer(config, api, configuration),
        sockets=[listener],
    )


if __name__ == "__main__":
    main()
