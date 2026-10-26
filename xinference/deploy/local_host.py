# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Keep local startup co-located; spawn component generations only on upgrade."""

import asyncio
import contextlib
import json
import logging
import os
import site
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

import xoscar as xo

from .api_host import (
    API_HOST_ENV,
    STARTUP_TIMEOUT,
    APIDiagnostics,
    APIHost,
    APIProcess,
    _wait_process,
)
from .process_host import start_process
from .supervisor_host import START_SUPERVISOR, SupervisorHost, SupervisorProcess
from .worker_host import HOST_PID, HOST_SOCKET, ProcessOwner, _enable_subreaper

logger = logging.getLogger(__name__)


def _runtime_environment() -> Dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if key != API_HOST_ENV
    }
    root = Path(__file__).resolve().parents[2]
    site_directories = [*site.getsitepackages(), site.getusersitepackages()]
    if not any(root.is_relative_to(Path(path).resolve()) for path in site_directories):
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(root), *filter(None, [environment.get("PYTHONPATH")])]
        )
    return environment


def _log_api_failure(task: asyncio.Task) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.error(
            "Local API failed", exc_info=(type(error), error, error.__traceback__)
        )


class InlineAPIProcess:
    def __init__(self):
        self.pid = os.getpid()
        self.server = None
        self.task: Optional[asyncio.Task] = None

    def poll(self) -> Optional[int]:
        if self.task is None or not self.task.done():
            return None
        return 1 if self.task.cancelled() or self.task.exception() is not None else 0

    def terminate(self) -> None:
        if self.server is not None:
            self.server.begin_drain()

    def kill(self) -> None:
        if self.server is not None:
            self.server.force_exit = True
            self.server.should_exit = True
            # Python 3.12+ waits for active transports in Server.wait_closed().
            # A second interrupt must release those transports and ASGI tasks.
            for connection in list(self.server.server_state.connections):
                connection.transport.close()
            for task in list(self.server.server_state.tasks):
                task.cancel()


class LocalHost(APIHost):
    """Reuse the API parent as service owner without an initial API subprocess."""

    serves_api_inline = True

    def __init__(self, listener, configuration, settings_store):
        super().__init__(listener, configuration, settings_store)
        self.owner = ProcessOwner()
        self.cluster: Optional[subprocess.Popen] = None
        self.cluster_exit: Optional[asyncio.Task] = None
        self.worker_socket = ""
        self.last_process_scan = 0.0

    async def start_components(self, environment: Dict[str, str]) -> None:
        os.environ[API_HOST_ENV] = json.dumps(self.connection)
        from ..constants import (
            XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD,
            XINFERENCE_HEALTH_CHECK_INTERVAL,
        )
        from ..core.supervisor import SupervisorActor
        from .local import LOCAL_SUPERVISOR_UID
        from .utils import health_check

        environment = {
            **_runtime_environment(),
            HOST_SOCKET: self.worker_socket,
            HOST_PID: str(os.getpid()),
        }
        path = self.directory / "local.json"
        path.write_text(
            json.dumps(
                {
                    "address": self.configuration["supervisor_address"],
                    "worker_address": self.configuration["worker_address"],
                    "metrics_exporter_host": self.configuration.get(
                        "metrics_exporter_host"
                    ),
                    "metrics_exporter_port": self.configuration.get(
                        "metrics_exporter_port"
                    ),
                    "logging_conf": self.configuration.get("logging_conf"),
                }
            )
        )
        path.chmod(0o600)
        script = self.directory / "start_local.py"
        script.write_text(
            "if __name__ == '__main__':\n    import runpy\n    runpy.run_module('xinference.deploy.local', run_name='__main__')\n"
        )
        diagnostics = APIDiagnostics()
        self.cluster = await start_process(
            [sys.executable, "-u", str(script), str(path)],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.cluster_exit = asyncio.create_task(
            _wait_process(self.cluster, diagnostics)
        )
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while self.cluster.pid not in self.owner.workers:
            if self.cluster.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError(
                    "Local cluster failed to start: " + diagnostics.read()
                )
            await asyncio.sleep(0.05)
        if not await asyncio.to_thread(
            health_check,
            self.configuration["supervisor_address"],
            XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD,
            XINFERENCE_HEALTH_CHECK_INTERVAL,
        ):
            raise RuntimeError("Local cluster failed its startup health check")
        host = self.supervisor_host = SupervisorHost(
            self.configuration, self.directory, self.settings_store
        )
        (self.directory / "start_supervisor.py").write_text(START_SUPERVISOR)
        host.gateway = await xo.actor_ref(
            address=self.configuration["supervisor_address"],
            uid=SupervisorActor.default_uid(),
        )
        source = await xo.actor_ref(
            address=self.configuration["supervisor_address"], uid=LOCAL_SUPERVISOR_UID
        )
        host.active = SupervisorProcess(
            self.cluster,
            source,
            sys.executable,
            environment,
            diagnostics,
            self.cluster_exit,
            shared=True,
        )
        host.children.append(host.active)
        await host.ready(host.active)
        await self.start_inline_api(environment)

    async def start_inline_api(self, environment: Dict[str, str]) -> None:
        from ..api.restful_api import RESTfulAPI
        from .api_worker import HostedServer
        from .utils import (
            update_all_formatter_addresses,
            update_logging_config_addresses,
        )

        process = InlineAPIProcess()

        class InlineServer(HostedServer):
            def run(self, sockets=None):
                process.server = self
                process.task = asyncio.create_task(self.serve(sockets=sockets))
                process.task.add_done_callback(_log_api_failure)

            @contextlib.contextmanager
            def capture_signals(self):
                # The existing API parent owns shutdown of the entire service.
                yield

            def install_signal_handlers(self):
                pass

        identity = uuid.uuid4().hex
        configuration = {
            **self.configuration,
            "identity": identity,
            "control": self.connection,
        }
        update_all_formatter_addresses(
            "local", self.configuration["supervisor_address"]
        )
        update_logging_config_addresses(
            self.configuration.get("logging_conf"),
            "local",
            self.configuration["supervisor_address"],
        )
        api = RESTfulAPI(
            self.configuration["supervisor_address"],
            self.configuration["host"],
            self.configuration["port"],
            api_key_cache_enabled=False,
        )
        api.serve(
            logging_conf=self.configuration.get("logging_conf"),
            server_factory=lambda config: InlineServer(config, api, configuration),
            sockets=[self.listener.dup()],
        )
        assert process.task is not None
        child = self.active = APIProcess(
            identity,
            process,
            sys.executable,
            environment,
            APIDiagnostics(),
            process.task,
        )
        self.children[identity] = child
        await child.wait(child.ready)
        await child.send("activate")
        await child.wait(child.serving)

    def check(self) -> None:
        super().check()
        now = time.monotonic()
        if now - self.last_process_scan >= 1:
            self.owner.remember()
            self.last_process_scan = now
        watched = {child.process.pid for child in self.children.values()}
        if self.supervisor_host is not None:
            watched.update(child.process.pid for child in self.supervisor_host.children)
        if self.cluster is not None:
            watched.add(self.cluster.pid)
        self.owner.reap(watched)
        if self.cluster is not None and self.cluster.poll() is not None:
            raise RuntimeError("Local service gateway exited unexpectedly")

    def force_exit(self) -> None:
        self.owner.kill()
        super().force_exit()

    async def run(self, stopping: asyncio.Event) -> None:
        _enable_subreaper()
        previous = os.environ.get(API_HOST_ENV)
        server, self.worker_socket = await self.owner.serve()
        try:
            await super().run(stopping)
        finally:
            server.close()
            await server.wait_closed()
            await self.owner.stop()
            if self.cluster_exit is not None:
                await self.cluster_exit
            if previous is None:
                os.environ.pop(API_HOST_ENV, None)
            else:
                os.environ[API_HOST_ENV] = previous
