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

import asyncio

import uvicorn
from aioprometheus import Counter, Gauge, MetricsMiddleware
from aioprometheus.asgi.starlette import metrics
from aioprometheus.mypy_types import LabelsType
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

DEFAULT_METRICS_SERVER_LOG_LEVEL = "warning"


generate_throughput = Gauge(
    "xinference:generate_tokens_per_s", "Generate throughput in tokens/s."
)
# Latency
first_token_latency = Gauge(
    "xinference:first_token_latency_ms", "First token latency in ms."
)
generate_latency = Gauge("xinference:generate_latency_ms", "Generate latency in ms.")
# Tokens counter
input_tokens_total_counter = Counter(
    "xinference:input_tokens_total_counter", "Total number of input tokens."
)
output_tokens_total_counter = Counter(
    "xinference:output_tokens_total_counter", "Total number of output tokens."
)
# RESTful API counter
requests_total_counter = Counter(
    "xinference:requests_total_counter", "Total number of requests received."
)
responses_total_counter = Counter(
    "xinference:responses_total_counter", "Total number of responses sent."
)
exceptions_total_counter = Counter(
    "xinference:exceptions_total_counter",
    "Total number of requested which generated an exception.",
)
status_codes_counter = Counter(
    "xinference:status_codes_counter", "Total number of response status codes."
)


def record_metrics(name, op, kwargs):
    collector = globals().get(name)
    getattr(collector, op)(**kwargs)


def launch_metrics_export_server(q, host=None, port=None):
    app = FastAPI()
    app.add_route("/metrics", metrics)

    @app.get("/")
    async def root():
        response = RedirectResponse(url="/metrics")
        return response

    async def main():
        if host is not None and port is not None:
            config = uvicorn.Config(
                app, host=host, port=port, log_level=DEFAULT_METRICS_SERVER_LOG_LEVEL
            )
        elif host is not None:
            config = uvicorn.Config(
                app, host=host, port=0, log_level=DEFAULT_METRICS_SERVER_LOG_LEVEL
            )
        elif port is not None:
            config = uvicorn.Config(
                app, port=port, log_level=DEFAULT_METRICS_SERVER_LOG_LEVEL
            )
        else:
            config = uvicorn.Config(app, log_level=DEFAULT_METRICS_SERVER_LOG_LEVEL)

        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve())

        while not server.started and not task.done():
            await asyncio.sleep(0.1)

        for server in server.servers:
            for socket in server.sockets:
                q.put(socket.getsockname())
        await task

    asyncio.run(main())


class RestfulAPIMetricsMiddleware(MetricsMiddleware):
    def __init__(self, restful_api, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.restful_api = restful_api

    def _counter_wrapper(self, name: str):
        class Counter:
            @staticmethod
            def inc(labels: LabelsType):
                supervisor_ref = self.restful_api._supervisor_ref
                if supervisor_ref is not None:
                    # May have performance issue.
                    coro = supervisor_ref.record_metrics(
                        name, "inc", {"labels": labels}
                    )
                    asyncio.create_task(coro)

        return Counter

    def create_metrics(self):
        """Create middleware metrics"""

        self.requests_counter = self._counter_wrapper("requests_total_counter")
        self.responses_counter = self._counter_wrapper("responses_total_counter")
        self.exceptions_counter = self._counter_wrapper("exceptions_total_counter")
        self.status_codes_counter = self._counter_wrapper("status_codes_counter")
        self.metrics_created = True
