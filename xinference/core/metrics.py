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

import uvicorn
from aioprometheus import Counter, Gauge, MetricsMiddleware
from aioprometheus.asgi.starlette import metrics
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

generate_throughput = Gauge(
    "xinference:generate_tokens_per_s", "Generate throughput in tokens/s."
)
first_token_latency = Gauge(
    "xinference:first_token_latency_ms", "First token latency in ms."
)
generate_latency = Gauge("xinference:generate_latency_ms", "Generate latency in ms.")
requests_throughput = Gauge("xinference:requests_per_s", "Requests per second.")
total_tokens_input = Counter(
    "xinference:total_tokens_input", "The counter for input tokens."
)
total_tokens_output = Counter(
    "xinference:total_tokens_output", "The counter for output tokens."
)


def update_metrics(name, op, kwargs):
    pass


def launch_metrics_export_server(host=None, port=None):
    app = FastAPI()

    app.add_middleware(MetricsMiddleware)
    app.add_route("/metrics", metrics)

    @app.get("/")
    async def root():
        response = RedirectResponse(url="/metrics")
        return response

    if host is not None or port is not None:
        uvicorn.run(app, host=host, port=port)
    else:
        uvicorn.run(app)
