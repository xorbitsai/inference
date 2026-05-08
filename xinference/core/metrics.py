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

import asyncio
import logging
import platform
from collections import defaultdict
from typing import Any, Dict, Set, Tuple

import uvicorn
from aioprometheus import REGISTRY, Counter, Gauge, Histogram
from aioprometheus.asgi.starlette import metrics
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

logger = logging.getLogger(__name__)

DEFAULT_METRICS_SERVER_LOG_LEVEL = "warning"

# ===========================================================================
# Worker-side inference metrics (LLM only)
# ===========================================================================
generate_throughput = Gauge(
    "xinference:generate_tokens_per_s",
    "Generate throughput in tokens/s (LLM only).",
)
time_to_first_token = Gauge(
    "xinference:time_to_first_token_ms",
    "First token latency in ms (LLM only).",
)
input_tokens_total_counter = Counter(
    "xinference:input_tokens_total_counter",
    "Total number of input tokens (LLM only).",
)
output_tokens_total_counter = Counter(
    "xinference:output_tokens_total_counter",
    "Total number of output tokens (LLM only).",
)

# ===========================================================================
# Worker-side model service quality metrics (all model types)
# ===========================================================================
model_request_total = Counter(
    "xinference:model_request_total",
    "Total number of model requests.",
)
model_request_errors_total = Counter(
    "xinference:model_request_errors_total",
    "Total number of failed model requests.",
)
model_request_duration_seconds = Histogram(
    "xinference:model_request_duration_seconds",
    "Model request duration in seconds.",
    buckets=(
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
        120.0,
        float("inf"),
    ),
)
model_serve_count = Gauge(
    "xinference:model_serve_count",
    "Number of requests currently being served.",
)
model_request_limit_gauge = Gauge(
    "xinference:model_request_limit",
    "Maximum concurrent request limit for the model.",
)

# ===========================================================================
# Supervisor-side cluster / worker / model info metrics
# ===========================================================================
supervisor_uptime = Gauge(
    "xinference:supervisor_uptime_seconds",
    "Supervisor uptime in seconds.",
)
workers_total = Gauge(
    "xinference:workers_total",
    "Number of online workers.",
)
models_loaded_total = Gauge(
    "xinference:models_loaded_total",
    "Number of loaded models by type.",
)
worker_cpu_utilization = Gauge(
    "xinference:worker_cpu_utilization",
    "Worker CPU utilization (0-1).",
)
worker_memory_used_bytes = Gauge(
    "xinference:worker_memory_used_bytes",
    "Worker memory used in bytes.",
)
worker_memory_total_bytes = Gauge(
    "xinference:worker_memory_total_bytes",
    "Worker total memory in bytes.",
)
worker_gpu_utilization_percent = Gauge(
    "xinference:worker_gpu_utilization_percent",
    "Worker GPU utilization (0-100).",
)
worker_gpu_memory_used_bytes = Gauge(
    "xinference:worker_gpu_memory_used_bytes",
    "Worker GPU memory used in bytes.",
)
worker_gpu_memory_total_bytes = Gauge(
    "xinference:worker_gpu_memory_total_bytes",
    "Worker GPU total memory in bytes.",
)
model_info_gauge = Gauge(
    "xinference:model_info",
    "Running model info (value=1).",
)
model_status_gauge = Gauge(
    "xinference:model_status",
    "Model lifecycle status (value=1).",
)
model_gpu_binding_gauge = Gauge(
    "xinference:model_gpu_binding",
    "Per-replica GPU binding (one series per GPU per replica, value=1).",
)
build_info_gauge = Gauge(
    "xinference:build_info",
    "Xinference build information (value=1).",
)
config_info_gauge = Gauge(
    "xinference:config_info",
    "Xinference configuration information (value=1).",
)

# ---------------------------------------------------------------------------
# Supervisor-only metric names — removed from Worker Registry at startup
# ---------------------------------------------------------------------------
_SUPERVISOR_ONLY_METRICS = {
    "xinference:supervisor_uptime_seconds",
    "xinference:workers_total",
    "xinference:models_loaded_total",
    "xinference:model_info",
    "xinference:model_status",
    "xinference:worker_cpu_utilization",
    "xinference:worker_memory_used_bytes",
    "xinference:worker_memory_total_bytes",
    "xinference:worker_gpu_utilization_percent",
    "xinference:worker_gpu_memory_used_bytes",
    "xinference:worker_gpu_memory_total_bytes",
    "xinference:model_gpu_binding",
}

# ---------------------------------------------------------------------------
# Stale-label tracking sets for update_cluster_metrics()
# ---------------------------------------------------------------------------
_prev_worker_labels: Set[Tuple[str, ...]] = set()
_prev_gpu_labels: Set[Tuple[str, ...]] = set()
_prev_model_labels: Set[Tuple[str, ...]] = set()
_prev_status_labels: Set[Tuple[str, ...]] = set()
_prev_gpu_binding_labels: Set[Tuple[str, ...]] = set()


def record_metrics(name, op, kwargs):
    collector = globals().get(name)
    if collector is not None:
        getattr(collector, op)(**kwargs)


def set_build_info(
    cluster: str = "",
    role: str = "",
    worker_address: str = "",
    supervisor_address: str = "",
) -> None:
    """Set xinference:build_info gauge with version and runtime information."""
    from xinference import __version__

    labels = {
        "version": __version__,
        "python_version": platform.python_version(),
    }
    if cluster:
        labels["cluster"] = cluster
    if role:
        labels["xinference_role"] = role
    if worker_address:
        labels["worker_address"] = worker_address
    if supervisor_address:
        labels["supervisor_address"] = supervisor_address
    build_info_gauge.set(labels, 1)


def set_config_info(
    xinference_home: str,
    role: str,
    cluster: str = "",
    worker_address: str = "",
    supervisor_address: str = "",
) -> None:
    """Set xinference:config_info gauge with configuration information."""
    labels = {
        "xinference_home": xinference_home,
        "xinference_role": role,
    }
    if cluster:
        labels["cluster"] = cluster
    if worker_address:
        labels["worker_address"] = worker_address
    if supervisor_address:
        labels["supervisor_address"] = supervisor_address
    config_info_gauge.set(labels, 1)


def update_cluster_metrics(
    cluster_data: Dict[str, Any],
    models_data: Dict[str, Dict[str, Any]],
    supervisor_address: str = "",
) -> None:
    """Refresh all Supervisor-side Prometheus gauges from in-memory data."""
    global _prev_worker_labels, _prev_gpu_labels
    global _prev_model_labels, _prev_status_labels, _prev_gpu_binding_labels

    # --- Build info (set once, labels are static) ---
    cluster_name = cluster_data.get("cluster", "")
    set_build_info(
        cluster=cluster_name, role="supervisor", supervisor_address=supervisor_address
    )

    # --- Config info for supervisor ---
    from xinference.constants import XINFERENCE_HOME as _xf_home

    set_config_info(
        xinference_home=_xf_home,
        role="supervisor",
        cluster=cluster_name,
        supervisor_address=supervisor_address,
    )

    # --- Supervisor uptime ---
    supervisor_uptime.set({}, cluster_data.get("uptime", 0))

    # --- Workers total ---
    workers_total.set({}, cluster_data.get("worker_count", 0))

    # --- Worker resources ---
    cur_worker_labels: Set[Tuple[str, ...]] = set()
    cur_gpu_labels: Set[Tuple[str, ...]] = set()

    for addr, status in cluster_data.get("workers", {}).items():
        w_labels = {"worker_address": addr}
        label_key = (addr,)
        cur_worker_labels.add(label_key)

        # status is Dict[str, Union[ResourceStatus, GPUStatus]] serialized as dict
        # "cpu" key -> ResourceStatus, integer keys -> GPUStatus
        cpu_res = status.get("cpu")
        if cpu_res:
            worker_cpu_utilization.set(w_labels, getattr(cpu_res, "usage", 0))
            worker_memory_used_bytes.set(w_labels, getattr(cpu_res, "memory_used", 0))
            worker_memory_total_bytes.set(w_labels, getattr(cpu_res, "memory_total", 0))

        for key, val in status.items():
            if key == "cpu":
                continue
            # GPU entries keyed by integer index
            gpu_idx = str(key)
            gpu_name = getattr(val, "name", "unknown")
            g_labels = {
                "worker_address": addr,
                "gpu_index": gpu_idx,
                "gpu_name": gpu_name,
            }
            g_key = (addr, gpu_idx, gpu_name)
            cur_gpu_labels.add(g_key)
            worker_gpu_utilization_percent.set(g_labels, getattr(val, "gpu_util", 0))
            worker_gpu_memory_used_bytes.set(
                g_labels, getattr(val, "gpu_mem_used", getattr(val, "mem_used", 0))
            )
            worker_gpu_memory_total_bytes.set(
                g_labels, getattr(val, "gpu_mem_total", getattr(val, "mem_total", 0))
            )

    # Clear stale worker labels
    for stale in _prev_worker_labels - cur_worker_labels:
        s = {"worker_address": stale[0]}
        worker_cpu_utilization.set(s, 0)
        worker_memory_used_bytes.set(s, 0)
        worker_memory_total_bytes.set(s, 0)
    _prev_worker_labels = cur_worker_labels

    for stale in _prev_gpu_labels - cur_gpu_labels:
        s = {"worker_address": stale[0], "gpu_index": stale[1], "gpu_name": stale[2]}
        worker_gpu_utilization_percent.set(s, 0)
        worker_gpu_memory_used_bytes.set(s, 0)
        worker_gpu_memory_total_bytes.set(s, 0)
    _prev_gpu_labels = cur_gpu_labels

    # --- Models loaded by type ---
    type_counts: Dict[str, int] = defaultdict(int)
    cur_model_labels: set = set()

    model_replica_dist = cluster_data.get("model_replica_distribution", {})

    for model_uid, info in models_data.items():
        model_type = info.get("model_type", "unknown")
        model_name = info.get("model_name", "unknown")
        type_counts[model_type] += 1

        dist = model_replica_dist.get(model_uid)
        if dist and dist.get("worker_distribution"):
            replica_total = str(dist["replica_total"])
            for w_addr, w_count in dist["worker_distribution"].items():
                m_labels = {
                    "model_uid": model_uid,
                    "model_name": model_name,
                    "model_type": model_type,
                    "worker_address": str(w_addr),
                    "replica_on_worker": str(w_count),
                    "replica_total": replica_total,
                }
                label_key = (  # type: ignore[assignment]
                    model_uid,
                    model_name,
                    model_type,
                    str(w_addr),
                    str(w_count),
                    replica_total,
                )
                cur_model_labels.add(label_key)
                model_info_gauge.set(m_labels, 1)
        else:
            worker_address = info.get("address", "unknown")
            replica = str(info.get("replica", 1))
            m_labels = {
                "model_uid": model_uid,
                "model_name": model_name,
                "model_type": model_type,
                "worker_address": str(worker_address),
                "replica_on_worker": replica,
                "replica_total": replica,
            }
            label_key = (  # type: ignore[assignment]
                model_uid,
                model_name,
                model_type,
                str(worker_address),
                replica,
                replica,
            )
            cur_model_labels.add(label_key)
            model_info_gauge.set(m_labels, 1)

    # Clear stale model labels
    for stale in _prev_model_labels - cur_model_labels:
        stale_labels = {
            "model_uid": stale[0],
            "model_name": stale[1],
            "model_type": stale[2],
            "worker_address": stale[3],
            "replica_on_worker": stale[4],
            "replica_total": stale[5],
        }
        model_info_gauge.set(stale_labels, 0)
    _prev_model_labels = cur_model_labels

    # --- Model GPU binding (per-replica, per-GPU) ---
    cur_gpu_binding_labels: set = set()

    for model_uid, info in models_data.items():
        model_type = info.get("model_type", "unknown")
        model_name = info.get("model_name", "unknown")
        dist = model_replica_dist.get(model_uid)
        if dist:
            for rep_idx, w_addr, gpu_list in dist.get("replica_gpu_details", []):
                for gpu_idx in gpu_list:
                    b_labels = {
                        "model_uid": model_uid,
                        "model_name": model_name,
                        "model_type": model_type,
                        "worker_address": str(w_addr),
                        "gpu_index": gpu_idx,
                        "replica_index": rep_idx,
                    }
                    label_key = (  # type: ignore[assignment]
                        model_uid,
                        model_name,
                        model_type,
                        str(w_addr),
                        gpu_idx,
                        rep_idx,
                    )
                    cur_gpu_binding_labels.add(label_key)
                    model_gpu_binding_gauge.set(b_labels, 1)

    # Clear stale gpu_binding labels
    for stale in _prev_gpu_binding_labels - cur_gpu_binding_labels:
        stale_labels = {
            "model_uid": stale[0],
            "model_name": stale[1],
            "model_type": stale[2],
            "worker_address": stale[3],
            "gpu_index": stale[4],
            "replica_index": stale[5],
        }
        model_gpu_binding_gauge.set(stale_labels, 0)
    _prev_gpu_binding_labels = cur_gpu_binding_labels

    # --- Models loaded total (by type) ---
    for mt, count in type_counts.items():
        models_loaded_total.set({"model_type": mt}, count)

    # --- Model lifecycle status ---
    cur_status_labels: Set[Tuple[str, ...]] = set()
    for inst in cluster_data.get("instance_infos", []):
        uid = inst.get("model_uid", "unknown")
        mname = inst.get("model_name", "unknown")
        status = inst.get("status", "unknown")
        s_labels = {"model_uid": uid, "model_name": mname, "status": status}
        s_key = (uid, mname, status)
        cur_status_labels.add(s_key)
        model_status_gauge.set(s_labels, 1)

    for stale in _prev_status_labels - cur_status_labels:
        stale_labels = {
            "model_uid": stale[0],
            "model_name": stale[1],
            "status": stale[2],
        }
        model_status_gauge.set(stale_labels, 0)
    _prev_status_labels = cur_status_labels


def launch_metrics_export_server(q, host=None, port=None):
    # Remove Supervisor-only metrics from Worker's Registry to avoid
    # empty HELP/TYPE headers on the Worker /metrics endpoint.
    for collector in list(REGISTRY.get_all()):
        if collector.name in _SUPERVISOR_ONLY_METRICS:
            REGISTRY.deregister(collector.name)

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
