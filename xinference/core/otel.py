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

"""
OpenTelemetry (OTEL) initialization for Xinference.

This module is only imported when ENABLE_OTEL=true.  All setup is performed
in a single call to ``setup_otel(app)``.  Requires the optional dependency
group:  ``pip install xinference[otel]``

Supported environment variables
--------------------------------
XINFERENCE_ENABLE_OTEL                         Enable OTEL (default: false)
XINFERENCE_OTLP_BASE_ENDPOINT                  Base OTLP endpoint  (default: http://localhost:4318)
XINFERENCE_OTLP_TRACE_ENDPOINT                 Override trace endpoint  (default: <base>/v1/traces)
XINFERENCE_OTLP_METRIC_ENDPOINT                Override metric endpoint (default: <base>/v1/metrics)
XINFERENCE_OTLP_API_KEY                        API key sent as "Authorization: Bearer <key>" header
XINFERENCE_OTEL_EXPORTER_OTLP_PROTOCOL         "http/protobuf" (default) or "grpc"
XINFERENCE_OTEL_EXPORTER_TYPE                  "otlp" (default) – reserved for future exporters
XINFERENCE_OTEL_SAMPLING_RATE                  TraceIdRatio sampler rate, 0.0-1.0 (default: 0.1)
XINFERENCE_OTEL_BATCH_EXPORT_SCHEDULE_DELAY    Batch span processor schedule delay ms (default: 5000)
XINFERENCE_OTEL_MAX_QUEUE_SIZE                 Max span queue size (default: 2048)
XINFERENCE_OTEL_MAX_EXPORT_BATCH_SIZE          Max spans per export batch (default: 512)
XINFERENCE_OTEL_METRIC_EXPORT_INTERVAL         Metric periodic reader interval ms (default: 60000)
XINFERENCE_OTEL_BATCH_EXPORT_TIMEOUT           Batch span export timeout ms (default: 10000)
XINFERENCE_OTEL_METRIC_EXPORT_TIMEOUT          Metric export timeout ms (default: 30000)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Sentinel so setup_otel() is idempotent (safe to call multiple times)
_otel_initialized = False


def setup_otel(
    app=None,
    service_name: str = "xinference",
    *,
    instrument_app: bool = True,
    register_worker_metrics: bool = True,
) -> None:  # type: ignore[type-arg]
    """
    Initialise OpenTelemetry tracing and metrics and instrument the FastAPI app.

    This function is a no-op when called a second time (idempotent).
    All OTEL SDK objects are created lazily inside this function so that the
    heavy imports are never executed when OTEL is disabled.

    Parameters
    ----------
    app:
        The FastAPI application instance.
    service_name:
        The OTEL ``service.name`` resource attribute (default: "xinference").
    instrument_app:
        Whether to attach FastAPI auto-instrumentation for the provided app.
    register_worker_metrics:
        Whether to register supervisor-side worker resource metrics.
    """
    global _otel_initialized
    if _otel_initialized:
        return

    from ..constants import (
        XINFERENCE_OTEL_BATCH_EXPORT_SCHEDULE_DELAY,
        XINFERENCE_OTEL_BATCH_EXPORT_TIMEOUT,
        XINFERENCE_OTEL_EXPORTER_OTLP_PROTOCOL,
        XINFERENCE_OTEL_MAX_EXPORT_BATCH_SIZE,
        XINFERENCE_OTEL_MAX_QUEUE_SIZE,
        XINFERENCE_OTEL_METRIC_EXPORT_INTERVAL,
        XINFERENCE_OTEL_METRIC_EXPORT_TIMEOUT,
        XINFERENCE_OTEL_SAMPLING_RATE,
        XINFERENCE_OTLP_API_KEY,
        XINFERENCE_OTLP_METRIC_ENDPOINT,
        XINFERENCE_OTLP_TRACE_ENDPOINT,
    )

    try:
        _setup_tracing(
            service_name=service_name,
            trace_endpoint=XINFERENCE_OTLP_TRACE_ENDPOINT,
            api_key=XINFERENCE_OTLP_API_KEY,
            protocol=XINFERENCE_OTEL_EXPORTER_OTLP_PROTOCOL,
            sampling_rate=XINFERENCE_OTEL_SAMPLING_RATE,
            schedule_delay_ms=XINFERENCE_OTEL_BATCH_EXPORT_SCHEDULE_DELAY,
            max_queue_size=XINFERENCE_OTEL_MAX_QUEUE_SIZE,
            max_export_batch_size=XINFERENCE_OTEL_MAX_EXPORT_BATCH_SIZE,
            export_timeout_ms=XINFERENCE_OTEL_BATCH_EXPORT_TIMEOUT,
        )
        _setup_metrics(
            service_name=service_name,
            metric_endpoint=XINFERENCE_OTLP_METRIC_ENDPOINT,
            api_key=XINFERENCE_OTLP_API_KEY,
            protocol=XINFERENCE_OTEL_EXPORTER_OTLP_PROTOCOL,
            export_interval_ms=XINFERENCE_OTEL_METRIC_EXPORT_INTERVAL,
            export_timeout_ms=XINFERENCE_OTEL_METRIC_EXPORT_TIMEOUT,
        )
        if instrument_app and app is not None:
            _instrument_fastapi(app)
        collector_id = None
        if register_worker_metrics:
            # Register worker metrics gauges (data is fed by supervisor)
            global _cluster_metrics_collector
            _cluster_metrics_collector = ClusterMetricsCollector()
            _cluster_metrics_collector.register()
            collector_id = id(_cluster_metrics_collector)

        _otel_initialized = True
        logger.info(
            "OpenTelemetry initialized. "
            "trace_endpoint=%s metric_endpoint=%s sampling_rate=%s collector_id=%s",
            XINFERENCE_OTLP_TRACE_ENDPOINT,
            XINFERENCE_OTLP_METRIC_ENDPOINT,
            XINFERENCE_OTEL_SAMPLING_RATE,
            collector_id,
        )
    except ImportError as exc:
        logger.warning(
            "OpenTelemetry packages are not installed. "
            "OTEL support is disabled. "
            "Install them with: pip install xinference[otel]\n"
            "Error: %s",
            exc,
        )
    except Exception:
        logger.exception(
            "Failed to initialise OpenTelemetry. OTEL support is disabled."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_headers(api_key: str) -> Optional[dict]:
    """Return Authorization headers when an API key is configured."""
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return None


def _setup_tracing(
    service_name: str,
    trace_endpoint: str,
    api_key: str,
    protocol: str,
    sampling_rate: float,
    schedule_delay_ms: int,
    max_queue_size: int,
    max_export_batch_size: int,
    export_timeout_ms: int,
) -> None:
    """Configure the global TracerProvider with a BatchSpanProcessor."""
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

    resource = Resource.create({"service.name": service_name})
    sampler = TraceIdRatioBased(sampling_rate)
    provider = TracerProvider(resource=resource, sampler=sampler)

    exporter = _build_span_exporter(
        endpoint=trace_endpoint,
        api_key=api_key,
        protocol=protocol,
        export_timeout_ms=export_timeout_ms,
    )

    processor = BatchSpanProcessor(
        exporter,
        schedule_delay_millis=schedule_delay_ms,
        max_queue_size=max_queue_size,
        max_export_batch_size=max_export_batch_size,
        export_timeout_millis=export_timeout_ms,
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)


def _build_span_exporter(
    endpoint: str,
    api_key: str,
    protocol: str,
    export_timeout_ms: int,
):
    """Return an OTLP span exporter based on the configured protocol."""
    headers = _build_headers(api_key)
    timeout_s = export_timeout_ms // 1000

    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(
            endpoint=endpoint,
            headers=headers,
            timeout=timeout_s,
        )
    else:
        # Default: http/protobuf
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(
            endpoint=endpoint,
            headers=headers,
            timeout=timeout_s,
        )


def _setup_metrics(
    service_name: str,
    metric_endpoint: str,
    api_key: str,
    protocol: str,
    export_interval_ms: int,
    export_timeout_ms: int,
) -> None:
    """Configure the global MeterProvider with a PeriodicExportingMetricReader."""
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create({"service.name": service_name})
    exporter = _build_metric_exporter(
        endpoint=metric_endpoint,
        api_key=api_key,
        protocol=protocol,
        export_timeout_ms=export_timeout_ms,
    )
    reader = PeriodicExportingMetricReader(
        exporter,
        export_interval_millis=export_interval_ms,
        export_timeout_millis=export_timeout_ms,
    )
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)


def _build_metric_exporter(
    endpoint: str,
    api_key: str,
    protocol: str,
    export_timeout_ms: int,
):
    """Return an OTLP metric exporter based on the configured protocol."""
    headers = _build_headers(api_key)
    timeout_s = export_timeout_ms // 1000

    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )

        return OTLPMetricExporter(
            endpoint=endpoint,
            headers=headers,
            timeout=timeout_s,
        )
    else:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )

        return OTLPMetricExporter(
            endpoint=endpoint,
            headers=headers,
            timeout=timeout_s,
        )


def _instrument_fastapi(app) -> None:  # type: ignore[type-arg]
    """Attach OpenTelemetry auto-instrumentation to the FastAPI app."""
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)
    logger.debug("FastAPI instrumented with OpenTelemetry.")


# ---------------------------------------------------------------------------
# Supervisor-side worker metrics (CPU / MEM / GPU)
# ---------------------------------------------------------------------------


class ClusterMetricsCollector:
    """
    Collects worker node metrics (CPU, memory, GPU) for ALL workers and
    reports them as OTEL ObservableGauge metrics.

    This is designed to run in the **supervisor** process, which already
    receives periodic ``report_worker_status(address, node_info)`` calls
    from every worker.  The supervisor's ``setup_otel()`` has already
    initialised the MeterProvider, so no additional provider setup is needed.

    Usage::

        collector = ClusterMetricsCollector()
        collector.register()

        # In supervisor.report_worker_status():
        collector.update(worker_address, node_info)
    """

    def __init__(self) -> None:
        # { worker_address: { "cpu": ResourceStatus, 0: GPUStatus, ... } }
        self._workers: dict = {}

    def update(self, worker_address: str, node_info: dict) -> None:
        """Store the latest ``gather_node_info()`` result for a worker."""
        self._workers[worker_address] = node_info

    def remove_worker(self, worker_address: str) -> None:
        """Stop reporting metrics for a removed/dead worker."""
        self._workers.pop(worker_address, None)

    def register(self) -> None:
        """Create OTEL ObservableGauge instruments and bind callbacks."""
        from opentelemetry import metrics

        meter = metrics.get_meter("xinference.worker")

        meter.create_observable_gauge(
            name="xinference.worker.cpu.utilization",
            callbacks=[self._cpu_utilization_cb],
            description="Worker CPU utilization (0.0 - 1.0)",
            unit="1",
        )
        meter.create_observable_gauge(
            name="xinference.worker.cpu.count",
            callbacks=[self._cpu_count_cb],
            description="Worker CPU core count",
            unit="{cores}",
        )
        meter.create_observable_gauge(
            name="xinference.worker.memory.used",
            callbacks=[self._memory_used_cb],
            description="Worker memory used in bytes",
            unit="By",
        )
        meter.create_observable_gauge(
            name="xinference.worker.memory.total",
            callbacks=[self._memory_total_cb],
            description="Worker total memory in bytes",
            unit="By",
        )
        meter.create_observable_gauge(
            name="xinference.worker.gpu.utilization",
            callbacks=[self._gpu_utilization_cb],
            description="Worker GPU utilization (0.0 - 1.0)",
            unit="1",
        )
        meter.create_observable_gauge(
            name="xinference.worker.gpu.memory.used",
            callbacks=[self._gpu_mem_used_cb],
            description="Worker GPU memory used in bytes",
            unit="By",
        )
        meter.create_observable_gauge(
            name="xinference.worker.gpu.memory.total",
            callbacks=[self._gpu_mem_total_cb],
            description="Worker GPU total memory in bytes",
            unit="By",
        )
        meter.create_observable_gauge(
            name="xinference.worker.gpu.memory.free",
            callbacks=[self._gpu_mem_free_cb],
            description="Worker GPU free memory in bytes",
            unit="By",
        )
    # -- CPU callbacks -------------------------------------------------------

    def _cpu_utilization_cb(self, options):  # type: ignore[no-untyped-def]
        from opentelemetry.metrics import Observation

        for addr, info in self._workers.items():
            cpu = info.get("cpu")
            if cpu is not None:
                yield Observation(cpu.usage, {"worker_address": addr})

    def _cpu_count_cb(self, options):  # type: ignore[no-untyped-def]
        from opentelemetry.metrics import Observation

        for addr, info in self._workers.items():
            cpu = info.get("cpu")
            if cpu is not None:
                yield Observation(cpu.total, {"worker_address": addr})

    # -- Memory callbacks ----------------------------------------------------

    def _memory_used_cb(self, options):  # type: ignore[no-untyped-def]
        from opentelemetry.metrics import Observation

        for addr, info in self._workers.items():
            cpu = info.get("cpu")
            if cpu is not None:
                yield Observation(cpu.memory_used, {"worker_address": addr})

    def _memory_total_cb(self, options):  # type: ignore[no-untyped-def]
        from opentelemetry.metrics import Observation

        for addr, info in self._workers.items():
            cpu = info.get("cpu")
            if cpu is not None:
                yield Observation(cpu.memory_total, {"worker_address": addr})

    # -- GPU callbacks -------------------------------------------------------

    def _gpu_utilization_cb(self, options):  # type: ignore[no-untyped-def]
        from opentelemetry.metrics import Observation

        for addr, info in self._workers.items():
            for key, gpu in info.items():
                if key == "cpu":
                    continue
                yield Observation(
                    gpu.gpu_util,
                    {
                        "worker_address": addr,
                        "gpu_index": str(key),
                        "gpu_name": gpu.name,
                    },
                )

    def _gpu_mem_used_cb(self, options):  # type: ignore[no-untyped-def]
        from opentelemetry.metrics import Observation

        for addr, info in self._workers.items():
            for key, gpu in info.items():
                if key == "cpu":
                    continue
                yield Observation(
                    gpu.mem_used,
                    {
                        "worker_address": addr,
                        "gpu_index": str(key),
                        "gpu_name": gpu.name,
                    },
                )

    def _gpu_mem_total_cb(self, options):  # type: ignore[no-untyped-def]
        from opentelemetry.metrics import Observation

        for addr, info in self._workers.items():
            for key, gpu in info.items():
                if key == "cpu":
                    continue
                yield Observation(
                    gpu.mem_total,
                    {
                        "worker_address": addr,
                        "gpu_index": str(key),
                        "gpu_name": gpu.name,
                    },
                )

    def _gpu_mem_free_cb(self, options):  # type: ignore[no-untyped-def]
        from opentelemetry.metrics import Observation

        for addr, info in self._workers.items():
            for key, gpu in info.items():
                if key == "cpu":
                    continue
                yield Observation(
                    gpu.mem_free,
                    {
                        "worker_address": addr,
                        "gpu_index": str(key),
                        "gpu_name": gpu.name,
                    },
                )


# Singleton instance (created by setup_otel, used by supervisor)
_cluster_metrics_collector: Optional[ClusterMetricsCollector] = None


def get_cluster_metrics_collector() -> Optional[ClusterMetricsCollector]:
    """Return the global ClusterMetricsCollector, or None if OTEL is disabled."""
    return _cluster_metrics_collector
