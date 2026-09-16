# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import asyncio

import pytest

from ..rpc_context import (
    RPC_METADATA_KEY,
    RpcMetadata,
    actor_call,
    correlate_model_ref,
    get_current_rpc_metadata,
    rpc_context,
)


def test_rpc_metadata_round_trip_and_rejects_malformed_payloads():
    metadata = RpcMetadata(
        correlation_id="http-id",
        operation_request_id="operation-id",
        actor_call_id="call-id",
        parent_call_id="parent-id",
    )
    assert RpcMetadata.from_payload(metadata.to_payload()) == metadata
    assert (
        RpcMetadata.from_payload({"version": 99, "correlation_id": "http-id"}) is None
    )
    assert (
        RpcMetadata.from_payload({"version": 1, "correlation_id": "bad\nvalue"}) is None
    )


@pytest.mark.asyncio
async def test_rpc_context_is_isolated_and_internal_metadata_is_consumed():
    seen = []

    @rpc_context
    async def operation(value, **kwargs):
        metadata = get_current_rpc_metadata()
        await asyncio.sleep(0)
        seen.append((value, metadata.correlation_id, kwargs))

    await asyncio.gather(
        operation(
            "a",
            **{
                RPC_METADATA_KEY: {
                    "version": 1,
                    "correlation_id": "correlation-a",
                    "actor_call_id": "call-a",
                }
            },
        ),
        operation(
            "b",
            **{
                RPC_METADATA_KEY: {
                    "version": 1,
                    "correlation_id": "correlation-b",
                    "actor_call_id": "call-b",
                }
            },
        ),
    )

    assert sorted(seen) == [
        ("a", "correlation-a", {}),
        ("b", "correlation-b", {}),
    ]
    assert get_current_rpc_metadata() is None


@pytest.mark.asyncio
async def test_actor_call_creates_parent_child_metadata_without_changing_request_id():
    calls = []

    class Child:
        async def infer(self, **kwargs):
            calls.append(kwargs)
            return "ok"

    @rpc_context
    async def parent(**kwargs):
        return await actor_call(Child(), "infer", request_id="operation-id")

    result = await parent(
        **{
            RPC_METADATA_KEY: {
                "version": 1,
                "correlation_id": "http-id",
                "actor_call_id": "parent-call",
            }
        }
    )

    assert result == "ok"
    assert calls[0]["request_id"] == "operation-id"
    metadata = calls[0][RPC_METADATA_KEY]
    assert metadata["correlation_id"] == "http-id"
    assert metadata["operation_request_id"] == "operation-id"
    assert metadata["parent_call_id"] == "parent-call"
    assert metadata["actor_call_id"] != "parent-call"


@pytest.mark.asyncio
async def test_stream_iteration_restores_rpc_context_and_closes_iterator():
    observations = []
    closed = False

    async def source():
        nonlocal closed
        try:
            observations.append(get_current_rpc_metadata().correlation_id)
            yield b"one"
            observations.append(get_current_rpc_metadata().correlation_id)
            yield b"two"
        finally:
            closed = True

    @rpc_context
    async def stream(**kwargs):
        return source()

    iterator = await stream(
        **{
            RPC_METADATA_KEY: {
                "version": 1,
                "correlation_id": "stream-id",
                "actor_call_id": "stream-call",
            }
        }
    )
    assert get_current_rpc_metadata() is None
    assert [chunk async for chunk in iterator] == [b"one", b"two"]
    assert observations == ["stream-id", "stream-id"]
    assert closed is True
    assert get_current_rpc_metadata() is None


@pytest.mark.asyncio
async def test_direct_rpc_without_metadata_preserves_behavior():
    received = []

    @rpc_context
    async def operation(value, **kwargs):
        received.append((value, kwargs, get_current_rpc_metadata()))
        return value

    assert await operation("ok") == "ok"
    assert received == [("ok", {}, None)]


@pytest.mark.asyncio
async def test_model_proxy_extracts_operation_id_from_generation_config():
    calls = []

    class Model:
        async def chat(self, messages, config, **kwargs):
            calls.append((messages, config, kwargs))
            return "ok"

    model = correlate_model_ref(Model(), "http-id")
    config = {"request_id": "operation-id", "stream": True}
    assert await model.chat([], config) == "ok"
    assert config == {"request_id": "operation-id", "stream": True}
    metadata = calls[0][2][RPC_METADATA_KEY]
    assert metadata["correlation_id"] == "http-id"
    assert metadata["operation_request_id"] == "operation-id"


@pytest.mark.asyncio
async def test_correlated_model_ref_only_wraps_inference_methods():
    calls = []

    class Model:
        uid = "model-uid"

        async def chat(self, **kwargs):
            calls.append(kwargs)
            return "ok"

        async def decrease_serve_count(self, **kwargs):
            calls.append(kwargs)

    raw = Model()
    model = correlate_model_ref(raw, "http-id")
    assert model.uid == "model-uid"
    assert await model.chat(request_id="operation-id") == "ok"
    await model.decrease_serve_count()

    assert calls[0]["request_id"] == "operation-id"
    assert calls[0][RPC_METADATA_KEY]["correlation_id"] == "http-id"
    assert calls[0][RPC_METADATA_KEY]["operation_request_id"] == "operation-id"
    assert calls[1] == {}


@pytest.mark.asyncio
async def test_malformed_metadata_is_removed_before_actor_method_runs():
    received = []

    @rpc_context
    async def operation(**kwargs):
        received.append((kwargs, get_current_rpc_metadata()))

    await operation(
        **{
            RPC_METADATA_KEY: {
                "version": 1,
                "correlation_id": "invalid\ncorrelation",
            }
        }
    )

    assert received == [({}, None)]


@pytest.mark.asyncio
async def test_correlated_model_abort_preserves_business_request_id():
    calls = []

    class Model:
        async def abort_request(self, request_id, block_duration=30, **kwargs):
            calls.append((request_id, block_duration, kwargs))
            return "DONE"

    model = correlate_model_ref(Model(), "http-id")
    assert await model.abort_request("operation-id", 5) == "DONE"

    request_id, block_duration, kwargs = calls[0]
    assert request_id == "operation-id"
    assert block_duration == 5
    metadata = kwargs[RPC_METADATA_KEY]
    assert metadata["correlation_id"] == "http-id"
    assert metadata["operation_request_id"] == "operation-id"


@pytest.mark.asyncio
async def test_supervisor_abort_propagates_parent_child_metadata():
    from ..supervisor import SupervisorActor

    calls = []

    class Model:
        @rpc_context
        async def abort_request(self, request_id, block_duration=30, **kwargs):
            metadata = get_current_rpc_metadata()
            calls.append(("model", request_id, block_duration, metadata))
            return "DONE"

    class Worker:
        @rpc_context
        async def get_model(self, model_uid, **kwargs):
            metadata = get_current_rpc_metadata()
            calls.append(("worker", model_uid, metadata))
            return Model()

    class Supervisor:
        _pd_model_mapping = {}
        _model_uid_to_replica_info = {"model": object()}
        _replica_model_uid_to_worker = {"model-1-0": Worker()}

        @staticmethod
        def _iter_active_replica_model_uids(model_uid):
            assert model_uid == "model"
            return iter(["model-1-0"])

    result = await SupervisorActor.abort_request(
        Supervisor(),
        "model",
        "operation-id",
        5,
        **{
            RPC_METADATA_KEY: {
                "version": 1,
                "correlation_id": "http-id",
                "operation_request_id": "operation-id",
                "actor_call_id": "supervisor-call",
            }
        },
    )

    assert result == {"msg": "DONE"}
    worker_metadata = calls[0][2]
    model_metadata = calls[1][3]
    assert calls[0][:2] == ("worker", "model-1-0")
    assert calls[1][:3] == ("model", "operation-id", 5)
    assert worker_metadata.correlation_id == "http-id"
    assert worker_metadata.operation_request_id == "operation-id"
    assert worker_metadata.parent_call_id == "supervisor-call"
    assert model_metadata.correlation_id == "http-id"
    assert model_metadata.operation_request_id == "operation-id"
    # Supervisor obtains the model ref from Worker and then calls ModelActor
    # directly, so both calls are siblings under the Supervisor call.
    assert model_metadata.parent_call_id == "supervisor-call"
    assert model_metadata.actor_call_id != worker_metadata.actor_call_id


@pytest.mark.asyncio
async def test_supervisor_pd_abort_propagates_metadata_without_changing_request_id():
    from ..supervisor import SupervisorActor

    calls = []

    class PDModel:
        @rpc_context
        async def abort_request(self, request_id, block_duration=30, **kwargs):
            calls.append(
                (request_id, block_duration, get_current_rpc_metadata(), kwargs)
            )
            return "DONE"

    class Supervisor:
        _pd_model_mapping = {"model": PDModel()}

    result = await SupervisorActor.abort_request(
        Supervisor(),
        "model",
        "operation-id",
        5,
        **{
            RPC_METADATA_KEY: {
                "version": 1,
                "correlation_id": "http-id",
                "actor_call_id": "supervisor-call",
            }
        },
    )

    assert result == {"msg": "DONE"}
    request_id, block_duration, metadata, kwargs = calls[0]
    assert (request_id, block_duration, kwargs) == ("operation-id", 5, {})
    assert metadata.correlation_id == "http-id"
    assert metadata.operation_request_id == "operation-id"
    assert metadata.parent_call_id == "supervisor-call"


@pytest.mark.asyncio
async def test_progress_query_propagates_metadata_without_changing_request_id():
    from ..supervisor import SupervisorActor

    calls = []

    class ProgressTracker:
        @rpc_context
        async def get_progress(self, request_id, **kwargs):
            calls.append((request_id, get_current_rpc_metadata(), kwargs))
            return 0.75

    class Supervisor:
        _progress_tracker = ProgressTracker()

    result = await SupervisorActor.get_progress(
        Supervisor(),
        "operation-id",
        **{
            RPC_METADATA_KEY: {
                "version": 1,
                "correlation_id": "http-id",
                "actor_call_id": "supervisor-call",
            }
        },
    )

    assert result == 0.75
    request_id, metadata, kwargs = calls[0]
    assert request_id == "operation-id"
    assert kwargs == {}
    assert metadata.correlation_id == "http-id"
    assert metadata.operation_request_id == "operation-id"
    assert metadata.parent_call_id == "supervisor-call"


@pytest.mark.asyncio
async def test_progressor_keeps_captured_context_for_threadsafe_updates():
    from ..progress_tracker import Progressor

    calls = []

    class ProgressTracker:
        async def start(self, request_id, **kwargs):
            calls.append(("start", request_id, kwargs[RPC_METADATA_KEY]))

        async def set_progress(self, request_id, progress, info, details, **kwargs):
            calls.append(
                ("set_progress", request_id, progress, kwargs[RPC_METADATA_KEY])
            )

    @rpc_context
    async def create_progressor(**kwargs):
        progressor = Progressor(
            "operation-id",
            ProgressTracker(),
            asyncio.get_running_loop(),
            upload_span=0,
        )
        await progressor.start()
        return progressor

    progressor = await create_progressor(
        **{
            RPC_METADATA_KEY: {
                "version": 1,
                "correlation_id": "http-id",
                "actor_call_id": "model-call",
            }
        }
    )
    assert get_current_rpc_metadata() is None
    progressor.set_progress(0.5)
    await asyncio.sleep(0.05)

    assert calls[0][0:2] == ("start", "operation-id")
    assert calls[1][0:3] == ("set_progress", "operation-id", 0.5)
    for metadata in (calls[0][2], calls[1][3]):
        assert metadata["correlation_id"] == "http-id"
        assert metadata["operation_request_id"] == "operation-id"
        assert metadata["parent_call_id"] == "model-call"


@pytest.mark.asyncio
async def test_actor_call_does_not_consume_ordinary_model_metadata_kwargs():
    calls = []

    class Model:
        async def infer(self, **kwargs):
            calls.append(kwargs)
            return "ok"

    assert (
        await actor_call(
            Model(),
            "infer",
            correlation_id="backend-correlation",
            operation_request_id="backend-operation",
            _rpc_correlation_id="http-id",
        )
        == "ok"
    )
    assert calls[0]["correlation_id"] == "backend-correlation"
    assert calls[0]["operation_request_id"] == "backend-operation"
    assert calls[0][RPC_METADATA_KEY]["correlation_id"] == "http-id"
