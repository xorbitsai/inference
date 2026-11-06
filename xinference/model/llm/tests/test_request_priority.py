# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
from concurrent.futures import Future

import pytest


@pytest.mark.parametrize("capacity", [1, 4, 16])
def test_transformers_capacity_tracks_actual_batching(capacity):
    from ..transformers.core import PytorchModel

    model = object.__new__(PytorchModel)
    model._pytorch_model_config = {"max_num_seqs": capacity}
    model._batch_scheduler = None
    assert model.get_request_priority_capacity() == 1
    model._batch_scheduler = object()
    assert model.get_request_priority_capacity() == capacity


def test_transformers_multimodal_capacity_without_batch_scheduler():
    from ..transformers.multimodal.core import PytorchMultiModalModel

    class VisionModel(PytorchMultiModalModel):
        def decide_device(self):
            pass

        def load_processor(self):
            pass

        def load_multimodal_model(self):
            pass

    model = object.__new__(VisionModel)
    model._pytorch_model_config = {"max_num_seqs": 16}
    assert not hasattr(model, "_batch_scheduler")
    assert model.get_request_priority_capacity() == 1


@pytest.mark.parametrize("vision", [False, True])
def test_mlx_capacity_tracks_batching_and_serial_vision(vision):
    from ..mlx.core import MLXChatModel, MLXVisionModel

    model = object.__new__(MLXVisionModel if vision else MLXChatModel)
    model._model_config = {"batch_size": 8}
    model._batch_model = None
    assert model.get_request_priority_capacity() == 1
    if not vision:
        model._batch_model = object()
        assert model.get_request_priority_capacity() == 8


def test_llama_cpp_capacity_uses_loaded_native_slots():
    from ..llama_cpp.core import XllamaCppModel

    model = object.__new__(XllamaCppModel)
    model._request_priority_capacity = 8
    assert model.get_request_priority_capacity() == 8


@pytest.mark.asyncio
@pytest.mark.parametrize("read", [False, True])
async def test_llama_cpp_stream_close_waits_for_native_worker(read):
    from ..llama_cpp.core import _owned_native_stream

    native = Future()
    stream = _owned_native_stream(iter(["chunk", "next"]), native)
    if read:
        assert next(stream) == "chunk"
    close = asyncio.create_task(asyncio.to_thread(stream.close))
    await asyncio.sleep(0)
    assert not close.done()
    native.set_result(None)
    await asyncio.wait_for(close, 3)
