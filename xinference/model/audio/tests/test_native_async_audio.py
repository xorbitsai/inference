# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# http://www.apache.org/licenses/LICENSE-2.0

# CI: linux

import asyncio
import sys
from contextlib import aclosing
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import numpy as np
import pytest

from ....core.realtime.protocol import RealtimeError, RealtimeLimits
from ..native_async import NativeAsyncAudioModel
from ..vllm import PCM_FORMAT, VLLMASRSession, VLLMQwen3ASRModel
from ..vllm_omni import VLLMOmniTTSModel


class Engine:
    def __init__(self):
        self.started = asyncio.Queue()
        self.release = asyncio.Event()
        self.abort_release = asyncio.Event()
        self.abort_release.set()
        self.aborted = []
        self.shutdown = lambda: None

    async def generate(self, *, request_id, **kwargs):
        await self.started.put(request_id)
        yield NS(outputs=[NS(text="hello")])
        await self.release.wait()
        yield NS(outputs=[NS(text="hello world")])

    async def abort(self, request_id):
        self.aborted.append(request_id)
        await self.abort_release.wait()


async def consume(model):
    async with aclosing(model._generate({})) as outputs:
        return [item async for item in outputs]


async def test_requests_overlap_and_cancel_isolated_until_abort_ack():
    model = NativeAsyncAudioModel("test", ".", NS())
    model._engine = engine = Engine()
    a, b = asyncio.create_task(consume(model)), asyncio.create_task(consume(model))
    first = await asyncio.wait_for(engine.started.get(), 1)
    second = await asyncio.wait_for(engine.started.get(), 1)
    assert first != second and len(model._requests) == 2
    engine.abort_release.clear()
    a.cancel()
    while not engine.aborted:
        await asyncio.sleep(0)
    a.cancel()
    await asyncio.sleep(0)
    assert not a.done() and len(model._requests) == 2
    assert not b.done() and engine.aborted == [first]
    engine.abort_release.set()
    with pytest.raises(asyncio.CancelledError):
        await a
    assert model._requests == {second} and not model._stopping
    engine.release.set()
    assert len(await b) == 2
    assert not model._requests and engine.aborted == [first]


async def test_generation_factory_failure_releases_tracking():
    model = NativeAsyncAudioModel("test", ".", NS())
    model._engine = engine = Engine()

    def fail(**kwargs):
        raise RuntimeError("cannot submit")

    engine.generate = fail
    with pytest.raises(RuntimeError, match="cannot submit"):
        await consume(model)
    assert not model._requests and len(engine.aborted) == 1


async def test_native_load_and_stop_use_actor_loop():
    model = NativeAsyncAudioModel("test", ".", NS())
    model.set_loop(asyncio.get_running_loop())

    async def load():
        model._engine = Engine()
        assert asyncio.get_running_loop() is model._loop

    model._load = load
    await asyncio.wait_for(asyncio.to_thread(model.load), 1)
    engine = model._engine
    engine.shutdown = AsyncMock()
    model._requests.add("active")
    await asyncio.wait_for(asyncio.to_thread(model.stop), 1)
    assert engine.aborted == ["active"]
    engine.shutdown.assert_awaited_once()
    assert model._engine is None


class Runtime:
    limits = RealtimeLimits()

    async def invoke(self, fn):
        return fn()


def asr_model(**options):
    model = VLLMQwen3ASRModel("asr", ".", NS(), realtime_options=options)
    # A stateful detector: nonzero PCM is speech. Real Silero is lazy/GPU-free.
    detectors = []

    def vad():
        detector = NS(calls=0)
        detectors.append(detector)

        def predict(frame, rate):
            detector.calls += 1
            return (frame.abs().max() > 0.01).float()

        return predict

    model._vad_factory = vad
    model.transcribe_audio = AsyncMock(return_value="Hello world")
    model.detectors = detectors
    return model


def config(**params):
    return {"input": {"audio": PCM_FORMAT}, "parameters": params}


def pcm(frames=1, speech=True, samples=None):
    return np.full(
        samples if samples is not None else frames * 512,
        3000 if speech else 0,
        dtype="<i2",
    ).tobytes()


async def collect(session):
    return [item async for item in session.iter_outputs()]


async def test_asr_vad_partial_endpoint_and_exact_final_tail():
    model = asr_model(partial_interval_ms=160, silence_ms=64)
    session = VLLMASRSession(model, config(), Runtime())
    reader = asyncio.create_task(collect(session))
    # Deliberately split inside a PCM sample and across VAD frame boundaries.
    audio = pcm(6) + pcm(2, False) + pcm(samples=2561)
    for offset in range(0, len(audio), 333):
        await session.receive_input(audio[offset : offset + 333])
    await session.finish_input()
    result = await asyncio.wait_for(reader, 1)
    assert any(item["partial"] == "Hello world" for item in result)
    assert result[-1]["is_final"] and result[-1]["duration_ms"] == 416
    assert len(result[-1]["sentences"]) == 2
    assert result[-1]["sentences"][0]["end"] == 256
    assert result[-1]["sentences"][1]["start"] == 256
    assert result[-1]["sentences"][1]["end"] == 416
    assert model.transcribe_audio.call_args.args[0].size == 2561
    assert model.transcribe_audio.call_args.args[1] == "en"
    await session.close("completed")


async def test_asr_noise_rejected_and_vad_state_per_connection():
    model = asr_model()
    sessions = [VLLMASRSession(model, config(), Runtime()) for _ in range(2)]
    for session in sessions:
        reader = asyncio.create_task(collect(session))
        await session.receive_input(pcm(2))  # below minimum speech duration
        await session.receive_input(pcm(20, False))
        await session.finish_input()
        assert (await reader)[-1]["text"] == ""
        await session.close("completed")
    assert len(model.detectors) == 2
    model.transcribe_audio.assert_not_called()


async def test_asr_decoder_error_reaches_writer_without_more_input():
    model = asr_model(partial_interval_ms=160)
    model.transcribe_audio.side_effect = RuntimeError("provider internal secret")
    session = VLLMASRSession(model, config(), Runtime())
    reader = asyncio.create_task(collect(session))
    try:
        await session.receive_input(pcm(5))
    except RuntimeError:
        pass  # The input processor can observe the same failure as the writer.
    with pytest.raises(RealtimeError, match="Speech recognition failed"):
        await asyncio.wait_for(reader, 1)
    await session.close("failed")


async def test_asr_cancel_waits_for_decode_and_fences_outputs():
    model = asr_model(partial_interval_ms=160)
    entered, release = asyncio.Event(), asyncio.Event()

    async def decode(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await release.wait()

    model.transcribe_audio = decode
    session = VLLMASRSession(model, config(), Runtime())
    await session.receive_input(pcm(6))
    await asyncio.wait_for(entered.wait(), 1)
    closing = asyncio.create_task(session.close("cancelled"))
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done() and session._closed and session._outputs.empty()
    release.set()
    await asyncio.wait_for(closing, 1)
    assert session._outputs.empty()


async def test_asr_odd_final_pcm_and_invalid_options():
    model = asr_model()
    session = VLLMASRSession(model, config(), Runtime())
    await session.receive_input(b"x")
    with pytest.raises(RealtimeError, match="Incomplete"):
        await session.finish_input()
    await session.close("failed")
    for options in (
        {"max_num_seqs": True},
        {"max_num_seqs": 0},
        {"realtime_options": {"silence_ms": "512"}},
    ):
        with pytest.raises(ValueError):
            VLLMQwen3ASRModel("a", ".", NS(), **options)
    with pytest.raises(RealtimeError):
        VLLMASRSession(model, config(hotwords="not a list"), Runtime())


@pytest.fixture
def native_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", NS(SamplingParams=lambda **kw: NS(**kw)))
    monkeypatch.setitem(
        sys.modules, "vllm.sampling_params", NS(RequestOutputKind=NS(DELTA="delta"))
    )


async def test_asr_native_prompt_contract(native_modules):
    model = asr_model()
    model._engine = engine = Engine()
    engine.model_config = object()
    engine.release.set()
    model._asr_class = NS(
        supported_languages={"en": "English"},
        get_generation_prompt=lambda params: {
            "multi_modal_data": {"audio": params.audio}
        },
        post_process_output=lambda text: text,
    )
    # Restore the method replaced by the session fixture.
    model.transcribe_audio = VLLMQwen3ASRModel.transcribe_audio.__get__(model)
    assert await model.transcribe_audio(np.zeros(800), "en") == "hello world"
    with pytest.raises(ValueError, match="Unsupported transcription language"):
        await model.transcribe_audio(np.zeros(800), "invalid")
    assert model.get_realtime_capabilities()["tasks"]["audio.transcription"][
        "request_batching"
    ]


def tts_model():
    model = VLLMOmniTTSModel("tts", ".", NS())
    model._config = NS(
        spk_id={"ryan": 0}, codec_language_id={"english": 1}, spk_is_dialect={}
    )
    model._tokenizer = lambda *args, **kwargs: {"input_ids": [1, 2]}
    model._estimate_prompt = lambda **kwargs: 12
    model._engine = NS(
        default_sampling_params_list=[NS(output_kind="cumulative", seed=42)]
    )
    return model


def audio_output(audio, rate=24000):
    return NS(request_output=NS(multimodal_output={"audio": audio, "sr": rate}))


async def test_tts_streams_before_completion_and_deduplicates(native_modules):
    model = tts_model()
    engine = Engine()
    engine.default_sampling_params_list = model._engine.default_sampling_params_list
    model._engine = engine
    first, second = np.array([0.25, -0.25]), np.array([0.5])
    seen = []

    async def generate(**kwargs):
        seen.append(kwargs)
        yield audio_output([first])
        await engine.release.wait()
        yield audio_output([first, second])

    engine.generate = generate
    stream = await model.async_speech("Hello", "ryan", "pcm", stream=True, seed=7)
    chunk = await asyncio.wait_for(anext(stream), 1)
    assert len(chunk) == 4 and not engine.release.is_set()
    engine.release.set()
    remaining = [item async for item in stream]
    assert list(map(len, remaining)) == [2]
    assert seen[0]["sampling_params_list"][0].seed == 7
    assert seen[0]["sampling_params_list"][0].output_kind == "delta"
    assert engine.default_sampling_params_list[0].seed == 42
    assert not model._requests


@pytest.mark.parametrize(
    "audio,rate",
    [
        (np.zeros((2, 2)), 24000),
        (np.array([np.nan]), 24000),
        (np.zeros(2), 16000),
        ([], 24000),
    ],
)
async def test_tts_rejects_invalid_audio_and_aborts(native_modules, audio, rate):
    model = tts_model()
    engine = Engine()
    engine.default_sampling_params_list = model._engine.default_sampling_params_list
    model._engine = engine

    async def generate(**kwargs):
        yield audio_output(audio, rate)
        await engine.release.wait()

    # Empty audio needs a terminal result to distinguish buffering from failure.
    if isinstance(audio, list):
        engine.release.set()
    engine.generate = generate
    with pytest.raises(RuntimeError):
        await model.async_speech("Hello", "ryan", "pcm")
    assert not model._requests
    assert len(engine.aborted) == (0 if isinstance(audio, list) else 1)


async def test_tts_validation_before_submission_and_stream_error(native_modules):
    model = tts_model()
    for kwargs in (
        {"voice": "missing"},
        {"language": "missing"},
        {"speed": 2},
        {"stream": True, "response_format": "mp3"},
        {"seed": True},
    ):
        with pytest.raises(ValueError):
            await model.async_speech("Hello", **kwargs)
    engine = Engine()
    engine.default_sampling_params_list = model._engine.default_sampling_params_list
    model._engine = engine

    async def generate(**kwargs):
        yield audio_output(np.array([0.1]))
        raise RuntimeError("generation failed")

    engine.generate = generate
    stream = await model.async_speech("Hello", response_format="pcm", stream=True)
    assert len(await anext(stream)) == 2
    with pytest.raises(RuntimeError, match="generation failed"):
        await anext(stream)
    assert len(engine.aborted) == 1 and not model._requests


async def test_actor_binary_stream_close_aborts_native_request(native_modules):
    from ....core.model import ModelActor

    model = tts_model()
    model.model_family = NS(to_description=lambda: {"model_type": "audio"})
    engine = Engine()
    engine.default_sampling_params_list = model._engine.default_sampling_params_list
    model._engine = engine

    async def generate(**kwargs):
        yield audio_output(np.array([0.25]))
        await engine.release.wait()

    engine.generate = generate
    actor = ModelActor("127.0.0.1:1", "127.0.0.1:2", model, "tts-0")
    actor.record_metrics = AsyncMock()
    stream = await actor._call_wrapper_binary(
        model.async_speech, "Hello", "ryan", "pcm", 1, True
    )
    assert len(await anext(stream)) == 2
    assert len(model._requests) == 1
    engine.abort_release.clear()
    closing = asyncio.create_task(stream.aclose())
    while not engine.aborted:
        await asyncio.sleep(0)
    assert not closing.done() and model._requests
    engine.abort_release.set()
    await asyncio.wait_for(closing, 1)
    assert not model._requests


async def test_stop_racing_client_close_shares_abort_ack():
    model = NativeAsyncAudioModel("test", ".", NS())
    model._engine = engine = Engine()
    stream = model._generate({})
    await anext(stream)
    engine.abort_release.clear()
    shutdown = []
    engine.shutdown = lambda: shutdown.append(True)
    stopping = asyncio.create_task(model._stop())
    while not engine.aborted:
        await asyncio.sleep(0)
    closing = asyncio.create_task(stream.aclose())
    await asyncio.sleep(0)
    assert len(engine.aborted) == 1 and not shutdown
    engine.abort_release.set()
    await asyncio.wait_for(asyncio.gather(stopping, closing), 1)
    assert len(engine.aborted) == 1 and shutdown == [True]
    assert not model._requests and not model._aborts


def test_loaded_audio_capabilities_come_from_adapter():
    from ....core.model import ModelActor

    model = tts_model()
    model.model_family = NS(to_description=lambda: {"model_type": "audio"})
    actor = ModelActor("127.0.0.1:1", "127.0.0.1:2", model, "tts-0")
    actor._model_state = "ready"
    caps = actor.get_audio_capabilities()
    assert caps["tts"]["streaming"] is True
    assert caps["tts"]["sample_rates"] == [24000]
    assert caps["tts"]["voices"] == [{"voice_id": "ryan", "name": "ryan"}]
    model._stopping = True
    assert actor.get_audio_capabilities() == {}


def test_existing_vllm_asr_batching_options_remain_accepted():
    model = VLLMQwen3ASRModel(
        "asr",
        ".",
        NS(),
        max_inference_batch_size=2,
        max_new_tokens=256,
        forced_aligner="unused",
        batch_interval=0.01,
    )
    assert model.realtime_max_sessions == 2 and model._max_tokens == 256
    assert (
        "forced_aligner" not in model._kwargs and "batch_interval" not in model._kwargs
    )
