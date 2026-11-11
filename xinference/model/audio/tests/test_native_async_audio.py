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
    assert model.transcribe_audio.call_args.args[1] == "English"
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
    model._samples_per_token = 1920
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


async def test_tts_streams_delta_audio_lists_before_completion(native_modules):
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
        yield audio_output([second])

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
    "parts,expected",
    [
        (["你好", "世界"], "你好世界"),
        (["最初。", "次です。"], "最初。次です。"),
        (["Hello", "world"], "Hello world"),
        (["你好！", "世界"], "你好！世界"),
        (["你好，", "世界"], "你好，世界"),
        (["最初！", "次です。"], "最初！次です。"),
        (["Hello！", "world"], "Hello！ world"),
        (["Ｈｅｌｌｏ！", "ｗｏｒｌｄ"], "Ｈｅｌｌｏ！ ｗｏｒｌｄ"),
    ],
)
async def test_file_and_realtime_join_cjk_without_extra_spaces(
    native_modules, parts, expected
):
    model, _ = native_asr([(text, "stop") for text in parts])
    result = await model.async_transcriptions(wav_bytes(30 * 16000 + 1))
    assert result["text"] == expected
    session = VLLMASRSession(model, config(), Runtime())
    session._sentences = [{"text": text} for text in parts]
    assert session._snapshot(final=True)["text"] == expected


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reason", ["length", "abort"])
async def test_tts_incomplete_termination_raises(native_modules, stream, reason):
    model = tts_model()
    model._engine.abort = AsyncMock()

    async def generate(**kwargs):
        yield audio_output(np.array([0.1]))
        yield NS(outputs=[NS(finish_reason=reason)])

    model._engine.generate = generate
    with pytest.raises(RuntimeError, match="incomplete"):
        result = await model.async_speech("Hello", response_format="pcm", stream=stream)
        if stream:
            assert await anext(result)
            await anext(result)
    assert not model._requests


@pytest.mark.parametrize("stream", [False, True])
async def test_tts_detects_talker_limit_despite_decoder_stop(native_modules, stream):
    model = tts_model()
    model._engine.abort = AsyncMock()

    async def generate(**kwargs):
        yield audio_output(np.zeros(1920))
        output = audio_output(np.zeros(1920))
        output.request_output.outputs = [NS(finish_reason="stop")]
        yield output

    model._engine.generate = generate
    with pytest.raises(RuntimeError, match="budget exhausted"):
        result = await model.async_speech(
            "Longer input", response_format="pcm", stream=stream, max_new_tokens=2
        )
        if stream:
            assert len(await anext(result)) == 3840
            await anext(result)
    assert not model._requests


async def test_tts_eos_token_at_budget_boundary_is_complete(native_modules):
    model = tts_model()

    async def generate(**kwargs):
        yield audio_output(np.zeros(1920))
        yield NS(outputs=[NS(finish_reason="stop")])

    model._engine.generate = generate
    assert (
        len(await model.async_speech("Short", response_format="pcm", max_new_tokens=2))
        == 3840
    )


@pytest.mark.parametrize("format", ["wav", "mp3", "flac", "ogg"])
async def test_tts_file_formats_do_not_require_torchaudio(
    native_modules, monkeypatch, format
):
    import io

    import soundfile as sf

    monkeypatch.setitem(sys.modules, "torchaudio", None)
    monkeypatch.setitem(sys.modules, "torchcodec", None)
    model = tts_model()

    async def generate(**kwargs):
        yield audio_output(np.zeros(24000))

    model._engine.generate = generate
    result = await model.async_speech("Hello", response_format=format)
    samples, rate = sf.read(io.BytesIO(result))
    assert rate == 24000 and len(samples) == 24000


async def test_stop_shuts_engine_down_even_when_abort_fails():
    model = NativeAsyncAudioModel("test", ".", NS())
    engine = model._engine = NS(
        abort=AsyncMock(side_effect=RuntimeError("abort failed")),
        shutdown=AsyncMock(),
    )
    model._requests.add("active")
    with pytest.raises(RuntimeError, match="abort failed"):
        await model._stop()
    engine.shutdown.assert_awaited_once()
    assert model._engine is None and model._stopping


def test_realtime_rejects_unsupported_language_before_start():
    with pytest.raises(RealtimeError) as exc:
        VLLMASRSession(asr_model(), config(language="not-a-language"), Runtime())
    assert exc.value.code == "invalid_config"


@pytest.mark.parametrize("value", [True, False])
def test_realtime_launch_flag_has_a_clear_validation_error(value):
    with pytest.raises(ValueError, match="realtime_options"):
        VLLMQwen3ASRModel("asr", ".", NS(), realtime=value)


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


async def test_file_transcription_resamples_with_declared_audio_dependencies():
    import io
    import wave

    pytest.importorskip("soundfile")
    pytest.importorskip("scipy")
    audio = io.BytesIO()
    with wave.open(audio, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(np.ones(2400, dtype="<i2").tobytes())
    model = asr_model()
    model._transcribe = AsyncMock(return_value=("Hello world", "English"))
    result = await model.async_transcriptions(
        audio.getvalue(), language="en", response_format="verbose_json"
    )
    samples = model._transcribe.call_args.args[0]
    assert samples.size == 1600 and samples.dtype == np.float32
    assert result["text"] == "Hello world" and result["duration"] == 0.1


@pytest.mark.parametrize(
    "options,expected",
    [
        ({"batch_size": 2}, 2),
        ({"max_inference_batch_size": -1}, 16),
        ({"batch_size": 2, "max_inference_batch_size": -1}, 2),
        ({"batch_size": 2, "request_batch_size": 3, "max_inference_batch_size": 4}, 2),
        ({"request_batch_size": 3, "max_inference_batch_size": 4}, 3),
        ({"batch_size": 2, "max_num_seqs": 5}, 5),
    ],
)
async def test_legacy_batch_options_do_not_reach_engine_load(
    native_modules, monkeypatch, options, expected
):
    engine_args = []

    def args(**kwargs):
        assert set(kwargs) == {"model", "max_num_seqs"}
        engine_args.append(kwargs)
        return kwargs

    sys.modules["vllm"].AsyncEngineArgs = args
    monkeypatch.setitem(sys.modules, "silero_vad", NS(load_silero_vad=lambda **_: None))
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models.qwen3_asr",
        NS(Qwen3ASRForConditionalGeneration=NS()),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.engine.async_llm",
        NS(AsyncLLM=NS(from_engine_args=lambda value: NS())),
    )
    model = VLLMQwen3ASRModel("asr", ".", NS(), **options)
    await model._load()
    assert model.realtime_max_sessions == expected
    assert engine_args == [{"model": ".", "max_num_seqs": expected}]


def native_asr(outputs):
    model = VLLMQwen3ASRModel("asr", ".", NS())
    calls = []

    async def generate(*, prompt, sampling_params, **kwargs):
        calls.append((prompt, sampling_params))
        text, reason = outputs[len(calls) - 1]
        yield NS(outputs=[NS(text=text, finish_reason=reason)])

    model._engine = NS(generate=generate, model_config=object(), abort=AsyncMock())
    model._asr_class = NS(
        get_generation_prompt=lambda params: params,
        post_process_output=lambda text: text.rsplit("<asr_text>", 1)[-1],
    )
    return model, calls


def wav_bytes(samples=1600):
    import io
    import wave

    data = io.BytesIO()
    with wave.open(data, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.zeros(samples, dtype="<i2").tobytes())
    return data.getvalue()


@pytest.mark.parametrize("language", ["Cantonese", "Filipino", "yue", "fil"])
async def test_qwen_languages_in_file_and_realtime(native_modules, language):
    model, calls = native_asr([("hello", "stop"), ("hello", "stop")])
    canonical = "Cantonese" if language in {"Cantonese", "yue"} else "Filipino"
    result = await model.async_transcriptions(
        wav_bytes(), language=language, response_format="verbose_json"
    )
    assert result["text"] == "hello" and result["language"] == canonical
    session = VLLMASRSession(model, config(language=language), Runtime())
    session._jobs.put_nowait((np.zeros(1600), 0, 1600, True))
    session._jobs.put_nowait(None)
    await session._decode_jobs()
    assert session._sentences[0]["text"] == "hello"
    assert [prompt.language for prompt, _ in calls] == [canonical, canonical]
    assert [params.max_tokens for _, params in calls] == [4096, 512]


@pytest.mark.parametrize("language", [None, "auto"])
async def test_file_preserves_detected_language_and_file_token_budget(
    native_modules, language
):
    text = "word " * 600
    model, calls = native_asr([("language English<asr_text>" + text, "stop")])
    result = await model.async_transcriptions(
        wav_bytes(), language=language, response_format="verbose_json"
    )
    assert result["language"] == "English" and result["text"] == text.strip()
    assert calls[0][1].max_tokens == 4096


async def test_file_merges_all_bounded_audio_chunks(native_modules):
    model, calls = native_asr(
        [
            ("language English<asr_text>first", "stop"),
            ("language Chinese<asr_text>second", "stop"),
            ("language English<asr_text>tail", "stop"),
        ]
    )
    result = await model.async_transcriptions(
        wav_bytes(65 * 16000), response_format="verbose_json"
    )
    assert result == {
        "text": "first second tail",
        "task": "transcribe",
        "language": "English,Chinese",
        "duration": 65,
    }
    assert [prompt.audio.size for prompt, _ in calls] == [480000, 480000, 80000]


@pytest.mark.parametrize("file", [False, True])
@pytest.mark.parametrize("reason", ["length", "abort"])
async def test_incomplete_termination_never_returns_a_complete_transcript(
    native_modules, file, reason
):
    model, _ = native_asr([("truncated prefix", reason)])
    with pytest.raises(RuntimeError, match="incomplete"):
        if file:
            await model.async_transcriptions(wav_bytes())
        else:
            await model.transcribe_audio(np.zeros(1600))


@pytest.mark.parametrize("terminal_output", [False, True])
async def test_file_shutdown_after_partial_never_returns_success(
    native_modules, terminal_output
):
    model, _ = native_asr([])
    partial, release, aborting, ack = [asyncio.Event() for _ in range(4)]

    async def generate(**kwargs):
        yield NS(outputs=[NS(text="partial prefix", finish_reason=None)])
        partial.set()
        await release.wait()
        if terminal_output:
            yield NS(outputs=[NS(text="partial prefix", finish_reason="abort")])

    async def abort(request_id):
        aborting.set()
        release.set()
        await ack.wait()

    model._engine.generate = generate
    model._engine.abort = abort
    model._engine.shutdown = lambda: None
    request = asyncio.create_task(model.async_transcriptions(wav_bytes()))
    stop = None
    try:
        await asyncio.wait_for(partial.wait(), 1)
        stop = asyncio.create_task(model._stop())
        await asyncio.wait_for(aborting.wait(), 1)
        assert not request.done() and not stop.done()
        ack.set()
        with pytest.raises(RuntimeError, match="interrupted"):
            await asyncio.wait_for(request, 1)
        await stop
    finally:
        ack.set()
        release.set()
        await asyncio.gather(
            *(task for task in (request, stop) if task), return_exceptions=True
        )


def test_tts_discovery_distinguishes_stream_and_file_formats():
    capabilities = tts_model().get_speech_capabilities()
    assert capabilities["formats"] == ["pcm"]
    assert set(capabilities["file_formats"]) == {"pcm", "wav", "mp3", "flac", "ogg"}
