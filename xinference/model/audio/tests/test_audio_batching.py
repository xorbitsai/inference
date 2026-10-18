# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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
import threading
import wave
from contextlib import suppress
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from ..batch import AudioRequestBatcher
from ..mlx_audio import MLXAudioTTSModel
from ..qwen3_tts import Qwen3TTSModel
from ..whisper import WhisperModel


async def _shutdown(model):
    task = model._batcher._process_batch_task
    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def _spec(name="Qwen3-TTS-CustomVoice", family="qwen3_tts"):
    return SimpleNamespace(
        model_name=name,
        model_family=family,
        multilingual=True,
        to_description=lambda: {"model_type": "audio", "model_name": name},
    )


class _QwenBackend:
    def __init__(self):
        self.calls = []

    def get_supported_languages(self):
        return ["chinese", "English"]

    def get_supported_speakers(self):
        return ["Vivian", "Ryan"]

    def generate_custom_voice(self, **kwargs):
        self.calls.append(kwargs)
        texts = kwargs["text"]
        if isinstance(texts, str):
            texts = [texts]
        if "ref_audio" in kwargs:
            assert Path(kwargs["ref_audio"]).read_bytes() == b"reference"
        return [np.full(24, int(text) / 10, dtype=np.float32) for text in texts], 24000

    generate_voice_clone = generate_custom_voice
    generate_voice_design = generate_custom_voice


def _qwen(name="Qwen3-TTS-CustomVoice", **kwargs):
    model = Qwen3TTSModel("tts", "/unused", _spec(name), **kwargs)
    model._model = _QwenBackend()
    return model


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["CustomVoice", "VoiceDesign", "Base"])
async def test_qwen_batches_native_variants_and_encodes_each_result(variant):
    model = _qwen("Qwen3-TTS-" + variant)
    kwargs = (
        dict(prompt_speech=b"reference", prompt_text="hello")
        if variant == "Base"
        else {}
    )
    try:
        outputs = await asyncio.gather(
            model.async_speech("1", "Vivian", response_format="wav", **kwargs),
            model.async_speech(
                "2", "Ryan", response_format="wav", language="English", **kwargs
            ),
        )
        assert len(model._model.calls) == 1
        call = model._model.calls[0]
        assert call["text"] == ["1", "2"]
        assert call["language"] == ["chinese", "English"]
        if variant == "CustomVoice":
            assert call["speaker"] == ["Vivian", "Ryan"]
        if variant == "Base":
            assert not Path(call["ref_audio"]).exists()
        values = []
        for output in outputs:
            with wave.open(BytesIO(output), "rb") as wav:
                assert wav.getframerate() == 24000
                assert wav.getnframes() == 24
                values.append(np.frombuffer(wav.readframes(24), dtype="<i2")[0])
        assert values[1] > values[0] > 0
    finally:
        await _shutdown(model)


@pytest.mark.asyncio
async def test_qwen_groups_sampling_settings_seeds_and_batch_limit():
    model = _qwen("Qwen3-TTS-VoiceDesign", request_batch_size=4)
    try:
        await asyncio.gather(
            model.async_speech("1", "Vivian", response_format="wav", top_k=10),
            model.async_speech("2", "Ryan", response_format="wav", top_k=20),
            model.async_speech("3", "Ryan", response_format="wav", top_k=10),
            model.async_speech("4", "Ryan", response_format="wav", seed=42),
            model.async_speech("5", "Ryan", response_format="wav", seed=42),
        )
        assert [call["text"] for call in model._model.calls] == [
            ["1", "3"],
            "2",
            "4",
            "5",
        ]
        assert [call.get("top_k") for call in model._model.calls] == [
            10,
            20,
            None,
            None,
        ]
        assert all("seed" not in call for call in model._model.calls)
    finally:
        await _shutdown(model)


class _WhisperBackend:
    def __init__(self):
        self.calls = []

    def __call__(self, audio, **kwargs):
        self.calls.append((audio, kwargs))

        def result(data):
            return {
                "text": data.decode(),
                "chunks": [{"text": data.decode(), "timestamp": (0, 1)}],
            }

        return (
            [result(data) for data in audio]
            if isinstance(audio, list)
            else result(audio)
        )


@pytest.mark.asyncio
async def test_whisper_groups_tasks_languages_and_timestamps():
    model = WhisperModel("asr", "/unused", _spec("whisper", "whisper"), batch_size=16)
    model._model = _WhisperBackend()
    try:
        results = await asyncio.gather(
            model.async_transcriptions(b"first", language="en"),
            model.async_translations(b"translated", language="en"),
            model.async_transcriptions(b"second", language="en"),
            model.async_transcriptions(
                b"words",
                language="en",
                response_format="verbose_json",
                timestamp_granularities=["word"],
            ),
            model.async_transcriptions(
                b"segments", language="zh", response_format="verbose_json"
            ),
        )
        assert [result["text"] for result in results] == [
            "first",
            "translated",
            "second",
            "words",
            "segments",
        ]
        assert model._model.calls[0][0] == [b"first", b"second"]
        assert model._model.calls[1][1]["generate_kwargs"]["task"] == "translate"
        assert results[3]["words"] == [{"word": "words", "start": 0, "end": 1}]
        assert results[4]["segments"][0]["end"] == 1
        assert results[4]["language"] == "zh"
        assert model._model_config["batch_size"] == 16
        assert len(model._model.calls) == 4
    finally:
        await _shutdown(model)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["transcriptions", "translations", "speech"])
async def test_audio_actor_routes_concurrent_requests_to_async_batch_methods(method):
    from ....core.model import ModelActor

    if method == "speech":
        model = _qwen()
        inputs = [("1", "Vivian"), ("2", "Ryan")]
        kwargs = {"response_format": "wav"}
    else:
        model = WhisperModel("asr", "/unused", _spec("whisper", "whisper"))
        model._model = _WhisperBackend()
        inputs = [(b"a",), (b"b",)]
        kwargs = {}
    actor = ModelActor("test:123", "test:345", model, "asr-0")
    actor._model_state = "ready"
    actor.record_metrics = AsyncMock()
    assert actor._lock is None
    try:
        await asyncio.gather(
            *[getattr(actor, method)(*args, **kwargs) for args in inputs]
        )
        assert len(model._model.calls) == 1
        if method == "speech":
            assert model._model.calls[0]["text"] == ["1", "2"]
        else:
            assert model._model.calls[0][0] == [b"a", b"b"]
    finally:
        await _shutdown(model)


class _MLXBackend:
    def __init__(self):
        self.calls = []
        self.threads = []

    def batch_generate(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        self.threads.append(threading.get_ident())
        if "ref_audio" in kwargs:
            assert Path(kwargs["ref_audio"]).read_bytes() == b"reference"
        for index in reversed(range(len(texts))):
            yield SimpleNamespace(
                sequence_idx=index,
                audio=np.full(100, 0.1 * (index + 1), dtype=np.float32),
                sample_rate=1000,
            )

    def generate(self, **kwargs):
        self.calls.append(([kwargs["text"]], kwargs))
        self.threads.append(threading.get_ident())
        yield SimpleNamespace(
            audio=np.full(100, 0.1, dtype=np.float32), sample_rate=1000
        )


def _mlx(**kwargs):
    model = MLXAudioTTSModel("mlx", "/unused", _spec(), **kwargs)
    model._model = _MLXBackend()
    return model


@pytest.mark.asyncio
async def test_mlx_batches_requests_and_segments_on_dedicated_thread():
    model = _mlx(request_batch_size=2)
    load_thread = model._run_on_mlx_thread(threading.get_ident)
    try:
        outputs = await asyncio.gather(
            model.async_speech(
                "第一句。第二句！", "Vivian", response_format="wav", instruct="calm"
            ),
            model.async_speech(
                "第三句。", "Ryan", response_format="pcm", instruct="happy"
            ),
        )
        assert [call[0] for call in model._model.calls] == [
            ["第一句。", "第二句！"],
            ["第三句。"],
        ]
        assert model._model.calls[0][1]["voices"] == ["Vivian", "Vivian"]
        assert model._model.calls[1][1]["instructs"] == ["happy"]
        with wave.open(BytesIO(outputs[0]), "rb") as wav:
            assert wav.getnframes() == 480
            audio = np.frombuffer(wav.readframes(480), dtype="<i2")
            assert audio[50] < audio[-50]
        assert len(outputs[1]) == 200
        assert set(model._model.threads) == {load_thread}
    finally:
        await _shutdown(model)


def test_mlx_batches_long_text_with_bounded_native_batches():
    model = _mlx(request_batch_size=2)
    result = model.speech("一。二。三。", "Vivian", response_format="pcm")
    assert [len(call[0]) for call in model._model.calls] == [2, 1]
    assert len(result) == (300 + 560) * 2


@pytest.mark.asyncio
async def test_mlx_groups_reference_language_and_sampling_settings():
    model = _mlx()
    try:
        await asyncio.gather(
            *[
                model.async_speech(
                    str(i),
                    "Vivian",
                    response_format="wav",
                    prompt_speech=b"reference",
                    prompt_text="ref",
                    language=language,
                    top_k=top_k,
                )
                for i, (language, top_k) in enumerate(
                    [("en", 10), ("zh", 10), ("en", 10), ("en", 20)]
                )
            ]
        )
        assert [call[0] for call in model._model.calls] == [["0", "2"], ["1"], ["3"]]
        assert all(
            not Path(call[1]["ref_audio"]).exists() for call in model._model.calls
        )
    finally:
        await _shutdown(model)


@pytest.mark.asyncio
async def test_batch_cancellation_failures_and_recovery():
    calls = []

    def run_batch(requests):
        values = [request["value"] for request in requests]
        calls.append(values)
        if "fail" in values:
            raise ValueError("native failure")
        if "short" in values:
            return []
        return values

    batcher = AudioRequestBatcher(
        run_batch, lambda request: request["group"], batch_interval=0.01
    )
    model = SimpleNamespace(_batcher=batcher)
    try:
        cancelled = asyncio.create_task(
            batcher.submit(dict(value="cancelled", group=0))
        )
        kept = asyncio.create_task(batcher.submit(dict(value="kept", group=0)))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert await kept == "kept"
        assert calls == [["kept"]]
        results = await asyncio.gather(
            batcher.submit(dict(value="fail", group=0)),
            batcher.submit(dict(value="ok", group=1)),
            return_exceptions=True,
        )
        assert isinstance(results[0], ValueError)
        assert results[1] == "ok"
        with pytest.raises(RuntimeError, match="result count"):
            await batcher.submit(dict(value="short", group=0))
        assert await batcher.submit(dict(value="after", group=0)) == "after"
    finally:
        await _shutdown(model)


@pytest.mark.asyncio
async def test_mlx_seeded_and_custom_split_requests_use_single_path(monkeypatch):
    from .. import utils

    seeds = []
    monkeypatch.setattr(
        utils,
        "apply_mlx_audio_seed",
        lambda kwargs: seeds.append(kwargs.pop("seed", None)),
    )
    model = _mlx()
    try:
        await asyncio.gather(
            model.async_speech("first", "Vivian", response_format="wav", seed=42),
            model.async_speech("second", "Vivian", response_format="wav", seed=42),
            model.async_speech(
                "third", "Ryan", response_format="wav", split_pattern=None
            ),
        )
        assert [call[0] for call in model._model.calls] == [
            ["first"],
            ["second"],
            ["third"],
        ]
        assert seeds == [42, 42, None]
        assert all("texts" not in call[1] for call in model._model.calls)
    finally:
        await _shutdown(model)


@pytest.mark.asyncio
async def test_request_batch_size_one_disables_merging():
    model = _qwen(request_batch_size=1, batch_interval=0)
    try:
        await asyncio.gather(
            *[
                model.async_speech(str(i), "Vivian", response_format="wav")
                for i in range(3)
            ]
        )
        assert [call["text"] for call in model._model.calls] == ["0", "1", "2"]
        assert model._batcher.batch_interval == 0
    finally:
        await _shutdown(model)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "indices, sample_rates, message",
    [
        ([-1, 1], [1000, 1000], "invalid batch sequence"),
        ([0, 0], [1000, 1000], "duplicate batch sequence"),
        ([0], [1000], "result count"),
        ([0, 1], [1000, 2000], "inconsistent sample rates"),
    ],
)
async def test_mlx_rejects_invalid_native_batch_mapping(indices, sample_rates, message):
    model = _mlx()

    def invalid_batch(**kwargs):
        for index, sample_rate in zip(indices, sample_rates):
            yield SimpleNamespace(
                sequence_idx=index,
                audio=np.ones(100, dtype=np.float32),
                sample_rate=sample_rate,
            )

    model._model.batch_generate = invalid_batch
    try:
        results = await asyncio.gather(
            model.async_speech("first", "Vivian", response_format="wav"),
            model.async_speech("second", "Ryan", response_format="wav"),
            return_exceptions=True,
        )
        assert all(isinstance(result, Exception) for result in results)
        assert all(message in str(result) for result in results)
        assert await model.async_speech("after", "Vivian", response_format="wav")
    finally:
        await _shutdown(model)


@pytest.mark.asyncio
async def test_audio_native_batches_do_not_overlap_after_waiter_cancellation():
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def run_batch(requests):
        values = [request["value"] for request in requests]
        calls.append(values)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        return values

    batcher = AudioRequestBatcher(run_batch, lambda request: 0, request_batch_size=2)
    model = SimpleNamespace(_batcher=batcher)
    try:
        cancelled = asyncio.create_task(batcher.submit(dict(value="cancelled")))
        kept = asyncio.create_task(batcher.submit(dict(value="kept")))
        assert await asyncio.to_thread(entered.wait, 5)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        later = asyncio.create_task(batcher.submit(dict(value="later")))
        await asyncio.sleep(0)
        assert len(calls) == 1
        release.set()
        assert await kept == "kept"
        assert await later == "later"
        assert calls == [["cancelled", "kept"], ["later"]]
    finally:
        release.set()
        await _shutdown(model)


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [_qwen, _mlx])
async def test_audio_batcher_survives_model_serialization(factory):
    import cloudpickle

    model = cloudpickle.loads(cloudpickle.dumps(factory()))
    try:
        outputs = await asyncio.gather(
            model.async_speech("1", "Vivian", response_format="wav"),
            model.async_speech("2", "Ryan", response_format="wav"),
        )
        assert all(output.startswith(b"RIFF") for output in outputs)
        assert len(model._model.calls) == 1
    finally:
        await _shutdown(model)
