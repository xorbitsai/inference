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

# CI: linux

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


def _whisper(**kwargs):
    model = WhisperModel("asr", "/unused", _spec("whisper", "whisper"), **kwargs)
    model._model = _WhisperBackend()
    # Ordinary grouping tests use text tokens rather than encoded audio;
    # decoder contract coverage below exercises real row/error mapping.
    model._decode_audio = lambda audio: audio
    return model


@pytest.mark.asyncio
async def test_whisper_groups_tasks_languages_and_timestamps():
    model = _whisper(batch_size=16)
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
@pytest.mark.parametrize("method", ["async_transcriptions", "async_translations"])
@pytest.mark.parametrize("layout", ["mixed", "one_valid", "all_invalid"])
async def test_whisper_decode_failures_are_local_to_each_upload(
    monkeypatch, method, layout
):
    from transformers.pipelines import audio_utils

    def wav_bytes(value):
        with BytesIO() as out:
            with wave.open(out, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(np.full(32, value * 1000, dtype="<i2").tobytes())
            return out.getvalue()

    def decode(payload, sampling_rate):
        assert sampling_rate == 16000
        try:
            with wave.open(BytesIO(payload), "rb") as wav:
                frames = wav.readframes(wav.getnframes())
            if not frames:
                raise ValueError("malformed audio")
            return np.frombuffer(frames, "<i2").astype(np.float32) / 32768
        except (wave.Error, EOFError) as error:
            raise ValueError("malformed audio") from error

    monkeypatch.setattr(audio_utils, "ffmpeg_read", decode)

    class Pipeline:
        feature_extractor = SimpleNamespace(sampling_rate=16000)

        def __init__(self):
            self.calls = []

        def __call__(self, audio, **kwargs):
            self.calls.append(audio)

            def result(item):
                # Like Transformers, materializing raw-byte list inputs can
                # fail the entire invocation during preprocessing.
                if isinstance(item, bytes):
                    item = audio_utils.ffmpeg_read(item, 16000)
                return {"text": str(round(float(item[0]) * 32768 / 1000))}

            return (
                [result(a) for a in audio] if isinstance(audio, list) else result(audio)
            )

    payloads, expected = {
        "mixed": ([wav_bytes(1), b"", wav_bytes(3)], ["1", None, "3"]),
        "one_valid": ([b"corrupt audio", wav_bytes(3)], [None, "3"]),
        "all_invalid": ([b"", b"corrupt audio"], [None, None]),
    }[layout]
    model = WhisperModel("asr", "/unused", _spec("whisper", "whisper"))
    model._model = Pipeline()
    try:
        results = await asyncio.gather(
            *[getattr(model, method)(audio, language="en") for audio in payloads],
            return_exceptions=True,
        )
        for result, text in zip(results, expected):
            if text is None:
                assert isinstance(result, ValueError)
            else:
                assert result == {"text": text}
        if layout == "all_invalid":
            assert not model._model.calls
        else:
            assert len(model._model.calls) == 1
            inputs = model._model.calls[0]
            arrays = inputs if isinstance(inputs, list) else [inputs]
            assert all(isinstance(array, np.ndarray) for array in arrays)
            assert await getattr(model, method)(wav_bytes(1), language="en") == {
                "text": "1"
            }
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
        model = _whisper()
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
    def __init__(self, variant="CustomVoice"):
        self.calls = []
        self.threads = []
        self.variant = variant

    def get_supported_speakers(self):
        return ["Vivian", "Ryan", "Chelsie"]

    def _validate_scalar(self, voice, instruct):
        if self.variant == "CustomVoice":
            if not voice:
                raise ValueError("CustomVoice model requires 'voice'")
            if voice.lower() not in [s.lower() for s in self.get_supported_speakers()]:
                raise ValueError(f"Speaker '{voice}' not supported")
        elif self.variant == "VoiceDesign" and not instruct:
            raise ValueError("VoiceDesign model requires 'instruct'")

    def batch_generate(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        self.threads.append(threading.get_ident())
        if "ref_audio" in kwargs:
            assert Path(kwargs["ref_audio"]).read_bytes() == b"reference"
        if kwargs.get("ref_audio") is not None and kwargs.get("ref_text") is not None:
            # Match the mlx-audio 0.4.6 batch ICL contract. The scalar ICL
            # dispatch ignores these fields, while batch_generate rejects them.
            if any(voice is not None for voice in kwargs.get("voices") or []):
                raise ValueError(
                    "Qwen3-TTS batch reference cloning does not support voices"
                )
            if any(instruct is not None for instruct in kwargs.get("instructs") or []):
                raise ValueError(
                    "Qwen3-TTS batch reference cloning does not support instructs"
                )
        for index in reversed(range(len(texts))):
            yield SimpleNamespace(
                sequence_idx=index,
                audio=np.full(100, 0.1 * (index + 1), dtype=np.float32),
                sample_rate=1000,
            )

    def generate(self, **kwargs):
        # Match the pinned scalar dispatcher. batch_generate intentionally
        # lacks these guards, as in mlx-audio 0.4.6.
        self._validate_scalar(kwargs.get("voice"), kwargs.get("instruct"))
        self.calls.append(([kwargs["text"]], kwargs))
        self.threads.append(threading.get_ident())
        yield SimpleNamespace(
            audio=np.full(100, 0.1, dtype=np.float32), sample_rate=1000
        )


def _mlx(name="Qwen3-TTS-CustomVoice", **kwargs):
    model = MLXAudioTTSModel("mlx", "/unused", _spec(name), **kwargs)
    model._model = _MLXBackend(name.rsplit("-", 1)[-1])
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
    model = _mlx("Qwen3-TTS-12Hz-0.6B-Base")
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
@pytest.mark.parametrize("size", ["0.6B", "1.7B"])
@pytest.mark.parametrize("path", ["sync_segments", "async_segments", "requests"])
async def test_mlx_base_clone_batches_ignore_voice_and_instruct(size, path):
    model = _mlx(f"Qwen3-TTS-12Hz-{size}-Base")
    kwargs = dict(
        response_format="wav",
        prompt_speech=b"reference",
        prompt_text="reference text",
        instruct="ignored in scalar Base ICL",
        language="English",
    )
    try:
        if path == "requests":
            outputs = await asyncio.gather(
                model.async_speech("第一句。", "Chelsie", **kwargs),
                model.async_speech("第二句！", "Vivian", **kwargs),
            )
            expected_frames = [100, 100]
        else:
            if path == "sync_segments":
                output = model.speech("第一句。第二句！", "Chelsie", **kwargs)
            else:
                output = await model.async_speech(
                    "第一句。第二句！", "Chelsie", **kwargs
                )
            outputs = [output]
            expected_frames = [480]
        assert len(model._model.calls) == 1
        texts, native_kwargs = model._model.calls[0]
        assert texts == ["第一句。", "第二句！"]
        assert native_kwargs["voices"] == [None, None]
        assert native_kwargs["instructs"] == [None, None]
        assert native_kwargs["ref_text"] == "reference text"
        assert native_kwargs["lang_code"] == "English"
        assert not Path(native_kwargs["ref_audio"]).exists()
        for output, frames in zip(outputs, expected_frames):
            with wave.open(BytesIO(output), "rb") as wav:
                assert wav.getframerate() == 1000
                assert wav.getnframes() == frames
    finally:
        await _shutdown(model)


@pytest.mark.asyncio
async def test_mlx_voice_design_batch_preserves_row_instructions():
    model = _mlx("Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    try:
        await asyncio.gather(
            model.async_speech("第一句。", "", response_format="wav", instruct="calm"),
            model.async_speech("第二句！", "", response_format="wav", instruct="happy"),
        )
        assert len(model._model.calls) == 1
        assert model._model.calls[0][1]["voices"] == [None, None]
        assert model._model.calls[0][1]["instructs"] == ["calm", "happy"]
    finally:
        await _shutdown(model)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["merged", "segments", "scalar"])
@pytest.mark.parametrize(
    "variant, voice, instruct",
    [
        ("CustomVoice", None, None),
        ("CustomVoice", "", None),
        ("CustomVoice", "unknown", None),
        ("VoiceDesign", "", None),
        ("VoiceDesign", "", ""),
    ],
)
async def test_mlx_batching_preserves_scalar_variant_validation(
    path, variant, voice, instruct
):
    model = _mlx(f"Qwen3-TTS-12Hz-1.7B-{variant}")
    kwargs = dict(response_format="wav", instruct=instruct)
    try:
        if path == "merged":
            valid_voice = "vIvIaN" if variant == "CustomVoice" else ""
            results = await asyncio.gather(
                model.async_speech(
                    "first", valid_voice, response_format="wav", instruct="calm"
                ),
                model.async_speech("bad", voice, **kwargs),
                model.async_speech(
                    "last", valid_voice, response_format="wav", instruct="happy"
                ),
                return_exceptions=True,
            )
            assert isinstance(results[1], ValueError)
            assert all(isinstance(results[i], bytes) for i in (0, 2))
            assert len(model._model.calls) == 1
            assert model._model.calls[0][0] == ["first", "last"]
        else:
            text = "一。二。" if path == "segments" else "short"
            with pytest.raises(ValueError):
                model.speech(text, voice, **kwargs)
            assert not model._model.calls
    finally:
        await _shutdown(model)


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [_qwen, _mlx], ids=["pytorch", "mlx"])
@pytest.mark.parametrize("bad_format", [None, "not-a-format"])
async def test_audio_batch_encoding_failure_does_not_fail_other_requests(
    factory, bad_format
):
    model = factory()
    try:
        results = await asyncio.gather(
            model.async_speech("1", "Vivian", response_format="wav"),
            model.async_speech("2", "Ryan", response_format=bad_format),
            model.async_speech("3", "Vivian", response_format="wav"),
            return_exceptions=True,
        )
        assert len(model._model.calls) == 1
        assert isinstance(results[1], Exception)
        peaks = []
        for output in (results[0], results[2]):
            assert isinstance(output, bytes)
            with wave.open(BytesIO(output), "rb") as wav:
                assert wav.getnframes() > 0
                peaks.append(
                    np.frombuffer(wav.readframes(wav.getnframes()), "<i2").max()
                )
        assert peaks[1] > peaks[0] > 0
        # A failed row must also leave the worker usable for later requests.
        assert isinstance(
            await model.async_speech("4", "Vivian", response_format="wav"), bytes
        )
        assert len(model._model.calls) == 2
    finally:
        await _shutdown(model)


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [_qwen, _mlx], ids=["pytorch", "mlx"])
@pytest.mark.parametrize("method", ["speech", "async_speech"])
async def test_audio_single_request_encoding_failure_is_raised(factory, method):
    model = factory()
    # The long MLX input also exercises the synchronous segment batch path.
    text = "1。2。" if factory is _mlx else "1"
    try:
        with pytest.raises(AttributeError):
            if method == "speech":
                model.speech(text, "Vivian", response_format=None)
            else:
                await model.async_speech(text, "Vivian", response_format=None)
    finally:
        await _shutdown(model)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled_index", [None, 0, 2])
async def test_zero_interval_drains_ready_audio_requests(cancelled_index):
    release_processor = asyncio.Event()
    calls = []

    class QueuedAudioRequestBatcher(AudioRequestBatcher):
        async def _process_batch(self):
            await release_processor.wait()
            await super()._process_batch()

    def run_batch(requests):
        values = [request["value"] for request in requests]
        calls.append(values)
        return values

    batcher = QueuedAudioRequestBatcher(
        run_batch, lambda request: 0, request_batch_size=3, batch_interval=0
    )
    tasks = [
        asyncio.create_task(batcher.submit(dict(value=value))) for value in range(7)
    ]
    try:
        await asyncio.sleep(0)
        assert batcher.queue.qsize() == 7
        if cancelled_index is not None:
            tasks[cancelled_index].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[cancelled_index]
        release_processor.set()
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=3
        )
        expected = [value for value in range(7) if value != cancelled_index]
        for value, result in enumerate(results):
            if value == cancelled_index:
                assert isinstance(result, asyncio.CancelledError)
            else:
                assert result == value
        assert calls == [
            expected[index : index + 3] for index in range(0, len(expected), 3)
        ]
        assert await asyncio.wait_for(batcher.submit(dict(value=7)), timeout=3) == 7
        assert calls[-1] == [7]
    finally:
        release_processor.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await _shutdown(SimpleNamespace(_batcher=batcher))


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
