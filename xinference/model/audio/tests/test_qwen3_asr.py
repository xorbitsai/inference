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
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import pytest

from ..qwen3_asr import Qwen3ASRModel


class _FakeQwenASR:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language, **kwargs):
        payloads = [Path(path).read_bytes().decode() for path in audio]
        self.calls.append(
            {"payloads": payloads, "language": list(language), "kwargs": kwargs}
        )
        return [
            SimpleNamespace(text=payload, language=lang or "Detected")
            for payload, lang in zip(payloads, language)
        ]


def _new_model(batch_size=8, batch_interval=0.01, **kwargs):
    model_spec = SimpleNamespace(
        model_name="Qwen3-ASR-0.6B",
        model_ability=["audio2text"],
        default_transcription_config={},
    )
    model = Qwen3ASRModel(
        "qwen3-asr",
        "/unused",
        model_spec,
        batch_size=batch_size,
        batch_interval=batch_interval,
        **kwargs,
    )
    model._model = _FakeQwenASR()
    return model


def test_qwen3_asr_uses_audio_batch_interval_default():
    model_spec = SimpleNamespace(
        model_name="Qwen3-ASR-0.6B",
        model_ability=["audio2text"],
        default_transcription_config={},
    )

    model = Qwen3ASRModel("default", "/unused", model_spec)
    overridden = Qwen3ASRModel("overridden", "/unused", model_spec, batch_interval=0.02)

    assert model.batch_interval == pytest.approx(0.1)
    assert overridden.batch_interval == pytest.approx(0.02)


async def _shutdown_batch_processor(model):
    task = model._process_batch_task
    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_qwen3_asr_batches_requests_and_preserves_order():
    model = _new_model()
    try:
        results = await asyncio.gather(
            model.transcriptions(b"first", language="Chinese"),
            model.transcriptions(
                b"second", language="English", response_format="verbose_json"
            ),
            model.transcriptions(b"third"),
        )

        assert results == [
            {"text": "first"},
            {"task": "transcribe", "language": "English", "text": "second"},
            {"text": "third"},
        ]
        assert model._model.calls == [
            {
                "payloads": ["first", "second", "third"],
                "language": ["Chinese", "English", None],
                "kwargs": {},
            }
        ]
    finally:
        await _shutdown_batch_processor(model)


@pytest.mark.asyncio
async def test_qwen3_asr_respects_batch_size():
    model = _new_model(batch_size=2)
    try:
        results = await asyncio.gather(
            model.transcriptions(b"first"),
            model.transcriptions(b"second"),
            model.transcriptions(b"third"),
        )

        assert results == [
            {"text": "first"},
            {"text": "second"},
            {"text": "third"},
        ]
        assert [call["payloads"] for call in model._model.calls] == [
            ["first", "second"],
            ["third"],
        ]
    finally:
        await _shutdown_batch_processor(model)


@pytest.mark.asyncio
async def test_qwen3_asr_groups_model_kwargs_and_survives_cancellation():
    model = _new_model()
    try:
        cancelled = asyncio.create_task(
            model.transcriptions(b"cancelled", language="Chinese", beam_size=1)
        )
        kept = asyncio.create_task(
            model.transcriptions(b"kept", language="English", beam_size=1)
        )
        other_group = asyncio.create_task(
            model.transcriptions(b"other", language="Japanese", beam_size=2)
        )
        await asyncio.sleep(0)
        cancelled.cancel()

        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert await kept == {"text": "kept"}
        assert await other_group == {"text": "other"}

        # A cancelled waiter must not stop the long-lived batch processor.
        assert await model.transcriptions(b"after") == {"text": "after"}

        assert model._model.calls[:2] == [
            {
                "payloads": ["kept"],
                "language": ["English"],
                "kwargs": {"beam_size": 1},
            },
            {
                "payloads": ["other"],
                "language": ["Japanese"],
                "kwargs": {"beam_size": 2},
            },
        ]
        assert model._model.calls[2]["payloads"] == ["after"]
    finally:
        await _shutdown_batch_processor(model)
