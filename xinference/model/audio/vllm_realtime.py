# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Qwen3-ASR sessions sharing vLLM's native asynchronous request scheduler.

Audio encoding is segmental: each connection owns its VAD and bounded utterance
buffer. Concurrent decodes enter AsyncLLM together, rather than a model-wide
Python inference lock or the single-stream qwen-asr SDK.
"""

from __future__ import annotations

import asyncio
import io
from collections import deque
from contextlib import aclosing, suppress
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np

from ...core.realtime.protocol import RealtimeError
from ..realtime import RealtimeSession
from .native_async import NativeAsyncAudioModel, settle_request

PCM_FORMAT = {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1}


class VLLMRealtimeASRModel(NativeAsyncAudioModel):
    def __init__(self, *args, **kwargs):
        self._options = kwargs.pop("realtime_options", {})
        if not isinstance(self._options, dict) or set(self._options) - {
            "partial_interval_ms",
            "silence_ms",
            "min_speech_ms",
            "max_segment_ms",
        }:
            raise ValueError("Invalid realtime_options")
        defaults = {
            "partial_interval_ms": 1000,
            "silence_ms": 512,
            "min_speech_ms": 160,
            "max_segment_ms": 6000,
        }
        defaults.update(self._options)
        for name, value in defaults.items():
            if type(value) is not int or not 32 <= value <= 30000:
                raise ValueError(f"realtime_options.{name} must be 32..30000 ms")
        if defaults["max_segment_ms"] < max(
            defaults["partial_interval_ms"],
            defaults["min_speech_ms"],
            defaults["silence_ms"],
        ):
            raise ValueError(
                "max_segment_ms must cover partial, speech and silence intervals"
            )
        self._options = defaults
        super().__init__(*args, **kwargs)
        self.realtime_max_sessions = self._kwargs.setdefault("max_num_seqs", 16)
        if (
            type(self.realtime_max_sessions) is not int
            or not 1 <= self.realtime_max_sessions <= 128
        ):
            raise ValueError("max_num_seqs must be 1..128")

    async def _load(self):
        from silero_vad import load_silero_vad
        from vllm import AsyncEngineArgs
        from vllm.model_executor.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        from vllm.v1.engine.async_llm import AsyncLLM

        self._vad_factory = lambda: load_silero_vad(onnx=True)
        self._asr_class = Qwen3ASRForConditionalGeneration
        self._engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(model=self._model_path, **self._kwargs)
        )

    def get_realtime_capabilities(self):
        if self._engine is None or self._stopping:
            return {}
        return {
            "protocol_versions": [1],
            "tasks": {
                "audio.transcription": {
                    "input_modalities": ["audio"],
                    "output_modalities": ["text"],
                    "input_formats": [PCM_FORMAT],
                    "partial_results": True,
                    "execution_mode": "segmental",
                    "server_vad": True,
                    "request_batching": self.realtime_max_sessions > 1,
                }
            },
        }

    def create_realtime_session(self, config, runtime):
        if not self.get_realtime_capabilities():
            raise ValueError("Realtime ASR engine is not ready")
        return VLLMASRSession(self, config, runtime)

    async def transcribe_audio(self, audio, language=None, prompt=None):
        from vllm import SamplingParams

        if language == "auto":
            language = None
        if (
            language is not None
            and language not in self._asr_class.supported_languages
            and language not in self._asr_class.supported_languages.values()
        ):
            raise ValueError("Unsupported transcription language")
        params = SimpleNamespace(
            audio=audio,
            model_config=self._engine.model_config,
            language=language,
            task_type="transcribe",
            request_prompt=prompt,
            to_language=None,
        )
        engine_prompt = await asyncio.to_thread(
            self._asr_class.get_generation_prompt, params
        )
        result = ""
        async with aclosing(
            self._generate(
                engine_prompt,
                sampling_params=SamplingParams(temperature=0, max_tokens=512),
            )
        ) as outputs:
            async for output in outputs:
                if output.outputs:
                    result = output.outputs[0].text
        return self._asr_class.post_process_output(result).strip()

    def transcriptions(self, *args, **kwargs):
        raise RuntimeError("Use the asynchronous transcription entrypoint")

    async def async_transcriptions(
        self,
        audio: bytes,
        language=None,
        prompt=None,
        response_format="json",
        temperature=0,
        timestamp_granularities=None,
        **kwargs,
    ):
        if timestamp_granularities is not None or kwargs:
            raise ValueError(
                "Realtime vLLM ASR does not support timestamp or extra options"
            )
        if response_format not in {"json", "verbose_json"}:
            raise ValueError(
                "Supported transcription formats are json and verbose_json"
            )

        def decode():
            import librosa
            import soundfile as sf

            samples, rate = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
            samples = samples.mean(axis=1)
            if rate != 16000:
                samples = librosa.resample(samples, orig_sr=rate, target_sr=16000)
            if not samples.size:
                raise ValueError("Empty audio")
            return samples

        samples = await asyncio.to_thread(decode)
        text = await self.transcribe_audio(samples, language, prompt)
        result = {"text": text}
        if response_format == "verbose_json":
            result.update(
                task="transcribe", language=language, duration=len(samples) / 16000
            )
        return result


class VLLMASRSession(RealtimeSession):
    def __init__(self, owner, config, runtime):
        super().__init__(runtime.limits.output_queue_size)
        if config.get("input") != {"audio": PCM_FORMAT}:
            raise RealtimeError("invalid_config", "Expected mono 16 kHz PCM16 input")
        parameters = config.get("parameters", {})
        if set(parameters) - {"language", "hotwords"}:
            raise RealtimeError("invalid_config", "Unsupported transcription parameter")
        self.language = parameters.get("language", "en")
        hotwords = parameters.get("hotwords", [])
        if not isinstance(self.language, str) or not 1 <= len(self.language) <= 32:
            raise RealtimeError("invalid_config", "Invalid language")
        if (
            not isinstance(hotwords, list)
            or len(hotwords) > 100
            or any(
                not isinstance(word, str) or not 1 <= len(word) <= 128
                for word in hotwords
            )
        ):
            raise RealtimeError("invalid_config", "Invalid hotwords")
        self.prompt = ", ".join(hotwords) or None
        self.owner, self.runtime = owner, runtime
        self._vad = None
        self._pending = bytearray()
        self._padding: deque = deque(maxlen=5)
        self._frames: list = []
        self._total = self._start = self._speech = self._silence = (
            self._last_partial
        ) = 0
        self._partial = ""
        self._sentences: list = []
        self._jobs: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._worker = None
        self._ended = False

    def _snapshot(self, final=False):
        result: Dict[str, Any] = {
            "sentences": [dict(item) for item in self._sentences],
            "partial": self._partial,
            "is_final": final,
            "duration_ms": self._total // 16,
        }
        if final:
            result["text"] = " ".join(item["text"] for item in self._sentences)
        return result

    async def _decode(self):
        try:
            await self._decode_jobs()
        except Exception:
            await self._outputs.put(
                RealtimeError("processing_error", "Speech recognition failed", 1011)
            )
            raise

    async def iter_outputs(self):
        async for output in super().iter_outputs():
            if isinstance(output, Exception):
                raise output
            yield output

    async def _decode_jobs(self):
        while True:
            job = await self._jobs.get()
            if job is None:
                return
            audio, start, end, final = job
            text = await self.owner.transcribe_audio(audio, self.language, self.prompt)
            if self._closed:
                return
            if final:
                if text:
                    self._sentences.append(
                        {"start": start // 16, "end": end // 16, "text": text}
                    )
                self._partial = ""
            else:
                self._partial = text
            await self.emit(self._snapshot())

    async def _enqueue(self, final):
        if not self._frames:
            return
        # A preview is replaceable; committed utterances are never discarded.
        if not final and (not self._jobs.empty() or self._worker.done()):
            if self._worker.done():
                self._worker.result()
            return
        audio = np.concatenate(self._frames)
        job = (audio, self._start, self._total, final)
        put = asyncio.create_task(self._jobs.put(job))
        try:
            done, _ = await asyncio.wait(
                {put, self._worker}, return_when=asyncio.FIRST_COMPLETED
            )
            if self._worker in done:
                self._worker.result()
                raise RuntimeError("ASR decoder stopped before accepting input")
            await put
        finally:
            if not put.done():
                put.cancel()
                with suppress(asyncio.CancelledError):
                    await settle_request(put)

    async def _frame(self, frame, valid_samples=None):
        import torch

        probability = await self.runtime.invoke(
            lambda: float(self._vad(torch.from_numpy(frame), 16000).item())
        )
        length = len(frame) if valid_samples is None else valid_samples
        frame = frame[:length]
        self._total += length
        options = self.owner._options
        if probability >= 0.6:
            if not self._frames:
                self._frames.extend(self._padding)
                self._start = self._total - length - sum(map(len, self._padding))
            self._speech += length
            self._silence = 0
        elif self._frames:
            self._silence += length
        if self._frames or probability >= 0.6:
            self._frames.append(frame)
            accepted = self._speech >= options["min_speech_ms"] * 16
            if (
                self._silence >= options["silence_ms"] * 16
                or self._total - self._start >= options["max_segment_ms"] * 16
            ):
                if accepted:
                    await self._enqueue(True)
                self._frames = []
                self._speech = self._silence = self._last_partial = 0
                self._padding.clear()
                return
            elif (
                accepted
                and self._total - self._start - self._last_partial
                >= options["partial_interval_ms"] * 16
            ):
                await self._enqueue(False)
                self._last_partial = self._total - self._start
        self._padding.append(frame)

    async def receive_input(self, frame: bytes):
        if self._closed or self._ended:
            raise RealtimeError("invalid_state", "Session no longer accepts audio")
        if self._vad is None:
            self._vad = await self.runtime.invoke(self.owner._vad_factory)
            self._worker = asyncio.create_task(self._decode())
        self._pending.extend(frame)
        while len(self._pending) >= 1024:
            pcm = bytes(self._pending[:1024])
            del self._pending[:1024]
            await self._frame(
                np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
            )

    async def finish_input(self):
        self._ended = True
        if len(self._pending) % 2:
            raise RealtimeError("invalid_audio", "Incomplete PCM16 sample")
        if self._pending:
            tail = (
                np.frombuffer(bytes(self._pending), dtype="<i2").astype(np.float32)
                / 32768
            )
            await self._frame(np.pad(tail, (0, 512 - len(tail))), len(tail))
            self._pending.clear()
        if self._frames and self._speech >= self.owner._options["min_speech_ms"] * 16:
            await self._enqueue(True)
        if self._worker is not None:
            done = asyncio.create_task(self._jobs.put(None))
            try:
                completed, _ = await asyncio.wait(
                    {done, self._worker}, return_when=asyncio.FIRST_COMPLETED
                )
                if self._worker in completed:
                    self._worker.result()
                else:
                    await self._worker
            finally:
                if not done.done():
                    done.cancel()
                    with suppress(asyncio.CancelledError):
                        await settle_request(done)
        await self.emit(self._snapshot(True))
        await self.end_outputs()

    def request_cancel(self):
        super().request_cancel()
        if self._worker is not None:
            self._worker.cancel()

    async def close(self, reason):
        self.request_cancel()
        if self._worker is not None:
            try:
                await settle_request(self._worker)
            except (Exception, asyncio.CancelledError):
                pass
        self._pending.clear()
        self._frames.clear()
        self._padding.clear()
