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
import re
from collections import deque
from contextlib import aclosing, suppress
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np

from ...core.exceptions import InvalidAudioInputError
from ...core.realtime.protocol import RealtimeError
from ..realtime import RealtimeSession
from .native_async import NativeAsyncAudioModel, settle_request

PCM_FORMAT = {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1}
# Qwen3-ASR's language contract differs from Whisper's language table.
QWEN_LANGUAGES = dict(
    zh="Chinese",
    en="English",
    yue="Cantonese",
    ar="Arabic",
    de="German",
    fr="French",
    es="Spanish",
    pt="Portuguese",
    id="Indonesian",
    it="Italian",
    ko="Korean",
    ru="Russian",
    th="Thai",
    vi="Vietnamese",
    ja="Japanese",
    tr="Turkish",
    hi="Hindi",
    ms="Malay",
    nl="Dutch",
    sv="Swedish",
    da="Danish",
    fi="Finnish",
    pl="Polish",
    cs="Czech",
    fil="Filipino",
    fa="Persian",
    el="Greek",
    ro="Romanian",
    hu="Hungarian",
    mk="Macedonian",
    tl="Filipino",
)


def transcription_language(language):
    if language is None:
        return None
    if isinstance(language, str):
        normalized = language.strip().lower()
        if normalized == "auto":
            return None
        for code, name in QWEN_LANGUAGES.items():
            if normalized in (code, name.lower()):
                return name
    raise InvalidAudioInputError("Unsupported transcription language")


def join_transcripts(texts):
    result = ""
    # Include fullwidth punctuation without classifying fullwidth Latin as CJK.
    cjk = (
        r"[\u3000-\u30ff\u3400-\u9fff\uf900-\ufaff"
        r"\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65]"
    )
    for text in texts:
        if not text:
            continue
        separator = (
            ""
            if not result or (re.search(cjk + r"$", result) and re.match(cjk, text))
            else " "
        )
        result += separator + text
    return result


class VLLMQwen3ASRModel(NativeAsyncAudioModel):
    def __init__(self, *args, **kwargs):
        if "realtime" in kwargs:
            raise ValueError(
                "Qwen3-ASR does not use the realtime launch flag; configure realtime_options instead"
            )
        # Preserve the SDK adapter's existing batching knobs while moving
        # inference into the native scheduler. Alignment wasn't exposed by
        # this adapter's transcription API, so don't load unused weights.
        batch_size = kwargs.pop("batch_size", None)
        request_batch_size = kwargs.pop("request_batch_size", None)
        sdk_batch_size = kwargs.pop("max_inference_batch_size", None)
        if batch_size is None:
            batch_size = request_batch_size
        # -1 disabled SDK list chunking; native scheduling still needs a
        # positive concurrency limit and does not use SDK batch chunks.
        if (
            batch_size is None
            and sdk_batch_size is not None
            and int(sdk_batch_size) > 0
        ):
            batch_size = int(sdk_batch_size)
        if batch_size is not None:
            kwargs.setdefault("max_num_seqs", batch_size)
        for name in (
            "batch_interval",
            "forced_aligner",
            "forced_aligner_kwargs",
            "device_map",
            "torch_dtype",
        ):
            kwargs.pop(name, None)
        self._max_tokens = kwargs.pop("max_new_tokens", 4096)
        if type(self._max_tokens) is not int or not 1 <= self._max_tokens <= 8192:
            raise ValueError("max_new_tokens must be 1..8192")
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

    async def _transcribe(self, audio, language, prompt, max_tokens):
        from vllm import SamplingParams

        language = transcription_language(language)
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
        finish_reason = None
        async with aclosing(
            self._generate(
                engine_prompt,
                sampling_params=SamplingParams(temperature=0, max_tokens=max_tokens),
            )
        ) as outputs:
            async for output in outputs:
                if output.outputs:
                    result = output.outputs[0].text
                    finish_reason = getattr(output.outputs[0], "finish_reason", None)
        if finish_reason in {"length", "abort"}:
            raise RuntimeError("Transcription did not complete; result is incomplete")
        detected = re.match(r"\s*language\s+([^<\n]+)<asr_text>", result)
        detected_language = detected.group(1).strip() if detected else language
        return self._asr_class.post_process_output(result).strip(), detected_language

    async def transcribe_audio(self, audio, language=None, prompt=None):
        text, _ = await self._transcribe(
            audio, language, prompt, min(self._max_tokens, 512)
        )
        return text

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
            import soundfile as sf

            try:
                samples, rate = sf.read(
                    io.BytesIO(audio), dtype="float32", always_2d=True
                )
            except (ValueError, RuntimeError) as exc:
                raise InvalidAudioInputError(
                    "Invalid audio file: cannot decode audio"
                ) from exc
            samples = samples.mean(axis=1)
            if rate != 16000:
                from math import gcd

                from scipy.signal import resample_poly

                divisor = gcd(rate, 16000)
                samples = resample_poly(
                    samples, 16000 // divisor, rate // divisor
                ).astype(np.float32, copy=False)
            if not samples.size:
                raise InvalidAudioInputError("Empty audio")
            return samples

        transcription_language(language)
        samples = await asyncio.to_thread(decode)
        texts, languages = [], []
        # Keep native audio prompts bounded while preserving every file chunk.
        for start in range(0, len(samples), 30 * 16000):
            text, detected_language = await self._transcribe(
                samples[start : start + 30 * 16000], language, prompt, self._max_tokens
            )
            texts.append(text)
            if detected_language and detected_language not in languages:
                languages.append(detected_language)
        result: Dict[str, Any] = {"text": join_transcripts(texts)}
        if response_format == "verbose_json":
            result.update(
                task="transcribe",
                language=",".join(languages),
                duration=len(samples) / 16000,
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
        try:
            self.language = transcription_language(self.language)
        except ValueError as exc:
            raise RealtimeError(
                "invalid_config", "Unsupported transcription language"
            ) from exc
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
            result["text"] = join_transcripts(item["text"] for item in self._sentences)
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
