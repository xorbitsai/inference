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


"""Opt-in segmental Nano realtime adapter, reusing PyTorch ASR/VAD weights.

Like the official realtime service, this uses streaming FSMN VAD and redecodes
the current utterance for partials. It is not incremental acoustic encoding.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Iterable, cast
from unicodedata import east_asian_width

from ...core.realtime.protocol import RealtimeError
from ..realtime import RealtimeSession

PCM_FORMAT = {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1}
CAPABILITIES = {
    "protocol_versions": [1],
    "tasks": {
        "audio.transcription": {
            "input_modalities": ["audio"],
            "output_modalities": ["text"],
            "input_formats": [PCM_FORMAT],
            "partial_results": True,
            "execution_mode": "segmental",
        }
    },
}


def _reject_json_constant(value: str) -> None:
    raise ValueError("Non-standard JSON constant")


@dataclass(frozen=True)
class NanoRealtimeOptions:
    """Model-level, session-pinned experiments; never passed to the ASR SDK."""

    optimize: bool = False
    end_priority: bool = True
    coalesce_partials: bool = True
    reuse_decode: bool = True
    first_partial_ms: int = 960
    partial_interval_ms: int = 960
    max_segment_ms: int = 15000
    preview_backlog_ms: int = 0

    @classmethod
    def from_dict(cls, value: Any) -> NanoRealtimeOptions:
        # Older Web UIs send JSON text from the extra-parameter input. Decode
        # once, never strip shell quotes or recursively decode JSON strings.
        if isinstance(value, str):
            try:
                value = json.loads(value, parse_constant=_reject_json_constant)
            except (ValueError, RecursionError):
                raise ValueError("realtime_options must contain valid JSON") from None
        if not isinstance(value, dict):
            raise ValueError("realtime_options must be a JSON object")
        if set(value) - {f.name for f in fields(cls)}:
            raise ValueError("realtime_options contains unsupported fields")
        options = cls(**value)
        for name in ("optimize", "end_priority", "coalesce_partials", "reuse_decode"):
            if type(getattr(options, name)) is not bool:
                raise ValueError(f"realtime_options.{name} must be a boolean")
        for name, low, high in (
            ("first_partial_ms", 120, 10000),
            ("partial_interval_ms", 120, 10000),
            ("max_segment_ms", 1000, 15000),
            ("preview_backlog_ms", 0, 10000),
        ):
            item = getattr(options, name)
            if type(item) is not int:
                raise ValueError(f"realtime_options.{name} must be an integer")
            if not low <= item <= high:
                raise ValueError(f"realtime_options.{name} must be {low}..{high} ms")
        if (
            max(options.first_partial_ms, options.partial_interval_ms)
            > options.max_segment_ms
        ):
            raise ValueError(
                "realtime_options: Partial intervals must not exceed max_segment_ms"
            )
        return options


def _json_config(value: Any) -> bool:
    """Do not equate tuples/objects/non-string keys through JSON coercion."""
    if value is None or type(value) in (str, bool, int, float):
        return True
    if type(value) is list:
        return all(_json_config(item) for item in value)
    if type(value) is dict:
        return all(
            type(key) is str and _json_config(item) for key, item in value.items()
        )
    return False


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if config["task"] != "audio.transcription":
        raise RealtimeError("unsupported_task", "Expected audio.transcription")
    audio_format = config["input"].get("audio")
    if (
        config["input"] != {"audio": PCM_FORMAT}
        or type(audio_format.get("sample_rate")) is not int
        or type(audio_format.get("channels")) is not int
    ):
        raise RealtimeError("unsupported_format", "Expected 16 kHz mono pcm_s16le")
    parameters = config.get("parameters", {})
    if set(parameters) - {"language", "hotwords"}:
        raise RealtimeError("invalid_config", "Unknown transcription parameter")
    language = parameters.get("language", "auto")
    hotwords = parameters.get("hotwords", [])
    if not isinstance(language, str) or not 1 <= len(language) <= 32:
        raise RealtimeError("invalid_config", "Invalid language")
    if (
        not isinstance(hotwords, list)
        or len(hotwords) > 100
        or any(
            not isinstance(word, str) or not 1 <= len(word) <= 128 for word in hotwords
        )
    ):
        raise RealtimeError("invalid_config", "Invalid hotwords")
    return {"language": language, "hotwords": list(hotwords)}


class _ExistingVAD:
    """AutoModel-like facade around already loaded FSMN weights, no model load."""

    def __init__(self, auto_model):
        self.auto_model = auto_model
        self.model = copy.copy(auto_model.vad_model)
        if hasattr(self.model, "vad_opts"):
            self.model.vad_opts = copy.deepcopy(self.model.vad_opts)
        self.kwargs = {
            key: copy.deepcopy(value) if isinstance(value, (dict, list, set)) else value
            for key, value in auto_model.vad_kwargs.items()
        }

    def generate(self, input, **kwargs):
        return self.auto_model.inference(
            input=input, model=self.model, kwargs=self.kwargs, **kwargs
        )


class NanoRealtimeSession(RealtimeSession):
    sample_rate = 16000
    vad_chunk_samples = 960  # 60 ms; independent of network frame boundaries.
    partial_samples = 15360  # 960 ms; tuned only after real-node measurement.
    max_segment_samples = 240000  # 15 s, prevents whole-recording redecoding.
    lookback_samples = 80000  # 5 s for delayed VAD onset.
    overlap_samples = 8000  # 500 ms context at forced boundaries.

    def __init__(self, owner, config, runtime):
        super().__init__(runtime.limits.output_queue_size)
        import numpy as np
        from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD

        self.owner = owner
        self.runtime = runtime
        self.parameters = validate_config(config)
        self.options = (
            getattr(owner, "_realtime_options", None) or NanoRealtimeOptions()
        )
        self.partial_samples = self.options.partial_interval_ms * 16
        self.first_partial_samples = self.options.first_partial_ms * 16
        self.max_segment_samples = self.options.max_segment_ms * 16
        self.vad = DynamicStreamingVAD(_ExistingVAD(owner._model))
        self._end_requested = threading.Event()
        self._cancel_requested = threading.Event()
        self._hint_lock = threading.Lock()
        self._received_bytes = 0
        self._decode_cache = None
        # Counter also stores fractional millisecond durations here, not only
        # integers. Expose that numeric mapping without changing its defaults.
        self._metrics = cast(Dict[str, float], Counter())
        self._decode_counts: Counter[str] = Counter()
        self._buffer_peak = 0
        self._backlog_peak = 0
        self._preview_count = 0
        self._pending = bytearray()
        self._audio = np.empty(0, dtype=np.float32)
        self._offset = 0
        self._total = 0
        self._last_decode = 0
        self._committed_until = 0
        self._continuation_start = None
        self._sentences = []
        self._partial = ""
        self._ended = False

    def notify_input(self, size: int) -> None:
        with self._hint_lock:
            self._received_bytes += size

    def request_input_end(self) -> None:
        self._end_requested.set()

    def request_cancel(self) -> None:
        super().request_cancel()
        self._cancel_requested.set()

    def observations(self) -> Dict[str, Any]:
        with self._hint_lock:
            received = max(self._received_bytes // 2, self._total)
        return {
            "received_samples": received,
            "processed_samples": self._total,
            "committed_samples": self._committed_until,
            "backlog_audio_ms": max(0, received - self._total) / 16,
            "backlog_peak_ms": self._backlog_peak / 16,
            "buffer_peak_samples": self._buffer_peak,
            "decode_counts": dict(self._decode_counts),
            "metrics": dict(self._metrics),
            "optimized": self.options.optimize,
            "options": asdict(self.options),
        }

    def _slice(self, start: int, end: int):
        return self._audio[max(0, start - self._offset) : max(0, end - self._offset)]

    def _decode(self, start: int, end: int, kind: str = "partial") -> str:
        if self._cancel_requested.is_set():
            raise RealtimeError("invalid_state", "Session cancelled")
        continuation = start == self._continuation_start
        actual_start = (
            max(self._offset, start - self.overlap_samples) if continuation else start
        )
        actual_start = max(self._offset, actual_start)
        audio = self._slice(actual_start, end)
        if not len(audio):
            return ""
        import torch
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        kwargs = copy.deepcopy(
            getattr(self.owner._model_spec, "default_transcription_config", None) or {}
        )
        kwargs.update(copy.deepcopy(self.parameters))
        kwargs["language"] = {
            "auto": None,
            "zh": "中文",
            "zh-CN": "中文",
            "en": "English",
            "ja": "日本語",
        }.get(self.parameters["language"], self.parameters["language"])
        # Nano supplies the target sample rate through frontend.fs internally.
        # Avoid passing fs again through **kwargs, including from defaults.
        kwargs.pop("fs", None)
        kwargs.update(cache={}, batch_size=1)
        cache_key = None
        if self.options.optimize and self.options.reuse_decode:
            try:
                configuration = {k: v for k, v in kwargs.items() if k != "cache"}
                if not _json_config(configuration):
                    raise TypeError("Non-JSON SDK options")
                effective = json.dumps(
                    configuration,
                    sort_keys=True,
                    allow_nan=False,
                )
                cache_key = (
                    self.runtime.generation,
                    id(self.owner._model),
                    self._offset,
                    actual_start,
                    start,
                    end,
                    self._total,
                    continuation,
                    len(self._sentences),
                    (
                        self._sentences[-1]["text"]
                        if continuation and self._sentences
                        else None
                    ),
                    effective,
                )
            except (TypeError, ValueError):
                pass  # Unknown SDK options cannot establish strict equivalence.
            if cache_key is not None and self._decode_cache is not None:
                previous_key, text = self._decode_cache
                if cache_key == previous_key:
                    self._metrics["decode_reused"] += 1
                    return text
        self._decode_cache = None
        # Bypass offline VAD: this segment was already selected by streaming VAD.
        # Nano accepts paths/Tensors, not ndarrays. Keep float32 CPU samples and
        # isolate the model input from the session's mutable audio buffer.
        waveform = torch.from_numpy(audio.copy())
        self._decode_counts[kind] += 1
        self._metrics["decode_samples"] += len(audio)
        self._metrics["max_decode_samples"] = max(
            self._metrics["max_decode_samples"], len(audio)
        )
        started = time.monotonic()
        self.runtime.record_event(
            "native_start", kind=kind, start_sample=actual_start, end_sample=end
        )
        succeeded = False
        try:
            result = self.owner._model.inference(input=waveform, **kwargs)
            succeeded = True
        finally:
            elapsed_ms = (time.monotonic() - started) * 1000
            self._metrics["decode_ms"] += elapsed_ms
            self.runtime.record_event(
                "native_end", kind=kind, elapsed_ms=elapsed_ms, succeeded=succeeded
            )
        if not isinstance(result, list) or any(
            not isinstance(x, dict) or "text" not in x for x in result
        ):
            raise RuntimeError("Invalid Nano inference result")
        text = rich_transcription_postprocess(
            "".join(x["text"] for x in result)
        ).strip()
        if continuation and self._sentences:
            text = trim_overlap(self._sentences[-1]["text"], text)
        if cache_key is not None and not self._cancel_requested.is_set():
            self._decode_cache = (cache_key, text)
        return text

    def _commit(self, start: int, end: int, kind: str = "vad_commit") -> None:
        # A forced cut may have already confirmed part of this VAD utterance.
        start = max(start, self._committed_until)
        end = min(end, self._total)
        if end <= start:
            return
        text = self._decode(start, end, kind)
        if text:
            self._sentences.append(
                {
                    "start": start * 1000 // self.sample_rate,
                    "end": end * 1000 // self.sample_rate,
                    "text": text,
                }
            )
        self._partial = ""
        self._committed_until = end
        self._continuation_start = None
        self._last_decode = self._total
        self._decode_cache = None

    def _snapshot(self, final: bool = False) -> Dict[str, Any]:
        data = {
            "sentences": [dict(sentence) for sentence in self._sentences],
            "partial": self._partial,
            "is_final": final,
            "duration_ms": self._total * 1000 // self.sample_rate,
        }
        if final:
            data["text"] = join_sentence_texts(
                sentence["text"] for sentence in self._sentences
            )
        return data

    def _feed(self, pcm: bytes, final: bool = False):
        import numpy as np
        import torch

        if self._cancel_requested.is_set():
            raise RealtimeError("invalid_state", "Session cancelled")
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self._audio = np.concatenate((self._audio, audio))
        self._total += len(audio)
        self._buffer_peak = max(self._buffer_peak, len(self._audio))
        with self._hint_lock:
            backlog = max(0, self._received_bytes // 2 - self._total)
        self._backlog_peak = max(self._backlog_peak, backlog)
        if self._total > int(self.runtime.limits.max_duration * self.sample_rate):
            raise RealtimeError("duration_exceeded", "Maximum audio duration exceeded")
        # FSMN's 25 ms frames, 10 ms shift and LFR right context need more than
        # a 10 ms empty flush. One VAD chunk of silence both initializes a new
        # frontend for sub-frame input and drains the frame-quantized tail.
        # Padding is VAD-only: _commit clips endpoints to accepted PCM. Natural
        # silence endpoints remain unchanged instead of extending every final
        # segment to the end of the recording.
        vad_audio = (
            np.concatenate((audio, np.zeros(self.vad_chunk_samples, dtype=np.float32)))
            if final
            else audio
        )
        vad_started = time.monotonic()
        segments = self.vad.feed(torch.from_numpy(vad_audio), is_final=final)
        self._metrics["vad_ms"] += (time.monotonic() - vad_started) * 1000
        changed = False
        for start_ms, end_ms in segments:
            end = int(end_ms) * 16
            self._commit(int(start_ms) * 16, end, "final" if final else "vad_commit")
            changed = True
        start_ms = self.vad.current_speech_start
        if start_ms is not None:
            start = max(self._offset, int(start_ms) * 16)
            while self._total - start >= self.max_segment_samples:
                cut = start + self.max_segment_samples
                self._commit(start, cut, "forced_commit")
                start = cut
                self._continuation_start = cut
                self.vad.current_speech_start = start // 16
                changed = True
            if final:
                self._commit(start, self._total, "final")
                self.vad.current_speech_start = None
                changed = True
            elif self._total - self._last_decode >= (
                self.first_partial_samples
                if not self._preview_count
                else self.partial_samples
            ):
                with self._hint_lock:
                    backlog = max(0, self._received_bytes // 2 - self._total)
                self._backlog_peak = max(self._backlog_peak, backlog)
                # The latest prefix is the sole implicit pending preview. Drain
                # VAD first when behind, rather than enqueueing stale ASR jobs.
                # No PCM or mandatory commit is skipped and no second model
                # thread is introduced. Recheck hints after acquiring invoke's lock.
                if (
                    self.options.optimize
                    and self.options.end_priority
                    and self._end_requested.is_set()
                ):
                    self._metrics["previews_skipped_end"] += 1
                elif (
                    self.options.optimize
                    and self.options.coalesce_partials
                    and backlog
                    >= max(self.vad_chunk_samples, self.options.preview_backlog_ms * 16)
                ):
                    self._metrics["previews_coalesced"] += 1
                else:
                    self._partial = self._decode(start, self._total)
                    self._last_decode = self._total
                    self._preview_count += 1
                    changed = True
        keep = (
            int(self.vad.current_speech_start) * 16
            if self.vad.current_speech_start is not None
            else max(0, self._total - self.lookback_samples)
        )
        if self._continuation_start is not None:
            keep = min(keep, max(0, self._continuation_start - self.overlap_samples))
        drop = max(0, min(keep, self._total) - self._offset)
        self._audio = self._audio[drop:].copy()
        self._offset += drop
        return self._snapshot(final) if changed or final else None

    async def receive_input(self, frame: bytes) -> None:
        if self._ended or self._closed:
            raise RealtimeError("invalid_state", "Input has ended")
        self._pending.extend(frame)
        size = self.vad_chunk_samples * 2
        while len(self._pending) >= size:
            pcm = bytes(self._pending[:size])
            del self._pending[:size]
            data = await self.runtime.invoke(self._feed, pcm)
            if data is not None:
                await self.emit(data)

    async def finish_input(self) -> None:
        if self._ended:
            raise RealtimeError("invalid_state", "Input already ended")
        self._ended = True
        if len(self._pending) % 2:
            raise RealtimeError("invalid_audio", "Incomplete PCM16 sample at input.end")
        data = await self.runtime.invoke(self._feed, bytes(self._pending), True)
        self._pending.clear()
        self.runtime.record_event("final_enqueue_start")
        await self.emit(data)
        self.runtime.record_event("final_enqueued")
        await self.end_outputs()

    async def close(self, reason: str) -> None:
        await super().close(reason)
        self._pending.clear()
        self._audio = self._audio[:0].copy()
        self._decode_cache = None
        self.vad.reset()


def join_sentence_texts(texts: Iterable[str]) -> str:
    """Keep word boundaries without adding spaces between CJK characters."""
    joined = ""
    for text in texts:
        if (
            joined
            and text
            and text[0].isalnum()
            and not joined[-1].isspace()
            and all(
                east_asian_width(char) not in {"W", "F"}
                for char in (joined[-1], text[0])
            )
        ):
            joined += " "
        joined += text
    return joined


def trim_overlap(previous: str, current: str) -> str:
    """Conservative exact suffix/prefix merge, only for overlapping audio.

    Never drop an entire new hypothesis or a single repeated character. Real
    boundary accuracy still needs annotated speech, not just synthetic tests.
    """
    for size in range(min(64, len(previous), len(current) - 1), 1, -1):
        prefix = current[:size]
        if previous.endswith(prefix) and (
            not prefix[-1].isascii()
            or not prefix[-1].isalnum()
            or (len(current) > size and not current[size].isalnum())
        ):
            return current[size:].lstrip()
    return current
