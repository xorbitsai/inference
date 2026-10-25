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
from typing import Any, Dict

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
        self.vad = DynamicStreamingVAD(_ExistingVAD(owner._model))
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

    def _slice(self, start: int, end: int):
        return self._audio[max(0, start - self._offset) : max(0, end - self._offset)]

    def _decode(self, start: int, end: int) -> str:
        continuation = start == self._continuation_start
        audio = self._slice(
            max(self._offset, start - self.overlap_samples) if continuation else start,
            end,
        )
        if not len(audio):
            return ""
        import torch
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        kwargs = dict(
            getattr(self.owner._model_spec, "default_transcription_config", None) or {}
        )
        kwargs.update(self.parameters)
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
        # Bypass offline VAD: this segment was already selected by streaming VAD.
        # Nano accepts paths/Tensors, not ndarrays. Keep float32 CPU samples and
        # isolate the model input from the session's mutable audio buffer.
        waveform = torch.from_numpy(audio.copy())
        result = self.owner._model.inference(input=waveform, **kwargs)
        if not isinstance(result, list) or any(
            not isinstance(x, dict) or "text" not in x for x in result
        ):
            raise RuntimeError("Invalid Nano inference result")
        text = rich_transcription_postprocess(
            "".join(x["text"] for x in result)
        ).strip()
        if continuation and self._sentences:
            text = trim_overlap(self._sentences[-1]["text"], text)
        return text

    def _commit(self, start: int, end: int) -> None:
        # A forced cut may have already confirmed part of this VAD utterance.
        start = max(start, self._committed_until)
        end = min(end, self._total)
        if end <= start:
            return
        text = self._decode(start, end)
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

    def _snapshot(self, final: bool = False) -> Dict[str, Any]:
        data = {
            "sentences": [dict(sentence) for sentence in self._sentences],
            "partial": self._partial,
            "is_final": final,
            "duration_ms": self._total * 1000 // self.sample_rate,
        }
        if final:
            data["text"] = "".join(sentence["text"] for sentence in self._sentences)
        return data

    def _feed(self, pcm: bytes, final: bool = False):
        import numpy as np
        import torch

        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self._audio = np.concatenate((self._audio, audio))
        self._total += len(audio)
        if self._total > int(self.runtime.limits.max_duration * self.sample_rate):
            raise RealtimeError("duration_exceeded", "Maximum audio duration exceeded")
        # Some FSMN releases require a nonempty final flush. Padding is VAD-only;
        # it never changes the accepted sample count or ASR tail.
        vad_audio = audio if len(audio) else np.zeros(160, dtype=np.float32)
        segments = self.vad.feed(torch.from_numpy(vad_audio), is_final=final)
        changed = False
        for start_ms, end_ms in segments:
            end = int(end_ms) * 16
            # VAD timestamps have millisecond resolution, PCM does not. Preserve
            # sub-millisecond accepted tail samples when flushing the last chunk.
            if final and 0 <= self._total - end < 16:
                end = self._total
            self._commit(int(start_ms) * 16, end)
            changed = True
        start_ms = self.vad.current_speech_start
        if start_ms is not None:
            start = max(self._offset, int(start_ms) * 16)
            while self._total - start >= self.max_segment_samples:
                cut = start + self.max_segment_samples
                self._commit(start, cut)
                start = cut
                self._continuation_start = cut
                self.vad.current_speech_start = start // 16
                changed = True
            if final:
                self._commit(start, self._total)
                self.vad.current_speech_start = None
                changed = True
            elif self._total - self._last_decode >= self.partial_samples:
                self._partial = self._decode(start, self._total)
                self._last_decode = self._total
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
        await self.emit(data)
        await self.end_outputs()

    async def close(self, reason: str) -> None:
        await super().close(reason)
        self._pending.clear()
        self._audio = self._audio[:0].copy()
        self.vad.reset()


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
