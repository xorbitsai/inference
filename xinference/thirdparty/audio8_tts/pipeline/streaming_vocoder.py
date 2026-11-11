# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import torch

from sglang_omni.executors import Executor
from xinference.thirdparty.audio8_tts.pipeline.state_io import load_state, store_state
from sglang_omni.proto import StagePayload


class Audio8StreamingVocoderExecutor(Executor):
    def __init__(
        self,
        codec: Any,
        *,
        device: str,
        eos_token_id: int,
        num_codebooks: int,
        chunk_frames: int = 12,
        context_frames: int = 128,
        guard_frames: int = 1,
        hop_length: int = 2048,
    ) -> None:
        self._codec = codec
        self._device = torch.device(device)
        self._eos_token_id = int(eos_token_id)
        self._num_codebooks = int(num_codebooks)
        self._chunk_frames = max(int(chunk_frames), 1)
        self._context_frames = max(int(context_frames), 0)
        self._guard_samples = max(int(guard_frames), 0) * int(hop_length)
        self._hop_length = int(hop_length)
        self._sample_rate = int(codec.sample_rate)
        self._stream_queue: Any | None = None
        self._done: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: dict[str, asyncio.Task[StagePayload]] = {}
        self._output_queues: dict[str, asyncio.Queue[dict[str, Any] | None]] = {}
        self._aborted: set[str] = set()
        self._gpu_lock = asyncio.Lock()

    async def add_request(self, payload: StagePayload) -> None:
        request_id = payload.request_id
        if request_id in self._aborted:
            return
        self._output_queues[request_id] = asyncio.Queue()
        if self._stream_queue is None:
            task = asyncio.create_task(self._run_batch(payload))
        else:
            task = asyncio.create_task(self._run_request(payload))
        self._tasks[request_id] = task
        task.add_done_callback(lambda _task: self._done.put_nowait(request_id))

    async def _run_batch(self, payload: StagePayload) -> StagePayload:
        """Non-streaming decode: decode the complete output_codes at once."""
        state = load_state(payload)
        codes = state.output_codes
        if codes is None:
            raise ValueError("Audio8 generation produced no codec frames")
        codes = torch.as_tensor(codes, device=self._device, dtype=torch.long)
        if codes.ndim != 2 or codes.shape[0] != self._num_codebooks or codes.shape[1] == 0:
            raise ValueError(
                f"unexpected Audio8 codes shape {tuple(codes.shape)}, "
                f"expected [{self._num_codebooks}, T>0]"
            )
        with torch.inference_mode():
            audio = self._codec.decode(codes.unsqueeze(0))[0, 0]
        arr = audio.detach().float().cpu().numpy().astype(np.float32, copy=False)
        self._output_queues.pop(payload.request_id, None)
        payload.data.update(
            {
                "audio_waveform": arr.tobytes(),
                "audio_waveform_shape": list(arr.shape),
                "audio_waveform_dtype": "float32",
                "sample_rate": self._sample_rate,
                "modality": "audio",
            }
        )
        return payload

    async def get_result(self) -> StagePayload:
        while True:
            request_id = await self._done.get()
            task = self._tasks.pop(request_id, None)
            if task is None:
                continue
            if request_id in self._aborted:
                self._output_queues.pop(request_id, None)
                continue
            try:
                return await task
            except Exception as exc:
                exc.request_id = request_id
                raise

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)
        task = self._tasks.pop(request_id, None)
        if task is not None:
            task.cancel()
        queue = self._output_queues.pop(request_id, None)
        if queue is not None:
            queue.put_nowait(None)

    async def stream(self, request_id: str):
        queue = self._output_queues.get(request_id)
        if queue is None:
            return
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item

    async def _run_request(self, payload: StagePayload) -> StagePayload:
        if self._stream_queue is None:
            raise RuntimeError("Audio8 streaming vocoder requires a stream queue")

        request_id = payload.request_id
        output_queue = self._output_queues[request_id]
        state = load_state(payload)
        frames: list[torch.Tensor] = []
        emitted_samples = 0

        try:
            while True:
                item = await self._stream_queue.get(request_id)
                if item is None:
                    break
                codes = torch.as_tensor(item.data, device=self._device, dtype=torch.long)
                if codes.ndim == 1:
                    codes = codes[:, None]
                semantic = int(codes[0, -1].item())
                if semantic == self._eos_token_id:
                    continue
                frame = codes[1 : self._num_codebooks + 1, -1].clone()
                if frame.numel() != self._num_codebooks:
                    raise ValueError(
                        f"Audio8 stream frame has {frame.numel()} codebooks, "
                        f"expected {self._num_codebooks}"
                    )
                frames.append(frame)
                if len(frames) % self._chunk_frames:
                    continue
                audio, absolute_start = await self._decode_stream_window(frames)
                stable_end = max(0, audio.size - self._guard_samples)
                begin = max(0, emitted_samples - absolute_start)
                if stable_end > begin:
                    chunk = np.ascontiguousarray(audio[begin:stable_end])
                    emitted_samples = absolute_start + stable_end
                    await output_queue.put(self._audio_payload(chunk))

            if not frames:
                await output_queue.put(None)
                return store_state(payload, state)

            tail, absolute_start = await self._decode_stream_window(frames)
            begin = max(0, emitted_samples - absolute_start)
            if begin < tail.size:
                chunk = np.ascontiguousarray(tail[begin:])
                emitted_samples = absolute_start + tail.size
                await output_queue.put(self._audio_payload(chunk))
            await output_queue.put(None)

            full_audio = await self._decode_frames(frames, 0, len(frames))
            payload.data = self._audio_payload(full_audio)
            prompt_tokens = len(state.input_ids) if state.input_ids is not None else 0
            payload.data["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(frames),
                "total_tokens": prompt_tokens + len(frames),
            }
            return payload
        finally:
            self._output_queues.pop(request_id, None)

    async def _decode_stream_window(
        self, frames: list[torch.Tensor]
    ) -> tuple[np.ndarray, int]:
        end = len(frames)
        start = max(0, end - self._context_frames - self._chunk_frames)
        return await self._decode_frames(frames, start, end), start * self._hop_length

    async def _decode_frames(
        self,
        frames: list[torch.Tensor],
        start: int,
        end: int,
    ) -> np.ndarray:
        loop = asyncio.get_running_loop()
        async with self._gpu_lock:
            return await loop.run_in_executor(
                None,
                self._decode_frames_sync,
                frames,
                start,
                end,
            )

    def _decode_frames_sync(
        self,
        frames: list[torch.Tensor],
        start: int,
        end: int,
    ) -> np.ndarray:
        torch.cuda.set_device(self._device)
        codes = torch.stack(frames[start:end], dim=1).unsqueeze(0)
        with torch.inference_mode():
            audio = self._codec.decode(codes)[0, 0]
        return audio.detach().float().cpu().numpy().copy()

    def _audio_payload(self, audio: np.ndarray) -> dict[str, Any]:
        value = audio.astype(np.float32, copy=False)
        return {
            "audio_waveform": value.tobytes(),
            "audio_waveform_shape": list(value.shape),
            "audio_waveform_dtype": "float32",
            "sample_rate": self._sample_rate,
            "modality": "audio",
        }
