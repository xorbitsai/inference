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

from __future__ import annotations

import asyncio
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Optional

from ...core.exceptions import InvalidModelRequestError
from .zonos3_mini import SAMPLE_RATE, Zonos3MiniModel

if TYPE_CHECKING:
    import numpy as np
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.proto import OmniRequest

SGLANG_MODEL_FILES = (
    "lm/zonos3_mini/model.pth",
    "lm/zonos3_mini/params.json",
    "lm/zonos3_mini/tokenizer/superbpe8192.json",
    "autoencoder/generator.pt",
    "autoencoder/heads/full_mr/step_0050000.pt",
    "autoencoder/teacher/config.json",
    "autoencoder/teacher/model.safetensors",
    "registries/codec_registry.json",
    "registries/silence_frame_opus_dual_3n_600k_sched600k_r4096_step600000.json",
)


class SGLangOmniZonos3MiniModel(Zonos3MiniModel):
    supports_streaming = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._runner: Optional[MultiProcessPipelineRunner] = None

    def load(self) -> None:
        if sys.platform != "linux" or not (3, 11) <= sys.version_info < (3, 13):
            raise ValueError(
                "ZONOS3-Mini SGLang-Omni requires Linux and Python 3.11 or 3.12."
            )

        import torch

        device = torch.device(self._device or "cuda:0")
        if (
            device.type != "cuda"
            or not torch.cuda.is_available()
            or torch.version.cuda is None
        ):
            raise ValueError("ZONOS3-Mini SGLang-Omni requires an NVIDIA CUDA device.")
        if self._kwargs:
            raise ValueError(
                "Unsupported ZONOS3-Mini SGLang-Omni launch arguments: "
                f"{sorted(self._kwargs)}"
            )

        model_path = Path(self._model_path).absolute()
        missing = [
            name for name in SGLANG_MODEL_FILES if not (model_path / name).is_file()
        ]
        if missing:
            raise ValueError(
                f"Incomplete local ZONOS3-Mini SGLang-Omni checkpoint at {model_path}: "
                f"{', '.join(missing)}"
            )

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=self._run_loop, args=(loop,), daemon=True)
        thread.start()
        self._loop, self._thread = loop, thread
        try:
            asyncio.run_coroutine_threadsafe(
                self._start(str(model_path), device.index or 0), loop
            ).result()
        except BaseException:
            self.stop()
            raise

    @staticmethod
    def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
            loop.close()

    async def _start(self, model_path: str, gpu_index: int) -> None:
        try:
            from sglang_omni.models.zonos3.config import Zonos3PipelineConfig
            from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
        except ImportError as exc:
            raise ImportError(
                "Install SGLang-Omni with ZONOS3-Mini support or enable the model "
                "virtual environment."
            ) from exc

        config = Zonos3PipelineConfig(model_path=model_path)
        for stage in config.stages:
            stage.gpu = gpu_index
        runner = MultiProcessPipelineRunner(config)
        timeout = float(os.environ.get("SGLANG_OMNI_STARTUP_TIMEOUT", "600"))
        await runner.start(timeout=timeout)
        self._runner = runner

    def stop(self) -> None:
        loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return
        try:
            if self._runner is not None:
                asyncio.run_coroutine_threadsafe(self._runner.stop(), loop).result()
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
            self._runner = self._loop = self._thread = None

    def _validate_engine_request(self, language: Optional[str], options: dict) -> None:
        if language is not None:
            raise InvalidModelRequestError(
                "ZONOS3-Mini SGLang-Omni does not accept an explicit language."
            )
        for name in ("top_k", "top_p"):
            if options.get(name) == 0:
                raise InvalidModelRequestError(
                    f"{name} must be positive for SGLang-Omni."
                )
        if options.get("max_new_tokens", 0) >= 8192:
            raise InvalidModelRequestError(
                "SGLang-Omni max_new_tokens must be below 8192."
            )

    def _make_request(
        self,
        text: str,
        voice: str,
        reference_path: Optional[Path],
        options: dict,
        *,
        stream: bool,
    ) -> OmniRequest:
        import numpy as np
        from sglang_omni.proto import OmniRequest

        if reference_path is None:
            voice_dir = Path(self._model_path) / "gguf" / "voices"
            codes = next(voice_dir.glob(f"*/{voice}_*.codes.npy"), None)
            if codes is None:
                raise ValueError(
                    f"ZONOS3-Mini preset audio codes are missing for {voice}."
                )
            reference = {"vq_codes": np.load(codes, allow_pickle=False).tolist()}
        else:
            reference = {"audio_path": str(reference_path)}

        return OmniRequest(
            inputs={"text": text, "references": [reference]},
            params={**options, "stream": stream},
        )

    @staticmethod
    def _waveform(payload: dict) -> np.ndarray:
        import numpy as np

        if payload["sample_rate"] != SAMPLE_RATE:
            raise RuntimeError("Unexpected ZONOS3-Mini SGLang-Omni sample rate.")
        return np.frombuffer(payload["audio_waveform"], dtype=np.float32)

    def _runtime(self) -> tuple[MultiProcessPipelineRunner, asyncio.AbstractEventLoop]:
        if self._runner is None or self._loop is None:
            raise RuntimeError("ZONOS3-Mini SGLang-Omni model is not loaded.")
        return self._runner, self._loop

    def _generate(
        self,
        text: str,
        voice: str,
        reference_path: Optional[Path],
        language: Optional[str],
        options: dict[str, Any],
    ) -> np.ndarray:
        runner, loop = self._runtime()
        request = self._make_request(text, voice, reference_path, options, stream=False)
        result = asyncio.run_coroutine_threadsafe(
            runner.coordinator.submit(uuid.uuid4().hex, request), loop
        )
        try:
            return self._waveform(result.result())
        except RuntimeError as exc:
            if "Zonos3 request:" in str(exc):
                raise InvalidModelRequestError(str(exc)) from exc
            raise

    def _generate_stream(
        self,
        text: str,
        voice: str,
        reference_path: Optional[Path],
        language: Optional[str],
        options: dict[str, Any],
    ) -> Generator[np.ndarray, None, None]:
        from sglang_omni.proto import StreamMessage

        runner, loop = self._runtime()
        request = self._make_request(text, voice, reference_path, options, stream=True)
        stream = runner.coordinator.stream(uuid.uuid4().hex, request)
        try:
            while (
                message := asyncio.run_coroutine_threadsafe(
                    anext(stream, None), loop
                ).result()
            ) is not None:
                if isinstance(message, StreamMessage) and message.modality == "audio":
                    yield self._waveform(message.chunk)
        except RuntimeError as exc:
            if "Zonos3 request:" in str(exc):
                raise InvalidModelRequestError(str(exc)) from exc
            raise
        finally:
            asyncio.run_coroutine_threadsafe(stream.aclose(), loop).result()
