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
import uuid
from typing import Any, Optional

from ...core.exceptions import InvalidModelRequestError
from .audio8_tts import Audio8TTSModel, _check_transformers_version


class SGLangOmniAudio8TTSModel(Audio8TTSModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loop = None
        self._thread = None
        self._runner = None

    def load(self):
        self._check_model_trust(self.model_family)

        import torch

        _check_transformers_version()
        self._device = self._device or "cuda:0"
        if not self._device.startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("Audio8-TTS SGLang-Omni requires an NVIDIA CUDA device.")
        if self._kwargs:
            raise ValueError(
                "Configure Audio8-TTS SGLang-Omni with the upstream AUDIO8_TTS_* "
                f"environment variables; unsupported launch arguments: {sorted(self._kwargs)}"
            )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        try:
            asyncio.run_coroutine_threadsafe(self._start(), self._loop).result()
        except BaseException:
            self.stop()
            raise

    def _run_loop(self):
        loop = self._loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _start(self):
        import torch
        from sglang_omni.config import build_pipeline_runner

        from ...thirdparty.audio8_tts.config import Audio8TTSPipelineConfig

        torch.cuda.set_device(self._device)
        config = Audio8TTSPipelineConfig(model_path=self._model_path)
        device_index = torch.device(self._device).index or 0
        for stage in config.stages:
            stage.executor.args["device"] = self._device
            if stage.name == "tts_engine":
                stage.executor.args["stream_enabled"] = False
            if stage.relay.device == "cuda":
                stage.relay.device = f"cuda:{device_index}"
            # Xinference exposes complete encoded audio, not Omni's SSE chunks.
            stage.stream_to = []
        config.gpu_placement = {stage.name: device_index for stage in config.stages}
        self._runner = build_pipeline_runner(config)
        await self._runner.start()

    def _generation_options(self, kwargs: dict) -> dict:
        if kwargs.get("seed") is not None and kwargs["seed"] != -1:
            raise InvalidModelRequestError(
                "Audio8-TTS SGLang-Omni does not support seed."
            )
        kwargs.pop("seed", None)
        options = super()._generation_options(kwargs)
        if not options.pop("do_sample"):
            options["temperature"] = 0.0
        return options

    async def _submit(self, request_id: str, request):
        try:
            return await self._runner.coordinator.submit(request_id, request)
        except BaseException:
            await self._runner.coordinator.abort(request_id)
            raise

    def _generate(
        self,
        text: str,
        reference: Optional[str],
        transcript: Optional[str],
        options: dict,
    ):
        import numpy as np
        import torch
        from sglang_omni.proto import OmniRequest

        if self._runner is None or self._loop is None:
            raise RuntimeError("Audio8-TTS SGLang-Omni model is not loaded.")
        inputs: dict[str, Any] = {"text": text}
        if reference is not None:
            inputs["references"] = [{"audio_path": reference, "text": transcript}]
        request = OmniRequest(
            inputs=inputs,
            params=options,
            metadata={"tts_params": {"response_format": "wav"}},
        )
        result = asyncio.run_coroutine_threadsafe(
            self._submit(str(uuid.uuid4()), request), self._loop
        ).result()
        audio = np.frombuffer(result["audio_waveform"], dtype="float32").copy()
        return torch.from_numpy(audio).unsqueeze(0), int(result["sample_rate"])

    def stop(self):
        loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return
        try:
            if self._runner is not None:
                future = asyncio.run_coroutine_threadsafe(self._runner.stop(), loop)
                try:
                    future.result(timeout=30)
                except TimeoutError:
                    future.cancel()
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
            self._runner = self._loop = self._thread = None
