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

"""Native MLX MiniMax-Music3 with the existing text-to-music API."""

from typing import Any, Optional

from .minimax_music3 import MiniMaxMusic3Model
from .utils import MLXModelThreadMixin


class MLXMiniMaxMusic3Model(MLXModelThreadMixin, MiniMaxMusic3Model):
    def load(self):
        self._run_on_mlx_thread(self._load)

    def _load(self):
        try:
            from mlx_audio.music import load
        except ImportError as exc:
            raise ImportError("MiniMax-Music3 MLX requires mlx-audio==0.5.0.") from exc
        self._model = load(self._model_path, **self._kwargs)

    def speech(
        self,
        input: str,
        voice: Optional[str] = None,
        response_format: Optional[str] = "wav",
        speed: Optional[float] = 1.0,
        stream: Optional[bool] = False,
        **kwargs: Any,
    ) -> bytes:
        return self._run_on_mlx_thread(
            self._speech, input, voice, response_format, speed, stream, **kwargs
        )

    def _speech(self, input, voice, response_format, speed, stream, **kwargs) -> bytes:
        import numpy as np

        assert self._model is not None
        instruct = kwargs.pop("instruct", None)
        seed = kwargs.pop("seed", 0)
        duration = kwargs.pop("duration", 60.0)
        steps = kwargs.pop("steps", 30)
        response_format = self._validate_speech_request(
            input,
            instruct,
            voice,
            response_format,
            speed,
            stream,
            seed,
            duration,
            kwargs,
        )
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or not 1 <= steps <= 30
        ):
            raise ValueError("MiniMax-Music3 `steps` must be an integer from 1 to 30.")
        chunks = []
        sample_rate = None
        for result in self._model.generate(
            text=instruct,
            lyrics=input,
            duration=float(duration),
            steps=steps,
            seed=seed,
        ):
            audio = np.asarray(result.audio, dtype=np.float32)
            # mlx-audio returns sample-major stereo. Do not flatten channels or
            # infer orientation from a short result with exactly two samples.
            if audio.ndim != 2 or audio.shape[1] != 2 or not np.isfinite(audio).all():
                raise RuntimeError("MiniMax-Music3 MLX returned invalid stereo audio.")
            if result.sample_rate != 44100:
                raise RuntimeError(
                    "MiniMax-Music3 MLX expected a 44100 Hz sample rate."
                )
            sample_rate = int(result.sample_rate)
            chunks.append(audio)
        if sample_rate is None or not chunks or not any(len(chunk) for chunk in chunks):
            raise RuntimeError("MiniMax-Music3 MLX returned no generated audio.")
        # The shared encoder accepts channel-major audio; transpose explicitly.
        return self._audio_to_bytes(
            np.concatenate(chunks, axis=0).T, sample_rate, response_format
        )
