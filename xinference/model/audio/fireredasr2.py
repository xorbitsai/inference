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

import logging
import os
import subprocess
from contextlib import nullcontext
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from ...core.exceptions import InvalidAudioInputError, InvalidModelRequestError
from ..utils import allow_trust_remote_code

if TYPE_CHECKING:
    from ...thirdparty.fireredasr2 import FireRedAsr2
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_VARIANTS = {"FireRedASR2-AED": ("aed", 60), "FireRedASR2-LLM": ("llm", 40)}
_FFMPEG_REQUIRED = (
    "FireRedASR2 requires an imageio-ffmpeg wheel with a bundled FFmpeg executable."
)


def _get_ffmpeg_executable() -> str:
    try:
        binaries = files("imageio_ffmpeg.binaries")
        executables = [
            entry
            for entry in binaries.iterdir()
            if entry.name.startswith("ffmpeg-") and entry.is_file()
        ]
    except (ModuleNotFoundError, FileNotFoundError) as exc:
        raise RuntimeError(_FFMPEG_REQUIRED) from exc
    if len(executables) != 1:
        raise RuntimeError(_FFMPEG_REQUIRED)
    executable = Path(str(executables[0])).resolve()
    if not os.access(executable, os.X_OK):
        raise RuntimeError(_FFMPEG_REQUIRED)
    return str(executable)


class FireRedASR2Model:
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "AudioModelFamilyV2",
        device: Optional[str] = None,
        use_half: bool = False,
        **kwargs,
    ):
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._use_half = use_half
        self._model: Optional["FireRedAsr2"] = None
        self._asr_type, self._max_seconds = _VARIANTS[model_spec.model_name]

    @property
    def model_ability(self):
        return self.model_family.model_ability

    def load(self):
        import torch

        # The upstream loader deserializes full PyTorch checkpoints.
        if not allow_trust_remote_code(self.model_family):
            raise ValueError(
                "FireRedASR2 loads pickled PyTorch checkpoints. For a custom model "
                "or explicit model_path, set XINFERENCE_TRUST_REMOTE_CODE=1 only "
                "if you trust the checkpoint source."
            )
        executable = _get_ffmpeg_executable()
        try:
            subprocess.run(
                [executable, "-version"], capture_output=True, check=True, timeout=60
            )
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            raise RuntimeError(
                "FireRedASR2 could not run the bundled FFmpeg executable."
            ) from exc
        device = torch.device(
            self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("FireRedASR2 supports CPU and CUDA devices.")
        use_gpu = device.type == "cuda"
        if use_gpu and not torch.cuda.is_available():
            raise ValueError("FireRedASR2 requires an available CUDA device.")
        self._device = str(device)

        from ...thirdparty.fireredasr2 import FireRedAsr2, FireRedAsr2Config

        config = FireRedAsr2Config(use_gpu=use_gpu, use_half=use_gpu and self._use_half)
        if self._asr_type == "llm":
            config.repetition_penalty = 3.0
            config.llm_length_penalty = 1.0
        with torch.cuda.device(device) if use_gpu else nullcontext():
            self._model = FireRedAsr2.from_pretrained(
                self._asr_type, self._model_path, config
            )

    def _decode_audio(self, audio: bytes) -> np.ndarray:
        if not audio:
            raise InvalidAudioInputError("Invalid audio file: audio is empty.")
        # Decode slightly beyond the limit to detect oversized uploads without
        # buffering the entire decoded waveform.
        try:
            result = subprocess.run(
                [
                    _get_ffmpeg_executable(),
                    "-nostdin",
                    "-v",
                    "error",
                    "-protocol_whitelist",
                    "pipe",
                    "-i",
                    "pipe:0",
                    "-map",
                    "0:a:0",
                    "-af",
                    "asetpts=N/SR/TB",
                    "-t",
                    str(self._max_seconds + 1),
                    "-ar",
                    str(_SAMPLE_RATE),
                    "-ac",
                    "1",
                    "-f",
                    "s16le",
                    "pipe:1",
                ],
                input=audio,
                capture_output=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(_FFMPEG_REQUIRED) from exc
        except subprocess.CalledProcessError as exc:
            raise InvalidAudioInputError(
                "Invalid audio file: unable to decode audio."
            ) from exc

        samples = np.frombuffer(result.stdout, dtype="<i2")
        if samples.size > self._max_seconds * _SAMPLE_RATE:
            raise InvalidAudioInputError(
                f"{self.model_family.model_name} supports audio up to "
                f"{self._max_seconds} seconds per request."
            )
        # Kaldi's 25 ms feature window requires at least 400 samples.
        if samples.size < 400:
            raise InvalidAudioInputError(
                "Invalid audio file: audio is shorter than 25 ms."
            )
        return samples

    def transcriptions(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[str]] = None,
        **kwargs,
    ) -> dict:
        import torch

        if response_format not in {"json", "verbose_json"}:
            raise InvalidModelRequestError(
                f"Unsupported response format: {response_format}"
            )
        if timestamp_granularities or temperature != 0 or kwargs:
            raise InvalidModelRequestError(
                "FireRedASR2 does not support timestamp granularities, nonzero "
                "temperature, or extra transcription options."
            )
        if prompt:
            logger.warning("FireRedASR2 ignores transcription prompts.")
        if self._model is None or self._device is None:
            raise RuntimeError("FireRedASR2 model is not loaded.")
        samples = self._decode_audio(audio)
        device = torch.device(self._device)
        # Upstream always transfers inputs to the current CUDA device.
        with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
            results = self._model.transcribe(["audio"], [(_SAMPLE_RATE, samples)])
        if (
            not results
            or not isinstance(results[0], dict)
            or not isinstance(results[0].get("text"), str)
        ):
            raise RuntimeError("FireRedASR2 failed to return a transcription.")
        text = results[0]["text"]
        if response_format == "json":
            return {"text": text}
        return {
            "task": "transcribe",
            "text": text,
            "duration": samples.size / _SAMPLE_RATE,
            "words": [],
            "segments": [],
        }

    def translations(self, *args, **kwargs):
        raise InvalidModelRequestError("FireRedASR2 does not support translations.")
