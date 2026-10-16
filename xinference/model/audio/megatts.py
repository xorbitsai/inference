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
import io
import logging
import math
from io import BytesIO
from typing import TYPE_CHECKING, Dict, Optional, Union

from ...core.exceptions import InvalidAudioInputError

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)

# Use conservative thresholds so that only effectively silent references are
# rejected. Digital silence is always rejected regardless of these values.
MEGATTS_MIN_PEAK_DBFS = -60.0
MEGATTS_MIN_RMS_DBFS = -70.0


class MegaTTSModel:
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "AudioModelFamilyV2",
        device: Optional[str] = None,
        **kwargs,
    ):
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._device = device
        self._model = None
        self._vocoder = None
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def load(self):
        import os
        import sys

        # The yaml config loaded from model has hard-coded the import paths. please refer to: load_hyperpyyaml
        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), "../../thirdparty/megatts3")
        )
        # For whisper
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../thirdparty"))

        from tts.infer_cli import MegaTTS3DiTInfer

        self._model = MegaTTS3DiTInfer(ckpt_root=self._model_path)

    def _validate_prompt_speech(
        self, prompt_speech: bytes
    ) -> Dict[str, Union[int, float]]:
        """Validate that a reference audio contains usable sound."""
        from pydub import AudioSegment

        try:
            audio_format = (
                "wav"
                if prompt_speech.startswith(b"RIFF") and prompt_speech[8:12] == b"WAVE"
                else None
            )
            audio = AudioSegment.from_file(BytesIO(prompt_speech), format=audio_format)
        except Exception as exc:
            logger.warning(
                "Rejecting MegaTTS3 prompt audio: model_uid=%s, bytes=%d, "
                "reason=decode_failed",
                self._model_uid,
                len(prompt_speech),
                exc_info=True,
            )
            raise InvalidAudioInputError(
                "Invalid prompt_speech: failed to decode reference audio."
            ) from exc

        duration_ms = len(audio)
        frame_rate = int(audio.frame_rate or 0)
        channels = int(audio.channels or 0)
        frame_count = int(audio.frame_count())
        rms = int(audio.rms)
        rms_dbfs = float(audio.dBFS)
        peak_dbfs = float(audio.max_dBFS)

        metrics: Dict[str, Union[int, float]] = {
            "bytes": len(prompt_speech),
            "duration_ms": duration_ms,
            "sample_rate": frame_rate,
            "channels": channels,
            "frames": frame_count,
            "rms": rms,
            "rms_dbfs": rms_dbfs,
            "peak_dbfs": peak_dbfs,
        }

        if duration_ms <= 0 or frame_count <= 0 or frame_rate <= 0 or channels <= 0:
            logger.warning(
                "Rejecting MegaTTS3 prompt audio: model_uid=%s, metrics=%s, "
                "reason=no_samples",
                self._model_uid,
                metrics,
            )
            raise InvalidAudioInputError(
                "Invalid prompt_speech: reference audio contains no samples."
            )

        digital_silence = (
            rms == 0 or not math.isfinite(rms_dbfs) or not math.isfinite(peak_dbfs)
        )
        near_silence = (
            peak_dbfs <= MEGATTS_MIN_PEAK_DBFS and rms_dbfs <= MEGATTS_MIN_RMS_DBFS
        )
        if digital_silence or near_silence:
            logger.warning(
                "Rejecting MegaTTS3 prompt audio: model_uid=%s, metrics=%s, "
                "reason=silent_audio",
                self._model_uid,
                metrics,
            )
            raise InvalidAudioInputError(
                "Invalid prompt_speech: no detectable audio was found in the "
                "reference audio."
            )

        logger.debug(
            "Validated MegaTTS3 prompt audio: model_uid=%s, metrics=%s",
            self._model_uid,
            metrics,
        )
        return metrics

    @staticmethod
    def _validate_output_audio(wav_bytes):
        import numpy as np

        if wav_bytes is None:
            raise RuntimeError("MegaTTS3 returned empty audio.")
        output = np.asarray(wav_bytes)
        if output.size == 0:
            raise RuntimeError("MegaTTS3 returned empty audio.")
        if not np.isfinite(output).all():
            raise RuntimeError("MegaTTS3 returned non-finite audio samples.")

    def speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ):
        from .utils import apply_audio_seed

        if stream:
            raise Exception("MegaTTS3 does not support stream generation.")
        apply_audio_seed(kwargs)
        if voice:
            raise Exception(
                "MegaTTS3 does not support voice, please specify prompt_speech and prompt_latent."
            )

        prompt_speech: Optional[bytes] = kwargs.pop("prompt_speech", None)
        prompt_latent: Optional[bytes] = kwargs.pop("prompt_latent", None)
        if not prompt_speech:
            raise InvalidAudioInputError(
                "Invalid prompt_speech: reference audio is empty."
            )
        if not prompt_latent:
            raise InvalidAudioInputError(
                "Invalid prompt_latent: reference latent is empty."
            )

        self._validate_prompt_speech(prompt_speech)

        assert self._model is not None
        with io.BytesIO(prompt_latent) as prompt_latent_io:
            resource_context = self._model.preprocess(
                prompt_speech, latent_file=prompt_latent_io
            )
        wav_bytes = self._model.forward(
            resource_context,
            input,
            time_step=kwargs.get("time_step", 32),
            p_w=kwargs.get("p_w", 1.6),
            t_w=kwargs.get("t_w", 2.5),
        )

        self._validate_output_audio(wav_bytes)

        # Save the generated audio
        import soundfile

        with BytesIO() as out:
            with soundfile.SoundFile(
                out, "w", self._model.sr, 1, format=response_format.upper()
            ) as f:
                f.write(wav_bytes)
            return out.getvalue()
