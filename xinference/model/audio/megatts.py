# Copyright 2022-2026 XProbe Inc.
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
from io import BytesIO
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


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

    def speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ):
        import soundfile

        if stream:
            raise Exception("MegaTTS3 does not support stream generation.")
        if voice:
            raise Exception(
                "MegaTTS3 does not support voice, please specify prompt_speech and prompt_latent."
            )

        prompt_speech: Optional[bytes] = kwargs.pop("prompt_speech", None)
        prompt_latent: Optional[bytes] = kwargs.pop("prompt_latent", None)
        if not prompt_speech:
            raise Exception("Please set prompt_speech for MegaTTS3.")
        if not prompt_latent:
            raise Exception("Please set prompt_latent for MegaTTS3.")

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

        # Save the generated audio
        with BytesIO() as out:
            with soundfile.SoundFile(
                out, "w", self._model.sr, 1, format=response_format.upper()
            ) as f:
                f.write(wav_bytes)
            return out.getvalue()
