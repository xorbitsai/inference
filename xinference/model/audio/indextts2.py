# Copyright 2022-2025 XProbe Inc.
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
import sys
from typing import TYPE_CHECKING, Optional

from ..utils import set_all_random_seed

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


class Indextts2:
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
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def load(self):
        # The yaml config loaded from model has hard-coded the import paths
        thirdparty_dir = os.path.join(os.path.dirname(__file__), "../../thirdparty")
        sys.path.insert(0, thirdparty_dir)

        from indextts.infer_v2 import IndexTTS2

        config_path = os.path.join(self._model_path, "config.yaml")
        use_fp16 = self._kwargs.get("use_fp16", False)
        use_deepspeed = self._kwargs.get("use_deepspeed", False)

        logger.info("Loading IndexTTS2 model...")
        self._model = IndexTTS2(
            cfg_path=config_path,
            model_dir=self._model_path,
            use_fp16=use_fp16,
            device=self._device,
            use_deepspeed=use_deepspeed,
        )

    def speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ):
        from io import BytesIO

        import soundfile

        if stream:
            raise Exception("IndexTTS2 does not support stream generation.")

        prompt_speech: Optional[bytes] = kwargs.pop("prompt_speech", None)
        emo_prompt_speech: Optional[bytes] = kwargs.pop("emo_prompt_speech", None)
        emo_alpha: float = kwargs.pop("emo_alpha", 1.0)
        emo_text: Optional[str] = kwargs.pop("emo_text", None)
        use_random: bool = kwargs.pop("use_random", False)
        emo_vector: Optional[list] = kwargs.pop("emo_vector", None)
        seed: Optional[int] = kwargs.pop("seed", 0)
        use_emo_text: bool = kwargs.pop("use_emo_text", False)

        if prompt_speech is None:
            # IndexTTS2 requires reference audio for voice cloning
            # We'll provide a helpful error message with usage examples
            raise ValueError(
                "IndexTTS2 requires a reference audio for voice cloning.\n"
                "Please provide a short audio sample (3-10 seconds) as 'prompt_speech' parameter.\n"
                "Example usage:\n"
                "  with open('reference.wav', 'rb') as f:\n"
                "      prompt_speech = f.read()\n"
                "  audio_bytes = model.speech(\n"
                "      input='Hello, world!',\n"
                "      voice='default',\n"
                "      prompt_speech=prompt_speech"
                "  )\n\n"
                "For emotion control, you can also add:\n"
                "  emo_prompt_speech=emotion_audio_bytes  # Optional: emotion reference\n"
                "  emo_text='happy and cheerful'  # Optional: emotion description\n"
                "  emo_alpha=1.5  # Optional: emotion intensity"
            )

        assert self._model is not None

        set_all_random_seed(seed)

        # Save prompt speech to temp file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_prompt:
            temp_prompt.write(prompt_speech)
            temp_prompt_path = temp_prompt.name

        emo_prompt_path = None
        if emo_prompt_speech is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_emo:
                temp_emo.write(emo_prompt_speech)
                emo_prompt_path = temp_emo.name

        try:
            # Generate audio
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as temp_output:
                output_path = temp_output.name

            self._model.infer(
                spk_audio_prompt=temp_prompt_path,
                text=input,
                output_path=output_path,
                emo_audio_prompt=emo_prompt_path,
                emo_alpha=emo_alpha,
                emo_text=emo_text,
                use_random=use_random,
                emo_vector=emo_vector,
                use_emo_text=use_emo_text,
            )

            # Read generated audio and convert to requested format
            audio, sample_rate = soundfile.read(output_path)

            with BytesIO() as out:
                with soundfile.SoundFile(
                    out, "w", sample_rate, 1, format=response_format.upper()
                ) as f:
                    f.write(audio)
                return out.getvalue()
        finally:
            # Clean up temp files
            try:
                os.unlink(temp_prompt_path)
                os.unlink(output_path)
                if emo_prompt_path:
                    os.unlink(emo_prompt_path)
            except:
                pass
