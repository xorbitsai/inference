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
import os
from typing import TYPE_CHECKING, Optional

from ..utils import is_flash_attn_available

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


class Qwen3TTSModel:
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
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            error_message = "Failed to import module 'qwen-tts'"
            installation_guide = [
                "Please make sure 'qwen-tts' is installed. ",
                "You can install it by `pip install qwen-tts`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        if is_flash_attn_available():
            self._model = Qwen3TTSModel.from_pretrained(
                self._model_path,
                device_map=self._device if self._device else "cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        else:
            self._model = Qwen3TTSModel.from_pretrained(
                self._model_path,
                device_map=self._device if self._device else "cuda:0",
                dtype=torch.bfloat16,
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
        assert self._model is not None

        import soundfile

        if stream:
            raise Exception("qwen-tts does not support stream generation.")

        prompt_speech: Optional[bytes] = kwargs.pop("prompt_speech", None)
        prompt_text: Optional[str] = kwargs.pop("prompt_text", None)

        language = kwargs.pop("language", "chinese")
        language_list = self._model.get_supported_languages()
        if language not in language_list:
            raise ValueError(
                f"Language '{language}' is not supported. Supported languages are: {', '.join(language_list)}"
            )
        instruct = kwargs.pop("instruct", None)
        if prompt_speech is None:
            if self.model_family.model_name.endswith("VoiceDesign"):
                audio, sample_rate = self._model.generate_voice_design(
                    text=input, language=language, instruct=instruct, **kwargs
                )
            elif self.model_family.model_name.endswith("CustomVoice"):
                speaker_list = self._model.get_supported_speakers()
                if voice and voice not in speaker_list:
                    raise ValueError(
                        f"voice '{voice}' is not supported for voice '{voice}'. Supported speakers are: {', '.join(speaker_list)}"
                    )
                audio, sample_rate = self._model.generate_custom_voice(
                    text=input,
                    language=language,
                    speaker=voice,
                    instruct=instruct,
                )
            else:
                raise RuntimeError(
                    "Base does not support generate_voice_design and generate_custom_voice, Please add prompt speech and prompt text for voice cloning."
                )

            # Non-streaming mode - return bytes directly
            with io.BytesIO() as out:
                with soundfile.SoundFile(
                    out, "w", sample_rate, 1, format=response_format.upper()
                ) as f:
                    f.write(audio[0])
                result = out.getvalue()

            return result
        else:
            if prompt_text is None:
                raise RuntimeError(
                    "prompt_text is required when prompt_speech is provided."
                )

            # Save prompt speech to temp file
            import tempfile

            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as temp_prompt:
                temp_prompt.write(prompt_speech)
                temp_prompt_path = temp_prompt.name

            try:
                audio, sample_rate = self._model.generate_voice_clone(
                    text=input,
                    language=language,
                    ref_audio=temp_prompt_path,
                    ref_text=prompt_text,
                )

                # Non-streaming mode - return bytes directly
                with io.BytesIO() as out:
                    with soundfile.SoundFile(
                        out, "w", sample_rate, 1, format=response_format.upper()
                    ) as f:
                        f.write(audio[0])
                    result = out.getvalue()

                return result
            finally:
                # Clean up temp files
                try:
                    os.unlink(temp_prompt_path)
                except Exception:
                    pass
