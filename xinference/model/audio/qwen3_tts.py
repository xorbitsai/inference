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
import os
import tempfile
from typing import TYPE_CHECKING, List, Optional, Union

from ...utils import make_hashable
from ..utils import is_flash_attn_available, resolve_media_seed
from .batch import AudioRequestBatcher

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


class Qwen3TTSModel:
    allow_batch = True

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
        self._batcher = AudioRequestBatcher(
            self._run_speech_batch,
            self._speech_batch_key,
            request_batch_size=int(kwargs.pop("request_batch_size", 4)),
            batch_interval=float(kwargs.pop("batch_interval", 0.01)),
        )

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

    def _prepare_speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ) -> dict:
        assert self._model is not None
        if stream:
            raise Exception("qwen-tts does not support stream generation.")
        seed = resolve_media_seed(kwargs.pop("seed", None))
        prompt_speech = kwargs.pop("prompt_speech", None)
        prompt_text = kwargs.pop("prompt_text", None)
        language = kwargs.pop("language", "chinese")
        language_list = self._model.get_supported_languages()
        if language not in language_list:
            raise ValueError(
                f"Language '{language}' is not supported. Supported languages are: {', '.join(language_list)}"
            )
        instruct = kwargs.pop("instruct", None)
        if prompt_speech is not None:
            if prompt_text is None:
                raise RuntimeError(
                    "prompt_text is required when prompt_speech is provided."
                )
            method = "generate_voice_clone"
        elif self.model_family.model_name.endswith("VoiceDesign"):
            method = "generate_voice_design"
        elif self.model_family.model_name.endswith("CustomVoice"):
            speaker_list = self._model.get_supported_speakers()
            if voice and voice not in speaker_list:
                raise ValueError(
                    f"voice '{voice}' is not supported. Supported speakers are: {', '.join(speaker_list)}"
                )
            method = "generate_custom_voice"
        else:
            raise RuntimeError(
                "Base does not support generate_voice_design and generate_custom_voice, Please add prompt speech and prompt text for voice cloning."
            )
        return dict(
            text=input,
            voice=voice,
            response_format=response_format,
            language=language,
            instruct=instruct,
            method=method,
            prompt_speech=prompt_speech,
            prompt_text=prompt_text,
            seed=seed,
            # Preserve the existing per-variant generation options.
            kwargs=kwargs if method == "generate_voice_design" else {},
        )

    @staticmethod
    def _speech_batch_key(request: dict):
        # Native generation shares an RNG across rows. Preserve explicit seeds
        # by executing these requests individually on the same batch worker.
        if request["seed"] is not None:
            return object()
        return make_hashable(
            (
                request["method"],
                request["prompt_speech"],
                request["prompt_text"],
                request["kwargs"],
            )
        )

    @staticmethod
    def _encode_audio(audio, sample_rate: int, response_format: str) -> bytes:
        import soundfile

        with io.BytesIO() as out:
            with soundfile.SoundFile(
                out, "w", sample_rate, 1, format=response_format.upper()
            ) as audio_file:
                audio_file.write(audio)
            return out.getvalue()

    def _run_speech_batch(self, requests: List[dict]) -> List[Union[bytes, Exception]]:
        assert self._model is not None
        from .utils import apply_audio_seed

        first = requests[0]
        apply_audio_seed({"seed": first["seed"]})
        batched = len(requests) > 1

        def values(name):
            items = [request[name] for request in requests]
            return items if batched else items[0]

        native_kwargs = dict(first["kwargs"])
        native_kwargs.update(text=values("text"), language=values("language"))
        method = first["method"]
        if method == "generate_custom_voice":
            native_kwargs.update(
                speaker=values("voice"),
                instruct=(
                    [request["instruct"] or "" for request in requests]
                    if batched
                    else first["instruct"]
                ),
            )
        elif method == "generate_voice_design":
            native_kwargs["instruct"] = values("instruct")

        with tempfile.TemporaryDirectory(prefix="xinference-qwen3-tts-") as temp_dir:
            if method == "generate_voice_clone":
                # Compatible requests share one reference, so native Qwen only
                # needs to extract its prompt features once for the entire batch.
                prompt_path = os.path.join(temp_dir, "reference.wav")
                with open(prompt_path, "wb") as prompt_file:
                    prompt_file.write(first["prompt_speech"])
                native_kwargs.update(
                    ref_audio=prompt_path, ref_text=first["prompt_text"]
                )
            audios, sample_rate = getattr(self._model, method)(**native_kwargs)
        if len(audios) != len(requests):
            raise RuntimeError("Qwen3-TTS returned an unexpected result count")
        outputs: List[Union[bytes, Exception]] = []
        for audio, request in zip(audios, requests):
            try:
                outputs.append(
                    self._encode_audio(audio, sample_rate, request["response_format"])
                )
            except Exception as error:
                if not batched:
                    raise
                outputs.append(error)
        return outputs

    def speech(self, *args, **kwargs):
        return self._run_speech_batch([self._prepare_speech(*args, **kwargs)])[0]

    async def async_speech(self, *args, **kwargs):
        request = self._prepare_speech(*args, **kwargs)
        return await self._batcher.submit(request)
