# Copyright 2022-2023 XProbe Inc.
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
from typing import TYPE_CHECKING, Dict, Optional

from xinference.device_utils import (
    get_available_device,
    get_device_preferred_dtype,
    is_device_available,
)

if TYPE_CHECKING:
    from .core import AudioModelFamilyV1

logger = logging.getLogger(__name__)


class WhisperModel:
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "AudioModelFamilyV1",
        device: Optional[str] = None,
        **kwargs,
    ):
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._device = device
        self._model = None
        self._kwargs = kwargs

    def load(self):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        if self._device is None:
            self._device = get_available_device()
        else:
            if not is_device_available(self._device):
                raise ValueError(f"Device {self._device} is not available!")

        torch_dtype = get_device_preferred_dtype(self._device)

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self._device)

        processor = AutoProcessor.from_pretrained(self._model_path)

        self._model = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=False,
            torch_dtype=torch_dtype,
            device=self._device,
        )

    def _call_model(
        self,
        audio: bytes,
        generate_kwargs: Dict,
        response_format: str,
    ):
        if response_format == "json":
            logger.debug("Call whisper model with generate_kwargs: %s", generate_kwargs)
            assert callable(self._model)
            result = self._model(audio, generate_kwargs=generate_kwargs)
            return {"text": result["text"]}
        else:
            raise ValueError(f"Unsupported response format: {response_format}")

    def transcriptions(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
    ):
        if temperature != 0:
            logger.warning(
                "Temperature for whisper transcriptions will be ignored: %s.",
                temperature,
            )
        if prompt is not None:
            logger.warning(
                "Prompt for whisper transcriptions will be ignored: %s", prompt
            )
        return self._call_model(
            audio=audio,
            generate_kwargs=(
                {"language": language, "task": "transcribe"}
                if language is not None
                else {"task": "transcribe"}
            ),
            response_format=response_format,
        )

    def translations(
        self,
        audio: bytes,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
    ):
        if not self._model_spec.multilingual:
            raise RuntimeError(
                f"Model {self._model_spec.model_name} is not suitable for translations."
            )
        if temperature != 0:
            logger.warning(
                "Temperature for whisper transcriptions will be ignored: %s.",
                temperature,
            )
        if prompt is not None:
            logger.warning(
                "Prompt for whisper transcriptions will be ignored: %s", prompt
            )
        return self._call_model(
            audio=audio,
            generate_kwargs={"task": "translate"},
            response_format=response_format,
        )
