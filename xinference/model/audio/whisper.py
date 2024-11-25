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
import os
from glob import glob
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from ...device_utils import (
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

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def load(self):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        if self._device is None:
            self._device = get_available_device()
        else:
            if not is_device_available(self._device):
                raise ValueError(f"Device {self._device} is not available!")

        torch_dtype = get_device_preferred_dtype(self._device)
        use_safetensors = any(glob(os.path.join(self._model_path, "*.safetensors")))

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=use_safetensors,
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
        temperature: float = 0,
        timestamp_granularities: Optional[List[str]] = None,
    ):
        if temperature != 0:
            generate_kwargs.update({"temperature": temperature, "do_sample": True})

        if response_format == "json":
            logger.debug("Call whisper model with generate_kwargs: %s", generate_kwargs)
            assert callable(self._model)
            result = self._model(audio, generate_kwargs=generate_kwargs)
            return {"text": result["text"]}
        elif response_format == "verbose_json":
            return_timestamps: Union[bool, str] = False
            if not timestamp_granularities:
                return_timestamps = True
            elif timestamp_granularities == ["segment"]:
                return_timestamps = True
            elif timestamp_granularities == ["word"]:
                return_timestamps = "word"
            else:
                raise Exception(
                    f"Unsupported timestamp_granularities: {timestamp_granularities}"
                )
            assert callable(self._model)
            results = self._model(
                audio,
                generate_kwargs=generate_kwargs,
                return_timestamps=return_timestamps,
            )

            language = generate_kwargs.get("language", "english")

            if return_timestamps is True:
                segments: List[dict] = []

                def _get_chunk_segment_json(idx, text, start, end):
                    find_start = 0
                    if segments:
                        find_start = segments[-1]["seek"] + len(segments[-1]["text"])
                    return {
                        "id": idx,
                        "seek": results["text"].find(text, find_start),
                        "start": start,
                        "end": end,
                        "text": text,
                        "tokens": [],
                        "temperature": temperature,
                        # We can't provide these values.
                        "avg_logprob": 0.0,
                        "compression_ratio": 0.0,
                        "no_speech_prob": 0.0,
                    }

                for idx, c in enumerate(results.get("chunks", [])):
                    text = c["text"]
                    start, end = c["timestamp"]
                    segments.append(_get_chunk_segment_json(idx, text, start, end))

                return {
                    "task": "transcribe",
                    "language": language,
                    "duration": segments[-1]["end"] if segments else 0,
                    "text": results["text"],
                    "segments": segments,
                }
            else:
                assert return_timestamps == "word"

                words = []
                for idx, c in enumerate(results.get("chunks", [])):
                    text = c["text"]
                    start, end = c["timestamp"]
                    words.append({"word": text, "start": start, "end": end})

                return {
                    "task": "transcribe",
                    "language": language,
                    "duration": words[-1]["end"] if words else 0,
                    "text": results["text"],
                    "words": words,
                }
        else:
            raise ValueError(f"Unsupported response format: {response_format}")

    def transcriptions(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[str]] = None,
    ):
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
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
        )

    def translations(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[str]] = None,
    ):
        if not self._model_spec.multilingual:
            raise RuntimeError(
                f"Model {self._model_spec.model_name} is not suitable for translations."
            )
        if prompt is not None:
            logger.warning(
                "Prompt for whisper transcriptions will be ignored: %s", prompt
            )
        return self._call_model(
            audio=audio,
            generate_kwargs=(
                {"language": language, "task": "translate"}
                if language is not None
                else {"task": "translate"}
            ),
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
        )
