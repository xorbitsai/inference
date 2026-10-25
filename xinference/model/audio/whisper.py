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
import typing
from glob import glob
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from typing_extensions import TypedDict

from ...device_utils import (
    get_available_device,
    get_device_preferred_dtype,
    is_device_available,
)
from ...utils import make_hashable
from .batch import AudioRequestBatcher

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


class WhisperModelConfig(TypedDict, total=False):
    chunk_length_s: Optional[float]
    stride_length_s: Optional[float]
    return_timestamps: Optional[bool]
    batch_size: Optional[int]


class WhisperModel:
    allow_batch = True

    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "AudioModelFamilyV2",
        device: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ):
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._device = device
        self._model = None
        self._max_new_tokens = max_new_tokens
        self._batcher = AudioRequestBatcher(
            self._run_transcription_batch,
            lambda request: make_hashable(request["pipeline_kwargs"]),
            request_batch_size=int(kwargs.pop("request_batch_size", 4)),
            batch_interval=float(kwargs.pop("batch_interval", 0.01)),
        )
        self._model_config: WhisperModelConfig = self._sanitize_model_config(
            typing.cast(WhisperModelConfig, kwargs)
        )

    def _sanitize_model_config(
        self, model_config: Optional[WhisperModelConfig]
    ) -> WhisperModelConfig:
        if model_config is None:
            model_config = WhisperModelConfig()
        model_config.setdefault("chunk_length_s", 30)
        model_config.setdefault("stride_length_s", None)
        model_config.setdefault("return_timestamps", False)
        model_config.setdefault("batch_size", 16)
        return model_config

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
            chunk_length_s=self._model_config.get("chunk_length_s"),
            stride_length_s=self._model_config.get("stride_length_s"),
            return_timestamps=self._model_config.get("return_timestamps"),
            batch_size=self._model_config.get("batch_size"),
            torch_dtype=torch_dtype,
            device=self._device,
        )

    @staticmethod
    def _pipeline_kwargs(
        generate_kwargs: Dict,
        response_format: str,
        temperature: float,
        timestamp_granularities: Optional[List[str]],
    ) -> dict:
        generate_kwargs = dict(generate_kwargs)
        if temperature != 0:
            generate_kwargs.update(temperature=temperature, do_sample=True)
        pipeline_kwargs: dict = {"generate_kwargs": generate_kwargs}
        if response_format == "verbose_json":
            if not timestamp_granularities or timestamp_granularities == ["segment"]:
                pipeline_kwargs["return_timestamps"] = True
            elif timestamp_granularities == ["word"]:
                pipeline_kwargs["return_timestamps"] = "word"
            else:
                raise Exception(
                    f"Unsupported timestamp_granularities: {timestamp_granularities}"
                )
        elif response_format != "json":
            raise ValueError(f"Unsupported response format: {response_format}")
        return pipeline_kwargs

    def _call_model(
        self,
        audio: bytes,
        generate_kwargs: Dict,
        response_format: str,
        temperature: float = 0,
        timestamp_granularities: Optional[List[str]] = None,
    ):
        pipeline_kwargs = self._pipeline_kwargs(
            generate_kwargs, response_format, temperature, timestamp_granularities
        )
        assert callable(self._model)
        result = self._model(audio, **pipeline_kwargs)
        return self._format_result(
            result, response_format, temperature, pipeline_kwargs
        )

    @staticmethod
    def _format_result(results, response_format, temperature, pipeline_kwargs):
        if response_format == "json":
            return {"text": results["text"]}
        generate_kwargs = pipeline_kwargs["generate_kwargs"]
        return_timestamps = pipeline_kwargs["return_timestamps"]
        language = generate_kwargs.get("language", "english")
        task = generate_kwargs.get("task", "transcribe")

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
                "task": task,
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
                "task": task,
                "language": language,
                "duration": words[-1]["end"] if words else 0,
                "text": results["text"],
                "words": words,
            }

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
        generate_kwargs: Dict = {"task": "transcribe"}
        # Let Whisper's generation config account for the decoder prefix by default.
        if self._max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = self._max_new_tokens
        if language is not None:
            generate_kwargs["language"] = language

        return self._call_model(
            audio=audio,
            generate_kwargs=generate_kwargs,
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

    def _decode_audio(self, audio: bytes):
        from transformers.pipelines.audio_utils import ffmpeg_read

        assert self._model is not None
        # Use the same decoder and target rate as the native pipeline, before
        # its list preprocessing can let one malformed upload abort its peers.
        return ffmpeg_read(audio, self._model.feature_extractor.sampling_rate)

    def _run_transcription_batch(
        self, requests: List[dict]
    ) -> List[Union[dict, Exception]]:
        assert callable(self._model)
        first = requests[0]
        outputs: List[Union[dict, Exception]] = [{} for _ in requests]
        audios = []
        indices = []
        for index, request in enumerate(requests):
            try:
                audio = request["audio"]
                if len(requests) > 1 and isinstance(audio, bytes):
                    audio = self._decode_audio(audio)
                audios.append(audio)
                indices.append(index)
            except Exception as error:
                outputs[index] = error
        if not audios:
            return outputs
        results = self._model(
            audios if len(audios) > 1 else audios[0], **first["pipeline_kwargs"]
        )
        if len(audios) == 1:
            results = [results]
        if not isinstance(results, list) or len(results) != len(indices):
            raise RuntimeError("Whisper returned an unexpected batch result count")
        for index, result in zip(indices, results):
            request = requests[index]
            outputs[index] = self._format_result(
                result,
                request["response_format"],
                request["temperature"],
                request["pipeline_kwargs"],
            )
        return outputs

    async def _async_transcription(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[str]] = None,
        *,
        task: str = "transcribe",
    ):
        if task == "translate" and not self._model_spec.multilingual:
            raise RuntimeError(
                f"Model {self._model_spec.model_name} is not suitable for translations."
            )
        if prompt is not None:
            logger.warning(
                "Prompt for whisper transcriptions will be ignored: %s", prompt
            )
        generate_kwargs: Dict = {"task": task}
        if task == "transcribe" and self._max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = self._max_new_tokens
        if language is not None:
            generate_kwargs["language"] = language
        pipeline_kwargs = self._pipeline_kwargs(
            generate_kwargs, response_format, temperature, timestamp_granularities
        )
        return await self._batcher.submit(
            dict(
                audio=audio,
                response_format=response_format,
                temperature=temperature,
                pipeline_kwargs=pipeline_kwargs,
            )
        )

    async def async_transcriptions(self, *args, **kwargs):
        return await self._async_transcription(*args, **kwargs)

    async def async_translations(self, *args, **kwargs):
        return await self._async_transcription(*args, task="translate", **kwargs)
