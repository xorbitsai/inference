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

import itertools
import logging
import tempfile
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .core import AudioModelFamilyV1

logger = logging.getLogger(__name__)


class WhisperMLXModel:
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
        try:
            import mlx.core as mx
            from mlx_whisper.transcribe import ModelHolder
        except ImportError:
            error_message = "Failed to import module 'mlx_whisper'"
            installation_guide = [
                "Please make sure 'mlx_whisper' is installed.\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        self._model = ModelHolder.get_model(self._model_path, mx.float16)

    def transcriptions(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[str]] = None,
    ):
        return self._call(
            audio,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            task="transcribe",
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
        return self._call(
            audio,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            task="translate",
        )

    def _call(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[str]] = None,
        task: str = "transcribe",
    ):
        from mlx_whisper import transcribe

        with tempfile.NamedTemporaryFile(delete=True) as f:
            f.write(audio)

            kwargs = {"task": task}
            if response_format == "verbose_json":
                if timestamp_granularities == ["word"]:
                    kwargs["word_timestamps"] = True  # type: ignore

            result = transcribe(
                f.name,
                path_or_hf_repo=self._model_path,
                language=language,
                temperature=temperature,
                initial_prompt=prompt,
                **kwargs,
            )
            text = result["text"]
            segments = result["segments"]
            language = result["language"]

            if response_format == "json":
                return {"text": text}
            elif response_format == "verbose_json":
                if not timestamp_granularities or timestamp_granularities == [
                    "segment"
                ]:
                    return {
                        "task": task,
                        "language": language,
                        "duration": segments[-1]["end"] if segments else 0,
                        "text": text,
                        "segments": segments,
                    }
                else:
                    assert timestamp_granularities == ["word"]

                    def _extract_word(word: dict) -> dict:
                        return {
                            "start": word["start"].item(),
                            "end": word["end"].item(),
                            "word": word["word"],
                        }

                    words = [
                        _extract_word(w)
                        for w in itertools.chain(*[s["words"] for s in segments])
                    ]
                    return {
                        "task": task,
                        "language": language,
                        "duration": words[-1]["end"] if words else 0,
                        "text": text,
                        "words": words,
                    }
            else:
                raise ValueError(f"Unsupported response format: {response_format}")
