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

import logging
import tempfile
from typing import TYPE_CHECKING, List, Optional

from ...device_utils import get_available_device, is_device_available

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


class FunASRModel:
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

    def convert_to_openai_format(self, input_data):
        if "timestamp" not in input_data:
            return {"task": "transcribe", "text": input_data["text"]}
        start_time = input_data["timestamp"][0][0] / 1000
        end_time = input_data["timestamp"][-1][1] / 1000
        duration = end_time - start_time
        word_timestamps = []
        for ts in input_data["timestamp"]:
            word_timestamps.append({"start": ts[0] / 1000, "end": ts[1] / 1000})
        if "sentence_info" not in input_data:
            return {
                "task": "transcribe",
                "text": input_data["text"],
                "words": word_timestamps,
                "duration": duration,
            }
        output = {
            "task": "transcribe",
            "duration": duration,
            "text": input_data["text"],
            "words": word_timestamps,
            "segments": [],
        }
        for sentence in input_data["sentence_info"]:
            seg_start = sentence["start"] / 1000
            seg_end = sentence["end"] / 1000
            output["segments"].append(
                {
                    "id": len(output["segments"]),
                    "start": seg_start,
                    "end": seg_end,
                    "text": sentence["text"],
                    "speaker": sentence["spk"],
                }
            )

        return output

    def load(self):
        try:
            from funasr import AutoModel
        except ImportError:
            error_message = "Failed to import module 'funasr'"
            installation_guide = [
                "Please make sure 'funasr' is installed. ",
                "You can install it by `pip install funasr`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        if self._device is None:
            self._device = get_available_device()
        else:
            if not is_device_available(self._device):
                raise ValueError(f"Device {self._device} is not available!")

        kwargs = (
            self._model_spec.default_model_config.copy()
            if getattr(self._model_spec, "default_model_config", None)
            else {}
        )
        kwargs.update(self._kwargs)
        logger.debug("Loading FunASR model with kwargs: %s", kwargs)
        self._model = AutoModel(model=self._model_path, device=self._device, **kwargs)

    def transcriptions(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[str]] = None,
        **kwargs,
    ):
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        if temperature != 0:
            raise RuntimeError("`temperature`is not supported for FunASR")
        if timestamp_granularities is not None:
            raise RuntimeError("`timestamp_granularities`is not supported for FunASR")
        if prompt is not None:
            logger.warning(
                "Prompt for funasr transcriptions will be ignored: %s", prompt
            )

        language = "auto" if language is None else language

        with tempfile.NamedTemporaryFile(buffering=0) as f:
            f.write(audio)

            kw = (
                self._model_spec.default_transcription_config.copy()  # type: ignore
                if getattr(self._model_spec, "default_transcription_config", None)
                else {}
            )
            kw.update(kwargs)
            logger.debug("Calling FunASR model with kwargs: %s", kw)
            result = self._model.generate(  # type: ignore
                input=f.name, cache={}, language=language, **kw
            )
            if not result or not isinstance(result, list):
                raise RuntimeError(f"FunASR returned empty or invalid result: {result}")
            if "text" not in result[0]:
                raise RuntimeError(f"Missing 'text' field in result[0]: {result[0]}")
            text = rich_transcription_postprocess(result[0]["text"])

            if response_format == "json":
                return {"text": text}
            elif response_format == "verbose_json":
                verbose = result[0]
                verbose["text"] = text
                return self.convert_to_openai_format(verbose)
            else:
                raise ValueError(f"Unsupported response format: {response_format}")

    def translations(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[str]] = None,
    ):
        raise RuntimeError("FunASR does not support translations API")
