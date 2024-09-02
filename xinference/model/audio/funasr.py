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
import tempfile
from typing import TYPE_CHECKING, List, Optional

from ...device_utils import get_available_device, is_device_available

if TYPE_CHECKING:
    from .core import AudioModelFamilyV1

logger = logging.getLogger(__name__)


class FunASRModel:
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

        kwargs = self._model_spec.default_model_config.copy()
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

            kw = self._model_spec.default_transcription_config.copy()  # type: ignore
            kw.update(kwargs)
            logger.debug("Calling FunASR model with kwargs: %s", kw)
            result = self._model.generate(  # type: ignore
                input=f.name, cache={}, language=language, **kw
            )
            text = rich_transcription_postprocess(result[0]["text"])

            if response_format == "json":
                return {"text": text}
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
