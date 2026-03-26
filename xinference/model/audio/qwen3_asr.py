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
from typing import TYPE_CHECKING, List, Optional, Tuple

from ...device_utils import (
    get_available_device,
    get_device_preferred_dtype,
    is_device_available,
)

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


class Qwen3ASRModel:
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
            from qwen_asr import Qwen3ASRModel as QwenASR
        except ImportError:
            error_message = "Failed to import module 'qwen_asr'"
            installation_guide = [
                "Please make sure 'qwen-asr' is installed. ",
                "You can install it by `pip install qwen-asr`\n",
            ]
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        if self._device is None:
            self._device = get_available_device()
        else:
            if not is_device_available(self._device):
                raise ValueError(f"Device {self._device} is not available!")

        init_kwargs = (
            self._model_spec.default_model_config.copy()
            if getattr(self._model_spec, "default_model_config", None)
            else {}
        )
        init_kwargs.update(self._kwargs)
        init_kwargs.setdefault("device_map", self._device)
        init_kwargs.setdefault("dtype", get_device_preferred_dtype(self._device))
        if "forced_aligner" in init_kwargs:
            forced_aligner_kwargs = init_kwargs.get("forced_aligner_kwargs") or {}
            forced_aligner_kwargs.setdefault("device_map", self._device)
            forced_aligner_kwargs.setdefault(
                "dtype", get_device_preferred_dtype(self._device)
            )
            init_kwargs["forced_aligner_kwargs"] = forced_aligner_kwargs
        logger.debug("Loading Qwen3-ASR model with kwargs: %s", init_kwargs)
        self._model = QwenASR.from_pretrained(self._model_path, **init_kwargs)

    def _extract_text_and_language(self, result) -> Tuple[str, Optional[str]]:
        if isinstance(result, list):
            if not result:
                return "", None
            result = result[0]

        if hasattr(result, "text"):
            text = result.text
            language = getattr(result, "language", None)
            return text, language

        if isinstance(result, dict):
            text = result.get("text") or result.get("transcript") or ""
            language = result.get("language")
            return text, language

        return str(result), None

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
        if temperature != 0:
            raise RuntimeError("`temperature` is not supported for Qwen3-ASR")
        if timestamp_granularities is not None:
            raise RuntimeError(
                "`timestamp_granularities` is not supported for Qwen3-ASR"
            )
        if prompt is not None:
            logger.warning(
                "Prompt for Qwen3-ASR transcriptions will be ignored: %s", prompt
            )

        kw = dict(getattr(self._model_spec, "default_transcription_config", None) or {})
        kw.update(kwargs)

        with tempfile.NamedTemporaryFile(buffering=0) as f:
            f.write(audio)
            assert self._model is not None
            result = self._model.transcribe(audio=f.name, language=language, **kw)
            text, detected_language = self._extract_text_and_language(result)

        if response_format == "json":
            return {"text": text}
        if response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": detected_language,
                "text": text,
            }
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
        raise RuntimeError("Qwen3-ASR does not support translations API")
