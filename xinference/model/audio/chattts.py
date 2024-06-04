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
from io import BytesIO
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .core import AudioModelFamilyV1

logger = logging.getLogger(__name__)


class ChatTTSModel:
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
        import torch

        from xinference.thirdparty import ChatTTS

        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.suppress_errors = True
        torch.set_float32_matmul_precision("high")
        self._model = ChatTTS.Chat()
        self._model.load_models(
            source="local", local_path=self._model_path, compile=True
        )

    def speech(
        self, input: str, voice: str, response_format: str = "mp3", speed: float = 1.0
    ):
        import torch
        import torchaudio

        default = 5
        infer_speed = int(default * speed)
        params_infer_code = {"prompt": f"[speed_{infer_speed}]"}

        assert self._model is not None
        wavs = self._model.infer([input], params_infer_code=params_infer_code)

        # Save the generated audio
        with BytesIO() as out:
            torchaudio.save(
                out, torch.from_numpy(wavs[0]), 24000, format=response_format
            )
            return out.getvalue()
