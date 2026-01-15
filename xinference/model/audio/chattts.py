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

import base64
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Optional

from ..utils import set_all_random_seed

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


class ChatTTSModel:
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
        import ChatTTS
        import torch

        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.suppress_errors = True
        torch.set_float32_matmul_precision("high")
        self._model = ChatTTS.Chat()
        logger.info("Load ChatTTS model with kwargs: %s", self._kwargs)
        ok = self._model.load(
            source="custom", custom_path=self._model_path, **self._kwargs
        )
        if not ok:
            raise Exception(f"The ChatTTS model is not correct: {self._model_path}")

    def speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
    ):
        import ChatTTS
        import numpy as np
        import torch
        import xxhash

        from .utils import audio_stream_generator, audio_to_bytes

        rnd_spk_emb = None

        if len(voice) > 400:
            try:
                assert self._model is not None
                b = base64.b64decode(voice)
                bio = BytesIO(b)
                tensor = torch.load(bio, map_location="cpu")
                rnd_spk_emb = self._model._encode_spk_emb(tensor)
                logger.info("Speech by input speaker")
            except Exception as e:
                logger.info("Fallback to random speaker due to %s", e)

        if rnd_spk_emb is None:
            seed = xxhash.xxh32_intdigest(voice)

            set_all_random_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            assert self._model is not None
            rnd_spk_emb = self._model.sample_random_speaker()
            logger.info("Speech by voice %s", voice)

        default = 5
        infer_speed = int(default * speed)
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            prompt=f"[speed_{infer_speed}]", spk_emb=rnd_spk_emb
        )

        assert self._model is not None

        output = self._model.infer(
            [input], params_infer_code=params_infer_code, stream=stream
        )
        if stream:

            def _gen_chunk():
                for it in output:
                    for chunk in it:
                        yield chunk

            return audio_stream_generator(
                response_format=response_format,
                sample_rate=24000,
                output_generator=_gen_chunk(),
                output_chunk_transformer=lambda c: torch.from_numpy(
                    np.array([c]).transpose()
                ),
            )
        else:
            return audio_to_bytes(
                response_format=response_format,
                sample_rate=24000,
                tensor=torch.from_numpy(output[0]).unsqueeze(0),
            )
