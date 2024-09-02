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

import base64
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Optional

from ..utils import set_all_random_seed

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
        self._model.load(source="custom", custom_path=self._model_path, compile=True)

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
        import torchaudio
        import xxhash

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
        if stream:
            iter = self._model.infer(
                [input], params_infer_code=params_infer_code, stream=True
            )

            def _generator():
                with BytesIO() as out:
                    writer = torchaudio.io.StreamWriter(out, format=response_format)
                    writer.add_audio_stream(sample_rate=24000, num_channels=1)
                    i = 0
                    last_pos = 0
                    with writer.open():
                        for it in iter:
                            for itt in it:
                                for chunk in itt:
                                    chunk = np.array([chunk]).transpose()
                                    writer.write_audio_chunk(i, torch.from_numpy(chunk))
                                    new_last_pos = out.tell()
                                    if new_last_pos != last_pos:
                                        out.seek(last_pos)
                                        encoded_bytes = out.read()
                                        yield encoded_bytes
                                        last_pos = new_last_pos

            return _generator()
        else:
            wavs = self._model.infer([input], params_infer_code=params_infer_code)

            # Save the generated audio
            with BytesIO() as out:
                torchaudio.save(
                    out, torch.from_numpy(wavs[0]), 24000, format=response_format
                )
                return out.getvalue()
