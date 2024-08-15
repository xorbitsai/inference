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
import os.path
from io import BytesIO
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .core import AudioModelFamilyV1

logger = logging.getLogger(__name__)
EVAL_RESULTS_FILE = os.path.join(os.path.dirname(__file__), "evaluation_results.npz")


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
        self._speakers = {}

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

        try:
            seed_id = int(voice)
            if not self._speakers:
                npzfiles = np.load(EVAL_RESULTS_FILE)
                self._speakers = dict(zip(npzfiles["seed_id"], npzfiles["emb_data"]))
            arr = self._speakers[seed_id]
            tensor = torch.Tensor(arr)
            assert self._model is not None
            rnd_spk_emb = self._model._encode_spk_emb(tensor)
            logger.info("Speech by eval speaker %s", seed_id)
        except (KeyError, ValueError) as e:
            if isinstance(e, KeyError):
                logger.info("Unrecognised speaker id %s, fallback to random.", voice)
            seed = xxhash.xxh32_intdigest(voice)

            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            assert self._model is not None
            rnd_spk_emb = self._model.sample_random_speaker()
            logger.info("Speech by speaker %s", voice)

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
                                        print(len(encoded_bytes))
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
