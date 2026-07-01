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
import io
import logging
import os
import re
from io import BytesIO
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


class F5TTSModel:
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
        self._vocoder = None
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def load(self):
        import os
        import sys

        # The yaml config loaded from model has hard-coded the import paths. please refer to: load_hyperpyyaml
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../thirdparty"))

        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from f5_tts.model import DiT

        vocoder_name = self._kwargs.get("vocoder_name", "vocos")
        vocoder_path = self._kwargs.get("vocoder_path")

        if vocoder_name not in ["vocos", "bigvgan"]:
            raise Exception(f"Unsupported vocoder name: {vocoder_name}")

        if vocoder_path is not None:
            self._vocoder = load_vocoder(
                vocoder_name=vocoder_name, is_local=True, local_path=vocoder_path
            )
        else:
            self._vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False)

        model_cls = DiT
        model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
        )
        if vocoder_name == "vocos":
            exp_name = "F5TTS_Base"
            ckpt_step = 1200000
        elif vocoder_name == "bigvgan":
            exp_name = "F5TTS_Base_bigvgan"
            ckpt_step = 1250000
        else:
            assert False
        ckpt_file = os.path.join(
            self._model_path, exp_name, f"model_{ckpt_step}.safetensors"
        )
        logger.info(f"Loading %s...", ckpt_file)
        self._model = load_model(
            model_cls, model_cfg, ckpt_file, mel_spec_type=vocoder_name
        )

    def _infer(self, ref_audio, ref_text, text_gen, model_obj, mel_spec_type, speed):
        import numpy as np
        from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

        config = {}
        main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
        if "voices" not in config:
            voices = {"main": main_voice}
        else:
            voices = config["voices"]
            voices["main"] = main_voice
        for voice in voices:
            (
                voices[voice]["ref_audio"],
                voices[voice]["ref_text"],
            ) = preprocess_ref_audio_text(
                voices[voice]["ref_audio"], voices[voice]["ref_text"]
            )
            logger.info("Voice:", voice)
            logger.info("Ref_audio:", voices[voice]["ref_audio"])
            logger.info("Ref_text:", voices[voice]["ref_text"])

        final_sample_rate = None
        generated_audio_segments = []
        reg1 = r"(?=\[\w+\])"
        chunks = re.split(reg1, text_gen)
        reg2 = r"\[(\w+)\]"
        for text in chunks:
            if not text.strip():
                continue
            match = re.match(reg2, text)
            if match:
                voice = match[1]
            else:
                logger.info("No voice tag found, using main.")
                voice = "main"
            if voice not in voices:
                logger.info(f"Voice {voice} not found, using main.")
                voice = "main"
            text = re.sub(reg2, "", text)
            gen_text = text.strip()
            ref_audio = voices[voice]["ref_audio"]
            ref_text = voices[voice]["ref_text"]
            logger.info(f"Voice: {voice}")
            audio, final_sample_rate, spectragram = infer_process(
                ref_audio,
                ref_text,
                gen_text,
                model_obj,
                self._vocoder,
                mel_spec_type=mel_spec_type,
                speed=speed,
            )
            generated_audio_segments.append(audio)

        if generated_audio_segments:
            final_wave = np.concatenate(generated_audio_segments)
            return final_sample_rate, final_wave
        return None, None

    def speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ):
        import f5_tts
        import soundfile
        import tomli

        if stream:
            raise Exception("F5-TTS does not support stream generation.")

        prompt_speech: Optional[bytes] = kwargs.pop("prompt_speech", None)
        prompt_text: Optional[str] = kwargs.pop("prompt_text", None)

        ref_audio: Union[str, io.BytesIO]
        if prompt_speech is None:
            base = os.path.dirname(f5_tts.__file__)
            config = os.path.join(base, "infer/examples/basic/basic.toml")
            with open(config, "rb") as f:
                config_dict = tomli.load(f)
            ref_audio = os.path.join(base, config_dict["ref_audio"])
            prompt_text = config_dict["ref_text"]
        else:
            ref_audio = io.BytesIO(prompt_speech)
            if prompt_text is None:
                raise ValueError("`prompt_text` cannot be empty")

        assert self._model is not None
        vocoder_name = self._kwargs.get("vocoder_name", "vocos")
        sample_rate, wav = self._infer(
            ref_audio=ref_audio,
            ref_text=prompt_text,
            text_gen=input,
            model_obj=self._model,
            mel_spec_type=vocoder_name,
            speed=speed,
        )

        # Save the generated audio
        with BytesIO() as out:
            with soundfile.SoundFile(
                out, "w", sample_rate, 1, format=response_format.upper()
            ) as f:
                f.write(wav)
            return out.getvalue()
