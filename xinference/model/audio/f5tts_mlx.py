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

import datetime
import io
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


class F5TTSMLXModel:
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
        self._model = None

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def load(self):
        try:
            import mlx.core as mx
            from f5_tts_mlx.cfm import F5TTS
            from f5_tts_mlx.dit import DiT
            from f5_tts_mlx.duration import DurationPredictor, DurationTransformer
            from vocos_mlx import Vocos
        except ImportError:
            error_message = "Failed to import module 'f5_tts_mlx'"
            installation_guide = [
                "Please make sure 'f5_tts_mlx' is installed.\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        path = Path(self._model_path)
        # vocab

        vocab_path = path / "vocab.txt"
        vocab = {v: i for i, v in enumerate(Path(vocab_path).read_text().split("\n"))}
        if len(vocab) == 0:
            raise ValueError(f"Could not load vocab from {vocab_path}")

        # duration predictor

        duration_model_path = path / "duration_v2.safetensors"
        duration_predictor = None

        if duration_model_path.exists():
            duration_predictor = DurationPredictor(
                transformer=DurationTransformer(
                    dim=512,
                    depth=8,
                    heads=8,
                    text_dim=512,
                    ff_mult=2,
                    conv_layers=2,
                    text_num_embeds=len(vocab) - 1,
                ),
                vocab_char_map=vocab,
            )
            weights = mx.load(duration_model_path.as_posix(), format="safetensors")
            duration_predictor.load_weights(list(weights.items()))

        # vocoder

        vocos = Vocos.from_pretrained("lucasnewman/vocos-mel-24khz")

        # model

        model_path = path / "model.safetensors"

        f5tts = F5TTS(
            transformer=DiT(
                dim=1024,
                depth=22,
                heads=16,
                ff_mult=2,
                text_dim=512,
                conv_layers=4,
                text_num_embeds=len(vocab) - 1,
            ),
            vocab_char_map=vocab,
            vocoder=vocos.decode,
            duration_predictor=duration_predictor,
        )

        weights = mx.load(model_path.as_posix(), format="safetensors")
        f5tts.load_weights(list(weights.items()))
        mx.eval(f5tts.parameters())

        self._model = f5tts

    def speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ):
        import mlx.core as mx
        import soundfile as sf
        import tomli
        from f5_tts_mlx.generate import (
            FRAMES_PER_SEC,
            SAMPLE_RATE,
            TARGET_RMS,
            convert_char_to_pinyin,
            split_sentences,
        )

        from .utils import ensure_sample_rate

        if stream:
            raise Exception("F5-TTS does not support stream generation.")

        prompt_speech: Optional[bytes] = kwargs.pop("prompt_speech", None)
        prompt_text: Optional[str] = kwargs.pop("prompt_text", None)
        duration: Optional[float] = kwargs.pop("duration", None)
        steps: Optional[int] = kwargs.pop("steps", 8)
        cfg_strength: Optional[float] = kwargs.pop("cfg_strength", 2.0)
        method: Literal["euler", "midpoint"] = kwargs.pop("method", "rk4")
        sway_sampling_coef: float = kwargs.pop("sway_sampling_coef", -1.0)
        seed: Optional[int] = kwargs.pop("seed", None)

        prompt_speech_path: Union[str, io.BytesIO]
        if prompt_speech is None:
            base = os.path.join(os.path.dirname(__file__), "../../thirdparty/f5_tts")
            config = os.path.join(base, "infer/examples/basic/basic.toml")
            with open(config, "rb") as f:
                config_dict = tomli.load(f)
            prompt_speech_path = os.path.join(base, config_dict["ref_audio"])
            prompt_text = config_dict["ref_text"]
        else:
            prompt_speech_path = io.BytesIO(prompt_speech)

            if prompt_text is None:
                raise ValueError("`prompt_text` cannot be empty")

        audio, sr = sf.read(prompt_speech_path)
        audio = ensure_sample_rate(audio, sr, SAMPLE_RATE)

        audio = mx.array(audio)
        ref_audio_duration = audio.shape[0] / SAMPLE_RATE
        logger.debug(
            f"Got reference audio with duration: {ref_audio_duration:.2f} seconds"
        )

        rms = mx.sqrt(mx.mean(mx.square(audio)))
        if rms < TARGET_RMS:
            audio = audio * TARGET_RMS / rms

        sentences = split_sentences(input)
        is_single_generation = len(sentences) <= 1 or duration is not None

        if is_single_generation:
            generation_text = convert_char_to_pinyin([prompt_text + " " + input])  # type: ignore

            if duration is not None:
                duration = int(duration * FRAMES_PER_SEC)

            start_date = datetime.datetime.now()

            wave, _ = self._model.sample(  # type: ignore
                mx.expand_dims(audio, axis=0),
                text=generation_text,
                duration=duration,
                steps=steps,
                method=method,
                speed=speed,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
            )

            wave = wave[audio.shape[0] :]
            mx.eval(wave)

            generated_duration = wave.shape[0] / SAMPLE_RATE
            print(
                f"Generated {generated_duration:.2f}s of audio in {datetime.datetime.now() - start_date}."
            )

        else:
            start_date = datetime.datetime.now()

            output = []

            for sentence_text in tqdm(split_sentences(input)):
                text = convert_char_to_pinyin([prompt_text + " " + sentence_text])  # type: ignore

                if duration is not None:
                    duration = int(duration * FRAMES_PER_SEC)

                wave, _ = self._model.sample(  # type: ignore
                    mx.expand_dims(audio, axis=0),
                    text=text,
                    duration=duration,
                    steps=steps,
                    method=method,
                    speed=speed,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    seed=seed,
                )

                # trim the reference audio
                wave = wave[audio.shape[0] :]
                mx.eval(wave)

                output.append(wave)

            wave = mx.concatenate(output, axis=0)

            generated_duration = wave.shape[0] / SAMPLE_RATE
            logger.debug(
                f"Generated {generated_duration:.2f}s of audio in {datetime.datetime.now() - start_date}."
            )

        # Save the generated audio
        with BytesIO() as out:
            with sf.SoundFile(
                out, "w", SAMPLE_RATE, 1, format=response_format.upper()
            ) as f:
                f.write(np.array(wave))
                return out.getvalue()
