# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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
import sys
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    import torch

    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


SUPPORTED_LANGUAGES = (
    "Arabic",
    "Cantonese",
    "Chinese",
    "Czech",
    "Dutch",
    "English",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hindi",
    "Indonesian",
    "Italian",
    "Japanese",
    "Korean",
    "Polish",
    "Portuguese",
    "Romanian",
    "Russian",
    "Spanish",
    "Thai",
    "Turkish",
    "Ukrainian",
    "Vietnamese",
    "ZH_Anhui",
    "ZH_Fujian",
    "ZH_Gansu",
    "ZH_Guizhou",
    "ZH_Hebei",
    "ZH_Henan",
    "ZH_Hubei",
    "ZH_Hunan",
    "ZH_Jiangxi",
    "ZH_Liaoning",
    "ZH_Minnan",
    "ZH_Ningxia",
    "ZH_Shaanxi",
    "ZH_Shandong",
    "ZH_Shanghai",
    "ZH_Shanxi",
    "ZH_Sichuan",
    "ZH_Tianjin",
    "ZH_Wenzhou",
    "ZH_Wu",
    "ZH_Yunnan",
)

_LANGUAGE_ALIASES = {
    "ar": "Arabic",
    "cs": "Czech",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "en-us": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
}


def _normalize_language(language: str) -> str:
    value = str(language).strip()
    normalized = _LANGUAGE_ALIASES.get(value.lower())
    if normalized is None:
        normalized = {item.lower(): item for item in SUPPORTED_LANGUAGES}.get(
            value.lower()
        )
    if normalized is None:
        raise ValueError(
            f"Unsupported FireRedTTS3 language {language!r}. Supported values: "
            f"{', '.join(SUPPORTED_LANGUAGES)}"
        )
    return normalized


class FireRedTTS3Model:
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
        self._normalizer = None
        self._is_instruct = model_spec.model_name.endswith("-Instruct")
        self._use_wetext = bool(kwargs.pop("use_wetext", True))
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def load(self):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("FireRedTTS3 requires a CUDA-capable GPU.")
        if self._device is not None and not str(self._device).startswith("cuda"):
            raise ValueError(
                f"FireRedTTS3 only supports CUDA devices, got {self._device!r}."
            )

        thirdparty_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../thirdparty")
        )
        if thirdparty_dir not in sys.path:
            sys.path.insert(0, thirdparty_dir)

        from fireredtts3.utils.text_normalize import build_wetext_normalizer

        if self._is_instruct:
            from fireredtts3.llm.fireredtts3_instruct import FireRedTTS3Instruct

            logger.info("Loading FireRedTTS3 Instruct model...")
            self._model = FireRedTTS3Instruct(self._model_path)
        else:
            from fireredtts3.llm.fireredtts3_base import FireRedTTS3Base

            logger.info("Loading FireRedTTS3 Base model...")
            self._model = FireRedTTS3Base(self._model_path)
        if self._use_wetext:
            self._normalizer = build_wetext_normalizer()

    @staticmethod
    def _load_prompt_audio(prompt_speech: bytes) -> Tuple["torch.Tensor", int]:
        import soundfile
        import torch

        audio, sample_rate = soundfile.read(
            io.BytesIO(prompt_speech), dtype="float32", always_2d=True
        )
        return torch.from_numpy(audio.T.copy()), int(sample_rate)

    def _prepare_sentences(
        self, text: str, language: Optional[str], kwargs: dict
    ) -> Tuple[str, List[str]]:
        from fireredtts3.utils.text_normalize import (
            clean_text,
            clean_tn_spaces,
            detect_language,
            split_paragraph,
        )

        do_clean = bool(kwargs.pop("do_clean", True))
        do_tn = bool(kwargs.pop("do_tn", True))
        do_split = bool(kwargs.pop("do_split", True))
        token_max_n = int(kwargs.pop("token_max_n", 80))
        token_min_n = int(kwargs.pop("token_min_n", 60))
        merge_len = int(kwargs.pop("merge_len", 20))

        if do_clean:
            text = clean_text(text)
        if not text or not text.strip():
            raise ValueError("input must be a non-empty string")

        language = _normalize_language(language or detect_language(text))
        if do_split:
            is_chinese = language in ("Chinese", "Cantonese") or language.startswith(
                "ZH_"
            )
            tokenize = None
            if not is_chinese:
                assert self._model is not None
                tokenize = lambda value: int(self._model._tokenize_text(value).shape[1])
            sentences = split_paragraph(
                text,
                tokenize=tokenize,
                lang="zh" if is_chinese else "en",
                token_max_n=token_max_n,
                token_min_n=token_min_n,
                merge_len=merge_len,
            )
        else:
            sentences = [text]

        can_use_wetext = language in (
            "Chinese",
            "English",
            "Cantonese",
        ) or language.startswith("ZH_")
        if do_tn and can_use_wetext and self._normalizer is not None:
            sentences = [
                clean_tn_spaces(self._normalizer(sentence)) for sentence in sentences
            ]
        sentences = [sentence for sentence in sentences if sentence.strip()]
        if not sentences:
            raise ValueError("input is empty after text normalization")
        return language, sentences

    @staticmethod
    def _join_segments(
        segments: List["torch.Tensor"], sample_rate: int, cross_fade_ms: float
    ) -> "torch.Tensor":
        audio = segments[0].detach().cpu()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if len(segments) == 1:
            return audio

        import torch

        fade_len = max(0, int(cross_fade_ms / 1000.0 * sample_rate))
        for segment in segments[1:]:
            segment = segment.detach().cpu()
            if segment.ndim == 1:
                segment = segment.unsqueeze(0)
            overlap_len = min(fade_len, audio.shape[-1], segment.shape[-1])
            if overlap_len <= 0:
                audio = torch.cat([audio, segment], dim=-1)
                continue
            ramp = torch.linspace(0.0, 1.0, overlap_len, dtype=audio.dtype).unsqueeze(0)
            overlap = (
                audio[:, -overlap_len:] * (1.0 - ramp) + segment[:, :overlap_len] * ramp
            )
            audio = torch.cat(
                [audio[:, :-overlap_len], overlap, segment[:, overlap_len:]], dim=-1
            )
        return audio

    @staticmethod
    def _audio_to_bytes(response_format: str, sample_rate: int, audio) -> bytes:
        from .utils import audio_to_bytes

        return audio_to_bytes(response_format, sample_rate, audio)

    def speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ):
        if stream:
            raise ValueError("FireRedTTS3 does not support streaming generation.")

        prompt_speech = kwargs.pop("prompt_speech", None)
        prompt_text = kwargs.pop("prompt_text", None)
        instruct = (
            kwargs.pop("instruct", None)
            or kwargs.pop("instruction", None)
            or kwargs.pop("instruct_text", None)
        )

        if self._is_instruct:
            if prompt_speech:
                if not prompt_text or not str(prompt_text).strip():
                    raise ValueError(
                        "FireRedTTS3-Instruct requires the reference audio "
                        "transcript in `prompt_text` when `prompt_speech` is provided."
                    )
                if instruct:
                    logger.warning(
                        "Ignoring FireRedTTS3-Instruct voice design instruction "
                        "because prompt_speech selects voice cloning."
                    )
            else:
                # Reuse the existing prompt_text UI field for voice design. With
                # reference audio it remains the prompt transcript.
                instruct = instruct or prompt_text
                if not instruct or not str(instruct).strip():
                    raise ValueError(
                        "FireRedTTS3-Instruct requires a voice design instruction "
                        "in `prompt_text` or `instruct`."
                    )
        else:
            if not prompt_speech:
                raise ValueError(
                    "FireRedTTS3-Base requires reference audio in `prompt_speech`."
                )
            if not prompt_text or not str(prompt_text).strip():
                raise ValueError(
                    "FireRedTTS3-Base requires the reference audio transcript in "
                    "`prompt_text`."
                )
        if speed != 1.0:
            logger.warning("FireRedTTS3 does not support speed; ignoring it.")

        assert self._model is not None
        language, sentences = self._prepare_sentences(
            input, kwargs.pop("language", None), kwargs
        )
        prompt_audio = None
        prompt_audio_sr = None
        if prompt_speech:
            prompt_audio, prompt_audio_sr = self._load_prompt_audio(prompt_speech)

        stop_threshold = float(kwargs.pop("stop_threshold", 0.5))
        n_timesteps = int(kwargs.pop("n_timesteps", 10))
        voice_design = self._is_instruct and prompt_audio is None
        inference_cfg = float(kwargs.pop("inference_cfg", 1.2 if voice_design else 2.0))
        seed = kwargs.pop("seed", 2 if voice_design else 1234)
        if seed is not None:
            seed = int(seed)
        cross_fade_ms = float(kwargs.pop("cross_fade_ms", 50.0))
        if cross_fade_ms < 0:
            raise ValueError("cross_fade_ms must be greater than or equal to 0")
        if kwargs:
            logger.warning("Ignoring unsupported FireRedTTS3 speech kwargs: %s", kwargs)

        segments = []
        sample_rate = None
        for sentence in sentences:
            if not self._is_instruct:
                audio, segment_sample_rate = self._model.generate(
                    language=language,
                    prompt_text=str(prompt_text),
                    prompt_audio=prompt_audio,
                    prompt_audio_sr=prompt_audio_sr,
                    text=sentence,
                    stop_threshold=stop_threshold,
                    n_timesteps=n_timesteps,
                    inference_cfg=inference_cfg,
                    seed=seed,
                )
            elif prompt_audio is not None:
                audio, segment_sample_rate = self._model.generate_tts(
                    prompt_text=str(prompt_text),
                    prompt_audio=prompt_audio,
                    prompt_audio_sr=prompt_audio_sr,
                    text=sentence,
                    stop_threshold=stop_threshold,
                    n_timesteps=n_timesteps,
                    inference_cfg=inference_cfg,
                    seed=seed,
                )
            else:
                audio, segment_sample_rate, _ = self._model.generate_voice_design(
                    instruction=str(instruct),
                    text=sentence,
                    n_timesteps=n_timesteps,
                    inference_cfg=inference_cfg,
                    seed=seed,
                )
            if sample_rate is not None and sample_rate != segment_sample_rate:
                raise RuntimeError("FireRedTTS3 returned inconsistent sample rates.")
            sample_rate = int(segment_sample_rate)
            segments.append(audio)

        assert sample_rate is not None
        audio = self._join_segments(segments, sample_rate, cross_fade_ms)
        return self._audio_to_bytes(response_format, sample_rate, audio)
