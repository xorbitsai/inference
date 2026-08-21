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

import logging
import os
import re
import tempfile
import wave
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .utils import MLXModelThreadMixin

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


_QWEN_ASR_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
}


def _read_result_field(result: Any, field: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(field, default)
    return getattr(result, field, default)


class MLXAudioSTTModel(MLXModelThreadMixin):
    """Run mlx-audio speech-to-text models behind Xinference's audio API."""

    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "AudioModelFamilyV2",
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
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
        self._run_on_mlx_thread(self._load)

    def _load(self):
        try:
            from mlx_audio.stt.utils import load
        except ImportError as exc:
            raise ImportError(
                "Failed to import 'mlx_audio'. Please install mlx-audio[stt]."
            ) from exc

        logger.info("Loading mlx-audio STT model from %s", self._model_path)
        self._model = load(self._model_path, **self._kwargs)

    def _decode_audio(self, audio: bytes):
        from mlx_audio.audio_io import read as audio_read

        assert self._model is not None
        sample_rate = getattr(self._model, "sample_rate", None)
        if sample_rate is None:
            config = getattr(self._model, "config", None)
            frontend_config = getattr(config, "frontend_conf", None)
            sample_rate = getattr(frontend_config, "fs", 16000)

        read_kwargs = {
            "dtype": "float32",
            "sample_rate": int(sample_rate),
            "nchannels": 1,
        }
        try:
            audio_data, _ = audio_read(BytesIO(audio), **read_kwargs)
        except ValueError as exc:
            # mlx-audio 0.4.6 only recognizes MP3 byte streams beginning with
            # ID3, FF FB, or FF FA. MPEG-2/2.5 Layer III files commonly begin
            # with other valid frame headers such as FF F3. Its path decoder
            # delegates these files to miniaudio and handles them correctly.
            is_mpeg_audio = (
                len(audio) >= 2
                and audio[0] == 0xFF
                and audio[1] & 0xE0 == 0xE0
                and audio[1] & 0x06 != 0
            )
            if (
                "Unable to detect audio format from bytes" not in str(exc)
                or not is_mpeg_audio
            ):
                raise

            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".mp3", delete=False
                ) as audio_file:
                    audio_file.write(audio)
                    temp_path = audio_file.name
                audio_data, _ = audio_read(temp_path, **read_kwargs)
            finally:
                if temp_path is not None:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
        return audio_data

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
        return self._run_on_mlx_thread(
            self._transcriptions,
            audio,
            language,
            prompt,
            response_format,
            temperature,
            timestamp_granularities,
            kwargs,
        )

    def _transcriptions(
        self,
        audio: bytes,
        language: Optional[str],
        prompt: Optional[str],
        response_format: str,
        temperature: float,
        timestamp_granularities: Optional[List[str]],
        kwargs: Dict[str, Any],
    ):
        assert self._model is not None
        if response_format not in ("json", "verbose_json"):
            raise ValueError(f"Unsupported response format: {response_format}")
        if timestamp_granularities is not None:
            raise RuntimeError(
                "`timestamp_granularities` is not supported by this mlx-audio model"
            )

        generate_kwargs = dict(
            getattr(self._model_spec, "default_transcription_config", None) or {}
        )
        generate_kwargs.update(kwargs)
        model_name = self._model_spec.model_name

        if language is not None:
            if self._model_spec.model_family == "qwen3_asr":
                language = _QWEN_ASR_LANGUAGE_NAMES.get(language.lower(), language)
            generate_kwargs["language"] = language

        if self._model_spec.model_family == "qwen3_asr":
            generate_kwargs["temperature"] = temperature
            if prompt:
                generate_kwargs["system_prompt"] = prompt
        elif model_name == "Fun-ASR-Nano-2512":
            generate_kwargs["temperature"] = temperature
            if prompt:
                generate_kwargs["context"] = prompt
        elif prompt:
            logger.warning("Prompt is ignored by mlx-audio model %s", model_name)

        audio_data = self._decode_audio(audio)
        result = self._model.generate(audio_data, **generate_kwargs)

        text = _read_result_field(result, "text", "")
        if response_format == "json":
            return {"text": text}

        response = {
            "task": "transcribe",
            "language": _read_result_field(result, "language"),
            "text": text,
        }
        segments = _read_result_field(result, "segments")
        if segments is not None:
            response["segments"] = segments
        return response

    def translations(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[str]] = None,
    ):
        raise RuntimeError("This mlx-audio model does not support translations API")


class MLXAudioTTSModel(MLXModelThreadMixin):
    """Run mlx-audio text-to-speech models behind Xinference's speech API."""

    _QWEN_MAX_SEGMENT_CHARS = 80
    _QWEN_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])|(?<=\.)\s+|\n+")
    _QWEN_FULL_STOP_PAUSE_SECONDS = 0.28
    _QWEN_SEMICOLON_PAUSE_SECONDS = 0.18
    _QWEN_COMMA_PAUSE_SECONDS = 0.12
    _QWEN_DEFAULT_PAUSE_SECONDS = 0.18
    _AUDIO_EDGE_FADE_SECONDS = 0.005

    _OPENAI_VOICES = {
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "fable",
        "nova",
        "onyx",
        "sage",
        "shimmer",
        "verse",
    }

    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "AudioModelFamilyV2",
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
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
        self._run_on_mlx_thread(self._load)

    def _load(self):
        try:
            from mlx_audio.tts.utils import load
        except ImportError as exc:
            raise ImportError(
                "Failed to import 'mlx_audio'. Please install mlx-audio[tts]."
            ) from exc

        logger.info("Loading mlx-audio TTS model from %s", self._model_path)
        self._model = load(self._model_path, **self._kwargs)

    @staticmethod
    def _save_temp_audio(audio: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
            audio_file.write(audio)
            return audio_file.name

    @staticmethod
    def _float_to_pcm16(audio):
        import numpy as np

        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")

    @classmethod
    def _audio_to_bytes(cls, response_format: str, sample_rate: int, audio) -> bytes:
        import numpy as np

        audio_array = np.asarray(audio, dtype=np.float32).squeeze()
        if audio_array.ndim != 1:
            raise ValueError(f"Unsupported generated audio shape: {audio_array.shape}")

        response_format = response_format.lower()
        if response_format == "pcm":
            return cls._float_to_pcm16(audio_array).tobytes()
        if response_format == "wav":
            with BytesIO() as out:
                with wave.open(out, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(cls._float_to_pcm16(audio_array).tobytes())
                return out.getvalue()

        import soundfile

        with BytesIO() as out:
            with soundfile.SoundFile(
                out, "w", sample_rate, 1, format=response_format.upper()
            ) as audio_file:
                audio_file.write(audio_array)
            return out.getvalue()

    @classmethod
    def _split_qwen_text(cls, text: str) -> List[str]:
        """Split long Qwen3-TTS input while retaining sentence punctuation."""
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        sentence_segments = [
            segment.strip()
            for segment in cls._QWEN_SENTENCE_BOUNDARY.split(text)
            if segment.strip()
        ]

        segments: List[str] = []
        max_chars = cls._QWEN_MAX_SEGMENT_CHARS
        for sentence in sentence_segments:
            remainder = sentence
            while len(remainder) > max_chars:
                window = remainder[: max_chars + 1]
                split_at = max_chars
                for separator in ("，", ",", "、", "：", ":", " "):
                    position = window.rfind(separator)
                    if position >= max_chars // 2:
                        split_at = position + 1
                        break
                segments.append(remainder[:split_at].strip())
                remainder = remainder[split_at:].strip()
            if remainder:
                segments.append(remainder)

        return segments or [text]

    @classmethod
    def _qwen_pause_seconds(cls, segment: str) -> float:
        final_character = segment.rstrip()[-1:] if segment.strip() else ""
        if final_character in "。.!！？?":
            return cls._QWEN_FULL_STOP_PAUSE_SECONDS
        if final_character in "；;":
            return cls._QWEN_SEMICOLON_PAUSE_SECONDS
        if final_character in "，,、：:":
            return cls._QWEN_COMMA_PAUSE_SECONDS
        return cls._QWEN_DEFAULT_PAUSE_SECONDS

    @classmethod
    def _fade_audio_edges(cls, audio, sample_rate: int):
        import numpy as np

        audio = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
        fade_samples = min(
            int(sample_rate * cls._AUDIO_EDGE_FADE_SECONDS), audio.size // 2
        )
        if fade_samples:
            audio[:fade_samples] *= np.linspace(
                0.0, 1.0, fade_samples, endpoint=True, dtype=np.float32
            )
            audio[-fade_samples:] *= np.linspace(
                1.0, 0.0, fade_samples, endpoint=True, dtype=np.float32
            )
        return audio

    @classmethod
    def _join_qwen_results(
        cls, results_by_segment: List[List[Any]], segments: List[str], sample_rate: int
    ):
        import numpy as np

        segment_audio = []
        for results in results_by_segment:
            chunks = []
            for result in results:
                result_sample_rate = int(
                    _read_result_field(result, "sample_rate", sample_rate)
                )
                if result_sample_rate != sample_rate:
                    raise ValueError(
                        "mlx-audio returned inconsistent sample rates: "
                        f"{sample_rate} and {result_sample_rate}"
                    )
                chunks.append(
                    np.asarray(_read_result_field(result, "audio")).reshape(-1)
                )
            segment_audio.append(np.concatenate(chunks))

        joined_parts = []
        for segment_index, audio in enumerate(segment_audio):
            joined_parts.append(cls._fade_audio_edges(audio, sample_rate))
            if segment_index < len(segment_audio) - 1:
                pause_samples = round(
                    sample_rate * cls._qwen_pause_seconds(segments[segment_index])
                )
                joined_parts.append(np.zeros(pause_samples, dtype=np.float32))

        if not joined_parts:
            raise RuntimeError("mlx-audio returned no generated audio")
        return np.concatenate(joined_parts)

    def _build_generation_kwargs(
        self,
        input: str,
        voice: str,
        speed: float,
        kwargs: Dict[str, Any],
        temp_files: List[str],
    ) -> Dict[str, Any]:
        from .utils import apply_mlx_audio_seed

        apply_mlx_audio_seed(kwargs)
        generate_kwargs = dict(kwargs)
        prompt_speech = generate_kwargs.pop("prompt_speech", None)
        prompt_text = generate_kwargs.pop("prompt_text", None)
        reference_speech = generate_kwargs.pop("reference_speech", None)
        language = generate_kwargs.pop("language", None)
        instruct = (
            generate_kwargs.pop("instruct", None)
            or generate_kwargs.pop("control_instruction", None)
            or generate_kwargs.pop("instruct_text", None)
        )

        prompt_audio_path = None
        if prompt_speech is not None:
            prompt_audio_path = self._save_temp_audio(prompt_speech)
            temp_files.append(prompt_audio_path)
        reference_audio_path = None
        if reference_speech is not None:
            reference_audio_path = self._save_temp_audio(reference_speech)
            temp_files.append(reference_audio_path)

        model_family = self._model_spec.model_family
        if model_family == "qwen3_tts":
            # mlx-audio defaults to 1200 codec tokens, which is only about nine
            # seconds of Qwen3-TTS audio and can truncate ordinary paragraphs.
            generate_kwargs.setdefault("max_tokens", 4096)
            generate_kwargs.update(
                text=input,
                voice=voice or None,
                speed=speed,
                lang_code=language or "auto",
            )
            if instruct:
                generate_kwargs["instruct"] = instruct
            if prompt_audio_path:
                if not prompt_text:
                    raise ValueError(
                        "prompt_text is required when prompt_speech is provided"
                    )
                generate_kwargs["ref_audio"] = prompt_audio_path
                generate_kwargs["ref_text"] = prompt_text
        elif model_family == "VoxCPM":
            generate_kwargs["text"] = input
            generate_kwargs["ref_audio"] = reference_audio_path or prompt_audio_path
            if prompt_audio_path and prompt_text:
                generate_kwargs["prompt_audio"] = prompt_audio_path
                generate_kwargs["prompt_text"] = prompt_text
            if instruct:
                generate_kwargs["instruct"] = instruct
            elif voice and voice.lower() not in self._OPENAI_VOICES:
                generate_kwargs["instruct"] = voice
            if speed != 1.0:
                logger.warning("VoxCPM2 MLX does not support speed; ignoring it")
        elif model_family == "MeloTTS":
            generate_kwargs.update(
                text=input,
                voice=voice or None,
                speed=speed,
                lang_code=language or "EN-US",
            )
        else:  # pragma: no cover - guarded by engine matching
            raise ValueError(f"Unsupported mlx-audio TTS family: {model_family}")
        return generate_kwargs

    def speech(self, *args, **kwargs):
        return self._run_on_mlx_thread(self._speech, *args, **kwargs)

    def _speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ):
        import numpy as np

        assert self._model is not None
        if stream:
            raise RuntimeError(
                "Streaming mlx-audio output is not yet supported by Xinference"
            )

        temp_files: List[str] = []
        try:
            text_segments = None
            if (
                self._model_spec.model_family == "qwen3_tts"
                and "split_pattern" not in kwargs
            ):
                text_segments = self._split_qwen_text(input)

            generate_kwargs = self._build_generation_kwargs(
                input, voice, speed, kwargs, temp_files
            )
            if text_segments is not None:
                results_by_segment = []
                for segment_index, segment in enumerate(text_segments):
                    segment_kwargs = dict(generate_kwargs)
                    segment_kwargs.update(text=segment, split_pattern=None)
                    results = list(self._model.generate(**segment_kwargs))
                    if not results:
                        raise RuntimeError(
                            "mlx-audio returned no generated audio for Qwen3-TTS "
                            f"segment {segment_index + 1}"
                        )
                    results_by_segment.append(results)
                sample_rate = int(
                    _read_result_field(results_by_segment[0][0], "sample_rate")
                )
                audio = self._join_qwen_results(
                    results_by_segment, text_segments, sample_rate
                )
            else:
                results = list(self._model.generate(**generate_kwargs))
                if not results:
                    raise RuntimeError("mlx-audio returned no generated audio")
                sample_rate = int(_read_result_field(results[0], "sample_rate"))
                audio = np.concatenate(
                    [
                        np.asarray(_read_result_field(result, "audio")).reshape(-1)
                        for result in results
                    ]
                )
            return self._audio_to_bytes(response_format, sample_rate, audio)
        finally:
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except OSError:
                    logger.warning("Failed to remove temporary audio %s", temp_file)
