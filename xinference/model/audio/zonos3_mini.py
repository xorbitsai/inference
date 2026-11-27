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


from contextlib import closing, contextmanager
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, Generator, Iterator, Optional, Tuple

from ...core.exceptions import InvalidAudioInputError, InvalidModelRequestError

if TYPE_CHECKING:
    import numpy as np

    from .core import AudioModelFamilyV2

SAMPLE_RATE = 44100
DEFAULT_VOICE = "Savannah"
VOICES = {
    name.lower(): name
    for name in (
        "Ananya",
        "Arjun",
        "Chaitanya",
        "Cillian",
        "Gaspard",
        "Ginevra",
        "Haoran",
        "Imogen",
        "Jett",
        "Leonor",
        "Liesel",
        "Lorenzo",
        "Meilin",
        "Minjun",
        "Morgana",
        "Rodrigo",
        "Rupert",
        "Sakura",
        "Savannah",
        "Seoyeon",
        "Sravya",
        "Thaddeus",
        "Thiago",
        "Willow",
        "Wolfgang",
        "Wyatt",
        "Ximena",
    )
}
LANGUAGES = frozenset(
    "en_us en_gb fr_fr es de it pt_br nl el sv da no fi pl cs sk hu ro bg ru uk "
    "tr ar he ka cmn ja ko vi id ms th tl hi bn ta te mr gu kn ml pa or ur".split()
)
LANGUAGE_ALIASES = {
    "en": "en_us",
    "fr": "fr_fr",
    "pt": "pt_br",
    "zh": "cmn",
    "zh_cn": "cmn",
}


def _decode_reference(reference: bytes) -> "Tuple[np.ndarray, int]":
    import numpy as np
    import soundfile

    try:
        with soundfile.SoundFile(BytesIO(reference)) as source:
            rate = source.samplerate
            audio = source.read(frames=60 * rate, dtype="float32", always_2d=True)
    except (ValueError, soundfile.LibsndfileError) as exc:
        raise InvalidAudioInputError("Unable to decode prompt_speech audio.") from exc
    if not audio.size or not np.isfinite(audio).all():
        raise InvalidAudioInputError("prompt_speech must contain finite audio samples.")

    return audio.mean(axis=1), rate


@contextmanager
def _reference_file(
    reference: "Optional[Tuple[np.ndarray, int]]",
) -> Iterator[Optional[Path]]:
    if reference is None:
        yield None
        return

    import soundfile

    audio, rate = reference
    with TemporaryDirectory() as directory:
        path = Path(directory) / "reference.wav"
        soundfile.write(path, audio, rate, subtype="PCM_16")
        yield path


def _encode_audio(audio: "np.ndarray", response_format: str) -> bytes:
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if response_format == "pcm":
        return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    import soundfile

    subtype = {"wav": "PCM_16", "flac": "PCM_16", "ogg": "VORBIS"}.get(response_format)
    with BytesIO() as output:
        soundfile.write(
            output,
            samples,
            SAMPLE_RATE,
            format=response_format.upper(),
            subtype=subtype,
        )
        return output.getvalue()


def _sampling_options(options: Dict[str, Any]) -> Dict[str, Any]:
    bounds = {
        "seed": (True, 0, 2**32 - 1),
        "max_new_tokens": (True, 1, 32768),
        "top_k": (True, 0, 4096),
        "top_p": (False, 0.0, 1.0),
        "temperature": (False, 0.0, 3.4028234663852886e38),
    }
    for name, value in options.items():
        if name not in bounds:
            raise InvalidModelRequestError(f"Unsupported ZONOS3-Mini argument: {name}")
        integer_only, minimum, maximum = bounds[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (integer_only and not isinstance(value, int))
            or not minimum <= value <= maximum
        ):
            raise InvalidModelRequestError(f"Invalid ZONOS3-Mini {name}: {value!r}")
    return options


class Zonos3MiniModel:
    supports_streaming = False

    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "AudioModelFamilyV2",
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model_family = self._model_spec = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def _generate(
        self,
        text: str,
        voice: str,
        reference_path: Optional[Path],
        language: Optional[str],
        options: Dict[str, Any],
    ) -> "np.ndarray":
        raise NotImplementedError

    def _generate_stream(
        self,
        text: str,
        voice: str,
        reference_path: Optional[Path],
        language: Optional[str],
        options: Dict[str, Any],
    ) -> Generator["np.ndarray", None, None]:
        raise NotImplementedError

    def _validate_engine_request(
        self, language: Optional[str], options: Dict[str, Any]
    ) -> None:
        pass

    def speech(
        self,
        input: str,
        voice: Optional[str],
        response_format: Optional[str] = "mp3",
        speed: Optional[float] = 1.0,
        stream: Optional[bool] = False,
        *,
        prompt_speech: Optional[bytes] = None,
        prompt_text: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ):
        if not isinstance(input, str) or not input.strip():
            raise InvalidModelRequestError("ZONOS3-Mini input must be non-empty text.")
        if speed is not None and speed != 1.0:
            raise InvalidModelRequestError("ZONOS3-Mini only supports speed=1.0.")
        if response_format is None:
            response_format = "mp3"
        if not isinstance(response_format, str):
            raise InvalidModelRequestError(
                "ZONOS3-Mini response_format must be a string."
            )
        response_format = response_format.lower()
        if response_format not in {"mp3", "wav", "flac", "ogg", "pcm"}:
            raise InvalidModelRequestError(
                f"Unsupported response format: {response_format}"
            )
        if stream and (not self.supports_streaming or response_format != "pcm"):
            raise InvalidModelRequestError(
                "ZONOS3-Mini streaming requires a streaming engine and response_format=pcm."
            )
        if prompt_text is not None:
            raise InvalidModelRequestError("ZONOS3-Mini does not use prompt_text.")
        if prompt_speech is not None and (
            not isinstance(prompt_speech, bytes) or not prompt_speech
        ):
            raise InvalidAudioInputError(
                "prompt_speech must contain non-empty audio bytes."
            )
        if voice is not None and not isinstance(voice, str):
            raise InvalidModelRequestError("ZONOS3-Mini voice must be a string.")
        if voice is None or voice.lower() in {"", "default", "alloy"}:
            voice = DEFAULT_VOICE
        elif voice.lower() in VOICES:
            voice = VOICES[voice.lower()]
        else:
            raise InvalidModelRequestError(f"Unknown ZONOS3-Mini voice: {voice!r}")
        if language is not None:
            if not isinstance(language, str):
                raise InvalidModelRequestError("ZONOS3-Mini language must be a string.")
            language = language.lower().replace("-", "_")
            language = LANGUAGE_ALIASES.get(language, language)
            if language not in LANGUAGES:
                raise InvalidModelRequestError(
                    f"Unsupported ZONOS3-Mini language: {language}"
                )

        text = input.strip()
        options = _sampling_options(kwargs)
        self._validate_engine_request(language, options)
        reference = None if prompt_speech is None else _decode_reference(prompt_speech)
        if stream:
            return self._stream(text, voice, reference, language, options)
        with _reference_file(reference) as reference_path:
            audio = self._generate(text, voice, reference_path, language, options)
        return _encode_audio(audio, response_format)

    def _stream(
        self,
        text: str,
        voice: str,
        reference: "Optional[Tuple[np.ndarray, int]]",
        language: Optional[str],
        options: Dict[str, Any],
    ) -> Iterator[bytes]:
        with _reference_file(reference) as reference_path:
            with closing(
                self._generate_stream(text, voice, reference_path, language, options)
            ) as chunks:
                for chunk in chunks:
                    pcm = _encode_audio(chunk, "pcm")
                    if pcm:
                        yield pcm
