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
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ...core.exceptions import InvalidModelRequestError
from ..utils import allow_trust_remote_code

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


def _check_transformers_version() -> None:
    import transformers
    from packaging.version import Version

    if not Version("4.57.0") <= Version(transformers.__version__) < Version("5"):
        raise ImportError(
            "Audio8-TTS requires transformers>=4.57.0,<5. "
            "Enable the model virtual environment or install a compatible version."
        )


class Audio8TTSModel:
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
        self._kwargs = kwargs
        self._model = None
        self._processor = None

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    @staticmethod
    def _check_model_trust(model_spec: "AudioModelFamilyV2") -> None:
        if not allow_trust_remote_code(model_spec):
            raise ValueError(
                "Loading a custom Audio8-TTS model or an explicit model_path can "
                "execute Python code from its snapshot. Set "
                "XINFERENCE_TRUST_REMOTE_CODE=1 on the worker before startup "
                "only if you trust this model source."
            )

    def load(self):
        self._check_model_trust(self.model_family)

        import torch
        from transformers import AutoModel, AutoProcessor

        _check_transformers_version()
        device = torch.device(
            self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Audio8-TTS supports CPU or CUDA devices.")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("Audio8-TTS requires an available CUDA device.")
        options = self._kwargs.copy()
        dtype = options.pop("dtype", options.pop("torch_dtype", "auto"))
        if dtype == "auto":
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        elif isinstance(dtype, str):
            if dtype not in {"float32", "float16", "bfloat16"}:
                raise ValueError(f"Unsupported Audio8-TTS dtype: {dtype}")
            dtype = getattr(torch, dtype)
        if device.type == "cpu" and dtype != torch.float32:
            raise ValueError("Audio8-TTS CPU inference requires float32.")
        if not (Path(self._model_path) / "codec.pth").is_file():
            raise FileNotFoundError("Audio8-TTS snapshot is missing codec.pth.")
        options.update(trust_remote_code=True, local_files_only=True, dtype=dtype)
        self._processor = AutoProcessor.from_pretrained(
            self._model_path, trust_remote_code=True, local_files_only=True
        )
        self._model = AutoModel.from_pretrained(self._model_path, **options)
        self._model = self._model.eval().to(device)
        self._device = str(device)

    def _generation_options(self, kwargs: dict) -> dict:
        options = {
            "max_new_tokens": kwargs.pop("max_new_tokens", 1024),
            "temperature": kwargs.pop("temperature", 0.8),
            "top_p": kwargs.pop("top_p", 0.95),
            "top_k": kwargs.pop("top_k", 50),
            "do_sample": kwargs.pop("do_sample", True),
        }
        if (
            not isinstance(options["max_new_tokens"], int)
            or options["max_new_tokens"] <= 0
        ):
            raise InvalidModelRequestError("max_new_tokens must be a positive integer.")
        if not isinstance(options["temperature"], (int, float)) or not (
            options["temperature"] > 0
        ):
            raise InvalidModelRequestError("temperature must be positive.")
        if not isinstance(options["top_p"], (int, float)) or not (
            0 < options["top_p"] <= 1
        ):
            raise InvalidModelRequestError("top_p must be in (0, 1].")
        if (
            not isinstance(options["top_k"], int)
            or isinstance(options["top_k"], bool)
            or options["top_k"] <= 0
        ):
            raise InvalidModelRequestError("top_k must be a positive integer.")
        if not isinstance(options["do_sample"], bool):
            raise InvalidModelRequestError("do_sample must be a boolean.")
        from ..utils import resolve_media_seed

        seed = resolve_media_seed(kwargs.pop("seed", None))
        if seed is not None:
            import torch

            options["generator"] = torch.Generator(device=self._device).manual_seed(
                seed
            )
        return options

    def _generate(
        self,
        text: str,
        reference: Optional[str],
        transcript: Optional[str],
        options: dict,
    ):
        import torch

        if self._model is None or self._processor is None:
            raise RuntimeError("Audio8-TTS model is not loaded.")
        processor_options = {"text": [text], "return_tensors": "pt"}
        if reference is not None:
            processor_options.update(
                reference_audio=[reference], reference_text=[transcript]
            )
        inputs = self._processor(**processor_options)
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        with torch.inference_mode():
            output = self._model.generate(
                **inputs, **options, return_dict_in_generate=True
            )
            waveforms, lengths = self._model.decode_audio(output.codes)
        audio = waveforms[0, : int(lengths[0])].detach().float().cpu().unsqueeze(0)
        return audio, int(self._model.config.codec_sample_rate)

    @staticmethod
    def _encode_audio(response_format: str, sample_rate: int, audio) -> bytes:
        fmt = response_format.lower()
        if fmt not in {"mp3", "wav", "flac", "ogg", "pcm"}:
            from .utils import audio_to_bytes

            return audio_to_bytes(response_format, sample_rate, audio)

        # The bundled SoundFile encoder avoids TorchCodec's Torch ABI and
        # platform wheel requirements for the common speech response formats.
        import soundfile

        options = {"format": fmt.upper()}
        if fmt == "pcm":
            options.update(format="RAW", subtype="PCM_16", endian="LITTLE")
        elif fmt == "ogg":
            options["subtype"] = "VORBIS"
        with io.BytesIO() as output:
            soundfile.write(
                output, audio.detach().float().cpu().numpy().T, sample_rate, **options
            )
            return output.getvalue()

    def speech(
        self,
        input: str,
        voice: Optional[str],
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ):
        if stream:
            raise InvalidModelRequestError(
                "Audio8-TTS does not support streaming in Xinference."
            )
        if not isinstance(input, str) or not input.strip():
            raise InvalidModelRequestError("Audio8-TTS input must be non-empty text.")
        if speed != 1.0:
            raise InvalidModelRequestError("Audio8-TTS only supports speed=1.0.")
        if voice:
            logger.warning(
                "Audio8-TTS ignores named voice=%r; use prompt_speech for cloning.",
                voice,
            )
        reference = kwargs.pop("prompt_speech", None)
        transcript = kwargs.pop("prompt_text", None)
        if reference is not None and (
            not isinstance(reference, bytes) or not reference
        ):
            raise InvalidModelRequestError(
                "prompt_speech must contain non-empty audio bytes."
            )
        if (reference is None) != (transcript is None):
            raise InvalidModelRequestError(
                "prompt_speech and prompt_text must be provided together."
            )
        if transcript is not None and (
            not isinstance(transcript, str) or not transcript.strip()
        ):
            raise InvalidModelRequestError("prompt_text must be non-empty text.")
        # Audio8 infers language from the text; no language token is accepted upstream.
        language = kwargs.pop("language", None)
        if language and (
            not isinstance(language, str)
            or (
                language.lower() != "auto"
                and language not in (self._model_spec.model_lang or [])
            )
        ):
            raise InvalidModelRequestError(
                f"Unsupported Audio8-TTS language: {language}"
            )
        options = self._generation_options(kwargs)
        if kwargs:
            raise InvalidModelRequestError(
                f"Unsupported Audio8-TTS arguments: {sorted(kwargs)}"
            )
        with tempfile.TemporaryDirectory() as directory:
            reference_path = None
            if reference is not None:
                reference_path = os.path.join(directory, "reference.wav")
                Path(reference_path).write_bytes(reference)
            audio, sample_rate = self._generate(
                input.strip(), reference_path, transcript, options
            )
        return self._encode_audio(response_format, sample_rate, audio)
