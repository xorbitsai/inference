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
import importlib
import logging
import math
import os
import sys
from typing import TYPE_CHECKING, Optional

from ..utils import set_all_random_seed

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)

_SPEED_FACTOR_MIN = 0.5
_SPEED_FACTOR_MAX = 2.0


def _validate_speed_factor(name: str, value: float) -> float:
    try:
        normalized_value = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be between {_SPEED_FACTOR_MIN} and "
            f"{_SPEED_FACTOR_MAX}, got {value!r}"
        ) from None

    if not (
        math.isfinite(normalized_value)
        and _SPEED_FACTOR_MIN <= normalized_value <= _SPEED_FACTOR_MAX
    ):
        raise ValueError(
            f"{name} must be between {_SPEED_FACTOR_MIN} and "
            f"{_SPEED_FACTOR_MAX}, got {value!r}"
        )
    return normalized_value


def _resolve_duration_factor(
    speed: Optional[float], duration_factor: Optional[float]
) -> float:
    normalized_speed = _validate_speed_factor("speed", 1.0 if speed is None else speed)
    if duration_factor is not None:
        return _validate_speed_factor("duration_factor", duration_factor)

    # OpenAI speed > 1 means faster, while IndexTTS duration_factor
    # > 1 means slower.
    return 1.0 / normalized_speed


def _load_indextts_2_5_runtime():
    previous_hf_cache = os.environ.get("HF_HUB_CACHE")
    try:
        # infer_v2_5 overwrites HF_HUB_CACHE at import time. Import Hub
        # constants first so that Xinference's configured cache remains in use.
        importlib.import_module("huggingface_hub.constants")
        from indextts.infer_v2_5 import IndexTTS2
    finally:
        if previous_hf_cache is None:
            os.environ.pop("HF_HUB_CACHE", None)
        else:
            os.environ["HF_HUB_CACHE"] = previous_hf_cache
    return IndexTTS2


class Indextts2:
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
        self._is_v2_5 = model_spec.model_name == "IndexTTS-2.5"

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def load(self):
        # The yaml config loaded from model has hard-coded the import paths
        thirdparty_dir = os.path.join(os.path.dirname(__file__), "../../thirdparty")
        sys.path.insert(0, thirdparty_dir)

        config_path = os.path.join(self._model_path, "config.yaml")
        use_deepspeed = self._kwargs.get("use_deepspeed", False)

        if getattr(self._model_spec, "model_hub", None) == "modelscope":
            os.environ.setdefault("USE_MODELSCOPE", "true")

        if self._is_v2_5:
            IndexTTS2 = _load_indextts_2_5_runtime()
            logger.info("Loading IndexTTS-2.5 model...")
            self._model = IndexTTS2(
                cfg_path=config_path,
                model_dir=self._model_path,
                use_bf16=self._kwargs.get("use_bf16", False),
                device=self._device,
                use_deepspeed=use_deepspeed,
                use_qwen_emo=self._kwargs.get("use_qwen_emo", False),
            )
            return

        from indextts.infer_v2 import IndexTTS2

        use_fp16 = self._kwargs.get("use_fp16", False)

        # Handle small model directory for offline deployment
        small_models_config = (
            self._model_spec.default_model_config
            if getattr(self._model_spec, "default_model_config", None)
            else {}
        )
        small_models_config.update(self._kwargs)

        small_models_dir = small_models_config.get("small_models_dir")
        logger.info(
            f"Loading IndexTTS2 model... (small_models_dir: {small_models_dir})"
        )
        self._model = IndexTTS2(
            cfg_path=config_path,
            model_dir=self._model_path,
            use_fp16=use_fp16,
            device=self._device,
            use_deepspeed=use_deepspeed,
            small_models_dir=small_models_dir,
        )

    def speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: Optional[float] = 1.0,
        stream: bool = False,
        **kwargs,
    ):
        from io import BytesIO

        import soundfile

        # Streaming support is now implemented

        prompt_speech: Optional[bytes] = kwargs.pop("prompt_speech", None)
        emo_prompt_speech: Optional[bytes] = kwargs.pop("emo_prompt_speech", None)
        emo_alpha: float = kwargs.pop("emo_alpha", 1.0)
        emo_text: Optional[str] = kwargs.pop("emo_text", None)
        use_random: bool = kwargs.pop("use_random", False)
        emo_vector: Optional[list] = kwargs.pop("emo_vector", None)
        seed: Optional[int] = kwargs.pop("seed", 0)
        use_emo_text: bool = kwargs.pop("use_emo_text", False)

        if self._is_v2_5:
            language = kwargs.pop("language", None)
            if language is None:
                language = kwargs.pop("lang", "ZH")
            else:
                kwargs.pop("lang", None)
            language = str(language).upper()
            duration_factor = _resolve_duration_factor(
                speed, kwargs.pop("duration_factor", None)
            )

        if prompt_speech is None:
            # IndexTTS2 requires reference audio for voice cloning
            # We'll provide a helpful error message with usage examples
            raise ValueError(
                "IndexTTS2 requires a reference audio for voice cloning.\n"
                "Please provide a short audio sample (3-10 seconds) as 'prompt_speech' parameter.\n"
                "Example usage:\n"
                "  with open('reference.wav', 'rb') as f:\n"
                "      prompt_speech = f.read()\n"
                "  audio_bytes = model.speech(\n"
                "      input='Hello, world!',\n"
                "      voice='default',\n"
                "      prompt_speech=prompt_speech"
                "  )\n\n"
                "For emotion control, you can also add:\n"
                "  emo_prompt_speech=emotion_audio_bytes  # Optional: emotion reference\n"
                "  emo_text='happy and cheerful'  # Optional: emotion description\n"
                "  emo_alpha=1.5  # Optional: emotion intensity"
            )

        assert self._model is not None

        set_all_random_seed(seed)

        # Save prompt speech to temp file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_prompt:
            temp_prompt.write(prompt_speech)
            temp_prompt_path = temp_prompt.name

        emo_prompt_path = None
        if emo_prompt_speech is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_emo:
                temp_emo.write(emo_prompt_speech)
                emo_prompt_path = temp_emo.name

        try:
            # Pass output_path=None so IndexTTS2 returns the generated waveform in
            # memory (as an int16 numpy array) instead of writing it to disk with
            # torchaudio.save. torchaudio.save routes through TorchCodec, whose
            # FFmpeg shared libraries are missing / ABI-mismatched in some images
            # (see #5201), which made /v1/audio/speech fail at the save step even
            # though inference had already succeeded. soundfile below performs the
            # actual encoding, so the on-disk WAV round-trip is not needed.
            if self._is_v2_5:
                sample_rate, audio = self._model.infer(
                    spk_audio_prompt=temp_prompt_path,
                    text=input,
                    output_path=None,
                    emo_audio_prompt=emo_prompt_path,
                    emo_alpha=emo_alpha,
                    emo_text=emo_text,
                    use_random=use_random,
                    emo_vector=emo_vector,
                    use_emo_text=use_emo_text,
                    lang=language,
                    duration_factor=duration_factor,
                    **kwargs,
                )
            else:
                sample_rate, audio = self._model.infer(
                    spk_audio_prompt=temp_prompt_path,
                    text=input,
                    output_path=None,
                    emo_audio_prompt=emo_prompt_path,
                    emo_alpha=emo_alpha,
                    emo_text=emo_text,
                    use_random=use_random,
                    emo_vector=emo_vector,
                    use_emo_text=use_emo_text,
                )

            if stream:
                # Streaming mode - return generator that yields chunks
                def audio_stream_generator():
                    with BytesIO() as out:
                        with soundfile.SoundFile(
                            out, "w", sample_rate, 1, format=response_format.upper()
                        ) as f:
                            f.write(audio)
                        complete_audio = out.getvalue()

                    # Yield the complete audio in chunks
                    chunk_size = 8192  # 8KB chunks
                    for i in range(0, len(complete_audio), chunk_size):
                        yield complete_audio[i : i + chunk_size]

                return audio_stream_generator()
            else:
                # Non-streaming mode - return bytes directly
                with BytesIO() as out:
                    with soundfile.SoundFile(
                        out, "w", sample_rate, 1, format=response_format.upper()
                    ) as f:
                        f.write(audio)
                    result = out.getvalue()

                return result
        finally:
            # Clean up temp files
            try:
                os.unlink(temp_prompt_path)
                if emo_prompt_path:
                    os.unlink(emo_prompt_path)
            except Exception:
                pass
