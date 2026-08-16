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

import importlib.util
import platform
from typing import TYPE_CHECKING, Tuple, Union

from ..utils import has_cuda_device, virtual_env_allows_missing_engine
from .engine_family import SUPPORTED_ENGINES, AudioEngineModel
from .f5tts import F5TTSModel
from .f5tts_mlx import F5TTSMLXModel
from .funasr import FunASRModel
from .kokoro import KokoroModel
from .kokoro_mlx import KokoroMLXModel
from .melotts import MeloTTSModel
from .mlx_audio import MLXAudioSTTModel, MLXAudioTTSModel
from .minimax_music3 import MiniMaxMusic3Model
from .qwen3_asr import Qwen3ASRModel
from .qwen3_tts import Qwen3TTSModel
from .vllm import VLLMQwen3ASRModel
from .voxcpm import VoxCPMModel
from .whisper import WhisperModel
from .whisper_mlx import WhisperMLXModel

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2


WHISPER_MLX_MODEL_NAMES = {
    "whisper-tiny",
    "whisper-tiny.en",
    "whisper-base",
    "whisper-base.en",
    "whisper-small",
    "whisper-small.en",
    "whisper-medium",
    "whisper-medium.en",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
}

MLX_AUDIO_STT_MODEL_NAMES = {
    "SenseVoiceSmall",
    "Fun-ASR-Nano-2512",
    "Qwen3-ASR-0.6B",
    "Qwen3-ASR-1.7B",
}

MLX_AUDIO_TTS_MODEL_NAMES = {
    "MeloTTS-English",
    "MeloTTS-English-v3",
    "Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "VoxCPM2",
}


def _is_engine(model_family: "AudioModelFamilyV2", engine: str) -> bool:
    return (getattr(model_family, "engine", "") or "").lower() == engine.lower()


def _is_engine_or_unspecified(model_family: "AudioModelFamilyV2", engine: str) -> bool:
    configured_engine = getattr(model_family, "engine", "") or ""
    return not configured_engine or configured_engine.lower() == engine.lower()


class TransformersQwen3ASRAudioModel(Qwen3ASRModel, AudioEngineModel):
    required_libs = ("qwen_asr",)

    @classmethod
    def match(cls, model_family: "AudioModelFamilyV2") -> bool:
        return model_family.model_family == "qwen3_asr" and _is_engine_or_unspecified(
            model_family, "transformers"
        )

    @classmethod
    def is_model_family_supported(cls, model_family: "AudioModelFamilyV2") -> bool:
        return model_family.model_family == "qwen3_asr"


class VLLMQwen3ASRAudioModel(VLLMQwen3ASRModel, AudioEngineModel):
    required_libs = ("qwen_asr", "vllm")

    @classmethod
    def match(cls, model_family: "AudioModelFamilyV2") -> bool:
        if platform.system() != "Linux":
            return False
        if not has_cuda_device():
            return False
        return model_family.model_family == "qwen3_asr"

    @classmethod
    def is_model_family_supported(cls, model_family: "AudioModelFamilyV2") -> bool:
        return model_family.model_family == "qwen3_asr"


class TransformersWhisperAudioModel(WhisperModel, AudioEngineModel):
    required_libs = ("transformers",)

    @classmethod
    def match(cls, model_family: "AudioModelFamilyV2") -> bool:
        return model_family.model_family == "whisper" and _is_engine(
            model_family, "transformers"
        )

    @classmethod
    def is_model_family_supported(cls, model_family: "AudioModelFamilyV2") -> bool:
        return model_family.model_family == "whisper"


class PyTorchF5TTSAudioModel(F5TTSModel, AudioEngineModel):
    required_libs = ("torch", "torchdiffeq", "x_transformers", "vocos")

    @classmethod
    def match(cls, model_family: "AudioModelFamilyV2") -> bool:
        return model_family.model_family == "F5-TTS" and _is_engine(
            model_family, "PyTorch"
        )

    @classmethod
    def is_model_family_supported(cls, model_family: "AudioModelFamilyV2") -> bool:
        return model_family.model_name == "F5-TTS"


class PyTorchKokoroAudioModel(KokoroModel, AudioEngineModel):
    required_libs = ("torch", "kokoro")

    @classmethod
    def match(cls, model_family: "AudioModelFamilyV2") -> bool:
        return model_family.model_family == "Kokoro" and _is_engine(
            model_family, "PyTorch"
        )

    @classmethod
    def is_model_family_supported(cls, model_family: "AudioModelFamilyV2") -> bool:
        return model_family.model_name == "Kokoro-82M"


class PyTorchFunASRAudioModel(FunASRModel, AudioEngineModel):
    required_libs = ("funasr",)

    @classmethod
    def match(cls, model_family: "AudioModelFamilyV2") -> bool:
        return (
            model_family.model_name in MLX_AUDIO_STT_MODEL_NAMES
            and model_family.model_family == "funasr"
            and _is_engine_or_unspecified(model_family, "PyTorch")
        )

    @classmethod
    def is_model_family_supported(cls, model_family: "AudioModelFamilyV2") -> bool:
        return (
            model_family.model_name in MLX_AUDIO_STT_MODEL_NAMES
            and model_family.model_family == "funasr"
        )


class PyTorchQwen3TTSAudioModel(Qwen3TTSModel, AudioEngineModel):
    required_libs = ("qwen_tts",)

    @classmethod
    def match(cls, model_family: "AudioModelFamilyV2") -> bool:
        return (
            model_family.model_name in MLX_AUDIO_TTS_MODEL_NAMES
            and model_family.model_family == "qwen3_tts"
            and _is_engine_or_unspecified(model_family, "PyTorch")
        )

    @classmethod
    def is_model_family_supported(cls, model_family: "AudioModelFamilyV2") -> bool:
        return (
            model_family.model_name in MLX_AUDIO_TTS_MODEL_NAMES
            and model_family.model_family == "qwen3_tts"
        )


class PyTorchMeloTTSAudioModel(MeloTTSModel, AudioEngineModel):
    required_libs = ("torch", "melo")

    @classmethod
    def match(cls, model_family: "AudioModelFamilyV2") -> bool:
        return (
            model_family.model_name in MLX_AUDIO_TTS_MODEL_NAMES
            and model_family.model_family == "MeloTTS"
            and _is_engine_or_unspecified(model_family, "PyTorch")
        )

    @classmethod
    def is_model_family_supported(cls, model_family: "AudioModelFamilyV2") -> bool:
        return (
            model_family.model_name in MLX_AUDIO_TTS_MODEL_NAMES
            and model_family.model_family == "MeloTTS"
        )


class PyTorchVoxCPMAudioModel(VoxCPMModel, AudioEngineModel):
    required_libs = ("voxcpm",)

    @classmethod
    def match(cls, model_family: "AudioModelFamilyV2") -> bool:
        return model_family.model_name == "VoxCPM2" and _is_engine_or_unspecified(
            model_family, "PyTorch"
        )

    @classmethod
    def is_model_family_supported(cls, model_family: "AudioModelFamilyV2") -> bool:
        return model_family.model_name == "VoxCPM2"


class MLXWhisperAudioModel(WhisperMLXModel, AudioEngineModel):
    required_libs = ("mlx",)

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        if importlib.util.find_spec("mlx") is None:
            return False, "Library 'mlx' is not installed"
        if (
            importlib.util.find_spec("lightning_whisper_mlx") is None
            and importlib.util.find_spec("mlx_whisper") is None
        ):
            return (
                False,
                "Neither 'lightning_whisper_mlx' nor 'mlx_whisper' is installed",
            )
        return True

    @classmethod
    def match(cls, model_family: "AudioModelFamilyV2") -> bool:
        return (
            platform.system() == "Darwin"
            and platform.processor() == "arm"
            and model_family.model_name in WHISPER_MLX_MODEL_NAMES
            and _is_engine(model_family, "MLX")
        )

    @classmethod
    def is_model_family_supported(cls, model_family: "AudioModelFamilyV2") -> bool:
        return model_family.model_name in WHISPER_MLX_MODEL_NAMES


class MLXF5TTSAudioModel(F5TTSMLXModel, AudioEngineModel):
    required_libs = ("mlx", "f5_tts_mlx", "vocos_mlx")

    @classmethod
    def match(cls, model_family: "AudioModelFamilyV2") -> bool:
        return (
            platform.system() == "Darwin"
            and platform.processor() == "arm"
            and model_family.model_family == "F5-TTS-MLX"
            and _is_engine(model_family, "MLX")
        )

    @classmethod
    def is_model_family_supported(cls, model_family: "AudioModelFamilyV2") -> bool:
        return model_family.model_name == "F5-TTS"


class MLXKokoroAudioModel(KokoroMLXModel, AudioEngineModel):
    required_libs = ("mlx", "mlx_audio")

    @classmethod
    def match(cls, model_family: "AudioModelFamilyV2") -> bool:
        return (
            platform.system() == "Darwin"
            and platform.processor() == "arm"
            and model_family.model_family == "Kokoro-MLX"
            and _is_engine(model_family, "MLX")
        )

    @classmethod
    def is_model_family_supported(cls, model_family: "AudioModelFamilyV2") -> bool:
        return model_family.model_name == "Kokoro-82M"


class MLXAudioSTTEngineModel(MLXAudioSTTModel, AudioEngineModel):
    required_libs = ("mlx", "mlx_audio")

    @classmethod
    def match(cls, model_family: "AudioModelFamilyV2") -> bool:
        return (
            platform.system() == "Darwin"
            and platform.processor() == "arm"
            and model_family.model_name in MLX_AUDIO_STT_MODEL_NAMES
            and _is_engine(model_family, "MLX")
        )

    @classmethod
    def is_model_family_supported(cls, model_family: "AudioModelFamilyV2") -> bool:
        return model_family.model_name in MLX_AUDIO_STT_MODEL_NAMES


class MLXAudioTTSEngineModel(MLXAudioTTSModel, AudioEngineModel):
    required_libs = ("mlx", "mlx_audio")

    @classmethod
    def match(cls, model_family: "AudioModelFamilyV2") -> bool:
        return (
            platform.system() == "Darwin"
            and platform.processor() == "arm"
            and model_family.model_name in MLX_AUDIO_TTS_MODEL_NAMES
            and _is_engine(model_family, "MLX")
        )

    @classmethod
    def is_model_family_supported(cls, model_family: "AudioModelFamilyV2") -> bool:
        return model_family.model_name in MLX_AUDIO_TTS_MODEL_NAMES


class DiffusersMiniMaxMusic3AudioModel(MiniMaxMusic3Model, AudioEngineModel):
    required_libs = ("diffusers",)

    @classmethod
    def check_lib(cls):
        if virtual_env_allows_missing_engine():
            return True
        return super().check_lib()

    @classmethod
    def match(cls, model_family: "AudioModelFamilyV2") -> bool:
        return model_family.model_family == "minimax_music3"


def register_builtin_audio_engines() -> None:
    # the first registered engine is the default one for a model
    SUPPORTED_ENGINES["transformers"] = [
        TransformersQwen3ASRAudioModel,
        TransformersWhisperAudioModel,
    ]
    SUPPORTED_ENGINES["vLLM"] = [VLLMQwen3ASRAudioModel]
    SUPPORTED_ENGINES["PyTorch"] = [
        PyTorchF5TTSAudioModel,
        PyTorchKokoroAudioModel,
        PyTorchFunASRAudioModel,
        PyTorchQwen3TTSAudioModel,
        PyTorchMeloTTSAudioModel,
        PyTorchVoxCPMAudioModel,
    ]
    SUPPORTED_ENGINES["MLX"] = [
        MLXWhisperAudioModel,
        MLXF5TTSAudioModel,
        MLXKokoroAudioModel,
        MLXAudioSTTEngineModel,
        MLXAudioTTSEngineModel,
    ]
    SUPPORTED_ENGINES["diffusers"] = [DiffusersMiniMaxMusic3AudioModel]
