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
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Union

from ..core import CacheableModelSpec, VirtualEnvSettings
from ..utils import ModelInstanceInfoMixin
from .chattts import ChatTTSModel
from .cosyvoice import CosyVoiceModel
from .engine_family import AudioEngineModel
from .f5tts import F5TTSModel
from .f5tts_mlx import F5TTSMLXModel
from .fish_speech import FishSpeechModel
from .funasr import FunASRModel
from .indextts2 import Indextts2
from .kokoro import KokoroModel
from .kokoro_mlx import KokoroMLXModel
from .kokoro_zh import KokoroZHModel
from .megatts import MegaTTSModel
from .melotts import MeloTTSModel
from .minimax_music3 import MiniMaxMusic3Model
from .mlx_audio import MLXAudioSTTModel, MLXAudioTTSModel
from .qwen3_asr import Qwen3ASRModel
from .qwen3_tts import Qwen3TTSModel
from .speaker_embedding import ModelScopeSpeakerEmbeddingModel
from .voxcpm import VoxCPMModel
from .whisper import WhisperModel
from .whisper_mlx import WhisperMLXModel

logger = logging.getLogger(__name__)

LEGACY_AUDIO_MODEL_ALIASES = {
    "whisper-tiny-mlx": ("whisper-tiny", "MLX"),
    "whisper-tiny.en-mlx": ("whisper-tiny.en", "MLX"),
    "whisper-base-mlx": ("whisper-base", "MLX"),
    "whisper-base.en-mlx": ("whisper-base.en", "MLX"),
    "whisper-small-mlx": ("whisper-small", "MLX"),
    "whisper-small.en-mlx": ("whisper-small.en", "MLX"),
    "whisper-medium-mlx": ("whisper-medium", "MLX"),
    "whisper-medium.en-mlx": ("whisper-medium.en", "MLX"),
    "whisper-large-v3-mlx": ("whisper-large-v3", "MLX"),
    "whisper-large-v3-turbo-mlx": ("whisper-large-v3-turbo", "MLX"),
    "F5-TTS-MLX": ("F5-TTS", "MLX"),
    "Kokoro-82M-MLX": ("Kokoro-82M", "MLX"),
}

# Init when registering all the builtin models.
AUDIO_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)


def get_audio_model_descriptions():
    import copy

    return copy.deepcopy(AUDIO_MODEL_DESCRIPTIONS)


class AudioModelFamilyV2(CacheableModelSpec, ModelInstanceInfoMixin):
    version: Literal[2]
    model_family: str
    model_name: str
    model_id: str
    model_revision: Optional[str]
    multilingual: bool
    language: Optional[str]
    model_ability: Optional[List[str]]
    default_model_config: Optional[Dict[str, Any]]
    default_transcription_config: Optional[Dict[str, Any]]
    engine: Optional[str]
    model_format: Optional[str]
    cache_name: Optional[str]
    virtualenv: Optional[VirtualEnvSettings]

    class Config:
        extra = "allow"

    def to_description(self):
        return {
            "model_type": "audio",
            "address": getattr(self, "address", None),
            "accelerators": getattr(self, "accelerators", None),
            "model_name": self.model_name,
            "model_family": self.model_family,
            "model_revision": self.model_revision,
            "model_ability": self.model_ability,
            "model_engine": getattr(self, "model_engine", None),
        }

    def to_version_info(self):
        from ..cache_manager import CacheManager

        cache_manager = CacheManager(self)

        return {
            "model_version": self.cache_name or self.model_name,
            "model_file_location": cache_manager.get_cache_dir(),
            "cache_status": cache_manager.get_cache_status(),
        }


def resolve_audio_model_name_and_engine(
    model_name: str,
    model_engine: Optional[str] = None,
    use_default_engine: bool = False,
) -> tuple[str, Optional[str]]:
    """Resolve retired MLX model names to the canonical model plus engine."""

    alias = LEGACY_AUDIO_MODEL_ALIASES.get(model_name)
    if alias is not None:
        canonical_name, alias_engine = alias
        if model_engine is not None and model_engine.lower() != alias_engine.lower():
            raise ValueError(
                f"Legacy audio model name {model_name} selects engine {alias_engine}; "
                f"it cannot be launched with engine {model_engine}."
            )
        model_name, model_engine = canonical_name, alias_engine

    if use_default_engine or model_engine is not None:
        from .engine_family import AUDIO_ENGINES

        available_engines = AUDIO_ENGINES.get(model_name)
        if available_engines and model_engine is None:
            model_engine = next(iter(available_engines))
        elif available_engines and model_engine is not None:
            model_engine = next(
                (
                    engine
                    for engine in available_engines
                    if engine.lower() == model_engine.lower()
                ),
                model_engine,
            )
    return model_name, model_engine


def generate_audio_description(
    audio_model: AudioModelFamilyV2,
) -> Dict[str, List[Dict]]:
    res = defaultdict(list)
    res[audio_model.model_name].append(audio_model.to_version_info())
    return res


def match_audio(
    model_name: str,
    download_hub: Optional[
        Literal["huggingface", "modelscope", "openmind_hub", "csghub"]
    ] = None,
    model_engine: Optional[str] = None,
) -> AudioModelFamilyV2:
    from ..utils import download_from_modelscope
    from . import BUILTIN_AUDIO_MODELS
    from .custom import get_user_defined_audios

    model_name, model_engine = resolve_audio_model_name_and_engine(
        model_name, model_engine
    )

    for model_spec in get_user_defined_audios():
        if model_spec.model_name == model_name:
            return model_spec

    if model_name in BUILTIN_AUDIO_MODELS:
        model_families = BUILTIN_AUDIO_MODELS[model_name]
        if model_engine is not None:
            engine_families = [
                family
                for family in model_families
                if (family.engine or "").lower() == model_engine.lower()
            ]
            if engine_families:
                model_families = engine_families
        if download_hub is not None:
            if download_hub == "modelscope":
                return (
                    [x for x in model_families if x.model_hub == "modelscope"]
                    + [x for x in model_families if x.model_hub == "huggingface"]
                )[0]
            else:
                return [x for x in model_families if x.model_hub == download_hub][0]
        else:
            if download_from_modelscope():
                return (
                    [x for x in model_families if x.model_hub == "modelscope"]
                    + [x for x in model_families if x.model_hub == "huggingface"]
                )[0]
            else:
                return (
                    [x for x in model_families if x.model_hub == "huggingface"]
                    + [x for x in model_families if x.model_hub == "modelscope"]
                )[0]

    else:
        raise ValueError(
            f"Audio model {model_name} not found, available"
            f"model list: {BUILTIN_AUDIO_MODELS.keys()}"
        )


def create_audio_model_instance(
    model_uid: str,
    model_name: str,
    download_hub: Optional[
        Literal["huggingface", "modelscope", "openmind_hub", "csghub"]
    ] = None,
    model_path: Optional[str] = None,
    model_engine: Optional[str] = None,
    **kwargs,
) -> Union[
    WhisperModel,
    WhisperMLXModel,
    FunASRModel,
    ChatTTSModel,
    CosyVoiceModel,
    FishSpeechModel,
    F5TTSModel,
    F5TTSMLXModel,
    MeloTTSModel,
    MiniMaxMusic3Model,
    KokoroModel,
    KokoroMLXModel,
    KokoroZHModel,
    MegaTTSModel,
    Indextts2,
    Qwen3ASRModel,
    Qwen3TTSModel,
    VoxCPMModel,
    ModelScopeSpeakerEmbeddingModel,
    MLXAudioSTTModel,
    MLXAudioTTSModel,
    AudioEngineModel,
]:
    from ..cache_manager import CacheManager
    from .engine_family import AUDIO_ENGINES

    enable_virtual_env = kwargs.pop("enable_virtual_env", None)
    model_name, model_engine = resolve_audio_model_name_and_engine(
        model_name, model_engine
    )
    model_spec = match_audio(model_name, download_hub, model_engine=model_engine)
    audio_cls = None

    # Engine-aware dispatch for model families with multiple engines
    # (e.g. qwen3_asr on transformers or vLLM). Families without registered
    # engines keep the legacy dispatch below.
    if model_spec.model_name in AUDIO_ENGINES or model_engine is not None:
        from .engine_family import (
            check_engine_by_model_name_and_engine,
            check_engine_by_model_name_and_engine_with_virtual_env,
        )

        if model_engine is None:
            # the first registered engine is the default one
            model_engine = next(iter(AUDIO_ENGINES[model_spec.model_name]))

        if model_spec.model_name not in AUDIO_ENGINES:
            logger.warning(
                "Audio model %s does not support engine selection, "
                "`model_engine=%s` is ignored.",
                model_spec.model_name,
                model_engine,
            )
        else:
            if enable_virtual_env is None:
                from ...constants import XINFERENCE_ENABLE_VIRTUAL_ENV

                enable_virtual_env = XINFERENCE_ENABLE_VIRTUAL_ENV
            if enable_virtual_env:
                audio_cls = check_engine_by_model_name_and_engine_with_virtual_env(
                    model_engine,
                    model_spec.model_name,
                    model_family=model_spec,
                )
            else:
                audio_cls = check_engine_by_model_name_and_engine(
                    model_engine,
                    model_spec.model_name,
                )

    if model_path is None:
        cache_manager = CacheManager(model_spec)
        model_path = cache_manager.cache()

    if audio_cls is not None:
        model_spec = model_spec.copy()
        model_spec.model_engine = model_engine
        return audio_cls(model_uid, model_path, model_spec, **kwargs)  # type: ignore

    model: Union[
        WhisperModel,
        WhisperMLXModel,
        FunASRModel,
        ChatTTSModel,
        CosyVoiceModel,
        FishSpeechModel,
        F5TTSModel,
        F5TTSMLXModel,
        MeloTTSModel,
        KokoroModel,
        KokoroMLXModel,
        KokoroZHModel,
        MegaTTSModel,
        Indextts2,
        Qwen3ASRModel,
        Qwen3TTSModel,
        VoxCPMModel,
        ModelScopeSpeakerEmbeddingModel,
        MLXAudioSTTModel,
        MLXAudioTTSModel,
    ]
    if model_spec.model_family == "whisper":
        if (model_spec.engine or "").lower() == "mlx":
            model = WhisperMLXModel(model_uid, model_path, model_spec, **kwargs)
        else:
            model = WhisperModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "funasr":
        if (model_spec.engine or "").lower() == "mlx":
            model = MLXAudioSTTModel(model_uid, model_path, model_spec, **kwargs)
        else:
            model = FunASRModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "ChatTTS":
        model = ChatTTSModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "CosyVoice":
        model = CosyVoiceModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "FishAudio":
        model = FishSpeechModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "F5-TTS":
        model = F5TTSModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "F5-TTS-MLX":
        model = F5TTSMLXModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "MeloTTS":
        if (model_spec.engine or "").lower() == "mlx":
            model = MLXAudioTTSModel(model_uid, model_path, model_spec, **kwargs)
        else:
            model = MeloTTSModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "Kokoro":
        model = KokoroModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "Kokoro-zh":
        model = KokoroZHModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "Kokoro-MLX":
        model = KokoroMLXModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "MegaTTS":
        model = MegaTTSModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "IndexTTS2":
        model = Indextts2(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "qwen3_asr":
        if (model_spec.engine or "").lower() == "mlx":
            model = MLXAudioSTTModel(model_uid, model_path, model_spec, **kwargs)
        else:
            model = Qwen3ASRModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "qwen3_tts":
        if (model_spec.engine or "").lower() == "mlx":
            model = MLXAudioTTSModel(model_uid, model_path, model_spec, **kwargs)
        else:
            model = Qwen3TTSModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "VoxCPM":
        if (model_spec.engine or "").lower() == "mlx":
            model = MLXAudioTTSModel(model_uid, model_path, model_spec, **kwargs)
        else:
            model = VoxCPMModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "campplus":
        model = ModelScopeSpeakerEmbeddingModel(
            model_uid, model_path, model_spec, **kwargs
        )
    else:
        raise Exception(f"Unsupported audio model family: {model_spec.model_family}")
    return model
