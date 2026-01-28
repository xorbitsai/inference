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
import logging
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Union

from ..core import CacheableModelSpec, VirtualEnvSettings
from ..utils import ModelInstanceInfoMixin
from .chattts import ChatTTSModel
from .cosyvoice import CosyVoiceModel
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
from .whisper import WhisperModel
from .whisper_mlx import WhisperMLXModel

logger = logging.getLogger(__name__)

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
        }

    def to_version_info(self):
        from ..cache_manager import CacheManager

        cache_manager = CacheManager(self)

        return {
            "model_version": self.model_name,
            "model_file_location": cache_manager.get_cache_dir(),
            "cache_status": cache_manager.get_cache_status(),
        }


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
) -> AudioModelFamilyV2:
    from ..utils import download_from_modelscope
    from . import BUILTIN_AUDIO_MODELS
    from .custom import get_user_defined_audios

    for model_spec in get_user_defined_audios():
        if model_spec.model_name == model_name:
            return model_spec

    if model_name in BUILTIN_AUDIO_MODELS:
        model_families = BUILTIN_AUDIO_MODELS[model_name]
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
                return [x for x in model_families if x.model_hub == "huggingface"][0]

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
    KokoroModel,
    KokoroMLXModel,
    KokoroZHModel,
    MegaTTSModel,
    Indextts2,
]:
    from ..cache_manager import CacheManager

    kwargs.pop("enable_virtual_env", None)
    model_spec = match_audio(model_name, download_hub)
    if model_path is None:
        cache_manager = CacheManager(model_spec)
        model_path = cache_manager.cache()
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
    ]
    if model_spec.model_family == "whisper":
        if not model_spec.engine:
            model = WhisperModel(model_uid, model_path, model_spec, **kwargs)
        else:
            model = WhisperMLXModel(model_uid, model_path, model_spec, **kwargs)
    elif model_spec.model_family == "funasr":
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
    else:
        raise Exception(f"Unsupported audio model family: {model_spec.model_family}")
    return model
