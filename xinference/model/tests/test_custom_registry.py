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

"""Custom models must not reuse the cache directory of another model."""

import pytest

from ... import constants
from ..cache_manager import CacheManager


@pytest.fixture
def isolated_cache_home(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "XINFERENCE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(constants, "XINFERENCE_MODEL_DIR", str(tmp_path / "model"))
    monkeypatch.setattr(CacheManager, "is_initialized", False)
    weights = tmp_path / "weights"
    weights.mkdir()
    return weights


def _volume_ignores_case(monkeypatch, ignores_case):
    from ..llm import custom as llm_custom

    # stands in for the cache volume, so the result does not depend on the host
    monkeypatch.setattr(llm_custom, "_volume_ignores_case", lambda path: ignores_case)


def _image(name, weights):
    from ..image.custom import CustomImageModelFamilyV2, ImageModelRegistry

    family = CustomImageModelFamilyV2(
        model_family="stable_diffusion",
        model_name=name,
        model_uri=str(weights),
        model_ability=["text2image"],
    )
    return ImageModelRegistry(), family


def _audio(name, weights, cache_name=None):
    from ..audio.custom import AudioModelRegistry, CustomAudioModelFamilyV2

    family = CustomAudioModelFamilyV2(
        model_family="whisper",
        model_name=name,
        model_uri=str(weights),
        multilingual=True,
        model_ability=["audio2text"],
        cache_name=cache_name,
    )
    return AudioModelRegistry(), family


def _embedding(name, weights):
    from ..embedding.core import TransformersEmbeddingSpecV1
    from ..embedding.custom import CustomEmbeddingModelFamilyV2, EmbeddingModelRegistry

    family = CustomEmbeddingModelFamilyV2(
        model_name=name,
        dimensions=1024,
        max_tokens=8192,
        language=["en"],
        model_specs=[
            TransformersEmbeddingSpecV1(
                model_format="pytorch", model_uri=str(weights), quantization="none"
            )
        ],
    )
    return EmbeddingModelRegistry(), family


def _rerank(name, weights):
    from ..rerank.core import TransformersRerankSpecV1
    from ..rerank.custom import CustomRerankModelFamilyV2, RerankModelRegistry

    family = CustomRerankModelFamilyV2(
        model_name=name,
        language=["en"],
        model_specs=[
            TransformersRerankSpecV1(
                model_format="pytorch", model_uri=str(weights), quantization="none"
            )
        ],
    )
    return RerankModelRegistry(), family


@pytest.mark.parametrize(
    "make, name, builtin",
    [
        # "." becomes "_" in cache directory names
        (_image, "FLUX_1-dev", "FLUX.1-dev"),
        (_audio, "whisper-base_en", "whisper-base.en"),
    ],
)
@pytest.mark.parametrize("ignores_case", [True, False])
def test_custom_model_cannot_share_a_builtin_cache_directory(
    isolated_cache_home, monkeypatch, make, name, builtin, ignores_case
):
    _volume_ignores_case(monkeypatch, ignores_case)
    registry, family = make(name, isolated_cache_home)

    with pytest.raises(ValueError, match=f"{builtin}.*same cache directory"):
        registry.register(family, persist=False)
    assert family not in registry.get_custom_models()


def test_custom_audio_cannot_reuse_a_builtin_cache_name(
    isolated_cache_home, monkeypatch
):
    from ..audio import BUILTIN_AUDIO_MODELS

    _volume_ignores_case(monkeypatch, False)
    builtin = next(
        family
        for families in BUILTIN_AUDIO_MODELS.values()
        for family in families
        if family.cache_name
    )
    # e.g. a definition copied from a built-in one and given a new name
    registry, family = _audio(
        "my-audio-model", isolated_cache_home, cache_name=builtin.cache_name
    )

    with pytest.raises(ValueError, match=f"{builtin.model_name}.*cache directory"):
        registry.register(family, persist=False)
    assert family not in registry.get_custom_models()


@pytest.mark.parametrize(
    "make, name, builtin",
    [
        (_image, "flux.1-DEV", "FLUX.1-dev"),
        (_audio, "Whisper-Base.en", "whisper-base.en"),
        (_embedding, "BGE-M3", "bge-m3"),
        (_rerank, "BGE-Reranker-v2-M3", "bge-reranker-v2-m3"),
    ],
)
@pytest.mark.parametrize("ignores_case", [True, False])
def test_custom_model_named_like_a_builtin_up_to_case(
    isolated_cache_home, monkeypatch, make, name, builtin, ignores_case
):
    _volume_ignores_case(monkeypatch, ignores_case)
    registry, family = make(name, isolated_cache_home)

    if ignores_case:  # one directory on such a volume
        with pytest.raises(ValueError, match=f"{builtin}.*same cache directory"):
            registry.register(family, persist=False)
        assert family not in registry.get_custom_models()
    else:
        registry.check_cache_identity(family)


@pytest.mark.parametrize("make", [_image, _audio, _embedding, _rerank])
def test_custom_model_with_its_own_cache_directory_is_accepted(
    isolated_cache_home, monkeypatch, make
):
    _volume_ignores_case(monkeypatch, True)
    registry, family = make("my-custom-model", isolated_cache_home)

    registry.check_cache_identity(family)
    # A registered custom model is checked like a built-in one.
    registry.models = [family]
    _, twin = make("MY-CUSTOM-MODEL", isolated_cache_home)
    with pytest.raises(ValueError, match="my-custom-model.*same cache directory"):
        registry.check_cache_identity(twin)
