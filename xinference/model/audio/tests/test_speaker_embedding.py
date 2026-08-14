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

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from ..core import create_audio_model_instance, match_audio
from ..speaker_embedding import ModelScopeSpeakerEmbeddingModel


def _new_model():
    spec = SimpleNamespace(
        model_name="speech_campplus_sv_zh-cn_16k-common",
        model_ability=["speaker_embedding"],
        model_family="campplus",
    )
    return ModelScopeSpeakerEmbeddingModel("replica-uid", "/unused", spec)


def test_create_speaker_embedding_from_audio_bytes():
    model = _new_model()
    observed = {}

    def fake_pipeline(inputs, output_emb):
        path = Path(inputs[0])
        observed["path"] = path
        observed["audio"] = path.read_bytes()
        observed["output_emb"] = output_emb
        return {
            "embs": np.array([[0.25, -0.5, 0.75]], dtype=np.float32),
            "outputs": {"text": "No similarity score output"},
        }

    model._pipeline = fake_pipeline

    result = model.create_embedding(b"encoded-audio", model_uid="speaker-model")

    assert result == {
        "object": "embedding",
        "model": "speaker-model",
        "dimensions": 3,
        "embedding": [0.25, -0.5, 0.75],
    }
    assert observed["audio"] == b"encoded-audio"
    assert observed["output_emb"] is True
    assert not observed["path"].exists()


@pytest.mark.parametrize(
    "pipeline_result, match",
    [
        ({}, "invalid speaker embedding result"),
        ({"embs": np.empty((0, 192))}, "unexpected speaker embedding shape"),
        ({"embs": np.array([[np.nan]])}, "non-finite speaker embedding"),
    ],
)
def test_create_speaker_embedding_validates_pipeline_output(pipeline_result, match):
    model = _new_model()
    model._pipeline = lambda *_args, **_kwargs: pipeline_result

    with pytest.raises(RuntimeError, match=match):
        model.create_embedding(b"encoded-audio")


def test_create_speaker_embedding_rejects_empty_audio():
    model = _new_model()
    model._pipeline = object()

    with pytest.raises(ValueError, match="must not be empty"):
        model.create_embedding(b"")


@pytest.mark.parametrize(
    "model_name",
    [
        "speech_campplus_sv_zh-cn_16k-common",
        "speech_campplus_sv_zh_en_16k-common_advanced",
    ],
)
def test_campplus_builtin_registration(model_name):
    spec = match_audio(model_name, download_hub="modelscope")

    assert spec.model_family == "campplus"
    assert spec.model_ability == ["speaker_embedding"]

    model = create_audio_model_instance("uid", model_name, model_path="/fake/path")

    assert isinstance(model, ModelScopeSpeakerEmbeddingModel)


def test_campplus_load_falls_back_from_mps_to_cpu():
    pipeline = Mock()
    pipelines_module = ModuleType("modelscope.pipelines")
    pipelines_module.pipeline = pipeline
    constant_module = ModuleType("modelscope.utils.constant")
    constant_module.Tasks = SimpleNamespace(speaker_verification="speaker-verification")

    model = _new_model()
    with (
        patch.dict(
            sys.modules,
            {
                "modelscope.pipelines": pipelines_module,
                "modelscope.utils.constant": constant_module,
            },
        ),
        patch(
            "xinference.model.audio.speaker_embedding.get_available_device",
            return_value="mps",
        ),
    ):
        model.load()

    pipeline.assert_called_once_with(
        task="speaker-verification",
        model="/unused",
        device="cpu",
    )
