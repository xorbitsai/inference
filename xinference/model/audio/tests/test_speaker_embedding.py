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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from .. import load_model_family_from_json
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


def test_campplus_factory_dispatch():
    spec = SimpleNamespace(
        model_name="speaker-model",
        model_family="campplus",
        model_ability=["speaker_embedding"],
    )
    with patch("xinference.model.audio.core.match_audio", return_value=spec):
        model = create_audio_model_instance(
            "uid", "speaker-model", model_path="/fake/path"
        )

    assert isinstance(model, ModelScopeSpeakerEmbeddingModel)


def test_builtin_campplus_specs_are_modelscope_only_and_match_by_default():
    models = {}
    load_model_family_from_json("model_spec.json", models)

    expected = {
        "speech_campplus_sv_zh-cn_16k-common",
        "speech_campplus_sv_zh_en_16k-common_advanced",
    }
    for model_name in expected:
        specs = models[model_name]
        assert len(specs) == 1
        assert specs[0].model_hub == "modelscope"
        assert specs[0].model_ability == ["speaker_embedding"]
        packages = specs[0].virtualenv.packages
        assert "modelscope[framework]>=1.19.0" in packages
        assert "addict" in packages
        assert "datasets>=3,<5" in packages

    with (
        patch("xinference.model.audio.BUILTIN_AUDIO_MODELS", models),
        patch("xinference.model.audio.custom.get_user_defined_audios", return_value=[]),
        patch("xinference.model.utils.download_from_modelscope", return_value=False),
    ):
        matched = match_audio("speech_campplus_sv_zh-cn_16k-common")

    assert matched.model_hub == "modelscope"
