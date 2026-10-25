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

import os
import tempfile
from types import SimpleNamespace

import pytest

from ..f5tts import F5TTSModel


@pytest.mark.parametrize(
    ("model_hub", "vocoder_name", "vocoder_path", "expected"),
    [
        ("modelscope", "vocos", None, "/cache/vocos"),
        ("modelscope", "vocos", "/custom/vocos", "/custom/vocos"),
        ("modelscope", "bigvgan", None, None),
        ("huggingface", "vocos", None, None),
    ],
)
def test_f5tts_resolves_modelscope_vocos(
    monkeypatch, model_hub, vocoder_name, vocoder_path, expected
):
    model = F5TTSModel(
        "f5",
        "/model",
        SimpleNamespace(model_hub=model_hub, model_ability=[]),
    )
    calls = []
    monkeypatch.setattr(
        model,
        "_download_modelscope_vocos",
        lambda: calls.append(True) or "/cache/vocos",
    )

    assert model._resolve_vocoder_path(vocoder_name, vocoder_path) == expected
    assert calls == ([True] if expected == "/cache/vocos" else [])


def test_f5tts_downloads_modelscope_vocos_artifact(monkeypatch):
    from ...utils import ModelArtifactSource

    calls = []

    def snapshot_download(self, model_id, **kwargs):
        calls.append((self.hub, model_id, kwargs))
        return "/cache/vocos"

    monkeypatch.setattr(ModelArtifactSource, "snapshot_download", snapshot_download)

    assert F5TTSModel._download_modelscope_vocos() == "/cache/vocos"
    assert calls == [
        (
            "modelscope",
            "pengzhendong/vocos-mel-24khz",
            {"allow_patterns": ["config.yaml", "pytorch_model.bin"]},
        )
    ]


def test_f5tts(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="F5-TTS",
        model_type="audio",
        download_hub="huggingface",
    )
    model = client.get_model(model_uid)
    input_string = (
        "chat T T S is a text to speech model designed for dialogue applications."
    )
    response = model.speech(input_string)
    assert type(response) is bytes
    assert len(response) > 0

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
        f.write(response)

    # Test openai API
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    with client.audio.speech.with_streaming_response.create(
        model=model_uid, input=input_string, voice="echo"
    ) as response:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
            response.stream_to_file(f.name)
            assert os.stat(f.name).st_size > 0
