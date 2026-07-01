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

import os.path
import platform
import sys

import pytest


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.processor() != "arm",
    reason="MLX only works for Apple silicon chip",
)
def test_restful_api_for_whisper(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="whisper-1",
        model_name="whisper-small-mlx",
        model_type="audio",
    )
    model = client.get_model(model_uid)
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")
    with open(audio_path, "rb") as f:
        audio = f.read()

    response = model.transcriptions(audio)
    transcription = response["text"].lower()
    assert "my fellow americans" in transcription
    assert "your country" in transcription
    assert "do for you" in transcription

    # Translation requires large-v3 model.
    zh_cn_audio_path = os.path.join(os.path.dirname(__file__), "zero_shot_prompt.wav")
    with open(zh_cn_audio_path, "rb") as f:
        zh_cn_audio = f.read()
    response = model.translations(zh_cn_audio)
    translation = response["text"].lower()
    assert "do better" in translation

    # If model multilingual is False, it can't be used for translations.
    model_uid2 = client.launch_model(
        model_uid="whisper-2",
        model_name="whisper-tiny.en-mlx",
        model_type="audio",
        n_gpu=None,
    )
    model2 = client.get_model(model_uid2)
    with pytest.raises(RuntimeError, match="translations"):
        model2.translations(zh_cn_audio)

    # Test openai API
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    with open(zh_cn_audio_path, "rb") as f:
        completion = client.audio.transcriptions.create(model=model_uid, file=f)
        assert "希望" in completion.text

        completion = client.audio.translations.create(model=model_uid, file=f)
        translation = completion.text.lower()
        assert "do better" in translation


def test_transcriptions_for_whisper(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="whisper-1",
        model_name="whisper-small-mlx",
        model_type="audio",
    )
    model = client.get_model(model_uid)
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")
    with open(audio_path, "rb") as f:
        audio = f.read()

    response = model.transcriptions(audio, response_format="verbose_json")
    assert response["text"]
    assert len(response["segments"]) == 1

    seek_set = set()
    for s in response["segments"]:
        if s["seek"] in seek_set:
            assert False, "incorrect seek"
        seek_set.add(s["seek"])

    response = model.transcriptions(
        audio, response_format="verbose_json", timestamp_granularities=["word"]
    )
    assert response["text"]
    assert len(response["words"]) == 22

    zh_cn_audio_path = os.path.join(
        os.path.dirname(__file__), "common_voice_zh-CN_38026095.mp3"
    )

    # Test openai API
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    with open(zh_cn_audio_path, "rb") as f:
        completion = client.audio.transcriptions.create(
            model=model_uid,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
        assert len(completion.segments) == 1

        completion = client.audio.transcriptions.create(
            model=model_uid,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )
        assert len(completion.words) == 11
