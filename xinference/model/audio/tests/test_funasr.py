# Copyright 2022-2023 XProbe Inc.
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

import requests


def test_restful_api_for_funasr(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="SenseVoiceSmall",
        model_name="SenseVoiceSmall",
        model_type="audio",
    )
    model = client.get_model(model_uid)
    response = requests.get("https://github.com/openai/whisper/raw/main/tests/jfk.flac")
    audio = response.content

    response = model.transcriptions(audio)
    transcription = response["text"].lower()
    assert "my fellow americans" in transcription
    assert "your country" in transcription
    assert "do for you" in transcription

    # Test openai API
    import openai

    zh_cn_audio_path = os.path.join(
        os.path.dirname(__file__), "common_voice_zh-CN_38026095.mp3"
    )
    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    with open(zh_cn_audio_path, "rb") as f:
        completion = client.audio.transcriptions.create(model=model_uid, file=f)
        assert "列表" in completion.text
        assert "香港" in completion.text
        assert "航空" in completion.text
