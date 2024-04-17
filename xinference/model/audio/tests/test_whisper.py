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
import tempfile

import pytest
import requests


def test_restful_api_for_whisper(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="whisper-1",
        model_name="whisper-large-v3",
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

    # Translation requires large-v3 model.
    zh_cn_audio_path = os.path.join(
        os.path.dirname(__file__), "common_voice_zh-CN_38026095.mp3"
    )
    with open(zh_cn_audio_path, "rb") as f:
        zh_cn_audio = f.read()
    response = model.translations(zh_cn_audio)
    translation = response["text"].lower()
    assert "list" in translation
    assert "airlines" in translation
    assert "hong kong" in translation

    # If model multilingual is False, it can't be used for translations.
    model_uid2 = client.launch_model(
        model_uid="whisper-2",
        model_name="whisper-tiny.en",
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
        assert "列表" in completion.text
        assert "香港" in completion.text
        assert "航空" in completion.text

        completion = client.audio.translations.create(model=model_uid, file=f)
        translation = completion.text.lower()
        assert "list" in translation
        assert "airlines" in translation
        assert "hong kong" in translation


def test_register_custom_audio():
    from ..custom import (
        CustomAudioModelFamilyV1,
        get_user_defined_audios,
        register_audio,
        unregister_audio,
    )

    # correct
    family_a = CustomAudioModelFamilyV1(
        model_family="my-whisper",
        model_name="custom_test_a",
        model_id="test/custom_test_a",
        multilingual=True,
    )

    register_audio(family_a, False)
    assert family_a in get_user_defined_audios()

    # name conflict
    family_b = CustomAudioModelFamilyV1(
        model_family="my-whisper",
        model_name="custom_test_b",
        model_id="test/custom_test_b",
        multilingual=True,
    )
    register_audio(family_b, False)
    assert family_b in get_user_defined_audios()
    with pytest.raises(ValueError):
        register_audio(family_b, False)

    # unregister
    unregister_audio(family_a.model_name)
    assert family_a not in get_user_defined_audios()
    unregister_audio(family_b.model_name)
    assert family_b not in get_user_defined_audios()


def test_persistent_custom_audio():
    from ....constants import XINFERENCE_MODEL_DIR
    from ..custom import (
        CustomAudioModelFamilyV1,
        get_user_defined_audios,
        register_audio,
        unregister_audio,
    )

    temp_dir = tempfile.mkdtemp()

    # correct
    family = CustomAudioModelFamilyV1(
        model_family="my-whisper",
        model_name="custom_test_a",
        model_id="test/custom_test_a",
        multilingual=True,
        model_uri=os.path.abspath(temp_dir),
    )

    register_audio(family, True)
    assert family in get_user_defined_audios()
    assert f"{family.model_name}.json" in os.listdir(
        os.path.join(XINFERENCE_MODEL_DIR, "audio")
    )

    unregister_audio(family.model_name)
    assert f"{family.model_name}.json" not in os.listdir(
        os.path.join(XINFERENCE_MODEL_DIR, "audio")
    )
