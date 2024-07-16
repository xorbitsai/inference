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


def test_cosyvoice(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="CosyVoice-SFT",
        model_type="audio",
    )
    model = client.get_model(model_uid)
    input_string = (
        "chat T T S is a text to speech model designed for dialogue applications."
    )
    response = model.speech(input_string)
    assert type(response) is bytes
    assert len(response) > 0

    # Test openai API
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    # ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
    response = client.audio.speech.create(
        model=model_uid, input=input_string, voice="英文女"
    )
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
        response.stream_to_file(f.name)
        assert os.stat(f.name).st_size > 0
