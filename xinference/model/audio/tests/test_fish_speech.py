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
import inspect
import os
import tempfile


def test_fish_speech(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="FishSpeech-1.5", model_type="audio", compile=False
    )
    model = client.get_model(model_uid)

    input_string = "你好，你是谁？"
    response = model.speech(input_string)
    assert type(response) is bytes
    assert len(response) > 0

    # Test copy voice
    prompt_speech_path = os.path.join(os.path.dirname(__file__), "basic_ref_en.wav")
    with open(prompt_speech_path, "rb") as f:
        prompt_speech = f.read()
    response = model.speech(
        "Hello",
        prompt_speech=prompt_speech,
        prompt_text="Some call me nature, others call me mother nature.",
    )
    assert type(response) is bytes
    assert len(response) > 0

    # Test stream
    input_string = "瑞典王国，通称瑞典，是一个位于斯堪的纳维亚半岛的北欧国家，首都及最大城市为斯德哥尔摩。"
    response = model.speech(input_string, chunk_length=20, stream=True)
    assert inspect.isgenerator(response)
    i = 0
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
        for chunk in response:
            f.write(chunk)
            i += 1
            assert type(chunk) is bytes
            assert len(chunk) > 0
        assert i > 5

    # Test openai API
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    with client.audio.speech.with_streaming_response.create(
        model=model_uid, input=input_string, voice="echo", response_format="pcm"
    ) as response:
        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=True) as f:
            response.stream_to_file(f.name)
            assert os.stat(f.name).st_size > 0
