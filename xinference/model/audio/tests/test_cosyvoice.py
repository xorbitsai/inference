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
import inspect
import os.path
import tempfile

import pytest


@pytest.mark.parametrize("model_name", ["CosyVoice-300M-SFT", "CosyVoice2-0.5B"])
def test_cosyvoice_sft(setup, model_name):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name=model_name, model_type="audio", download_hub="huggingface"
    )
    model = client.get_model(model_uid)
    input_string = "你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？"

    # inference_sft
    response = model.speech(input_string)
    assert type(response) is bytes
    assert len(response) > 0

    # Test openai API
    import openai

    openai_client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    # ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
    response = openai_client.audio.speech.create(
        model=model_uid, input=input_string, voice="英文女"
    )
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
        response.stream_to_file(f.name)
        assert os.stat(f.name).st_size > 0

    if "CosyVoice2" in model_name:
        client.terminate_model(model_uid)
        model_uid = client.launch_model(
            model_name=model_name,
            model_type="audio",
            download_hub="modelscope",
            use_flow_cache=True,
        )
        model = client.get_model(model_uid)

    # inference_sft
    response = model.speech(input_string, stream=True)
    assert inspect.isgenerator(response)
    i = 0
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
        for chunk in response:
            f.write(chunk)
            i += 1
            assert type(chunk) is bytes
            assert len(chunk) > 0
        assert i > 5


@pytest.mark.parametrize("model_name", ["CosyVoice-300M", "CosyVoice2-0.5B"])
def test_cosyvoice(setup, model_name):
    endpoint, _ = setup
    from ....client import Client

    zero_shot_prompt_file = os.path.join(
        os.path.dirname(__file__), "zero_shot_prompt.wav"
    )
    cross_lingual_prompt_file = os.path.join(
        os.path.dirname(__file__), "cross_lingual_prompt.wav"
    )

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name=model_name,
        model_type="audio",
        download_hub="modelscope",
    )
    model = client.get_model(model_uid)
    with open(zero_shot_prompt_file, "rb") as f:
        zero_shot_prompt = f.read()
    with open(cross_lingual_prompt_file, "rb") as f:
        cross_lingual_prompt = f.read()
    input_string = (
        "<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in "
        "line with the asset that's coming into the family is a reason why sometimes we don't buy the whole thing.",
    )

    # inference_cross_lingual
    response = model.speech(input_string, prompt_speech=cross_lingual_prompt)
    assert type(response) is bytes, response
    assert len(response) > 0
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
        f.write(response)
        assert os.stat(f.name).st_size > 0

    # inference_zero_shot
    response = model.speech(
        "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
        prompt_text="希望你以后能够做的比我还好呦。",
        prompt_speech=zero_shot_prompt,
    )
    assert type(response) is bytes, response
    assert len(response) > 0
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
        f.write(response)
        assert os.stat(f.name).st_size > 0


@pytest.mark.parametrize("model_name", ["CosyVoice-300M-Instruct", "CosyVoice2-0.5B"])
def test_cosyvoice_instruct(setup, model_name):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name=model_name,
        model_type="audio",
        download_hub="modelscope",
    )
    model = client.get_model(model_uid)

    if "CosyVoice2" in model_name:
        zero_shot_prompt_file = os.path.join(
            os.path.dirname(__file__), "zero_shot_prompt.wav"
        )
        with open(zero_shot_prompt_file, "rb") as f:
            zero_shot_prompt = f.read()
        # inference with prompt speech
        response = model.speech(
            "在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。",
            prompt_speech=zero_shot_prompt,
        )
        assert type(response) is bytes
        assert len(response) > 0
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
            f.write(response)
            assert os.stat(f.name).st_size > 0
    else:
        # inference without instruction
        response = model.speech(
            "在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。", voice="中文男"
        )
        assert type(response) is bytes
        assert len(response) > 0
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
            f.write(response)
            assert os.stat(f.name).st_size > 0
        # inference_instruct
        response = model.speech(
            "在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。",
            voice="中文男",
            instruct_text="Theo 'Crimson', is a fiery, passionate rebel leader. "
            "Fights with fervor for justice, but struggles with impulsiveness.",
        )
        assert type(response) is bytes
        assert len(response) > 0
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
            f.write(response)
            assert os.stat(f.name).st_size > 0
