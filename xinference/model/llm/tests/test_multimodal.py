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
import base64

import pytest
import requests


@pytest.mark.skip(reason="Cost too many resources.")
@pytest.mark.parametrize(
    "model_format, quantization", [("pytorch", None), ("gptq", "Int4")]
)
def test_restful_api_for_qwen_vl(setup, model_format, quantization):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="qwen-vl-chat",
        model_name="qwen-vl-chat",
        model_format=model_format,
        quantization=quantization,
    )
    model = client.get_model(model_uid)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    },
                },
            ],
        }
    ]
    response = model.chat(messages)
    assert "grass" in response["choices"][0]["message"]["content"]
    assert "tree" in response["choices"][0]["message"]["content"]
    assert "sky" in response["choices"][0]["message"]["content"]

    # openai client
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    completion = client.chat.completions.create(
        model=model_uid,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                        },
                    },
                ],
            }
        ],
    )
    assert "grass" in completion.choices[0].message.content
    assert "tree" in completion.choices[0].message.content
    assert "sky" in completion.choices[0].message.content
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这是什么?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                },
            ],
        }
    ]
    completion = client.chat.completions.create(model=model_uid, messages=messages)
    assert "女" in completion.choices[0].message.content
    assert "狗" in completion.choices[0].message.content
    assert "沙滩" in completion.choices[0].message.content
    messages.append(completion.choices[0].message.model_dump())
    messages.append({"role": "user", "content": "框出图中击掌的位置"})
    completion = client.chat.completions.create(model=model_uid, messages=messages)
    assert "击掌" in completion.choices[0].message.content
    assert "<ref>" in completion.choices[0].message.content
    assert "<box>" in completion.choices[0].message.content

    # Test base64 image
    response = requests.get(
        "http://i.epochtimes.com/assets/uploads/2020/07/shutterstock_675595789-600x400.jpg"
    )

    # https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images
    # Function to encode the image
    b64_img = base64.b64encode(response.content).decode("utf-8")

    completion = client.chat.completions.create(
        model=model_uid,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "图中有几条鱼？"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_img}",
                        },
                    },
                ],
            }
        ],
    )
    assert "四条" in completion.choices[0].message.content


@pytest.mark.skip(reason="Cost too many resources.")
@pytest.mark.parametrize("model_format, quantization", [("pytorch", None)])
def test_restful_api_for_yi_vl(setup, model_format, quantization):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="yi-vl-chat",
        model_name="yi-vl-chat",
        model_format=model_format,
        quantization=quantization,
    )
    model = client.get_model(model_uid)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    },
                },
            ],
        }
    ]
    response = model.chat(messages)
    assert "green" in response["choices"][0]["message"]["content"]
    assert "tree" in response["choices"][0]["message"]["content"]
    assert "sky" in response["choices"][0]["message"]["content"]

    # openai client
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    completion = client.chat.completions.create(
        model=model_uid,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                        },
                    },
                ],
            }
        ],
    )
    assert "green" in completion.choices[0].message.content
    assert "tree" in completion.choices[0].message.content
    assert "sky" in completion.choices[0].message.content

    # Test base64 image
    response = requests.get(
        "http://i.epochtimes.com/assets/uploads/2020/07/shutterstock_675595789-600x400.jpg"
    )

    # https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images
    # Function to encode the image
    b64_img = base64.b64encode(response.content).decode("utf-8")

    completion = client.chat.completions.create(
        model=model_uid,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "图中有几条鱼？"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_img}",
                        },
                    },
                ],
            }
        ],
    )
    assert "两条" in completion.choices[0].message.content


@pytest.mark.skip(reason="Cost too many resources.")
@pytest.mark.parametrize("model_format, quantization", [("pytorch", None)])
def test_restful_api_for_deepseek_vl(setup, model_format, quantization):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="deepseek-vl-chat",
        model_name="deepseek-vl-chat",
        model_format=model_format,
        quantization=quantization,
        temperature=0.0,
    )
    model = client.get_model(model_uid)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    },
                },
            ],
        }
    ]
    response = model.chat(messages)
    assert any(
        green in response["choices"][0]["message"]["content"]
        for green in ["grass", "green"]
    )
    assert any(
        tree in response["choices"][0]["message"]["content"]
        for tree in ["tree", "wooden"]
    )
    assert "sky" in response["choices"][0]["message"]["content"]

    # openai client
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    completion = client.chat.completions.create(
        model=model_uid,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                        },
                    },
                ],
            }
        ],
    )
    assert any(
        green in response["choices"][0]["message"]["content"]
        for green in ["grass", "green"]
    )
    assert any(
        tree in response["choices"][0]["message"]["content"]
        for tree in ["tree", "wooden"]
    )
    assert "sky" in completion.choices[0].message.content

    # Test base64 image
    response = requests.get(
        "http://i.epochtimes.com/assets/uploads/2020/07/shutterstock_675595789-600x400.jpg"
    )

    # https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images
    # Function to encode the image
    b64_img = base64.b64encode(response.content).decode("utf-8")

    completion = client.chat.completions.create(
        model=model_uid,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "图中有几条鱼？"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_img}",
                        },
                    },
                ],
            }
        ],
    )
    assert any(
        count in completion.choices[0].message.content for count in ["两条", "四条"]
    )


@pytest.mark.skip(reason="Cost too many resources.")
def test_restful_api_for_qwen_audio(setup):
    model_name = "qwen2-audio-instruct"

    endpoint, _ = setup
    url = f"{endpoint}/v1/models"

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 0

    # launch
    payload = {
        "model_uid": "test_audio",
        "model_name": model_name,
        "model_engine": "transformers",
        "model_size_in_billions": 7,
        "model_format": "pytorch",
        "quantization": "none",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_audio"

    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 1

    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": model_uid_res,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3",
                    },
                    {"type": "text", "text": "What's that sound?"},
                ],
            },
            {"role": "assistant", "content": "It is the sound of glass shattering."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What can you do when you hear that?"},
                ],
            },
            {
                "role": "assistant",
                "content": "Stay alert and cautious, and check if anyone is hurt or if there is any damage to property.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac",
                    },
                    {"type": "text", "text": "What does the person say?"},
                ],
            },
        ],
    }
    response = requests.post(url, json=payload)
    completion = response.json()
    assert len(completion["choices"][0]["message"]) > 0
