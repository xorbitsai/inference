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
    prompt = [
        {"type": "text", "text": "What’s in this image?"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            },
        },
    ]
    response = model.chat(prompt=prompt)
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
    prompt = [
        {"type": "text", "text": "What’s in this image?"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            },
        },
    ]
    response = model.chat(prompt=prompt)
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
