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
# CI: linux
import base64
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests


@pytest.mark.parametrize("content_type", ["string", "text-parts", "image-plus-text"])
def test_glm4v_processed_content_blocks(monkeypatch, content_type):
    from ..transformers.multimodal import glm4_1v

    image = object()
    decode_image = Mock(return_value=image)
    monkeypatch.setattr(glm4_1v, "_decode_image", decode_image)
    content = [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "world"},
    ]
    expected = [{"type": "text", "text": "hello world"}]
    if content_type == "string":
        content = "hello world"
    elif content_type == "image-plus-text":
        content.insert(1, {"type": "image_url", "image_url": {"url": "test-image.png"}})
        expected.insert(0, {"type": "image", "image": image})

    messages = [{"role": "user", "content": content}]
    assert glm4_1v.Glm4_1VModel._get_processed_msgs(messages) == [
        {"role": "user", "content": expected}
    ]
    if content_type == "image-plus-text":
        decode_image.assert_called_once_with("test-image.png")
    else:
        decode_image.assert_not_called()


@pytest.mark.parametrize(
    "content",
    ["hello world", [{"type": "text", "text": "hello world"}]],
    ids=["string", "text-parts"],
)
def test_glm4v_processor_accepts_text_content(content):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        Glm4vImageProcessor,
        Glm4vProcessor,
        Glm4vVideoProcessor,
        PreTrainedTokenizerFast,
    )

    from ..transformers.multimodal.glm4_1v import Glm4_1VModel

    backend = Tokenizer(
        WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    processor = Glm4vProcessor(
        image_processor=Glm4vImageProcessor(),
        tokenizer=tokenizer,
        video_processor=Glm4vVideoProcessor(),
        # Content-block traversal must succeed before any template is rendered.
        chat_template="{{ 'hello world' }}",
    )
    adapter = object.__new__(Glm4_1VModel)
    adapter._processor = processor
    adapter._model = SimpleNamespace(device="cpu")
    adapter.reasoning_parser = None

    inputs = adapter.build_inputs_from_messages(
        [{"role": "user", "content": content}], {}
    )
    assert inputs["input_ids"].tolist() == [[1, 2]]


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
