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


def test_restful_api_for_qwen_vl(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="my_controlnet",
        model_name="qwen-vl-chat",
        model_type="multimodal",
        device="cpu",
    )
    model = client.get_model(model_uid)

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
