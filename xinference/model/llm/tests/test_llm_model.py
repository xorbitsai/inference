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

import pytest


@pytest.mark.skip(reason="Cost too many resources.")
@pytest.mark.parametrize(
    "model_size_in_billions", "model_format, quantization", [(7, "pytorch", None)]
)
def test_restful_api_for_qwen_vl(
    setup, model_size_in_billions, model_format, quantization
):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="deepseek-r1",
        model_name="deepseek-r1-distill-qwen",
        model_size_in_billions=model_size_in_billions,
        model_format=model_format,
        quantization=quantization,
    )
    model = client.get_model(model_uid)
    messages = [{"role": "user", "content": "What is k8s?"}]
    response = model.chat(messages)
    assert "reasoning_content" in response["choices"][0]["message"]

    # openai client
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    completion = client.chat.completions.create(
        model=model_uid,
        messages=[
            {
                "role": "user",
                "content": "What is k8s?",
            }
        ],
    )
    assert "reasoning_content" in completion.choices[0].message
