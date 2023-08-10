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

from typing import Dict, List

import gradio as gr

from xinference.client import RESTfulClient


class ChatInterface:
    def __init__(
        self,
        endpoint: str,
        model_uid: str,
    ):
        self.client = RESTfulClient(endpoint)
        self.endpoint = endpoint
        self.model_uid = model_uid

    def build_interface(self):
        model = self.client.get_model(self.model_uid)
        model_info = self.client.get_model_info(self.model_uid)
        model_name = model_info["model_name"]
        model_format = model_info["model_format"]
        model_size_in_billions = model_info["model_size_in_billions"]
        quantization = model_info["quantization"]

        def flatten(matrix: List[List[str]]) -> List[str]:
            flat_list = []
            for row in matrix:
                flat_list += row
            return flat_list

        def to_chat(lst: List[str]) -> List[Dict[str, str]]:
            res = []
            for i in range(len(lst)):
                role = "assistant" if i % 2 == 1 else "user"
                res.append(
                    {
                        "role": role,
                        "content": lst[i],
                    }
                )
            return res

        def generate_wrapper(
            message: str,
            history: List[List[str]],
            max_response_length: int,
            temperature: float,
        ) -> str:
            output = model.chat(
                prompt=message,
                chat_history=to_chat(flatten(history)),
                generate_config={
                    "max_tokens": int(max_response_length),
                    "temperature": temperature,
                    "stream": False,
                },
            )
            return output["choices"][0]["message"]["content"]

        return gr.ChatInterface(
            fn=generate_wrapper,
            # TODO: Change min/max based on model context size
            additional_inputs=[
                gr.Slider(
                    minimum=1, maximum=2048, value=1024, step=1, label="Max Tokens"
                ),
                gr.Slider(
                    minimum=0, maximum=1, value=0.8, step=0.01, label="Temperature"
                ),
            ],
            title=f"🚀 Xinference Chat Bot : {model_name} 🚀",
            css="""
            .center{
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 0px;
                color: #9ea4b0 !important;
            }
            """,
            description=f"""
            <div class="center">
            Model ID: {self.model_uid}
            </div>
            <div class="center">
            Model Size: {model_size_in_billions} Billion Parameters
            </div>
            <div class="center">
            Model Format: {model_format}
            </div>
            <div class="center">
            Model Quantization: {quantization}
            </div>
            """,
        )
