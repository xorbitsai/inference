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

import os
from typing import Dict, List

import gradio as gr
from gradio.components import Markdown, Textbox
from gradio.layouts import Accordion, Column, Row

from ..client import RESTfulClient


class LLMInterface:
    def __init__(
        self,
        endpoint: str,
        model_uid: str,
    ):
        self.client = RESTfulClient(endpoint)
        self.endpoint = endpoint
        self.model_uid = model_uid

    def build(self) -> "gr.Blocks":
        model = self.client.get_model(self.model_uid)
        model_info = self.client.describe_model(self.model_uid)
        model_ability = model_info["model_ability"]

        if "chat" in model_ability:
            interface = self.build_chat_interface(
                model,
                model_info,
            )
        else:
            interface = self.build_generate_interface(
                model,
                model_info,
            )

        favicon_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.path.pardir,
            "web",
            "ui",
            "public",
            "favicon.svg",
        )
        interface.favicon_path = favicon_path
        return interface

    def build_chat_interface(
        self,
        model,
        model_info,
    ) -> "gr.Blocks":
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
                    minimum=0, maximum=2, value=1, step=0.01, label="Temperature"
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
            analytics_enabled=False,
        )

    def build_generate_interface(
        self,
        model,
        model_info,
    ):
        model_name = model_info["model_name"]
        model_format = model_info["model_format"]
        model_size_in_billions = model_info["model_size_in_billions"]
        quantization = model_info["quantization"]

        def undo(text, hist):
            if len(hist) == 0:
                return {
                    textbox: "",
                    history: [text],
                }
            if text == hist[-1]:
                hist = hist[:-1]

            return {
                textbox: hist[-1] if len(hist) > 0 else "",
                history: hist,
            }

        def clear(text, hist):
            if len(hist) == 0 or (len(hist) > 0 and text != hist[-1]):
                hist.append(text)
            hist.append("")
            return {
                textbox: "",
                history: hist,
            }

        def complete(text, hist, length, temperature):
            if len(hist) == 0 or (len(hist) > 0 and text != hist[-1]):
                hist.append(text)
            response = model.generate(
                prompt=text,
                generate_config={"max_tokens": length, "temperature": temperature},
            )
            text_gen = text + response["choices"][0]["text"]
            hist.append(text_gen)
            return {
                textbox: text_gen,
                history: hist,
            }

        def retry(text, hist, length, temperature):
            if len(hist) == 0 or (len(hist) > 0 and text != hist[-1]):
                hist.append(text)
            text = hist[-2] if len(hist) > 1 else ""
            response = model.generate(
                prompt=text,
                generate_config={"max_tokens": length, "temperature": temperature},
            )
            text_gen = text + response["choices"][0]["text"]
            return {
                textbox: text_gen,
                history: hist,
            }

        with gr.Blocks(
            title=f"🚀 Xinference Generate Bot : {model_name} 🚀",
            css="""
            .center{
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 0px;
                color: #9ea4b0 !important;
            }
            """,
            analytics_enabled=False,
        ) as generate_interface:
            history = gr.State([])

            Markdown(
                f"""
                <h1 style='text-align: center; margin-bottom: 1rem'>🚀 Xinference Generate Bot : {model_name} 🚀</h1>
                """
            )
            Markdown(
                f"""
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
                """
            )

            with Column(variant="panel"):
                textbox = Textbox(
                    container=False,
                    show_label=False,
                    label="Message",
                    placeholder="Type a message...",
                    lines=21,
                    max_lines=50,
                )

                with Row():
                    btn_generate = gr.Button("Generate", variant="primary")
                with Row():
                    btn_undo = gr.Button("↩️  Undo")
                    btn_retry = gr.Button("🔄  Retry")
                    btn_clear = gr.Button("🗑️  Clear")
                with Accordion("Additional Inputs", open=False):
                    length = gr.Slider(
                        minimum=1, maximum=1024, value=10, step=1, label="Max Tokens"
                    )
                    temperature = gr.Slider(
                        minimum=0, maximum=2, value=1, step=0.01, label="Temperature"
                    )

                btn_generate.click(
                    fn=complete,
                    inputs=[textbox, history, length, temperature],
                    outputs=[textbox, history],
                )

                btn_undo.click(
                    fn=undo,
                    inputs=[textbox, history],
                    outputs=[textbox, history],
                )

                btn_retry.click(
                    fn=retry,
                    inputs=[textbox, history, length, temperature],
                    outputs=[textbox, history],
                )

                btn_clear.click(
                    fn=clear,
                    inputs=[textbox, history],
                    outputs=[textbox, history],
                )

        return generate_interface
