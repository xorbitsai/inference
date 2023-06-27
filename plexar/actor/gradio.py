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

from typing import List

import gradio as gr

from ..client import Client
from ..model.llm.core import ChatHistory


class GradioApp:
    def __init__(self, xoscar_endpoint: str):
        self._xoscar_endpoint = xoscar_endpoint
        self._api = Client(xoscar_endpoint)
        self._models = self._api.list_models()

    async def generate(
        self,
        model: str,
        message: str,
        chat: List,
        max_token: int,
        temperature: float,
        top_p: float,
    ):
        if not message:
            yield message, chat
        if not self._models:
            raise gr.Error(f"Please create model first")
        inputs = [c[0] for c in chat]
        outputs = [c[1] for c in chat]
        history = ChatHistory(inputs=inputs, outputs=outputs)
        generate_config = dict(
            max_tokens=max_token,
            temperature=temperature,
            top_p=top_p,
        )
        chat += [[message, ""]]
        model_ref = self._api.get_model(model)
        chat_generator = await model_ref.chat(
            message,
            chat_history=history,
            generate_config=generate_config,
        )
        async for chunk in chat_generator:
            chat[-1][1] += chunk["text"]
            yield "", chat

    def _build_chatbot(self, model_uid: str, model_name: str):
        with gr.Accordion("Parameters", open=False):
            max_token = gr.Slider(
                128,
                512,
                value=128,
                step=1,
                label="Max tokens",
                info="The maximum number of tokens to generate.",
            )
            temperature = gr.Slider(
                0.2,
                1,
                value=0.8,
                step=0.01,
                label="Temperature",
                info="The temperature to use for sampling.",
            )
            top_p = gr.Slider(
                0.2,
                1,
                value=0.95,
                step=0.01,
                label="Top P",
                info="The top-p value to use for sampling.",
            )
        chat = gr.Chatbot(label=model_name)
        text = gr.Textbox(visible=False)
        model = gr.Textbox(model_uid, visible=False)
        text.change(
            self.generate,
            [model, text, chat, max_token, temperature, top_p],
            [text, chat],
        )
        return text, chat, max_token, temperature, top_p

    def build(self):
        with gr.Blocks() as blocks:
            gr.Markdown("# Chat with LLMs")
            model_components = []
            with gr.Box():
                with gr.Row():
                    chats = []
                    texts = []
                    for model in self._models:
                        with gr.Column():
                            components = self._build_chatbot(model[0], model[1].name)
                            model_components.append(components)
                            texts.append(components[0])
                            chats.append(components[1])
                with gr.Column():
                    msg = gr.Textbox()
                    gr.ClearButton(components=[msg] + chats)

                    def _pass_to_all(msg, *text):
                        return [""] + [msg] * len(text)

                    msg.submit(_pass_to_all, [msg] + texts, [msg] + texts)
        return blocks
