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

import asyncio
from typing import List, Tuple

import gradio as gr
import xoscar as xo

from ..api import API
from ..model import MODEL_SPECS
from ..model.config import model_config
from ..model.llm.core import ChatHistory

name_to_spec = dict((spec.name, spec) for spec in MODEL_SPECS)


class GradioApp:
    def __init__(self, xoscar_endpoint: str):
        self._xoscar_endpoint = xoscar_endpoint
        self._api = API(xoscar_endpoint)
        self._model_limits = 2
        self._models: List[Tuple[str, xo.ActorRef]] = []

    async def select_models(self, models: List[str]):
        if len(models) != self._model_limits:
            raise gr.Error("Please choose 2 models")
        if self._models:
            destroy_tasks = []
            for _, ref in self._models:
                destroy_tasks.append(ref.destroy())
            await asyncio.gather(*destroy_tasks)
        create_tasks = []
        for model in models:
            kwargs = model_config[model]
            cls = name_to_spec[model].cls
            create_tasks.append(self._api.create_model(cls, **kwargs))
        model_refs = await asyncio.gather(*create_tasks)
        self._models = list(zip(models, model_refs))

    async def generate(
        self,
        model_idx: int,
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

        chat_generator = await self._models[int(model_idx)][1].chat(
            message,
            chat_history=history,
            generate_config=generate_config,
        )
        async for chunk in chat_generator:
            chat[-1][1] += chunk["text"]
            yield "", chat

    def _build_chatbot(self, model_idx: int):
        with gr.Column():
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
            chat = gr.Chatbot(show_label=False)
            text = gr.Textbox(visible=False)
            model_idx = gr.Number(model_idx, visible=False)
            text.change(
                self.generate,
                [model_idx, text, chat, max_token, temperature, top_p],
                [text, chat],
            )
        return text, chat, max_token, temperature, top_p

    def build(self):
        with gr.Blocks() as blocks:
            gr.Markdown("# Chat with LLMs")
            choice = gr.CheckboxGroup(
                list(model_config.keys()),
                label="Choose models to deploy",
            )
            create_button = gr.Button("create")
            model_components = []
            with gr.Box():
                with gr.Row():
                    chats = []
                    texts = []
                    for model_idx in range(self._model_limits):
                        with gr.Column():
                            components = self._build_chatbot(model_idx)
                            model_components.append(components)
                            texts.append(components[0])
                            chats.append(components[1])
                with gr.Column():
                    msg = gr.Textbox()
                    gr.ClearButton(components=[msg] + chats)

                    def _pass_to_all(msg, *text):
                        return [""] + [msg] * len(text)

                    msg.submit(_pass_to_all, [msg] + texts, [msg] + texts)
            create_button.click(self.select_models, [choice])
        return blocks
