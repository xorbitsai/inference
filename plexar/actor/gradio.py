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
from typing import Dict, List

import gradio as gr
import xoscar as xo

from ..api import API
from ..model import MODEL_SPECS
from ..model.config import model_config

name_to_spec = dict((spec.name, spec) for spec in MODEL_SPECS)


class GradioApp:
    def __init__(self, xoscar_endpoint: str):
        self._xoscar_endpoint = xoscar_endpoint
        self._api = API(xoscar_endpoint)
        self._model_limits = 2
        self._models: Dict[str, xo.ActorRef] = dict()

    async def select_models(self, models: List[str]):
        if len(models) != self._model_limits:
            raise gr.Error("Please choose 2 models")
        if self._models:
            destroy_tasks = []
            for ref in self._models.values():
                destroy_tasks.append(ref.destroy())
            await asyncio.gather(*destroy_tasks)
        create_tasks = []
        for model in models:
            kwargs = model_config[model]
            cls = name_to_spec[model].cls
            create_tasks.append(self._api.create_model(cls, **kwargs))
        model_refs = await asyncio.gather(*create_tasks)
        self._models = dict(zip(models, model_refs))

    async def generate(self, message, *chats: List):
        if not self._models:
            raise gr.Error("Please create models first")
        chat_tasks = []
        for ref in self._models.values():
            chat_tasks.append(ref.chat(message))
        answers = await asyncio.gather(*chat_tasks)
        for answer, chat in zip(answers, chats):
            chat.append((message, answer["text"]))
        return message, *chats

    async def clear(self):
        for ref in self._models.values():
            await ref.clear()

    def build(self):
        with gr.Blocks() as blocks:
            gr.Markdown("# Chat with LLMs")
            choice = gr.CheckboxGroup(
                list(model_config.keys()),
                label="Choose models to deploy",
            )
            create_button = gr.Button("create")
            with gr.Box():
                with gr.Row():
                    chats = [
                        gr.Chatbot(show_label=False) for _ in range(self._model_limits)
                    ]
                with gr.Column():
                    msg = gr.Textbox()
                    clear_button = gr.ClearButton(components=[msg] + chats)
                    clear_button.click(self.clear)
                    msg.submit(self.generate, [msg] + chats, [msg] + chats)
            create_button.click(self.select_models, [choice])
        return blocks
