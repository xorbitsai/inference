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

import uuid
from typing import List

import gradio as gr
import xoscar as xo

from ..client import Client
from ..model.llm.core import ChatHistory


class GradioApp:
    def __init__(self, xoscar_endpoint: str):
        self._xoscar_endpoint = xoscar_endpoint
        self._api = Client(xoscar_endpoint)
        # model string to model uid
        self._models = dict((str(m[1]), m[0]) for m in self._api.list_models())

    def _refresh_and_get_models(self) -> List[str]:
        self._models = dict((str(m[1]), m[0]) for m in self._api.list_models())
        return list(self._models.keys())

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
        model_uid = gr.Textbox(model_uid, visible=False)
        text.change(
            self.generate,
            [model_uid, text, chat, max_token, temperature, top_p],
            [text, chat],
        )
        return text, chat, max_token, temperature, top_p, model_uid

    def build_multiple(self):
        with gr.Blocks() as blocks:
            gr.Markdown("# Chat with LLMs")
            with gr.Box():
                with gr.Row():
                    chats = []
                    texts = []
                    for model in self._models:
                        with gr.Column():
                            components = self._build_chatbot(model[0], model[1].name)
                            texts.append(components[0])
                            chats.append(components[1])
                with gr.Column():
                    msg = gr.Textbox()
                    gr.ClearButton(components=[msg] + chats)

                    def _pass_to_all(msg, *text):
                        return [""] + [msg] * len(text)

                    msg.submit(_pass_to_all, [msg] + texts, [msg] + texts)
        return blocks

    def build(self):
        with gr.Blocks() as blocks:
            gr.Markdown("# Chat with LLM")

            selected_model = gr.Dropdown(
                choices=self._refresh_and_get_models(),
                label="select launched model",
            )

            # It's a trick, create an invisible Number with callable value
            # and set every to 5 to trigger update every 5 seconds
            def _refresh_models():
                return gr.Dropdown.update(choices=self._refresh_and_get_models())

            n = gr.Text(value=lambda *_: str(uuid.uuid4()), visible=False, every=5)
            n.change(
                _refresh_models, inputs=None, outputs=[selected_model], queue=False
            )

            with gr.Box():
                with gr.Column():
                    components = self._build_chatbot("", "")
                    msg = gr.Textbox()
                    model_text = components[0]
                    chat, model_uid = components[1], components[-1]
                    gr.ClearButton(components=[chat, msg, model_text])

                    def update_message(text_in: str):
                        return "", text_in

                    msg.submit(update_message, inputs=[msg], outputs=[msg, model_text])

            def select_model(model_name: str):
                uid = self._models[model_name]
                return gr.Chatbot.update(label=model_name, value=[]), gr.Textbox.update(
                    value=uid
                )

            selected_model.change(
                select_model,
                inputs=[selected_model],
                outputs=[chat, model_uid],
                postprocess=False,
            )
        return blocks


class GradioActor(xo.Actor):
    def __init__(self, xoscar_endpoint: str, host: str, port: int, share: bool):
        super().__init__()
        self._gradio_cls = GradioApp(xoscar_endpoint)
        self._host = host
        self._port = port
        self._share = share

    def launch(self):
        demo = self._gradio_cls.build()
        demo.queue(concurrency_count=20)
        demo.launch(
            share=self._share,
            server_name=self._host,
            server_port=self._port,
            prevent_thread_lock=True,
        )
