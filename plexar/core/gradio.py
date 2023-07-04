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

from typing import Dict, List, Optional, Tuple

import gradio as gr
import xoscar as xo

from ..client import Client
from ..model import MODEL_FAMILIES, ModelSpec
from ..model.llm.core import ChatHistory

MODEL_TO_FAMILIES = dict(
    (model_family.model_name, model_family) for model_family in MODEL_FAMILIES
)


class GradioApp:
    def __init__(
        self,
        xoscar_endpoint: str,
        gladiator_num: int = 2,
        max_model_num: int = 2,
        use_launched_model: bool = False,
    ):
        self._xoscar_endpoint = xoscar_endpoint
        self._api = Client(xoscar_endpoint)
        self._gladiator_num = gladiator_num
        self._max_model_num = max_model_num
        self._use_launched_model = use_launched_model

    def _create_model(
        self,
        model_name: str,
        model_size_in_billions: Optional[int] = None,
        model_format: Optional[str] = None,
        quantization: Optional[str] = None,
    ):
        models = self._api.list_models()
        if len(models) >= self._max_model_num:
            self._api.terminate_model(models[0][0])
        return self._api.launch_model(
            model_name, model_size_in_billions, model_format, quantization
        )

    async def generate(
        self,
        model: str,
        message: str,
        chat: List[List[str]],
        max_token: int,
        temperature: float,
        top_p: float,
        window_size: int,
        show_finish_reason: bool,
    ):
        if not message:
            yield message, chat
        else:
            try:
                model_ref = self._api.get_model(model)
            except KeyError:
                raise gr.Error(f"Please create model first")
            inputs = []
            outputs = []
            for c in chat:
                inputs.append(c[0])
                out = c[1]
                # remove stop reason
                finish_reason_idx = out.find("[stop reason: ")
                if finish_reason_idx == -1:
                    outputs.append(out)
                else:
                    outputs.append(out[:finish_reason_idx])
            chat = list([i, o] for i, o in zip(inputs, outputs))
            if window_size == 0:
                history = ChatHistory()
            else:
                history = ChatHistory(
                    inputs=inputs[-window_size:], outputs=outputs[-window_size:]
                )
            generate_config = dict(
                max_tokens=max_token,
                temperature=temperature,
                top_p=top_p,
            )
            chat += [[message, ""]]
            chat_generator = await model_ref.chat(
                message,
                chat_history=history,
                generate_config=generate_config,
            )
            chunk = None
            async for chunk in chat_generator:
                chat[-1][1] += chunk["choices"][0]["text"]
                yield "", chat
            if show_finish_reason and chunk is not None:
                chat[-1][1] += f"[ stop reason: {chunk['choices'][0]['finish_reason']}]"
                yield "", chat

    def _build_chatbot(self, model_uid: str, model_name: str):
        with gr.Accordion("Parameters", open=False):
            max_token = gr.Slider(
                128,
                1024,
                value=256,
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
            window_size = gr.Slider(
                0,
                50,
                value=10,
                step=1,
                label="Window size",
                info="Window size of chat history.",
            )
            show_finish_reason = gr.Checkbox(label="show stop reason")
        chat = gr.Chatbot(label=model_name)
        text = gr.Textbox(visible=False)
        model_uid = gr.Textbox(model_uid, visible=False)
        text.change(
            self.generate,
            [
                model_uid,
                text,
                chat,
                max_token,
                temperature,
                top_p,
                window_size,
                show_finish_reason,
            ],
            [text, chat],
        )
        return (
            text,
            chat,
            max_token,
            temperature,
            top_p,
            show_finish_reason,
            window_size,
            model_uid,
        )

    def _build_chat_column(self):
        with gr.Column():
            with gr.Row():
                model_name = gr.Dropdown(
                    choices=[m.model_name for m in MODEL_FAMILIES],
                    label="model name",
                    scale=2,
                )
                model_format = gr.Dropdown(
                    choices=[],
                    interactive=False,
                    label="model format",
                    scale=2,
                )
                model_size_in_billions = gr.Dropdown(
                    choices=[],
                    interactive=False,
                    label="model size in billions",
                    scale=1,
                )
                quantization = gr.Dropdown(
                    choices=[],
                    interactive=False,
                    label="quantization",
                    scale=1,
                )
            create_model = gr.Button(value="create")

            def select_model_name(model_name: str):
                if model_name:
                    model_family = MODEL_TO_FAMILIES[model_name]
                    formats = [model_family.model_format]
                    model_sizes_in_billions = [
                        str(b) for b in model_family.model_sizes_in_billions
                    ]
                    quantizations = model_family.quantizations
                    return (
                        gr.Dropdown.update(
                            choices=formats,
                            interactive=True,
                            value=model_family.model_format,
                        ),
                        gr.Dropdown.update(
                            choices=model_sizes_in_billions,
                            interactive=True,
                            value=model_sizes_in_billions[0],
                        ),
                        gr.Dropdown.update(
                            choices=quantizations,
                            interactive=True,
                            value=quantizations[0],
                        ),
                    )
                else:
                    return (
                        gr.Dropdown.update(),
                        gr.Dropdown.update(),
                        gr.Dropdown.update(),
                    )

            model_name.change(
                select_model_name,
                inputs=[model_name],
                outputs=[model_format, model_size_in_billions, quantization],
            )

            components = self._build_chatbot("", "")
            model_text = components[0]
            chat, model_uid = components[1], components[-1]

        def select_model(
            _model_name: str,
            _model_format: str,
            _model_size_in_billions: str,
            _quantization: str,
        ):
            model_uid = self._create_model(
                _model_name, int(_model_size_in_billions), _model_format, _quantization
            )
            return gr.Chatbot.update(
                label="-".join(
                    [_model_name, _model_size_in_billions, _model_format, _quantization]
                ),
                value=[],
            ), gr.Textbox.update(value=model_uid)

        create_model.click(
            select_model,
            inputs=[model_name, model_format, model_size_in_billions, quantization],
            outputs=[chat, model_uid],
            postprocess=False,
        )
        return chat, model_text

    def _build_arena(self):
        with gr.Box():
            with gr.Row():
                chat_and_text = [
                    self._build_chat_column() for _ in range(self._gladiator_num)
                ]
                chats = [c[0] for c in chat_and_text]
                texts = [c[1] for c in chat_and_text]

            msg = gr.Textbox()

            def update_message(text_in: str):
                return "", text_in, text_in

            msg.submit(update_message, inputs=[msg], outputs=[msg] + texts)

        gr.ClearButton(components=[msg] + chats + texts)

    def _build_single(self):
        chat, model_text = self._build_chat_column()

        msg = gr.Textbox()

        def update_message(text_in: str):
            return "", text_in

        msg.submit(update_message, inputs=[msg], outputs=[msg, model_text])
        gr.ClearButton(components=[chat, msg, model_text])

    def _build_single_with_launched(
        self, models: List[Tuple[str, ModelSpec]], default_index: int
    ):
        uid_to_model_spec: Dict[str, ModelSpec] = dict((m[0], m[1]) for m in models)
        choices = [
            "-".join(
                [
                    s.model_name,
                    str(s.model_size_in_billions),
                    s.model_format,
                    s.quantization,
                ]
            )
            for s in uid_to_model_spec.values()
        ]
        choice_to_uid = dict(zip(choices, uid_to_model_spec.keys()))
        model_selection = gr.Dropdown(
            label="select model", choices=choices, value=choices[default_index]
        )
        components = self._build_chatbot(
            models[default_index][0], choices[default_index]
        )
        model_text = components[0]
        model_uid = components[-1]
        chat = components[1]

        def select_model(model_name):
            uid = choice_to_uid[model_name]
            return gr.Chatbot.update(label=model_name), uid

        model_selection.change(
            select_model, inputs=[model_selection], outputs=[chat, model_uid]
        )
        return chat, model_text

    def _build_arena_with_launched(self, models: List[Tuple[str, ModelSpec]]):
        with gr.Box():
            with gr.Row():
                chat_and_text = [
                    self._build_single_with_launched(models, i)
                    for i in range(self._gladiator_num)
                ]
                chats = [c[0] for c in chat_and_text]
                texts = [c[1] for c in chat_and_text]

            msg = gr.Textbox()

            def update_message(text_in: str):
                return "", text_in, text_in

            msg.submit(update_message, inputs=[msg], outputs=[msg] + texts)

        gr.ClearButton(components=[msg] + chats + texts)

    def build(self):
        if self._use_launched_model:
            models = self._api.list_models()
            with gr.Blocks() as blocks:
                with gr.Tab("Chat"):
                    chat, model_text = self._build_single_with_launched(models, 0)
                    msg = gr.Textbox()

                    def update_message(text_in: str):
                        return "", text_in

                    msg.submit(update_message, inputs=[msg], outputs=[msg, model_text])
                    gr.ClearButton(components=[chat, msg, model_text])
                if len(models) > 2:
                    with gr.Tab("Arena"):
                        self._build_arena_with_launched(models)
            return blocks
        else:
            with gr.Blocks() as blocks:
                with gr.Tab("Chat"):
                    self._build_single()
                with gr.Tab("Arena"):
                    self._build_arena()
            return blocks


class GradioActor(xo.Actor):
    def __init__(
        self,
        xoscar_endpoint: str,
        host: str,
        port: int,
        share: bool,
        use_launched_model: bool = False,
        gladiator_num: int = 2,
    ):
        super().__init__()
        self._gradio_cls = GradioApp(
            xoscar_endpoint, gladiator_num, use_launched_model=use_launched_model
        )
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
