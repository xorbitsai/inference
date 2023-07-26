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
import urllib.request
import uuid
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import gradio as gr
from huggingface_hub import snapshot_download

from ..constants import XINFERENCE_CACHE_DIR
from ..locale.utils import Locale
from ..model import MODEL_FAMILIES, ModelSpec
from .api import SyncSupervisorAPI

if TYPE_CHECKING:
    from ..types import ChatCompletionChunk, ChatCompletionMessage

MODEL_TO_FAMILIES = dict(
    (model_family.model_name, model_family)
    for model_family in MODEL_FAMILIES
    if model_family.model_name not in ["baichuan", "baichuan-base", "llama-2"]
)


class GradioApp:
    def __init__(
        self,
        supervisor_address: str,
        gladiator_num: int = 2,
        max_model_num: int = 3,
        use_launched_model: bool = False,
    ):
        self._api = SyncSupervisorAPI(supervisor_address)
        self._gladiator_num = gladiator_num
        self._max_model_num = max_model_num
        self._use_launched_model = use_launched_model
        self._locale = Locale()

    def _create_model(
        self,
        model_name: str,
        model_size_in_billions: Optional[int] = None,
        model_format: Optional[str] = None,
        quantization: Optional[str] = None,
    ):
        model_uid = str(uuid.uuid1())
        models = self._api.list_models()
        if len(models) >= self._max_model_num:
            self._api.terminate_model(models[0][0])
        return self._api.launch_model(
            model_uid, model_name, model_size_in_billions, model_format, quantization
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
                raise gr.Error(self._locale(f"Please create model first"))

            history: "List[ChatCompletionMessage]" = []
            for c in chat:
                history.append({"role": "user", "content": c[0]})

                out = c[1]
                finish_reason_idx = out.find(f"[{self._locale('stop reason')}: ")
                if finish_reason_idx != -1:
                    out = out[:finish_reason_idx]
                history.append({"role": "assistant", "content": out})

            if window_size != 0:
                history = history[-(window_size // 2) :]

            # chatglm only support even number of conversation history.
            if len(history) % 2 != 0:
                history = history[1:]

            generate_config = dict(
                max_tokens=max_token,
                temperature=temperature,
                top_p=top_p,
                stream=True,
            )
            chat += [[message, ""]]
            chat_generator = await model_ref.chat(
                message,
                chat_history=history,
                generate_config=generate_config,
            )

            chunk: Optional["ChatCompletionChunk"] = None
            async for chunk in chat_generator:
                assert chunk is not None
                delta = chunk["choices"][0]["delta"]
                if "content" not in delta:
                    continue
                else:
                    chat[-1][1] += delta["content"]
                    yield "", chat
            if show_finish_reason and chunk is not None:
                chat[-1][
                    1
                ] += f"[{self._locale('stop reason')}: {chunk['choices'][0]['finish_reason']}]"
                yield "", chat

    def _build_chatbot(self, model_uid: str, model_name: str):
        with gr.Accordion(self._locale("Parameters"), open=False):
            max_token = gr.Slider(
                128,
                1024,
                value=256,
                step=1,
                label=self._locale("Max tokens"),
                info=self._locale("The maximum number of tokens to generate."),
            )
            temperature = gr.Slider(
                0.2,
                1,
                value=0.8,
                step=0.01,
                label=self._locale("Temperature"),
                info=self._locale("The temperature to use for sampling."),
            )
            top_p = gr.Slider(
                0.2,
                1,
                value=0.95,
                step=0.01,
                label=self._locale("Top P"),
                info=self._locale("The top-p value to use for sampling."),
            )
            window_size = gr.Slider(
                0,
                50,
                value=10,
                step=1,
                label=self._locale("Window size"),
                info=self._locale("Window size of chat history."),
            )
            show_finish_reason = gr.Checkbox(
                label=f"{self._locale('Show stop reason')}"
            )
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
                    choices=list(MODEL_TO_FAMILIES.keys()),
                    label=self._locale("model name"),
                    scale=2,
                )
                model_format = gr.Dropdown(
                    choices=[],
                    interactive=False,
                    label=self._locale("model format"),
                    scale=2,
                )
                model_size_in_billions = gr.Dropdown(
                    choices=[],
                    interactive=False,
                    label=self._locale("model size in billions"),
                    scale=1,
                )
                quantization = gr.Dropdown(
                    choices=[],
                    interactive=False,
                    label=self._locale("quantization"),
                    scale=1,
                )
            create_model = gr.Button(value=self._locale("create"))

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
            progress=gr.Progress(),
        ):
            model_family = MODEL_TO_FAMILIES[_model_name]
            cache_path = model_family.generate_cache_path(
                int(_model_size_in_billions), _quantization
            )
            url = model_family.url_generator(
                int(_model_size_in_billions), _quantization
            )
            if _model_format == "pytorch":
                try:
                    full_name = (
                        f"{_model_name}-{_model_format}-{_model_size_in_billions}b-any"
                    )
                    save_dir = os.path.join(XINFERENCE_CACHE_DIR, full_name)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    snapshot_download(
                        repo_id=url,
                        revision="main",
                        local_dir=save_dir,
                        local_files_only=True,
                    )
                except:
                    raise gr.Error(self._locale(f"Download failed, please retry."))
            elif not (os.path.exists(cache_path)):
                try:
                    urllib.request.urlretrieve(
                        url,
                        cache_path,
                        reporthook=lambda block_num, block_size, total_size: progress(
                            block_num * block_size / total_size,
                            desc=self._locale("Downloading"),
                        ),
                    )
                except:
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                    raise gr.Error(self._locale(f"Download failed, please retry."))

            model_uid = self._create_model(
                _model_name, int(_model_size_in_billions), _model_format, _quantization
            )
            return gr.Chatbot.update(
                label="-".join(
                    [_model_name, _model_size_in_billions, _model_format, _quantization]
                ),
                value=[],
            ), gr.Textbox.update(value=model_uid)

        def clear_chat(
            _model_name: str,
            _model_format: str,
            _model_size_in_billions: str,
            _quantization: str,
        ):
            full_name = "-".join(
                [_model_name, _model_size_in_billions, _model_format, _quantization]
            )
            return str(uuid.uuid4()), gr.Chatbot.update(
                label=full_name,
                value=[],
            )

        invisible_text = gr.Textbox(visible=False)
        create_model.click(
            clear_chat,
            inputs=[model_name, model_format, model_size_in_billions, quantization],
            outputs=[invisible_text, chat],
        )

        invisible_text.change(
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

            msg = gr.Textbox(label=self._locale("Input"))

            def update_message(text_in: str):
                return "", text_in, text_in

            msg.submit(update_message, inputs=[msg], outputs=[msg] + texts)

        gr.ClearButton(components=[msg] + chats + texts)

    def _build_single(self):
        chat, model_text = self._build_chat_column()

        msg = gr.Textbox(label=self._locale("Input"))

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
            label=self._locale("select model"),
            choices=choices,
            value=choices[default_index],
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
        chat_and_text = []
        with gr.Row():
            for i in range(self._gladiator_num):
                with gr.Column():
                    chat_and_text.append(self._build_single_with_launched(models, i))

        chats = [c[0] for c in chat_and_text]
        texts = [c[1] for c in chat_and_text]

        msg = gr.Textbox(label=self._locale("Input"))

        def update_message(text_in: str):
            return "", text_in, text_in

        msg.submit(update_message, inputs=[msg], outputs=[msg] + texts)

        gr.ClearButton(components=[msg] + chats + texts)

    def build(self):
        if self._use_launched_model:
            models = self._api.list_models()
            with gr.Blocks() as blocks:
                with gr.Tab(self._locale("Chat")):
                    chat, model_text = self._build_single_with_launched(models, 0)
                    msg = gr.Textbox(label=self._locale("Input"))

                    def update_message(text_in: str):
                        return "", text_in

                    msg.submit(update_message, inputs=[msg], outputs=[msg, model_text])
                    gr.ClearButton(components=[chat, msg, model_text])
                if len(models) >= 2:
                    with gr.Tab(self._locale("Arena")):
                        self._build_arena_with_launched(models)
        else:
            with gr.Blocks() as blocks:
                with gr.Tab(self._locale("Chat")):
                    self._build_single()
                with gr.Tab(self._locale("Arena")):
                    self._build_arena()
        blocks.queue(concurrency_count=40)
        return blocks
