# Copyright 2022-2026 XProbe Inc.
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

import base64
import html
import logging
import os
import tempfile
from io import BytesIO
from typing import Generator, List, Optional

import gradio as gr
import PIL.Image
from gradio import ChatMessage
from gradio.components import Markdown, Textbox
from gradio.layouts import Accordion, Column, Row

from ...client.restful.restful_client import (
    RESTfulChatModelHandle,
    RESTfulGenerateModelHandle,
)

logger = logging.getLogger(__name__)


class GradioInterface:
    def __init__(
        self,
        endpoint: str,
        model_uid: str,
        model_name: str,
        model_size_in_billions: int,
        model_type: str,
        model_format: str,
        quantization: str,
        context_length: int,
        model_ability: List[str],
        model_description: str,
        model_lang: List[str],
        access_token: Optional[str],
    ):
        self.endpoint = endpoint
        self.model_uid = model_uid
        self.model_name = model_name
        self.model_size_in_billions = model_size_in_billions
        self.model_type = model_type
        self.model_format = model_format
        self.quantization = quantization
        self.context_length = context_length
        self.model_ability = model_ability
        self.model_description = model_description
        self.model_lang = model_lang
        self._access_token = (
            access_token.replace("Bearer ", "") if access_token is not None else None
        )

    def build(self) -> "gr.Blocks":
        if "vision" in self.model_ability:
            interface = self.build_chat_multimodel_interface()
        elif "chat" in self.model_ability:
            interface = self.build_chat_interface()
        else:
            interface = self.build_generate_interface()

        interface.queue(default_concurrency_limit=os.cpu_count())
        # Gradio initiates the queue during a startup event, but since the app has already been
        # started, that event will not run, so manually invoke the startup events.
        # See: https://github.com/gradio-app/gradio/issues/5228
        try:
            interface.run_startup_events()
        except AttributeError:
            # compatibility
            interface.startup_events()
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

    def build_chat_interface(self) -> "gr.Blocks":
        def generate_wrapper(
            message: str,
            history: List[ChatMessage],
            max_tokens: int,
            temperature: float,
            lora_name: str,
            stream: bool,
        ) -> Generator:
            from ...client import RESTfulClient

            client = RESTfulClient(self.endpoint)
            client._set_token(self._access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulChatModelHandle)

            generate_config = {
                "temperature": temperature,
                "stream": stream,
                "lora_name": lora_name,
            }
            if max_tokens > 0:
                generate_config["max_tokens"] = max_tokens

            # Convert history to messages format
            messages = []
            for msg in history:
                # ignore thinking content
                if msg["metadata"]:
                    continue
                messages.append({"role": msg["role"], "content": msg["content"]})

            if stream:
                response_content = ""
                reasoning_content = ""
                is_first_reasoning_content = True
                is_first_content = True
                for chunk in model.chat(
                    messages=messages,
                    generate_config=generate_config,  # type: ignore
                ):
                    assert isinstance(chunk, dict)
                    if not chunk["choices"]:
                        continue
                    delta = chunk["choices"][0]["delta"]

                    if (
                        "reasoning_content" in delta
                        and delta["reasoning_content"] is not None
                        and is_first_reasoning_content
                    ):
                        reasoning_content += html.escape(delta["reasoning_content"])
                        history.append(
                            ChatMessage(
                                role="assistant",
                                content=reasoning_content,
                                metadata={"title": "💭 Thinking Process"},
                            )
                        )
                        is_first_reasoning_content = False
                    elif (
                        "reasoning_content" in delta
                        and delta["reasoning_content"] is not None
                    ):
                        reasoning_content += html.escape(delta["reasoning_content"])
                        history[-1] = ChatMessage(
                            role="assistant",
                            content=reasoning_content,
                            metadata={"title": "💭  Thinking Process"},
                        )
                    elif (
                        "content" in delta
                        and delta["content"] is not None
                        and is_first_content
                    ):
                        response_content += html.escape(delta["content"])
                        history.append(
                            ChatMessage(role="assistant", content=response_content)
                        )
                        is_first_content = False
                    elif "content" in delta and delta["content"] is not None:
                        response_content += html.escape(delta["content"])
                        # Replace thinking message with actual response
                        history[-1] = ChatMessage(
                            role="assistant", content=response_content
                        )
                    yield history
            else:
                result = model.chat(
                    messages=messages,
                    generate_config=generate_config,  # type: ignore
                )
                assert isinstance(result, dict)
                mg = result["choices"][0]["message"]
                if "reasoning_content" in mg:
                    reasoning_content = mg["reasoning_content"]
                    if reasoning_content is not None:
                        reasoning_content = html.escape(str(reasoning_content))
                        history.append(
                            ChatMessage(
                                role="assistant",
                                content=reasoning_content,
                                metadata={"title": "💭 Thinking Process"},
                            )
                        )

                content = mg["content"]
                response_content = (
                    html.escape(str(content)) if content is not None else ""
                )
                history.append(ChatMessage(role="assistant", content=response_content))
                yield history

        with gr.Blocks(
            title=f"🚀 Xinference Chat Bot : {self.model_name} 🚀",
            css="""
            .center{
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 0px;
                color: #9ea4b0 !important;
            }
            .main-container {
                display: flex;
                flex-direction: column;
                padding: 0.5rem;
                box-sizing: border-box;
                gap: 0.25rem;
                flex-grow: 1;
                min-width: min(320px, 100%);
                height: calc(100vh - 70px)!important;
            }
            .header {
                flex-grow: 0!important;
            }
            .header h1 {
                margin: 0.5rem 0;
                font-size: 1.5rem;
            }
            .center {
                font-size: 0.9rem;
                margin: 0.1rem 0;
            }
            .chat-container {
                flex: 1;
                display: flex;
                min-height: 0;
                margin: 0.25rem 0;
            }
            .chat-container .block {
                height: 100%!important;
            }
            .input-container {
                flex-grow: 0!important;
            }
            """,
            analytics_enabled=False,
        ) as chat_interface:
            with gr.Column(elem_classes="main-container"):
                # Header section
                with gr.Column(elem_classes="header"):
                    gr.Markdown(
                        f"""<h1 style='text-align: center; margin-bottom: 1rem'>🚀 Xinference Chat Bot : {self.model_name} 🚀</h1>"""
                    )
                    gr.Markdown(
                        f"""
                        <div class="center">Model ID: {self.model_uid}</div>
                        <div class="center">Model Size: {self.model_size_in_billions} Billion Parameters</div>
                        <div class="center">Model Format: {self.model_format}</div>
                        <div class="center">Model Quantization: {self.quantization}</div>
                        """
                    )

                # Chat container
                with gr.Column(elem_classes="chat-container"):
                    chatbot = gr.Chatbot(
                        type="messages",
                        label=self.model_name,
                        show_label=True,
                        render_markdown=True,
                        container=True,
                    )

                # Input container
                with gr.Column(elem_classes="input-container"):
                    with gr.Row():
                        with gr.Column(scale=12):
                            textbox = gr.Textbox(
                                show_label=False,
                                placeholder="Type a message...",
                                container=False,
                            )
                        with gr.Column(scale=1, min_width=50):
                            submit_btn = gr.Button("Enter", variant="primary")

                    with gr.Accordion("Additional Inputs", open=False):
                        max_tokens = gr.Slider(
                            minimum=0,
                            maximum=self.context_length,
                            value=0,
                            step=1,
                            label="Max Tokens",
                        )
                        temperature = gr.Slider(
                            minimum=0,
                            maximum=2,
                            value=1,
                            step=0.01,
                            label="Temperature",
                        )
                        stream = gr.Checkbox(label="Stream", value=True)
                        lora_name = gr.Text(label="LoRA Name")

            # deal with message submit
            textbox.submit(
                lambda m, h: ("", h + [ChatMessage(role="user", content=m)]),
                [textbox, chatbot],
                [textbox, chatbot],
            ).then(
                generate_wrapper,
                [textbox, chatbot, max_tokens, temperature, lora_name, stream],
                chatbot,
            )

            submit_btn.click(
                lambda m, h: ("", h + [ChatMessage(role="user", content=m)]),
                [textbox, chatbot],
                [textbox, chatbot],
            ).then(
                generate_wrapper,
                [textbox, chatbot, max_tokens, temperature, lora_name, stream],
                chatbot,
            )

            return chat_interface

    def build_chat_multimodel_interface(
        self,
    ) -> "gr.Blocks":
        def predict(history, bot, max_tokens, temperature, stream):
            from ...client import RESTfulClient

            generate_config = {
                "temperature": temperature,
                "stream": stream,
            }
            if max_tokens > 0:
                generate_config["max_tokens"] = max_tokens

            client = RESTfulClient(self.endpoint)
            client._set_token(self._access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulChatModelHandle)

            if stream:
                response_content = ""
                for chunk in model.chat(
                    messages=history,
                    generate_config=generate_config,
                ):
                    assert isinstance(chunk, dict)
                    delta = chunk["choices"][0]["delta"]
                    if "content" not in delta:
                        continue
                    else:
                        response_content += html.escape(delta["content"])
                        bot[-1][1] = response_content
                        yield history, bot
                history.append(
                    {
                        "content": response_content,
                        "role": "assistant",
                    }
                )
                bot[-1][1] = response_content
                yield history, bot
            else:
                response = model.chat(
                    messages=history,
                    generate_config=generate_config,
                )
                history.append(response["choices"][0]["message"])
                if "audio" in history[-1]:
                    # audio output
                    audio_bytes = base64.b64decode(history[-1]["audio"]["data"])
                    audio_file = tempfile.NamedTemporaryFile(
                        delete=False, suffix=".wav"
                    )
                    audio_file.write(audio_bytes)
                    audio_file.close()

                    def audio_to_base64(audio_path):
                        with open(audio_path, "rb") as audio_file:
                            return base64.b64encode(audio_file.read()).decode("utf-8")

                    def generate_html_audio(audio_path):
                        base64_audio = audio_to_base64(audio_path)
                        audio_format = audio_path.split(".")[-1]
                        return (
                            f"<audio controls style='max-width:100%;'>"
                            f"<source src='data:audio/{audio_format};base64,{base64_audio}' type='audio/{audio_format}'>"
                            f"Your browser does not support the audio tag.</audio>"
                        )

                    bot[-1] = (bot[-1][0], history[-1]["content"])
                    yield history, bot

                    # append html audio tag instead of gr.Audio
                    bot.append((None, generate_html_audio(audio_file.name)))
                    yield history, bot
                else:
                    bot[-1][1] = history[-1]["content"]
                    yield history, bot

        def add_text(history, bot, text, image, video, audio):
            logger.debug(
                "Add text, text: %s, image: %s, video: %s, audio: %s",
                text,
                image,
                video,
                audio,
            )
            if image:
                buffered = BytesIO()
                with PIL.Image.open(image) as img:
                    img.thumbnail((500, 500))
                    img.save(buffered, format="JPEG")
                img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                display_content = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />\n{text}'
                message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64_str}"
                            },
                        },
                    ],
                }
            elif video:

                def video_to_base64(video_path):
                    with open(video_path, "rb") as video_file:
                        encoded_string = base64.b64encode(video_file.read()).decode(
                            "utf-8"
                        )
                    return encoded_string

                def generate_html_video(video_path):
                    base64_video = video_to_base64(video_path)
                    video_format = video_path.split(".")[-1]
                    html_code = f"""
                    <video controls>
                        <source src="data:video/{video_format};base64,{base64_video}" type="video/{video_format}">
                        Your browser does not support the video tag.
                    </video>
                    """
                    return html_code

                display_content = f"{generate_html_video(video)}\n{text}"
                message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "video_url",
                            "video_url": {"url": video},
                        },
                    ],
                }

            elif audio:

                def audio_to_base64(audio_path):
                    with open(audio_path, "rb") as audio_file:
                        encoded_string = base64.b64encode(audio_file.read()).decode(
                            "utf-8"
                        )
                    return encoded_string

                def generate_html_audio(audio_path):
                    base64_audio = audio_to_base64(audio_path)
                    audio_format = audio_path.split(".")[-1]
                    return (
                        f"<audio controls style='max-width:100%;'>"
                        f"<source src='data:audio/{audio_format};base64,{base64_audio}' type='audio/{audio_format}'>"
                        f"Your browser does not support the audio tag.</audio>"
                    )

                display_content = f"{generate_html_audio(audio)}<br>{text}"
                message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "audio_url",
                            "audio_url": {"url": audio},
                        },
                    ],
                }

            else:
                display_content = text
                message = {"role": "user", "content": text}
            history = history + [message]
            bot = bot + [[display_content, None]]
            return history, bot, "", None, None, None

        def clear_history():
            logger.debug("Clear history.")
            return [], None, "", None, None, None

        def update_button(text):
            return gr.update(interactive=bool(text))

        has_vision = "vision" in self.model_ability
        has_audio = "audio" in self.model_ability

        with gr.Blocks(
            title=f"🚀 Xinference Chat Bot : {self.model_name} 🚀",
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
        ) as chat_vl_interface:
            Markdown(
                f"""
                <h1 style='text-align: center; margin-bottom: 1rem'>🚀 Xinference Chat Bot : {self.model_name} 🚀</h1>
                """
            )
            Markdown(
                f"""
                <div class="center">
                Model ID: {self.model_uid}
                </div>
                <div class="center">
                Model Size: {self.model_size_in_billions} Billion Parameters
                </div>
                <div class="center">
                Model Format: {self.model_format}
                </div>
                <div class="center">
                Model Quantization: {self.quantization}
                </div>
                """
            )

            state = gr.State([])
            with gr.Row():
                chatbot = gr.Chatbot(
                    elem_id="chatbot", label=self.model_name, scale=7, min_height=900
                )
                with gr.Column(scale=3):
                    if has_vision:
                        imagebox = gr.Image(type="filepath")
                        videobox = gr.Video()
                    else:
                        imagebox = gr.Image(type="filepath", visible=False)
                        videobox = gr.Video(visible=False)

                    if has_audio:
                        audiobox = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            visible=True,
                        )
                    else:
                        audiobox = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            visible=False,
                        )

                    textbox = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press ENTER",
                        container=False,
                    )
                    submit_btn = gr.Button(
                        value="Send", variant="primary", interactive=False
                    )
                    clear_btn = gr.Button(value="Clear")

            with gr.Accordion("Additional Inputs", open=False):
                max_tokens = gr.Slider(
                    minimum=0,
                    maximum=self.context_length,
                    value=0,
                    step=1,
                    label="Max Tokens (0 stands for maximum possible tokens)",
                )
                temperature = gr.Slider(
                    minimum=0, maximum=2, value=1, step=0.01, label="Temperature"
                )
                stream = gr.Checkbox(label="Stream", value=False)

            textbox.change(update_button, [textbox], [submit_btn], queue=False)

            textbox.submit(
                add_text,
                [state, chatbot, textbox, imagebox, videobox, audiobox],
                [state, chatbot, textbox, imagebox, videobox, audiobox],
                queue=False,
            ).then(
                predict,
                [state, chatbot, max_tokens, temperature, stream],
                [state, chatbot],
            )

            submit_btn.click(
                add_text,
                [state, chatbot, textbox, imagebox, videobox, audiobox],
                [state, chatbot, textbox, imagebox, videobox, audiobox],
                queue=False,
            ).then(
                predict,
                [state, chatbot, max_tokens, temperature, stream],
                [state, chatbot],
            )

            clear_btn.click(
                clear_history,
                None,
                [state, chatbot, textbox, imagebox, videobox, audiobox],
                queue=False,
            )

        return chat_vl_interface

    def build_generate_interface(
        self,
    ):
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

        def complete(text, hist, max_tokens, temperature, lora_name) -> Generator:
            from ...client import RESTfulClient

            client = RESTfulClient(self.endpoint)
            client._set_token(self._access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulGenerateModelHandle)

            if len(hist) == 0 or (len(hist) > 0 and text != hist[-1]):
                hist.append(text)

            generate_config = {
                "temperature": temperature,
                "stream": True,
                "lora_name": lora_name,
            }
            if max_tokens > 0:
                generate_config["max_tokens"] = max_tokens

            response_content = text
            for chunk in model.generate(
                prompt=text,
                generate_config=generate_config,  # type: ignore
            ):
                assert isinstance(chunk, dict)
                choice = chunk["choices"][0]
                if "text" not in choice:
                    continue
                else:
                    response_content += choice["text"]
                    yield {
                        textbox: response_content,
                        history: hist,
                    }

            hist.append(response_content)
            return {  # type: ignore
                textbox: response_content,
                history: hist,
            }

        def retry(text, hist, max_tokens, temperature, lora_name) -> Generator:
            from ...client import RESTfulClient

            generate_config = {
                "temperature": temperature,
                "stream": True,
                "lora_name": lora_name,
            }
            if max_tokens > 0:
                generate_config["max_tokens"] = max_tokens

            client = RESTfulClient(self.endpoint)
            client._set_token(self._access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulGenerateModelHandle)

            if len(hist) == 0 or (len(hist) > 0 and text != hist[-1]):
                hist.append(text)
            text = hist[-2] if len(hist) > 1 else ""

            response_content = text
            for chunk in model.generate(prompt=text, generate_config=generate_config):  # type: ignore
                assert isinstance(chunk, dict)
                choice = chunk["choices"][0]
                if "text" not in choice:
                    continue
                else:
                    response_content += choice["text"]
                    yield {
                        textbox: response_content,
                        history: hist,
                    }

            hist.append(response_content)
            return {  # type: ignore
                textbox: response_content,
                history: hist,
            }

        with gr.Blocks(
            title=f"🚀 Xinference Generate Bot : {self.model_name} 🚀",
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
                <h1 style='text-align: center; margin-bottom: 1rem'>🚀 Xinference Generate Bot : {self.model_name} 🚀</h1>
                """
            )
            Markdown(
                f"""
                <div class="center">
                Model ID: {self.model_uid}
                </div>
                <div class="center">
                Model Size: {self.model_size_in_billions} Billion Parameters
                </div>
                <div class="center">
                Model Format: {self.model_format}
                </div>
                <div class="center">
                Model Quantization: {self.quantization}
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
                        minimum=0,
                        maximum=self.context_length,
                        value=0,
                        step=1,
                        label="Max Tokens (0 stands for maximum possible tokens)",
                    )
                    temperature = gr.Slider(
                        minimum=0, maximum=2, value=1, step=0.01, label="Temperature"
                    )
                    lora_name = gr.Text(label="LoRA Name")

                btn_generate.click(
                    fn=complete,
                    inputs=[textbox, history, length, temperature, lora_name],
                    outputs=[textbox, history],
                )

                btn_undo.click(
                    fn=undo,
                    inputs=[textbox, history],
                    outputs=[textbox, history],
                )

                btn_retry.click(
                    fn=retry,
                    inputs=[textbox, history, length, temperature, lora_name],
                    outputs=[textbox, history],
                )

                btn_clear.click(
                    fn=clear,
                    inputs=[textbox, history],
                    outputs=[textbox, history],
                )

        return generate_interface
