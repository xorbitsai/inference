# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# -*- coding:utf-8 -*-

import base64
from io import BytesIO

import gradio as gr
import torch
from app_modules.gradio_utils import (
    cancel_outputing,
    delete_last_conversation,
    reset_state,
    reset_textbox,
    transfer_input,
    wrap_gen_fn,
)
from app_modules.overwrites import reload_javascript
from app_modules.presets import CONCURRENT_COUNT, description, description_top, title
from app_modules.utils import configure_logger, is_variable_assigned, strip_stop_words

from deepseek_vl.serve.inference import (
    convert_conversation_to_prompts,
    deepseek_generate,
    load_model,
)
from deepseek_vl.utils.conversation import SeparatorStyle


def load_models():
    models = {
        "DeepSeek-VL 7B": "deepseek-ai/deepseek-vl-7b-chat",
    }

    for model_name in models:
        models[model_name] = load_model(models[model_name])

    return models


logger = configure_logger()
models = load_models()
MODELS = sorted(list(models.keys()))


def generate_prompt_with_history(
    text, image, history, vl_chat_processor, tokenizer, max_length=2048
):
    """
    Generate a prompt with history for the deepseek application.

    Args:
        text (str): The text prompt.
        image (str): The image prompt.
        history (list): List of previous conversation messages.
        tokenizer: The tokenizer used for encoding the prompt.
        max_length (int): The maximum length of the prompt.

    Returns:
        tuple: A tuple containing the generated prompt, image list, conversation, and conversation copy. If the prompt could not be generated within the max_length limit, returns None.
    """

    sft_format = "deepseek"
    user_role_ind = 0
    bot_role_ind = 1

    # Initialize conversation
    conversation = vl_chat_processor.new_chat_template()

    if history:
        conversation.messages = history

    if image is not None:
        if "<image_placeholder>" not in text:
            text = (
                "<image_placeholder>" + "\n" + text
            )  # append the <image_placeholder> in a new line after the text prompt
        text = (text, image)

    conversation.append_message(conversation.roles[user_role_ind], text)
    conversation.append_message(conversation.roles[bot_role_ind], "")

    # Create a copy of the conversation to avoid history truncation in the UI
    conversation_copy = conversation.copy()
    logger.info("=" * 80)
    logger.info(get_prompt(conversation))

    rounds = len(conversation.messages) // 2

    for _ in range(rounds):
        current_prompt = get_prompt(conversation)
        current_prompt = (
            current_prompt.replace("</s>", "")
            if sft_format == "deepseek"
            else current_prompt
        )

        if torch.tensor(tokenizer.encode(current_prompt)).size(-1) <= max_length:
            return conversation_copy

        if len(conversation.messages) % 2 != 0:
            gr.Error("The messages between user and assistant are not paired.")
            return

        try:
            for _ in range(2):  # pop out two messages in a row
                conversation.messages.pop(0)
        except IndexError:
            gr.Error("Input text processing failed, unable to respond in this round.")
            return None

    gr.Error("Prompt could not be generated within max_length limit.")
    return None


def to_gradio_chatbot(conv):
    """Convert the conversation to gradio chatbot format."""
    ret = []
    for i, (role, msg) in enumerate(conv.messages[conv.offset :]):
        if i % 2 == 0:
            if type(msg) is tuple:
                msg, image = msg
                if isinstance(image, str):
                    with open(image, "rb") as f:
                        data = f.read()
                    img_b64_str = base64.b64encode(data).decode()
                    image_str = f'<video src="data:video/mp4;base64,{img_b64_str}" controls width="426" height="240"></video>'
                    msg = msg.replace("\n".join(["<image_placeholder>"] * 4), image_str)
                else:
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = msg.replace("<image_placeholder>", img_str)
            ret.append([msg, None])
        else:
            ret[-1][-1] = msg
    return ret


def to_gradio_history(conv):
    """Convert the conversation to gradio history state."""
    return conv.messages[conv.offset :]


def get_prompt(conv) -> str:
    """Get the prompt for generation."""
    system_prompt = conv.system_template.format(system_message=conv.system_message)
    if conv.sep_style == SeparatorStyle.DeepSeek:
        seps = [conv.sep, conv.sep2]
        if system_prompt == "" or system_prompt is None:
            ret = ""
        else:
            ret = system_prompt + seps[0]
        for i, (role, message) in enumerate(conv.messages):
            if message:
                if type(message) is tuple:  # multimodal message
                    message, _ = message
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret
    else:
        return conv.get_prompt


@wrap_gen_fn
def predict(
    text,
    image,
    chatbot,
    history,
    top_p,
    temperature,
    repetition_penalty,
    max_length_tokens,
    max_context_length_tokens,
    model_select_dropdown,
):
    """
    Function to predict the response based on the user's input and selected model.

    Parameters:
    user_text (str): The input text from the user.
    user_image (str): The input image from the user.
    chatbot (str): The chatbot's name.
    history (str): The history of the chat.
    top_p (float): The top-p parameter for the model.
    temperature (float): The temperature parameter for the model.
    max_length_tokens (int): The maximum length of tokens for the model.
    max_context_length_tokens (int): The maximum length of context tokens for the model.
    model_select_dropdown (str): The selected model from the dropdown.

    Returns:
    generator: A generator that yields the chatbot outputs, history, and status.
    """
    print("running the prediction function")
    try:
        tokenizer, vl_gpt, vl_chat_processor = models[model_select_dropdown]

        if text == "":
            yield chatbot, history, "Empty context."
            return
    except KeyError:
        yield [[text, "No Model Found"]], [], "No Model Found"
        return

    conversation = generate_prompt_with_history(
        text,
        image,
        history,
        vl_chat_processor,
        tokenizer,
        max_length=max_context_length_tokens,
    )
    prompts = convert_conversation_to_prompts(conversation)

    stop_words = conversation.stop_str
    gradio_chatbot_output = to_gradio_chatbot(conversation)

    full_response = ""
    with torch.no_grad():
        for x in deepseek_generate(
            prompts=prompts,
            vl_gpt=vl_gpt,
            vl_chat_processor=vl_chat_processor,
            tokenizer=tokenizer,
            stop_words=stop_words,
            max_length=max_length_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
        ):
            full_response += x
            response = strip_stop_words(full_response, stop_words)
            conversation.update_last_message(response)
            gradio_chatbot_output[-1][1] = response
            yield gradio_chatbot_output, to_gradio_history(
                conversation
            ), "Generating..."

    print("flushed result to gradio")
    torch.cuda.empty_cache()

    if is_variable_assigned("x"):
        print(f"{model_select_dropdown}:\n{text}\n{'-' * 80}\n{x}\n{'=' * 80}")
        print(
            f"temperature: {temperature}, top_p: {top_p}, repetition_penalty: {repetition_penalty}, max_length_tokens: {max_length_tokens}"
        )

    yield gradio_chatbot_output, to_gradio_history(conversation), "Generate: Success"


def retry(
    text,
    image,
    chatbot,
    history,
    top_p,
    temperature,
    repetition_penalty,
    max_length_tokens,
    max_context_length_tokens,
    model_select_dropdown,
):
    if len(history) == 0:
        yield (chatbot, history, "Empty context")
        return

    chatbot.pop()
    history.pop()
    text = history.pop()[-1]
    if type(text) is tuple:
        text, image = text

    yield from predict(
        text,
        image,
        chatbot,
        history,
        top_p,
        temperature,
        repetition_penalty,
        max_length_tokens,
        max_context_length_tokens,
        model_select_dropdown,
    )


def build_demo(MODELS):
    with open("deepseek_vl/serve/assets/custom.css", "r", encoding="utf-8") as f:
        customCSS = f.read()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        history = gr.State([])
        input_text = gr.State()
        input_image = gr.State()

        with gr.Row():
            gr.HTML(title)
            status_display = gr.Markdown("Success", elem_id="status_display")
        gr.Markdown(description_top)

        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                with gr.Row():
                    chatbot = gr.Chatbot(
                        elem_id="deepseek_chatbot",
                        show_share_button=True,
                        likeable=True,
                        bubble_full_width=False,
                        height=600,
                    )
                with gr.Row():
                    with gr.Column(scale=4):
                        text_box = gr.Textbox(
                            show_label=False, placeholder="Enter text", container=False
                        )
                    with gr.Column(
                        min_width=70,
                    ):
                        submitBtn = gr.Button("Send")
                    with gr.Column(
                        min_width=70,
                    ):
                        cancelBtn = gr.Button("Stop")
                with gr.Row():
                    emptyBtn = gr.Button(
                        "🧹 New Conversation",
                    )
                    retryBtn = gr.Button("🔄 Regenerate")
                    delLastBtn = gr.Button("🗑️ Remove Last Turn")

            with gr.Column():
                image_box = gr.Image(type="pil")

                with gr.Tab(label="Parameter Setting") as parameter_row:
                    top_p = gr.Slider(
                        minimum=-0,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                        interactive=True,
                        label="Top-p",
                    )
                    temperature = gr.Slider(
                        minimum=0,
                        maximum=1.0,
                        value=0.1,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    repetition_penalty = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.1,
                        step=0.1,
                        interactive=True,
                        label="Repetition penalty",
                    )
                    max_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=2048,
                        step=8,
                        interactive=True,
                        label="Max Generation Tokens",
                    )
                    max_context_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=4096,
                        step=128,
                        interactive=True,
                        label="Max History Tokens",
                    )
                    model_select_dropdown = gr.Dropdown(
                        label="Select Models",
                        choices=MODELS,
                        multiselect=False,
                        value=MODELS[0],
                        interactive=True,
                    )

        examples_list = [
            [
                "deepseek_vl/serve/examples/rap.jpeg",
                "Can you write me a master rap song that rhymes very well based on this image?",
            ],
            [
                "deepseek_vl/serve/examples/app.png",
                "What is this app about?",
            ],
            [
                "deepseek_vl/serve/examples/pipeline.png",
                "Help me write a python code based on the image.",
            ],
            [
                "deepseek_vl/serve/examples/chart.png",
                "Could you help me to re-draw this picture with python codes?",
            ],
            [
                "deepseek_vl/serve/examples/mirror.png",
                "How many people are there in the image. Why?",
            ],
            [
                "deepseek_vl/serve/examples/puzzle.png",
                "Can this 2 pieces combine together?",
            ],
        ]
        gr.Examples(examples=examples_list, inputs=[image_box, text_box])
        gr.Markdown(description)

        input_widgets = [
            input_text,
            input_image,
            chatbot,
            history,
            top_p,
            temperature,
            repetition_penalty,
            max_length_tokens,
            max_context_length_tokens,
            model_select_dropdown,
        ]
        output_widgets = [chatbot, history, status_display]

        transfer_input_args = dict(
            fn=transfer_input,
            inputs=[text_box, image_box],
            outputs=[input_text, input_image, text_box, image_box, submitBtn],
            show_progress=True,
        )

        predict_args = dict(
            fn=predict,
            inputs=input_widgets,
            outputs=output_widgets,
            show_progress=True,
        )

        retry_args = dict(
            fn=retry,
            inputs=input_widgets,
            outputs=output_widgets,
            show_progress=True,
        )

        reset_args = dict(
            fn=reset_textbox, inputs=[], outputs=[text_box, status_display]
        )

        predict_events = [
            text_box.submit(**transfer_input_args).then(**predict_args),
            submitBtn.click(**transfer_input_args).then(**predict_args),
        ]

        emptyBtn.click(reset_state, outputs=output_widgets, show_progress=True)
        emptyBtn.click(**reset_args)
        retryBtn.click(**retry_args)

        delLastBtn.click(
            delete_last_conversation,
            [chatbot, history],
            output_widgets,
            show_progress=True,
        )

        cancelBtn.click(cancel_outputing, [], [status_display], cancels=predict_events)

    return demo


if __name__ == "__main__":
    demo = build_demo(MODELS)
    demo.title = "DeepSeek-VL Chatbot"

    reload_javascript()
    demo.queue(concurrency_count=CONCURRENT_COUNT).launch(
        share=False,
        favicon_path="deepseek_vl/serve/assets/favicon.ico",
        inbrowser=False,
        server_name="0.0.0.0",
        server_port=8122,
    )
