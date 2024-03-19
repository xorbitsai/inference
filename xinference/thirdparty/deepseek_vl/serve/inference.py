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

from threading import Thread
from typing import List

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.conversation import Conversation


def load_model(model_path):
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    return tokenizer, vl_gpt, vl_chat_processor


def convert_conversation_to_prompts(conversation: Conversation):
    prompts = []
    messages = conversation.messages

    for i in range(0, len(messages), 2):
        prompt = {
            "role": messages[i][0],
            "content": (
                messages[i][1][0]
                if isinstance(messages[i][1], tuple)
                else messages[i][1]
            ),
            "images": [messages[i][1][1]] if isinstance(messages[i][1], tuple) else [],
        }
        response = {"role": messages[i + 1][0], "content": messages[i + 1][1]}
        prompts.extend([prompt, response])

    return prompts


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ):
        for stop in self.stops:
            if input_ids.shape[-1] < len(stop):
                continue
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True

        return False


@torch.inference_mode()
def deepseek_generate(
    prompts: list,
    vl_gpt: torch.nn.Module,
    vl_chat_processor,
    tokenizer: transformers.PreTrainedTokenizer,
    stop_words: list,
    max_length: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty=1.1,
):
    prompts = prompts
    pil_images = list()
    for message in prompts:
        if "images" not in message:
            continue
        for pil_img in message["images"]:
            pil_images.append(pil_img)

    prepare_inputs = vl_chat_processor(
        conversations=prompts, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    return generate(
        vl_gpt,
        tokenizer,
        prepare_inputs,
        max_length,
        temperature,
        repetition_penalty,
        top_p,
        stop_words,
    )


@torch.inference_mode()
def generate(
    vl_gpt,
    tokenizer,
    prepare_inputs,
    max_gen_len: int = 256,
    temperature: float = 0,
    repetition_penalty=1.1,
    top_p: float = 0.95,
    stop_words: List[str] = [],
):
    """Stream the text output from the multimodality model with prompt and image inputs."""
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    streamer = TextIteratorStreamer(tokenizer)

    stop_words_ids = [
        torch.tensor(tokenizer.encode(stop_word)) for stop_word in stop_words
    ]
    stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids)]
    )

    generation_config = dict(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_gen_len,
        do_sample=True,
        use_cache=True,
        streamer=streamer,
        stopping_criteria=stopping_criteria,
    )

    if temperature > 0:
        generation_config.update(
            {
                "do_sample": True,
                "top_p": top_p,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
            }
        )
    else:
        generation_config["do_sample"] = False

    thread = Thread(target=vl_gpt.language_model.generate, kwargs=generation_config)
    thread.start()

    yield from streamer
