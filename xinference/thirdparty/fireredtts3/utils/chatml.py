from typing import List, Dict


CHATML_LATENT_IN_PAD_SYM = '<|image_pad|>'
CHATML_LATENT_IN_PAD_ID = 151655
CHATML_LATENT_OUT_PAD_SYM = '<|video_pad|>'
CHATML_LATENT_OUT_PAD_ID = 151656


def convert_to_chatml(
    # User input
    text_in: str,   # Required
    latent_in_len: int = 0,
    # Assistant output
    text_out: str = "",
    latent_out_len: int = 0,
    # Placeholder(will be replaced with audio latents)
    latent_in_pad: str = CHATML_LATENT_IN_PAD_SYM,
    latent_out_pad:str = CHATML_LATENT_OUT_PAD_SYM,
):
    # System prompt
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    # Input 
    input_message = {
        "role": "user", 
        "content": [
            {
                "type": "text", 
                "text": text_in + ' /no_think', # NOTE should add /no_think for Qwen3(not Qwen3.5)
            }
        ]
    }
    if latent_in_len>0:
        input_message['content'].insert(0, 
            {
                "type": "audio", 
                "audio": latent_in_pad * latent_in_len,
            }
        )
    messages.append(input_message)
    # Output 
    output_message = {
        "role": "assistant", 
        "content": [
            {
                "type": "text", 
                "text": f"<think>\n\n</think>\n\n" + text_out,  # text_out should be wrapped in <|sot|><|eot|>
            }
        ]
    }
    if latent_out_len>0:
        output_message['content'].append(
            {
                "type": "audio", 
                "audio": latent_out_pad * latent_out_len,
            }
        )
    messages.append(output_message)
    # Convert to chatml string
    chatml_str_list: List[str] = []
    for msg in messages:
        if isinstance(msg['content'], str):
            chatml_str_list.append(
                f'<|im_start|>{msg["role"]}\n{msg["content"]}<|im_end|>\n'
            )
        else:
            chatml_str_list.append(f'<|im_start|>{msg["role"]}\n')
            for content in msg["content"]:
                if content["type"] == "text":
                    chatml_str_list.append(content["text"])
                elif content["type"] == "audio":
                    chatml_str_list.append(
                        f"<|sosp|>{content['audio']}<|eosp|>\n"
                    )
            chatml_str_list.append(f"<|im_end|>\n")
    chatml_str = ''.join(chatml_str_list)
    return chatml_str


def compose_generate_input_tts(
    prompt_latent_len: int,
    prompt_text: str,
    text: str,
):
    text_in = "Convert text to speech.\n{}".format(prompt_text+text)
    chatml_str = convert_to_chatml(text_in=text_in, latent_out_len=prompt_latent_len)
    chatml_str = chatml_str.removesuffix('<|eosp|>\n<|im_end|>\n')  # Remove ending tags
    return chatml_str


def compose_generate_input_voice_design(
    instruction: str,
    text: str,
):
    text_in = "{}\n\n根据上述音色描述，首先整理成语音属性，再合成以下文本对应的音频：\n{}".format(instruction, text)
    chatml_str = convert_to_chatml(text_in=text_in, text_out="<|sot|>")
    chatml_str = chatml_str.removesuffix('<|im_end|>\n')  # Remove ending tags
    return chatml_str


def compose_generate_input_semantic_edit(
    instruction: str,
    audio_in_latent_len: int,
):
    text_in = 'Identify the content of the audio. {}'.format(instruction.strip())
    chatml_str = convert_to_chatml(text_in=text_in, latent_in_len=audio_in_latent_len, text_out="<|sot|>")
    chatml_str = chatml_str.removesuffix('<|im_end|>\n')  # Remove ending tags
    return chatml_str


def compose_generate_input_acoustic_edit(
    instruction: str,
    audio_in_latent_len: int,
):
    chatml_str = convert_to_chatml(text_in=instruction, latent_in_len=audio_in_latent_len, latent_out_len=1)
    chatml_str = chatml_str.removesuffix('<|video_pad|><|eosp|>\n<|im_end|>\n')  # Remove ending tags
    return chatml_str

