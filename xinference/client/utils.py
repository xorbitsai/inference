from typing import List, Optional, Union, no_type_check

from ..types import ChatCompletionMessage


def convert_float_to_int_or_str(model_size: float) -> Union[int, str]:
    """convert float to int or string

    if float can be presented as int, convert it to int, otherwise convert it to string
    """
    if int(model_size) == model_size:
        return int(model_size)
    else:
        return str(model_size)


@no_type_check
def handle_system_prompts(
    chat_history: List["ChatCompletionMessage"], system_prompt: Optional[str]
) -> List["ChatCompletionMessage"]:
    history_system_prompts = [
        ch["content"] for ch in chat_history if ch["role"] == "system"
    ]
    if system_prompt is not None:
        history_system_prompts.append(system_prompt)

    # remove all the system prompt in the chat_history
    chat_history = list(filter(lambda x: x["role"] != "system", chat_history))
    # insert all system prompts at the beginning
    chat_history.insert(
        0, {"role": "system", "content": ". ".join(history_system_prompts)}
    )
    return chat_history
