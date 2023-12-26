from typing import Dict, Iterator, List, Optional, Union

from ...types import ChatCompletion, ChatCompletionChunk
from .core import LVLM, LVLMFamilyV1, LVLMSpecV1


class QwenVLChat(LVLM):
    @classmethod
    def match(
        cls, model_family: "LVLMFamilyV1", model_spec: "LVLMSpecV1", quantization: str
    ) -> bool:
        if "qwen" in model_family.model_name:
            return True
        return False

    def load(self):
        raise NotImplementedError

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[Dict]] = None,
        generate_config: Optional[Dict] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        raise NotImplementedError
