# Copyright 2022-2025 XProbe Inc.
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
from abc import abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from .....types import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionChunk,
    PytorchGenerateConfig,
)
from ....utils import cache_clean
from ...utils import generate_chat_completion, generate_completion_chunk
from ..core import PytorchChatModel


class PytorchMultiModalModel(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = None
        self._device = None
        self._processor = None
        self._model = None

    @abstractmethod
    def decide_device(self):
        """
        Update self._device
        """
        pass

    @abstractmethod
    def load_processor(self):
        """
        Load self._processor and self._tokenizer
        """
        pass

    @abstractmethod
    def load_multimodal_model(self):
        """
        Load self._model
        """
        pass

    def load(self):
        self.decide_device()
        self.load_processor()
        self.load_multimodal_model()

    @abstractmethod
    def build_inputs_from_messages(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ):
        """
        Convert from input OpenAI-formatted messages to
        actual parameters needed for inference,
        e.g. input_ids, attention_masks, etc.
        """
        pass

    @abstractmethod
    def build_generate_kwargs(
        self,
        generate_config: Dict,
    ) -> Dict[str, Any]:
        """
        Hyperparameters needed for generation,
        e.g. temperature, max_new_tokens, etc.
        """
        pass

    @abstractmethod
    def build_streaming_iter(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ) -> Tuple[Iterator, int]:
        """
        Return the iterator needed for streaming inference and the length of prompt token for statisticians.
        The length of prompt token usually comes from the input_ids.
        In this interface you need to call the `build_inputs_from_messages` and `build_generate_kwargs`.
        """
        pass

    def get_stop_strs(self) -> List[str]:
        return []

    def check_conditions(self, new_text: str) -> Tuple[str, bool]:
        stop_strs = self.get_stop_strs()
        for ss in stop_strs:
            if new_text.endswith(ss):
                new_text = new_text[: -len(ss)]
                break
        return new_text, False

    def generate_non_streaming(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> ChatCompletion:
        generate_config = generate_config if generate_config else {}  # type: ignore
        streamer, prompt_tokens = self.build_streaming_iter(messages, generate_config)  # type: ignore
        completion_tokens, total_tokens = 0, 0
        res = ""
        for i, new_text in enumerate(streamer):
            new_text, should_stop = self.check_conditions(new_text)
            if should_stop:
                break
            completion_tokens = i
            total_tokens = prompt_tokens + completion_tokens
            res += new_text
        return generate_chat_completion(
            self.model_uid,
            res,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens if prompt_tokens != -1 else -1,
            total_tokens=total_tokens if prompt_tokens != -1 else -1,
        )

    def generate_streaming(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Iterator[CompletionChunk]:
        generate_config = generate_config if generate_config else {}  # type: ignore
        streamer, prompt_tokens = self.build_streaming_iter(messages, generate_config)  # type: ignore
        stream_options = generate_config.pop("stream_options", None)
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )

        completion_id = str(uuid.uuid1())
        completion_tokens, total_tokens = 0, 0
        for i, new_text in enumerate(streamer):
            new_text, should_stop = self.check_conditions(new_text)
            if should_stop:
                break
            completion_tokens = i
            total_tokens = prompt_tokens + completion_tokens
            yield generate_completion_chunk(
                chunk_text=new_text,
                finish_reason=None,
                chunk_id=completion_id,
                model_uid=self.model_uid,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens if prompt_tokens != -1 else -1,
                total_tokens=total_tokens if prompt_tokens != -1 else -1,
                has_choice=True,
                has_content=True,
            )
        yield generate_completion_chunk(
            chunk_text=None,
            finish_reason="stop",
            chunk_id=completion_id,
            model_uid=self.model_uid,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens if prompt_tokens != -1 else -1,
            total_tokens=total_tokens if prompt_tokens != -1 else -1,
            has_choice=True,
            has_content=False,
        )
        if include_usage:
            yield generate_completion_chunk(
                chunk_text=None,
                finish_reason=None,
                chunk_id=completion_id,
                model_uid=self.model_uid,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens if prompt_tokens != -1 else -1,
                total_tokens=total_tokens if prompt_tokens != -1 else -1,
                has_choice=False,
                has_content=False,
            )

    @cache_clean
    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        stream = generate_config.get("stream", False) if generate_config else False
        return (
            self._to_chat_completion_chunks(
                self.generate_streaming(messages, generate_config)
            )
            if stream
            else self.generate_non_streaming(messages, generate_config)
        )
