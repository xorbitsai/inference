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

import functools
import uuid
from typing import List, Optional, Tuple


class InferenceRequest:
    def __init__(
        self,
        prompt_or_messages,
        future_or_queue,
        is_prefill,
        call_ability,
        *args,
        **kwargs,
    ):
        # original prompt, prompt(str) for generate model and messages(List[Dict]) for chat model
        self._prompt = prompt_or_messages
        # full prompt that contains chat history and applies chat template
        self._full_prompt = None
        # whether the current request is in the prefill phase
        self._is_prefill = is_prefill
        # the ability that the user calls this model for, that is `generate` / `chat` for now,
        # which is for results formatting
        self._call_ability = call_ability
        # full prompt tokens
        self._prompt_tokens = None
        # all new generated tokens during decode phase
        self._new_tokens = []
        # kv_cache used in decode phase
        self._kv_cache = None
        # use passed args from upstream interface
        self._inference_args = args
        # use passed kwargs from upstream interface, currently for getting raw generate config from upstream,
        # which is useful for some special models
        self._inference_kwargs = kwargs
        # should this request be stopped
        self._stopped = False
        # finish reason. If this is set, self._stopped is True.
        self._finish_reason = None
        # should this request be aborted
        # note that when this flag is True, assert self._stopped is True
        self._aborted = False
        # sanitized generate config
        self._sanitized_generate_config = None
        # Chunk id for results. In stream mode, all the chunk ids should be same.
        self._stream_chunk_id = str(uuid.uuid4())
        # For calculate attention mask if needed
        self.padding_len = 0
        # Use in stream mode
        self.last_output_length = 0
        # For tool call
        self.tools = None
        # Currently, for storing tool call streaming results.
        self.outputs: List[str] = []  # type: ignore
        # inference results,
        # it is a list type because when stream=True,
        # self.completion contains all the results in a decode round.
        self.completion = []
        # The way upstream gets the returned results,
        # when stream=True, it is an asyncio.Queue,
        # and when stream=False, it is an asyncio future.
        self.future_or_queue = future_or_queue
        # Record error message when this request has error.
        # Must set stopped=True when this field is set.
        self.error_msg: Optional[str] = None  # type: ignore
        # For compatibility. Record some extra parameters for some special cases.
        self.extra_kwargs = {}

        # check the integrity of args passed upstream
        self._check_args()

        # for reasoning_content using
        self.previous_texts = [""]

    def _check_args(self):
        assert len(self._inference_args) == 1
        # generate config
        assert self._inference_args[0] is None or isinstance(
            self._inference_args[0], dict
        )

    @property
    def prompt(self):
        """
        prompt for generate model and messages for chat model
        """
        return self._prompt

    @prompt.setter
    def prompt(self, value: str):
        self._prompt = value

    @property
    def call_ability(self):
        return self._call_ability

    @property
    def full_prompt(self):
        return self._full_prompt

    @full_prompt.setter
    def full_prompt(self, value: str):
        self._full_prompt = value

    @property
    def is_prefill(self):
        return self._is_prefill

    @is_prefill.setter
    def is_prefill(self, value: bool):
        self._is_prefill = value

    @property
    def prompt_tokens(self):
        return self._prompt_tokens

    @prompt_tokens.setter
    def prompt_tokens(self, value: List[int]):
        self._prompt_tokens = value

    @property
    def kv_cache(self):
        return self._kv_cache

    @kv_cache.setter
    def kv_cache(self, value):
        self._kv_cache = value

    @property
    def new_tokens(self):
        return self._new_tokens

    def append_new_token(self, token: int):
        self._new_tokens.append(token)

    @property
    def generate_config(self):
        return self._inference_args[0]

    @property
    def sanitized_generate_config(self):
        return self._sanitized_generate_config

    @sanitized_generate_config.setter
    def sanitized_generate_config(self, value: dict):
        self._sanitized_generate_config = value

    @property
    def inference_kwargs(self):
        return self._inference_kwargs

    @property
    def stopped(self):
        return self._stopped

    @stopped.setter
    def stopped(self, value: bool):
        self._stopped = value

    @property
    def finish_reason(self):
        return self._finish_reason

    @finish_reason.setter
    def finish_reason(self, value: Optional[str]):
        self._finish_reason = value

    @property
    def chunk_id(self):
        return self._stream_chunk_id

    @property
    def stream(self) -> bool:
        return (
            False
            if self.generate_config is None
            else self.generate_config.get("stream", False)
        )

    @property
    def stream_interval(self) -> int:
        return self.sanitized_generate_config.get("stream_interval", 2)

    @property
    def include_usage(self) -> bool:
        stream_options = self.sanitized_generate_config.get("stream_options", None)
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )
        return include_usage

    @property
    def aborted(self) -> bool:
        return self._aborted

    @aborted.setter
    def aborted(self, value: bool):
        self._aborted = value

    @property
    def request_id(self) -> Optional[str]:
        return (
            None
            if self.generate_config is None
            else self.generate_config.get("request_id", None)
        )

    @functools.lru_cache
    def get_generate_configs(
        self, eos_token_id: int, builtin_stop_token_ids: Optional[Tuple[int]] = None
    ):
        from ...types import max_tokens_field

        max_new_tokens = int(
            self.sanitized_generate_config.get("max_tokens", max_tokens_field.default)
            or 0
        )
        stream_interval = self.sanitized_generate_config.get("stream_interval", 2)
        include_usage = self.include_usage
        stop_str = self.sanitized_generate_config.get("stop", None)
        stop_token_ids = (
            self.sanitized_generate_config.get("stop_token_ids", None) or []
        )
        stop_token_ids = set(stop_token_ids)
        stop_token_ids.add(eos_token_id)
        stop_token_ids.update(builtin_stop_token_ids or [])
        temperature = float(self.sanitized_generate_config.get("temperature", 1.0))
        repetition_penalty = float(
            self.sanitized_generate_config.get("repetition_penalty", 1.0)
        )
        top_p = float(self.sanitized_generate_config.get("top_p", 1.0))
        top_k = int(self.sanitized_generate_config.get("top_k", -1))  # -1 means disable
        return (
            max_new_tokens,
            stream_interval,
            include_usage,
            stop_str,
            stop_token_ids,
            temperature,
            repetition_penalty,
            top_p,
            top_k,
        )
