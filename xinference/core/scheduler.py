# Copyright 2022-2024 XProbe Inc.
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

import asyncio
from collections import deque
from typing import List, Optional

import xoscar as xo

XINFERENCE_STREAMING_DONE_FLAG = "<XINFERENCE_STREAMING_DONE>"
XINFERENCE_STREAMING_ERROR_FLAG = "<XINFERENCE_STREAMING_ERROR>"


class InferenceRequest:
    def __init__(self, prompt, future_or_queue, is_prefill, *args, **kwargs):
        # original prompt
        self._prompt = prompt
        # full prompt that contains chat history and applies chat template
        self._full_prompt = None
        # whether the current request is in the prefill phase
        self._is_prefill = is_prefill
        # full prompt tokens
        self._prompt_tokens = None
        # all new generated tokens during decode phase
        self._new_tokens = []
        # kv_cache used in decode phase
        self._kv_cache = None
        # use passed args from `chat` interface
        self._inference_args = args
        # use passed kwargs from `chat` interface, basically not used for now
        self._inference_kwargs = kwargs
        # should this request be stopped
        self._stopped = False
        # sanitized generate config
        self._sanitized_generate_config = None
        # inference results,
        # it is a list type because when stream=True, for the first time you need to return two chunks
        self.completion = []
        # The way upstream gets the returned results,
        # when stream=True, it is an asyncio.Queue,
        # and when stream=False, it is an asyncio future.
        self.future_or_queue = future_or_queue
        # Record error message when this request has error.
        # Must set stopped=True when this field is set.
        self.error_msg: Optional[str] = None

        # check the integrity of args passed upstream
        self._check_args()

    def _check_args(self):
        assert len(self._inference_args) == 3
        # system prompt
        assert self._inference_args[0] is None or isinstance(
            self._inference_args[0], str
        )
        # chat history
        assert self._inference_args[1] is None or isinstance(
            self._inference_args[1], list
        )
        # generate config
        assert self._inference_args[2] is None or isinstance(
            self._inference_args[2], dict
        )

    @property
    def prompt(self):
        return self._prompt

    @property
    def system_prompt(self):
        return self._inference_args[0]

    @property
    def chat_history(self):
        return self._inference_args[1]

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
        return self._inference_args[2]

    @property
    def sanitized_generate_config(self):
        return self._sanitized_generate_config

    @sanitized_generate_config.setter
    def sanitized_generate_config(self, value: dict):
        self._sanitized_generate_config = value

    @property
    def stopped(self):
        return self._stopped

    @stopped.setter
    def stopped(self, value: bool):
        self._stopped = value

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


class SchedulerActor(xo.StatelessActor):
    @classmethod
    def gen_uid(cls, model_uid: str, replica_id: str):
        return f"{model_uid}-{replica_id}-scheduler-actor"

    def __init__(self):
        super().__init__()
        self._waiting_queue: deque[InferenceRequest] = deque()
        self._running_queue: deque[InferenceRequest] = deque()
        self._model = None

    def set_model(self, model):
        self._model = model

    def get_max_num_seqs(self):
        assert self._model is not None
        return self._model.get_max_num_seqs()

    def _handle_request(self) -> Optional[List[InferenceRequest]]:
        if self._model is None:
            return None
        max_num_seqs = self.get_max_num_seqs()
        # currently, FCFS strategy
        running_list: List[InferenceRequest] = []
        while len(self._running_queue) > 0:
            if len(running_list) == max_num_seqs:
                break
            running_list.append(self._running_queue.popleft())

        waiting_list: List[InferenceRequest] = []
        if len(running_list) < max_num_seqs:
            while len(self._waiting_queue) > 0:
                waiting_list.append(self._waiting_queue.popleft())
                if len(running_list) + len(waiting_list) == max_num_seqs:
                    break
        # must waiting_list in front
        return waiting_list + running_list

    async def step(self):
        req_list = self._handle_request()
        if not req_list:
            return
        batch_size = len(req_list)
        self._model.batch_inference(req_list)

        stop_before_prefill_cnt = 0
        for r in req_list:
            if r.stream:
                for completion in r.completion:
                    await r.future_or_queue.put(completion)

            if not r.stopped:
                self._running_queue.append(r)
            else:
                if r.error_msg is None:  # normal stop
                    if not r.stream:
                        r.future_or_queue.set_result(r.completion[0])
                    else:
                        await r.future_or_queue.put(XINFERENCE_STREAMING_DONE_FLAG)
                # Abnormal stop, currently indicates that the parameter check does not pass,
                # and does not participate in the inference
                else:
                    stop_before_prefill_cnt += 1
                    if not r.stream:
                        r.future_or_queue.set_exception(ValueError(r.error_msg))
                    else:
                        await r.future_or_queue.put(
                            XINFERENCE_STREAMING_ERROR_FLAG + r.error_msg
                        )

        if len(self._running_queue) > 0:
            batch_size_after_one_step = len(self._running_queue)
            batch_size_before_one_step = batch_size - stop_before_prefill_cnt
            if batch_size_before_one_step != batch_size_after_one_step:
                assert batch_size_after_one_step < batch_size_before_one_step
                for r in self._running_queue:
                    r.kv_cache = None

    async def add_request(self, prompt: str, future_or_queue, *args, **kwargs):
        req = InferenceRequest(prompt, future_or_queue, True, *args, **kwargs)
        self._waiting_queue.append(req)

    async def run(self):
        while True:
            await self.step()
            # wait 10ms
            await asyncio.sleep(0.01)
