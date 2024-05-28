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
import logging
from collections import deque
from enum import Enum
from typing import List, Optional, Set

import xoscar as xo

logger = logging.getLogger(__name__)

XINFERENCE_STREAMING_DONE_FLAG = "<XINFERENCE_STREAMING_DONE>"
XINFERENCE_STREAMING_ERROR_FLAG = "<XINFERENCE_STREAMING_ERROR>"
XINFERENCE_STREAMING_ABORT_FLAG = "<XINFERENCE_STREAMING_ABORT>"


class AbortRequestMessage(Enum):
    NOT_FOUND = 1
    DONE = 2
    NO_OP = 3


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
        # should this request be aborted
        # note that when this flag is True, assert self._stopped is True
        self._aborted = False
        # sanitized generate config
        self._sanitized_generate_config = None
        # Use in stream mode
        self.last_output_length = 0
        # inference results,
        # it is a list type because when stream=True, for the first time you need to return two chunks,
        # or when include_usage is True, the last results should contain usage information
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


class SchedulerActor(xo.StatelessActor):
    @classmethod
    def gen_uid(cls, model_uid: str, replica_id: str):
        return f"{model_uid}-{replica_id}-scheduler-actor"

    def __init__(self):
        super().__init__()
        self._waiting_queue: deque[InferenceRequest] = deque()
        self._running_queue: deque[InferenceRequest] = deque()
        self._model = None
        self._id_to_req = {}
        self._abort_req_ids: Set[str] = set()

    def set_model(self, model):
        self._model = model

    def get_max_num_seqs(self):
        assert self._model is not None
        return self._model.get_max_num_seqs()

    def _check_request_aborted(self, req: InferenceRequest):
        if req.request_id and req.request_id in self._abort_req_ids:
            req.aborted = True
            req.stopped = True

    def _handle_request(self) -> Optional[List[InferenceRequest]]:
        if self._model is None:
            return None
        max_num_seqs = self.get_max_num_seqs()
        # currently, FCFS strategy
        running_list: List[InferenceRequest] = []
        while len(self._running_queue) > 0:
            if len(running_list) == max_num_seqs:
                break
            req = self._running_queue.popleft()
            self._check_request_aborted(req)
            running_list.append(req)

        waiting_list: List[InferenceRequest] = []
        if len(running_list) < max_num_seqs:
            while len(self._waiting_queue) > 0:
                req = self._waiting_queue.popleft()
                self._check_request_aborted(req)
                waiting_list.append(req)
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
                rid = r.request_id
                # clear data structure
                if rid is not None:
                    self._id_to_req.pop(rid, None)
                    self._abort_req_ids.discard(rid)

                if r.aborted:  # stop due to abort
                    # handle abort result
                    if r.stream:
                        await r.future_or_queue.put(XINFERENCE_STREAMING_ABORT_FLAG)
                    else:
                        r.future_or_queue.cancel()
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
        rid = req.request_id
        if rid is not None:
            if rid in self._id_to_req:
                raise KeyError(f"Request id: {rid} has already existed!")
            self._id_to_req[rid] = req
        self._waiting_queue.append(req)

    async def abort_request(self, req_id: str) -> str:
        """
        Abort a request.
        Abort a submitted request. If the request is finished or not found, this method will be a no-op.
        """
        if req_id not in self._id_to_req:
            logger.info(f"Request id: {req_id} not found. No-op for xinference.")
            return AbortRequestMessage.NOT_FOUND.name
        else:
            self._abort_req_ids.add(req_id)
            logger.info(f"Request id: {req_id} found to be aborted.")
            return AbortRequestMessage.DONE.name

    async def run(self):
        while True:
            await self.step()
            # wait 10ms
            await asyncio.sleep(0.01)
