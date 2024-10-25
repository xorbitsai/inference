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
import functools
import logging
import uuid
from collections import deque
from typing import Dict, List, Optional, Set, Tuple, Union

import xoscar as xo

from .utils import AbortRequestMessage

logger = logging.getLogger(__name__)

XINFERENCE_STREAMING_DONE_FLAG = "<XINFERENCE_STREAMING_DONE>"
XINFERENCE_STREAMING_ERROR_FLAG = "<XINFERENCE_STREAMING_ERROR>"
XINFERENCE_STREAMING_ABORT_FLAG = "<XINFERENCE_STREAMING_ABORT>"
XINFERENCE_NON_STREAMING_ABORT_FLAG = "<XINFERENCE_NON_STREAMING_ABORT>"


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
        from ..types import max_tokens_field

        max_new_tokens = int(
            self.sanitized_generate_config.get("max_tokens", max_tokens_field.default)
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


def _get_valid_batch_kv_cache(data, skipped_indexes: Set[int]):
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache.from_legacy_cache(data)
    batch_size = cache.key_cache[0].shape[0]
    batch_slices = [num for num in range(batch_size) if num not in skipped_indexes]
    for idx in range(len(cache)):
        cache.key_cache[idx] = cache.key_cache[idx][batch_slices, ::]
        cache.value_cache[idx] = cache.value_cache[idx][batch_slices, ::]
    return cache.to_legacy_cache()


class SchedulerActor(xo.StatelessActor):
    @classmethod
    def gen_uid(cls, model_uid: str, replica_id: str):
        return f"{model_uid}-{replica_id}-scheduler-actor"

    def __init__(self):
        super().__init__()
        self._waiting_queue: deque[InferenceRequest] = deque()  # type: ignore
        self._running_queue: deque[InferenceRequest] = deque()  # type: ignore
        self._model = None
        self._id_to_req = {}
        self._abort_req_ids: Set[str] = set()  # type: ignore
        self._isolation = None

    async def __post_create__(self):
        from ..isolation import Isolation

        self._isolation = Isolation(
            asyncio.new_event_loop(), threaded=True, daemon=True
        )
        self._isolation.start()
        asyncio.run_coroutine_threadsafe(self.run(), loop=self._isolation.loop)

    async def __pre_destroy__(self):
        try:
            assert self._isolation is not None
            self._isolation.stop()
            del self._isolation
        except Exception as e:
            logger.debug(
                f"Destroy scheduler actor failed, address: {self.address}, error: {e}"
            )

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

    @staticmethod
    def _empty_cache():
        from ..model.llm.transformers.utils import empty_cache

        empty_cache()

    async def step(self):
        req_list = self._handle_request()
        if not req_list:
            return
        self._model.batch_inference(req_list)

        stopped_batch_indexes = set()
        for idx, r in enumerate(req_list):
            if r.stream:
                for completion in r.completion:
                    await r.future_or_queue.put(completion)
                r.completion = []

            if not r.stopped:
                self._running_queue.append(r)
            else:
                if r.new_tokens:
                    stopped_batch_indexes.add(idx)
                # set kv_cache to None for collection
                r.kv_cache = None
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
                        r.future_or_queue.set_result(
                            XINFERENCE_NON_STREAMING_ABORT_FLAG
                        )
                else:
                    if r.error_msg is None:  # normal stop
                        if not r.stream:
                            r.future_or_queue.set_result(r.completion[0])
                        else:
                            await r.future_or_queue.put(XINFERENCE_STREAMING_DONE_FLAG)
                    # Abnormal stop, currently indicates that the parameter check does not pass,
                    # and does not participate in the inference
                    else:
                        if not r.stream:
                            r.future_or_queue.set_exception(ValueError(r.error_msg))
                        else:
                            await r.future_or_queue.put(
                                XINFERENCE_STREAMING_ERROR_FLAG + r.error_msg
                            )

        # Some requests have been completed. Batch size needs to be reduced for kv cache.
        if stopped_batch_indexes and len(self._running_queue) > 0:
            kv_cache = self._running_queue[0].kv_cache
            reduced_kv_cache = _get_valid_batch_kv_cache(
                kv_cache, stopped_batch_indexes
            )
            for r in self._running_queue:
                r.kv_cache = reduced_kv_cache

        self._empty_cache()

    async def add_request(
        self,
        prompt_or_messages: Union[str, List[Dict]],
        future_or_queue,
        call_ability,
        *args,
        **kwargs,
    ):
        req = InferenceRequest(
            prompt_or_messages, future_or_queue, True, call_ability, *args, **kwargs
        )
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
        try:
            while True:
                # wait 10ms
                await asyncio.sleep(0.01)
                await self.step()
        except Exception as e:
            logger.exception(
                f"Scheduler actor uid: {self.uid}, address: {self.address} run with error: {e}"
            )
