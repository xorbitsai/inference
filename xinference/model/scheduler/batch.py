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

import asyncio
import logging
from collections import deque
from typing import Dict, List, Optional, Set, Union

from .core import (
    XINFERENCE_NON_STREAMING_ABORT_FLAG,
    XINFERENCE_STREAMING_ABORT_FLAG,
    XINFERENCE_STREAMING_DONE_FLAG,
    XINFERENCE_STREAMING_ERROR_FLAG,
    AbortRequestMessage,
)
from .request import InferenceRequest

logger = logging.getLogger(__name__)


class BatchScheduler:
    def __init__(self, model):
        self._waiting_queue: deque[InferenceRequest] = deque()  # type: ignore
        self._running_queue: deque[InferenceRequest] = deque()  # type: ignore
        self._model = model
        self._id_to_req = {}
        self._abort_req_ids: Set[str] = set()  # type: ignore
        self._running = False
        self._task = None

    async def start(self):
        """Start the scheduler background task"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        """Stop the scheduler background task"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

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
        # Function-level import to avoid circular dependency
        from ..llm.transformers.utils import empty_cache

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
            reduced_kv_cache = self._model.build_reduced_kv_cache(
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

    async def _run(self):
        try:
            while self._running:
                # wait 10ms
                await asyncio.sleep(0.01)
                await self.step()
        except asyncio.CancelledError:
            logger.debug(f"Scheduler stopped")
        except Exception as e:
            logger.exception(f"Scheduler run with error: {e}")
