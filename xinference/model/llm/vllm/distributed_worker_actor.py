# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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

import logging
import os
from typing import Callable, Union

import xoscar as xo

logger = logging.getLogger(__name__)

DEBUG_EXECUTOR = bool(int(os.getenv("XINFERENCE_DEBUG_VLLM_EXECUTOR", "0")))


# Lives apart from distributed_executor_v1 because the worker main pool
# deserializes this class when a sub-pool registers it, and that process may
# have no vllm installed (per-model virtualenv). Nothing here may import vllm
# at module level.
class WorkerActor(xo.StatelessActor):
    def __init__(self, rpc_rank: int = 0, **kwargs):
        super().__init__(**kwargs)
        from vllm.v1.worker.worker_base import WorkerWrapperBase

        self._worker = WorkerWrapperBase(rpc_rank=rpc_rank)

    async def __post_create__(self):
        try:
            # Change process title for model
            import setproctitle

            _uid = os.environ.get("XINFERENCE_MODEL_UID", "")
            setproctitle.setproctitle(
                f"Xinf vLLM worker: {self._worker.rpc_rank} [{_uid}]"
            )
        except ImportError:
            pass

    def __getattr__(self, item):
        from xoscar.core import NO_LOCK_ATTRIBUTE_HINT

        if item == NO_LOCK_ATTRIBUTE_HINT:
            return True
        return getattr(self._worker, item)

    @classmethod
    def gen_uid(cls, rank):
        return f"VllmWorker_{rank}"

    def execute_method(self, method: Union[str, Callable], *args, **kwargs):
        if DEBUG_EXECUTOR:
            # NOTE: too many logs, but useful for debug
            logger.debug(
                "Calling method %s in vllm worker %s, args: %s, kwargs: %s",
                method,
                self.uid,
                args,
                kwargs,
            )
        if isinstance(method, str):
            if method != "sample_tokens":
                return getattr(self._worker, method)(*args, **kwargs)
            else:
                result = getattr(self._worker, method)(*args, **kwargs)
                return self._sanitize_result(result)
        else:
            return method(self._worker, *args, **kwargs)

    def _sanitize_result(self, obj):
        if obj is None:
            return obj
        output = obj.get_output()
        return output
