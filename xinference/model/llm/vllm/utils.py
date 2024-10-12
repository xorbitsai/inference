# Copyright 2022-2023 XProbe Inc.
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
import logging
import os

logger = logging.getLogger(__name__)


def vllm_check(fn):
    try:
        from vllm.engine.async_llm_engine import AsyncEngineDeadError
    except:
        return fn

    @functools.wraps(fn)
    async def _async_wrapper(self, *args, **kwargs):
        try:
            return await fn(self, *args, **kwargs)
        except AsyncEngineDeadError:
            logger.info("Detecting vLLM is not health, prepare to quit the process")
            try:
                self.stop()
            except:
                # ignore error when stop
                pass
            # Just kill the process and let xinference auto-recover the model
            os._exit(1)

    return _async_wrapper
