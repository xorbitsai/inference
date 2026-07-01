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
import dataclasses
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import xoscar as xo

TO_REMOVE_PROGRESS_INTERVAL = float(
    os.getenv("XINFERENCE_REMOVE_PROGRESS_INTERVAL", 5 * 60)
)  # 5min
CHECK_PROGRESS_INTERVAL = float(
    os.getenv("XINFERENCE_CHECK_PROGRESS_INTERVAL", 1 * 60)
)  # 1min
UPLOAD_PROGRESS_SPAN = float(
    os.getenv("XINFERENCE_UPLOAD_PROGRESS_SPAN", 0.05)
)  # not upload when change less than 0.1

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _ProgressInfo:
    progress: float
    last_updated: float
    info: Optional[str] = None


class ProgressTrackerActor(xo.StatelessActor):
    _request_id_to_progress: Dict[str, _ProgressInfo]

    @classmethod
    def default_uid(cls) -> str:
        return "progress_tracker"

    def __init__(
        self,
        to_remove_interval: float = TO_REMOVE_PROGRESS_INTERVAL,
        check_interval: float = CHECK_PROGRESS_INTERVAL,
    ):
        super().__init__()

        self._request_id_to_progress = {}
        self._clear_finished_task = None
        self._to_remove_interval = to_remove_interval
        self._check_interval = check_interval

    async def __post_create__(self):
        self._clear_finished_task = asyncio.create_task(self._clear_finished())

    async def __pre_destroy__(self):
        if self._clear_finished_task:
            self._clear_finished_task.cancel()

    async def _clear_finished(self):
        to_remove_request_ids = []
        while True:
            now = time.time()
            for request_id, progress in self._request_id_to_progress.items():
                if abs(progress.progress - 1.0) > 1e-5:
                    continue

                # finished
                if now - progress.last_updated > self._to_remove_interval:
                    to_remove_request_ids.append(request_id)

            for rid in to_remove_request_ids:
                del self._request_id_to_progress[rid]

            if to_remove_request_ids:
                logger.debug(
                    "Remove requests %s due to it's finished for over %s seconds",
                    to_remove_request_ids,
                    self._to_remove_interval,
                )

            await asyncio.sleep(self._check_interval)

    def start(self, request_id: str, info: Optional[str] = None):
        self._request_id_to_progress[request_id] = _ProgressInfo(
            progress=0.0, last_updated=time.time(), info=info
        )

    def set_progress(
        self, request_id: str, progress: float, info: Optional[str] = None
    ):
        assert progress <= 1.0
        info_ = self._request_id_to_progress[request_id]
        info_.progress = progress
        info_.last_updated = time.time()
        if info:
            info_.info = info
        logger.debug(
            "Setting progress, request id: %s, progress: %s", request_id, progress
        )

    def get_progress(self, request_id: str) -> float:
        return self._request_id_to_progress[request_id].progress

    def get_progress_info(self, request_id: str) -> Tuple[float, Optional[str]]:
        info = self._request_id_to_progress[request_id]
        return info.progress, info.info


class Progressor:
    _sub_progress_stack: List[Tuple[float, float]]

    def __init__(
        self,
        request_id: str,
        progress_tracker_ref: xo.ActorRefType["ProgressTrackerActor"],
        loop: asyncio.AbstractEventLoop,
        upload_span: float = UPLOAD_PROGRESS_SPAN,
    ):
        self.request_id = request_id
        self.progress_tracker_ref = progress_tracker_ref
        self.loop = loop
        # uploading when progress changes over this span
        # to prevent from frequently uploading
        self._upload_span = upload_span

        self._last_report_progress = 0.0
        self._current_progress = 0.0
        self._sub_progress_stack = [(0.0, 1.0)]
        self._current_sub_progress_start = 0.0
        self._current_sub_progress_end = 1.0

    async def start(self):
        if self.request_id:
            await self.progress_tracker_ref.start(self.request_id)

    def split_stages(self, n_stage: int, stage_weight: Optional[List[float]] = None):
        if self.request_id:
            if stage_weight is not None:
                if len(stage_weight) != n_stage + 1:
                    raise ValueError(
                        f"stage_weight should have size {n_stage + 1}, got {len(stage_weight)}"
                    )
                progresses = stage_weight
            else:
                progresses = np.linspace(
                    self._current_sub_progress_start,
                    self._current_sub_progress_end,
                    n_stage + 1,
                )
            spans = [(progresses[i], progresses[i + 1]) for i in range(n_stage)]
            self._sub_progress_stack.extend(spans[::-1])

    def __enter__(self):
        if self.request_id:
            (
                self._current_sub_progress_start,
                self._current_sub_progress_end,
            ) = self._sub_progress_stack[-1]

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.request_id:
            self._sub_progress_stack.pop()
            # force to set progress to 1.0 for this sub progress
            # nevertheless it is done or not
            self.set_progress(1.0)
        return False

    def set_progress(self, progress: float, info: Optional[str] = None):
        if self.request_id:
            self._current_progress = (
                self._current_sub_progress_start
                + (self._current_sub_progress_end - self._current_sub_progress_start)
                * progress
            )
            if (
                self._current_progress - self._last_report_progress >= self._upload_span
                or 1.0 - progress < 1e-5
            ) or info:
                set_progress = self.progress_tracker_ref.set_progress(
                    self.request_id, self._current_progress
                )
                asyncio.run_coroutine_threadsafe(set_progress, self.loop)  # type: ignore
                self._last_report_progress = self._current_progress
