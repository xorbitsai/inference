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
from enum import Enum
from logging import getLogger
from typing import Dict, List, Optional

import xoscar as xo

from .._compat import BaseModel

logger = getLogger(__name__)


class LaunchStatus(Enum):
    CREATING = 1
    UPDATING = 2
    TERMINATING = 3
    TERMINATED = 4
    READY = 5
    ERROR = 6


class InstanceInfo(BaseModel):
    model_name: str
    model_uid: str
    model_version: Optional[str]
    model_ability: List[str]
    replica: int
    status: str
    instance_created_ts: int

    def update(self, **kwargs):
        for field, value in kwargs.items():
            setattr(self, field, value)


class StatusGuardActor(xo.StatelessActor):
    def __init__(self):
        super().__init__()
        self._model_uid_to_info: Dict[str, InstanceInfo] = {}  # type: ignore

    @classmethod
    def default_uid(cls) -> str:
        return "status_guard"

    @staticmethod
    def _drop_terminated_info(instance_infos: List[InstanceInfo]) -> List[InstanceInfo]:
        return [
            info
            for info in instance_infos
            if info.status != LaunchStatus.TERMINATED.name
        ]

    def set_instance_info(self, model_uid: str, info: InstanceInfo):
        self._model_uid_to_info[model_uid] = info

    def get_instance_info(
        self, model_name: Optional[str] = None, model_uid: Optional[str] = None
    ) -> List[InstanceInfo]:
        if model_uid is not None:
            return (
                self._drop_terminated_info([self._model_uid_to_info[model_uid]])
                if model_uid in self._model_uid_to_info
                else []
            )
        all_infos: List[InstanceInfo] = list(self._model_uid_to_info.values())
        filtered_infos: List[InstanceInfo] = list(
            filter(lambda info: info.model_name == model_name, all_infos)
        )
        return (
            self._drop_terminated_info(filtered_infos)
            if model_name is not None
            else self._drop_terminated_info(all_infos)
        )

    def get_instance_count(self, model_name: str) -> int:
        return len(self.get_instance_info(model_name=model_name))

    def update_instance_info(self, model_uid: str, info: Dict):
        self._model_uid_to_info[model_uid].update(**info)
