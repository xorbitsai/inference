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

from typing import Dict

import xoscar as xo

from ..model.llm.core import Model


class ModelManagerActor(xo.Actor):
    models: Dict[str, xo.ActorRef] = dict()

    def add_model(self, model_uid: str, ref: xo.ActorRef):
        self.models[model_uid] = ref

    def get_model(self, model_uid: str):
        return self.models[model_uid]


class ModelActor(xo.Actor):
    @classmethod
    def gen_uid(cls, model: Model):
        return f"{model.__class__}-model-actor"

    def __init__(self, model: Model):
        super().__init__()
        self._model = model

    async def __post_create__(self):
        self._model.load()

    def __getattr__(self, item):
        return getattr(self._model, item)
