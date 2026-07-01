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

from ..core import FlexibleModel, FlexibleModelSpec


class ModelScopePipelineModel(FlexibleModel):
    def load(self):
        # we have to move import here,
        # modelscope cannot be compatible with datasets>3.2.0
        # if put outside, it will just raise error
        # when enabled virtualenv,
        # we can make sure mdoelscope works well
        from modelscope.pipelines import pipeline

        config = dict(self.config or {})
        if self._device:
            config["device"] = self._device
        self._pipeline = pipeline(model=self._model_path, **config)

    def infer(self, *args, **kwargs):
        return self._pipeline(*args, **kwargs)


def launcher(model_uid: str, model_spec: FlexibleModelSpec, **kwargs) -> FlexibleModel:
    device = kwargs.get("device")
    if not kwargs.get("task"):
        raise ValueError("modelscope launcher requires `task`")

    model_path = model_spec.model_uri
    if model_path is None:
        raise ValueError("model_path required")

    return ModelScopePipelineModel(
        model_uid=model_uid,
        model_path=model_path,
        model_family=model_spec,
        device=device,
        config=kwargs,
    )
