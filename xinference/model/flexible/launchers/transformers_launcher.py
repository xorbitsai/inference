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

from ..core import FlexibleModel, FlexibleModelSpec


class MockModel(FlexibleModel):
    def infer(self, *args, **kwargs):
        return kwargs


class AutoModel(FlexibleModel):
    def load(self) -> None:
        from transformers import pipeline

        config = dict(self.config or {})
        # The worker injects this runtime option; it is not a pipeline argument.
        config.pop("enable_virtual_env", None)
        config.setdefault("device", self.device)
        self._pipeline = pipeline(model=self.model_path, **config)

    def infer(self, *args, **kwargs):
        return self._pipeline(*args, **kwargs)


class TransformersTextClassificationModel(AutoModel):
    """Run sequence classification with the Transformers pipeline interface."""


def launcher(model_uid: str, model_spec: FlexibleModelSpec, **kwargs) -> FlexibleModel:
    task = kwargs.get("task")
    device = kwargs.get("device")

    model_path = model_spec.model_uri
    if model_path is None:
        raise ValueError("model_path required")

    if task == "text-classification":
        return TransformersTextClassificationModel(
            model_uid=model_uid,
            model_path=model_path,
            model_family=model_spec,
            device=device,
            config=kwargs,
        )
    elif task == "mock":
        return MockModel(
            model_uid=model_uid,
            model_path=model_path,
            model_family=model_spec,
            device=device,
            config=kwargs,
        )
    else:
        return AutoModel(
            model_uid=model_uid,
            model_path=model_path,
            model_family=model_spec,
            device=device,
            config=kwargs,
        )
