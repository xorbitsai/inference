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

import base64
import inspect
import io
import json

import PIL.Image

from ..core import FlexibleModel, FlexibleModelSpec


class UltralyticsModel(FlexibleModel):
    def load(self):
        from ultralytics import YOLO

        config = dict(self.config or {})
        if self._device:
            config["device"] = self._device

        self._model = YOLO(model=self._model_path, **config)

    def infer(self, *args, **kwargs):
        predict_func = self._model.predict

        sig = inspect.signature(predict_func)
        bound_args = sig.bind_partial(*args, **kwargs)  # 或 bind() 视场景选择
        bound_args.apply_defaults()

        if "source" in bound_args.arguments:
            source = bound_args.arguments["source"]
            decoded = base64.b64decode(source)
            img = PIL.Image.open(io.BytesIO(decoded))

            bound_args.arguments["source"] = img

        results = predict_func(*bound_args.args, **bound_args.kwargs)
        return [json.loads(r.to_json()) for r in results]


def launcher(model_uid: str, model_spec: FlexibleModelSpec, **kwargs) -> FlexibleModel:
    device = kwargs.get("device")

    model_path = model_spec.model_uri
    if model_path is None:
        raise ValueError("model_path required")

    return UltralyticsModel(
        model_uid=model_uid,
        model_path=model_path,
        model_family=model_spec,
        device=device,
        config=kwargs,
    )
