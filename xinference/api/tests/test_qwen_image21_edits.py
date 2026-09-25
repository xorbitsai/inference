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

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image
from starlette.datastructures import UploadFile

from ..restful_api import RESTfulAPI


@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen-Image-2.1",
        "Qwen-Image-Edit-2511",
        "Ming-Image-0.1-Design",
        "Ming-Image-0.1-Design-Layer",
    ],
)
def test_image_edits_alpha_depends_on_model_name(monkeypatch, model_name):
    model_ref = SimpleNamespace(image_to_image=AsyncMock(return_value=b'{"data": []}'))
    supervisor = SimpleNamespace(
        describe_model=AsyncMock(return_value={"model_name": model_name})
    )
    api = SimpleNamespace(
        _get_supervisor_ref=AsyncMock(return_value=supervisor),
        _report_error_event=AsyncMock(),
        _set_trace_model=Mock(),
        _set_trace_model_type=Mock(),
        _check_model_access=Mock(),
        _add_running_task=Mock(),
    )
    monkeypatch.setattr(
        "xinference.api.restful_api.require_model",
        AsyncMock(return_value=model_ref),
    )
    uploads = []
    for _ in range(2):
        buffer = io.BytesIO()
        Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(buffer, format="PNG")
        buffer.seek(0)
        uploads.append(UploadFile(buffer, filename="reference.png"))
    response = asyncio.run(
        RESTfulAPI.create_image_edits(
            api,
            request=Mock(),
            prompt="Edit",
            images=uploads,
            mask=None,
            model="custom-model-uid",
            n=1,
            size="original",
            response_format="b64_json",
            stream=False,
        )
    )
    assert response.status_code == 200
    supervisor.describe_model.assert_awaited_once_with("custom-model-uid")
    kwargs = model_ref.image_to_image.call_args.kwargs
    for image in [kwargs["image"], *kwargs["reference_images"]]:
        if model_name in {
            "Qwen-Image-2.1",
            "Ming-Image-0.1-Design",
            "Ming-Image-0.1-Design-Layer",
        }:
            assert image.mode == "RGBA"
            assert image.getpixel((0, 0)) == (0, 0, 0, 0)
        else:
            assert image.mode == "RGB"
            assert image.getpixel((0, 0)) == (255, 255, 255)
