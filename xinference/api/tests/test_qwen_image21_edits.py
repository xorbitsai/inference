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

# CI: linux

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import UploadFile

from ...core.exceptions import InvalidModelRequestError
from ..restful_api import RESTfulAPI


def _upload_image():
    buffer = io.BytesIO()
    Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(buffer, format="PNG")
    buffer.seek(0)
    return UploadFile(buffer, filename="reference.png")


@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen-Image-2.1",
        "Qwen-Image-2.1-Turbo",
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
    response = asyncio.run(
        RESTfulAPI.create_image_edits(
            api,
            request=Mock(),
            prompt="Edit",
            images=[_upload_image(), _upload_image()],
            mask=None,
            model="custom-model-uid",
            n=1,
            size="original",
            response_format="b64_json",
            stream=False,
            use_lightning=False if model_name == "Qwen-Image-Edit-2511" else None,
            kwargs=None,
        )
    )
    assert response.status_code == 200
    supervisor.describe_model.assert_awaited_once_with("custom-model-uid")
    kwargs = model_ref.image_to_image.call_args.kwargs
    if model_name == "Qwen-Image-Edit-2511":
        assert kwargs["use_lightning"] is False
    else:
        assert "use_lightning" not in kwargs
    for image in [kwargs["image"], *kwargs["reference_images"]]:
        if model_name in {
            "Qwen-Image-2.1",
            "Qwen-Image-2.1-Turbo",
            "Ming-Image-0.1-Design",
            "Ming-Image-0.1-Design-Layer",
        }:
            assert image.mode == "RGBA"
            assert image.getpixel((0, 0)) == (0, 0, 0, 0)
        else:
            assert image.mode == "RGB"
            assert image.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "method"),
    [
        ("create_images", "text_to_image"),
        ("create_variations", "image_to_image"),
        ("create_inpainting", "inpainting"),
        ("create_image_edits", "image_to_image"),
        ("sdapi_txt2img", "txt2img"),
    ],
)
async def test_invalid_lightning_request_returns_http_400(
    monkeypatch, endpoint, method
):
    message = (
        "Launch the model with a Lightning adapter before using use_lightning=true"
    )
    generate = AsyncMock(side_effect=InvalidModelRequestError(message))
    supervisor = SimpleNamespace(
        describe_model=AsyncMock(return_value={"model_name": "Qwen-Image"})
    )
    api = SimpleNamespace(
        _get_supervisor_ref=AsyncMock(return_value=supervisor),
        _report_error_event=AsyncMock(),
        _set_trace_model=Mock(),
        _set_trace_model_type=Mock(),
        _check_model_access=Mock(),
        _add_running_task=Mock(),
        _parse_sdapi_request=RESTfulAPI._parse_sdapi_request,
    )
    monkeypatch.setattr(
        "xinference.api.restful_api.require_model",
        AsyncMock(return_value=SimpleNamespace(**{method: generate})),
    )
    model, prompt = "image-model", "test"
    kwargs = '{"use_lightning": true}'
    body = {"model": model, "prompt": prompt, "kwargs": kwargs}
    options = {}
    if endpoint == "sdapi_txt2img":
        body = {"model": model, "prompt": prompt, "use_lightning": True}
    elif endpoint == "create_image_edits":
        options = dict(
            model=model,
            prompt=prompt,
            images=[_upload_image()],
            mask=None,
            n=1,
            size="original",
            response_format="url",
            stream=False,
            use_lightning=True,
            kwargs=None,
        )
    elif endpoint in ("create_variations", "create_inpainting"):
        options = dict(
            model=model,
            prompt=prompt,
            negative_prompt=None,
            n=1,
            size="8*8",
            response_format="url",
            kwargs=kwargs,
        )
        if endpoint == "create_variations":
            options["image"] = [_upload_image()]
        else:
            options.update(image=_upload_image(), mask_image=_upload_image())

    request = SimpleNamespace(json=AsyncMock(return_value=body))
    with pytest.raises(HTTPException) as error:
        await getattr(RESTfulAPI, endpoint)(api, request, **options)
    assert error.value.status_code == 400
    assert error.value.detail == message
    assert generate.await_args.kwargs["use_lightning"] is True


@pytest.mark.parametrize(
    "with_mask, extra_options",
    [
        (False, '{"image": null}'),
        (True, '{"mask_image": null}'),
    ],
)
def test_image_edits_uploaded_files_take_precedence_over_extra_options(
    monkeypatch, with_mask, extra_options
):
    model_ref = SimpleNamespace(
        image_to_image=AsyncMock(return_value=b'{"data": []}'),
        inpainting=AsyncMock(return_value=b'{"data": []}'),
    )
    supervisor = SimpleNamespace(
        describe_model=AsyncMock(return_value={"model_name": "Qwen-Image-Edit-2511"})
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
    response = asyncio.run(
        RESTfulAPI.create_image_edits(
            api,
            request=Mock(),
            prompt="Edit",
            images=[_upload_image()],
            mask=_upload_image() if with_mask else None,
            model="custom-model-uid",
            n=1,
            size="original",
            response_format="b64_json",
            stream=False,
            use_lightning=None,
            kwargs=extra_options,
        )
    )

    assert response.status_code == 200
    called = model_ref.inpainting if with_mask else model_ref.image_to_image
    call_kwargs = called.call_args.kwargs
    assert call_kwargs["image"].size == (8, 8)
    if with_mask:
        assert call_kwargs["mask_image"].size == (8, 8)
