"""Test SDAPI routing, validated payloads and actor dispatch without a GPU."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter
from starlette.requests import Request

from ..restful_api import RESTfulAPI
from ..routers.images import register_routes
from ..schemas import SDAPIControlNetDetect, SDAPIImg2imgRequst, SDAPITxt2imgRequst


def request(payload=None, query=b""):
    async def receive():
        return {
            "type": "http.request",
            "body": json.dumps(payload or {}).encode(),
            "more_body": False,
        }

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [],
            "query_string": query,
        },
        receive,
    )


@pytest.fixture
def api():
    api = object.__new__(RESTfulAPI)
    api._get_supervisor_ref = AsyncMock()
    api._check_model_access = lambda *a: None
    api._set_trace_model = lambda *a: None
    api._set_trace_model_type = lambda *a: None
    api._report_error_event = AsyncMock()
    return api


def test_routes_keep_auth_dependencies():
    handlers = SimpleNamespace(
        _router=APIRouter(), _auth_service=lambda: None, is_authenticated=lambda: True
    )
    handlers.__dict__.update(
        {
            name: (lambda: None)
            for name in [
                "create_images",
                "create_variations",
                "create_inpainting",
                "create_ocr",
                "create_image_edits",
                "sdapi_options",
                "sdapi_sd_models",
                "sdapi_samplers",
                "sdapi_loras",
                "sdapi_upscalers",
                "sdapi_progress",
                "sdapi_interrupt",
                "sdapi_txt2img",
                "sdapi_img2img",
                "sdapi_controlnet_model_list",
                "sdapi_controlnet_module_list",
                "sdapi_controlnet_control_types",
                "sdapi_controlnet_detect",
            ]
        }
    )
    register_routes(handlers)
    routes = {route.path: route for route in handlers._router.routes}
    assert (
        len([path for path in routes if path.startswith(("/sdapi", "/controlnet"))])
        == 13
    )
    assert all(route.dependencies for route in routes.values())


def test_schema_preserves_extensions_and_rejects_invalid_sizes():
    body = SDAPITxt2imgRequst.parse_obj(
        {
            "alwayson_scripts": {"ControlNet": {"args": []}},
            "subseed": 42,
            "enable_hr": True,
        }
    )
    assert body.subseed == 42 and body.hr_scale == 2
    with pytest.raises(ValueError):
        SDAPITxt2imgRequst(width=0)
    with pytest.raises(ValueError):
        SDAPIImg2imgRequst(init_images=[])
    assert SDAPIControlNetDetect().controlnet_module == "none"


@pytest.mark.asyncio
async def test_txt2img_passes_extension_parameters(api):
    actor = SimpleNamespace(
        uid="image", txt2img=AsyncMock(return_value='{"images": []}')
    )
    api._get_supervisor_ref.return_value.get_model = AsyncMock(return_value=actor)
    result = await api.sdapi_txt2img(
        request(
            {
                "model": "image",
                "request_id": "req",
                "seed": 9,
                "subseed": 10,
                "batch_size": 2,
                "alwayson_scripts": {"ControlNet": {"args": []}},
            }
        )
    )
    assert result.status_code == 200
    kwargs = actor.txt2img.call_args.kwargs
    assert kwargs["request_id"] == "req" and kwargs["batch_size"] == 2
    assert "ControlNet" in kwargs["alwayson_scripts"]


@pytest.mark.asyncio
async def test_progress_and_interrupt(api):
    supervisor = api._get_supervisor_ref.return_value
    supervisor.get_progress = AsyncMock(return_value=0.5)
    result = await api.sdapi_progress(request(query=b"request_id=r"))
    assert json.loads(result.body)["progress"] == 0.5
    actor = SimpleNamespace(abort_request=AsyncMock(return_value="DONE"))
    supervisor.get_model = AsyncMock(return_value=actor)
    result = await api.sdapi_interrupt(request({"model": "sd", "request_id": "r"}))
    assert json.loads(result.body)["status"] == "DONE"
    actor.abort_request.assert_awaited_once_with("r")


@pytest.mark.asyncio
async def test_controlnet_detect_alias(api):
    supervisor = api._get_supervisor_ref.return_value
    supervisor.list_models = AsyncMock(
        return_value={"sd": {"model_type": "image", "controlnet": [{}]}}
    )
    actor = SimpleNamespace(
        controlnet_detect=AsyncMock(return_value='{"images": ["ok"]}')
    )
    supervisor.get_model = AsyncMock(return_value=actor)
    await api.sdapi_controlnet_detect(request({"controlnet_images": ["encoded"]}))
    assert actor.controlnet_detect.call_args.kwargs["controlnet_input_images"] == [
        "encoded"
    ]
    assert "controlnet_images" not in actor.controlnet_detect.call_args.kwargs
