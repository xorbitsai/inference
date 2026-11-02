# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# http://www.apache.org/licenses/LICENSE-2.0

# CI: linux

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Response

from ..routers.audio import register_routes


async def test_capabilities_preserve_model_access_before_actor_call():
    actor = SimpleNamespace(
        get_audio_capabilities=AsyncMock(
            return_value={"asr": {"streaming": True, "server_vad": True}}
        )
    )
    supervisor = SimpleNamespace(get_model=AsyncMock(return_value=actor))

    async def unused() -> Response:
        return Response()

    api = SimpleNamespace(
        _router=APIRouter(),
        _auth_service=None,
        is_authenticated=lambda: False,
        _check_model_access=Mock(),
        _get_supervisor_ref=AsyncMock(return_value=supervisor),
        _report_error_event=AsyncMock(),
        create_audio_embedding=unused,
        create_transcriptions=unused,
        create_translations=unused,
        create_speech=unused,
    )
    register_routes(api)
    app = FastAPI()
    app.include_router(api._router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.get("/v1/models/custom/audio/capabilities")
        assert response.status_code == 200 and response.json()["asr"]["server_vad"]
        assert api._check_model_access.call_args.args[1:] == ("custom", "audio")
        api._check_model_access.side_effect = HTTPException(403, "Model access denied")
        response = await client.get("/v1/models/custom/audio/capabilities")
        assert response.status_code == 403
        actor.get_audio_capabilities.assert_awaited_once()
