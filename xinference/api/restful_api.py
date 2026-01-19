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
import inspect
import ipaddress
import json
import logging
import multiprocessing
import os
import pprint
import time
import uuid
import warnings
from typing import Any, Dict, List, Optional, Union

import gradio as gr
import xoscar as xo
from aioprometheus import REGISTRY, MetricsMiddleware
from aioprometheus.asgi.starlette import metrics
from fastapi import (
    APIRouter,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    Security,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from sse_starlette.sse import EventSourceResponse
from starlette.responses import JSONResponse as StarletteJSONResponse
from starlette.responses import PlainTextResponse, RedirectResponse
from uvicorn import Config, Server
from xoscar.utils import get_next_port

from .._compat import BaseModel, Field
from .._version import get_versions
from ..constants import (
    XINFERENCE_ALLOWED_IPS,
    XINFERENCE_DEFAULT_CANCEL_BLOCK_DURATION,
    XINFERENCE_DEFAULT_ENDPOINT_PORT,
    XINFERENCE_DISABLE_METRICS,
    XINFERENCE_SSE_PING_ATTEMPTS_SECONDS,
)
from ..core.event import Event, EventCollectorActor, EventType
from ..core.supervisor import SupervisorActor
from ..core.utils import CancelMixin, json_dumps

# Import Anthropic-related types and availability flag
from ..types import (
    ANTHROPIC_AVAILABLE,
    AnthropicMessage,
    ChatCompletion,
    Completion,
    CreateChatCompletion,
    CreateCompletion,
    CreateMessage,
    ImageList,
    PeftModelConfig,
    SDAPIResult,
    VideoList,
    max_tokens_field,
)
from .oauth2.auth_service import AuthService
from .oauth2.types import LoginUserForm

logger = logging.getLogger(__name__)


class JSONResponse(StarletteJSONResponse):  # type: ignore # noqa: F811
    def render(self, content: Any) -> bytes:
        return json_dumps(content)


class CreateCompletionRequest(CreateCompletion):
    class Config:
        schema_extra = {
            "example": {
                "prompt": "\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n",
                "stop": ["\n", "###"],
            }
        }


class CreateEmbeddingRequest(BaseModel):
    model: str
    input: Union[
        str, List[str], List[int], List[List[int]], Dict[str, str], List[Dict[str, str]]
    ] = Field(description="The input to embed.")
    user: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "input": "The food was delicious and the waiter...",
            }
        }


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: Optional[bool] = False
    return_len: Optional[bool] = False
    max_chunks_per_doc: Optional[int] = None
    kwargs: Optional[str] = None


class TextToImageRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]] = Field(description="The input to embed.")
    n: Optional[int] = 1
    response_format: Optional[str] = "url"
    size: Optional[str] = "1024*1024"
    kwargs: Optional[str] = None
    user: Optional[str] = None


class SDAPIOptionsRequest(BaseModel):
    sd_model_checkpoint: Optional[str] = None


class SDAPITxt2imgRequst(BaseModel):
    model: Optional[str]
    prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    steps: Optional[int] = None
    seed: Optional[int] = -1
    cfg_scale: Optional[float] = 7.0
    override_settings: Optional[dict] = {}
    width: Optional[int] = 512
    height: Optional[int] = 512
    sampler_name: Optional[str] = None
    denoising_strength: Optional[float] = None
    kwargs: Optional[str] = None
    user: Optional[str] = None


class SDAPIImg2imgRequst(BaseModel):
    model: Optional[str]
    init_images: Optional[list]
    prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    steps: Optional[int] = None
    seed: Optional[int] = -1
    cfg_scale: Optional[float] = 7.0
    override_settings: Optional[dict] = {}
    width: Optional[int] = 512
    height: Optional[int] = 512
    sampler_name: Optional[str] = None
    denoising_strength: Optional[float] = None
    kwargs: Optional[str] = None
    user: Optional[str] = None


class TextToVideoRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]] = Field(description="The input to embed.")
    n: Optional[int] = 1
    kwargs: Optional[str] = None
    user: Optional[str] = None


class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: Optional[str]
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0
    stream: Optional[bool] = False
    kwargs: Optional[str] = None


class RegisterModelRequest(BaseModel):
    model: str
    worker_ip: Optional[str]
    persist: bool


class AutoConfigLLMRequest(BaseModel):
    model_path: str
    model_family: str


class UpdateModelRequest(BaseModel):
    model_type: str


class BuildGradioInterfaceRequest(BaseModel):
    model_type: str
    model_name: str
    model_size_in_billions: int
    model_format: str
    quantization: str
    context_length: int
    model_ability: List[str]
    model_description: str
    model_lang: List[str]


class BuildGradioMediaInterfaceRequest(BaseModel):
    model_type: str
    model_name: str
    model_family: str
    model_id: str
    controlnet: Union[None, List[Dict[str, Union[str, dict, None]]]]
    model_revision: Optional[str]
    model_ability: List[str]


class RESTfulAPI(CancelMixin):
    # Add new class attributes
    _allowed_ip_list: Optional[List[ipaddress.IPv4Network]] = None

    def __init__(
        self,
        supervisor_address: str,
        host: str,
        port: int,
        auth_config_file: Optional[str] = None,
    ):
        super().__init__()
        self._supervisor_address = supervisor_address
        self._host = host
        self._port = port
        self._supervisor_ref = None
        self._event_collector_ref = None
        self._auth_service = AuthService(auth_config_file)
        self._router = APIRouter()
        self._app = FastAPI()
        # Initialize allowed IP list once
        self._init_allowed_ip_list()

    def _init_allowed_ip_list(self):
        """Initialize the allowed IP list from environment variable."""
        if RESTfulAPI._allowed_ip_list is None:
            # ie: export XINFERENCE_ALLOWED_IPS=192.168.1.0/24
            allowed_ips = XINFERENCE_ALLOWED_IPS
            if allowed_ips:
                RESTfulAPI._allowed_ip_list = []
                for ip in allowed_ips.split(","):
                    ip = ip.strip()
                    try:
                        # Try parsing as network/CIDR
                        if "/" in ip:
                            RESTfulAPI._allowed_ip_list.append(ipaddress.ip_network(ip))
                        else:
                            # Parse as single IP
                            RESTfulAPI._allowed_ip_list.append(
                                ipaddress.ip_network(f"{ip}/32")
                            )
                    except ValueError:
                        logger.error(
                            f"Invalid IP address or network: {ip}", exc_info=True
                        )
                        continue

    def _is_ip_allowed(self, ip: str) -> bool:
        """Check if an IP is allowed based on configured rules."""
        if not RESTfulAPI._allowed_ip_list:
            return True

        try:
            client_ip = ipaddress.ip_address(ip)
            return any(
                client_ip in allowed_net for allowed_net in RESTfulAPI._allowed_ip_list
            )
        except ValueError:
            return False

    def is_authenticated(self):
        return False if self._auth_service.config is None else True

    @staticmethod
    def handle_request_limit_error(e: Exception):
        if "Rate limit reached" in str(e):
            raise HTTPException(status_code=429, detail=str(e))

    async def _get_supervisor_ref(self) -> xo.ActorRefType[SupervisorActor]:
        if self._supervisor_ref is None:
            self._supervisor_ref = await xo.actor_ref(
                address=self._supervisor_address, uid=SupervisorActor.default_uid()
            )
        return self._supervisor_ref

    async def _get_event_collector_ref(self) -> xo.ActorRefType[EventCollectorActor]:
        if self._event_collector_ref is None:
            self._event_collector_ref = await xo.actor_ref(
                address=self._supervisor_address, uid=EventCollectorActor.default_uid()
            )
        return self._event_collector_ref

    async def _report_error_event(self, model_uid: str, content: str):
        try:
            event_collector_ref = await self._get_event_collector_ref()
            await event_collector_ref.report_event(
                model_uid,
                Event(
                    event_type=EventType.ERROR,
                    event_ts=int(time.time()),
                    event_content=content,
                ),
            )
        except Exception:
            logger.exception(
                "Report error event failed, model: %s, content: %s", model_uid, content
            )

    async def login_for_access_token(self, request: Request) -> JSONResponse:
        form_data = LoginUserForm.parse_obj(await request.json())
        result = self._auth_service.generate_token_for_user(
            form_data.username, form_data.password
        )
        return JSONResponse(content=result)

    async def is_cluster_authenticated(self) -> JSONResponse:
        return JSONResponse(content={"auth": self.is_authenticated()})

    def serve(self, logging_conf: Optional[dict] = None):
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self._app.middleware("http")
        async def ip_restriction_middleware(request: Request, call_next):
            client_ip = request.client.host
            if not self._is_ip_allowed(client_ip):
                return PlainTextResponse(
                    status_code=403, content=f"Access denied for IP: {client_ip}\n"
                )
            response = await call_next(request)
            return response

        @self._app.exception_handler(500)
        async def internal_exception_handler(request: Request, exc: Exception):
            logger.exception("Handling request %s failed: %s", request.url, exc)
            return PlainTextResponse(
                status_code=500, content=f"Internal Server Error: {exc}"
            )

        # internal interface
        self._router.add_api_route("/status", self.get_status, methods=["GET"])
        # conflict with /v1/models/{model_uid} below, so register this first
        self._router.add_api_route(
            "/v1/models/prompts", self._get_builtin_prompts, methods=["GET"]
        )
        self._router.add_api_route(
            "/v1/models/families", self._get_builtin_families, methods=["GET"]
        )
        self._router.add_api_route(
            "/v1/models/llm/auto-register",
            self.build_llm_registration_from_config,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:register"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/models/vllm-supported",
            self.list_vllm_supported_model_families,
            methods=["GET"],
        )
        self._router.add_api_route(
            "/v1/cluster/info",
            self.get_cluster_device_info,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["admin"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/cluster/version",
            self.get_cluster_version,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["admin"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/cluster/devices",
            self._get_devices_count,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:list"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route("/v1/address", self.get_address, methods=["GET"])

        # user interface
        self._router.add_api_route(
            "/v1/ui/{model_uid}",
            self.build_gradio_interface,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/ui/images/{model_uid}",
            self.build_gradio_media_interface,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/ui/audios/{model_uid}",
            self.build_gradio_media_interface,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/ui/videos/{model_uid}",
            self.build_gradio_media_interface,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/token", self.login_for_access_token, methods=["POST"]
        )
        self._router.add_api_route(
            "/v1/cluster/auth", self.is_cluster_authenticated, methods=["GET"]
        )
        # just for compatibility, LLM only
        self._router.add_api_route(
            "/v1/engines/{model_name}",
            self.query_engines_by_model_name,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:list"])]
                if self.is_authenticated()
                else None
            ),
        )
        # engines for all model types
        self._router.add_api_route(
            "/v1/engines/{model_type}/{model_name}",
            self.query_engines_by_model_name,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:list"])]
                if self.is_authenticated()
                else None
            ),
        )
        # running instances
        self._router.add_api_route(
            "/v1/models/instances",
            self.get_instance_info,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:list"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/models/{model_type}/{model_name}/versions",
            self.get_model_versions,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:list"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/models",
            self.list_models,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:list"])]
                if self.is_authenticated()
                else None
            ),
        )

        self._router.add_api_route(
            "/v1/models/{model_uid}",
            self.describe_model,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:list"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/models/{model_uid}/events",
            self.get_model_events,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/models/{model_uid}/replicas",
            self.get_model_replicas,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:list"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/models/{model_uid}/requests/{request_id}/abort",
            self.abort_request,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/models/instance",
            self.launch_model_by_version,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:start"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/models",
            self.launch_model,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:start"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/models/{model_uid}",
            self.terminate_model,
            methods=["DELETE"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:stop"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/models/{model_uid}/progress",
            self.get_launch_model_progress,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/models/{model_uid}/cancel",
            self.cancel_launch_model,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:stop"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/completions",
            self.create_completion,
            methods=["POST"],
            response_model=Completion,
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        # Register messages endpoint only if Anthropic is available
        if ANTHROPIC_AVAILABLE:
            self._router.add_api_route(
                "/anthropic/v1/messages",
                self.create_message,
                methods=["POST"],
                response_model=AnthropicMessage,
                dependencies=(
                    [Security(self._auth_service, scopes=["models:read"])]
                    if self.is_authenticated()
                    else None
                ),
            )
            # Register Anthropic models endpoints
            self._router.add_api_route(
                "/anthropic/v1/models",
                self.anthropic_list_models,
                methods=["GET"],
                dependencies=(
                    [Security(self._auth_service, scopes=["models:list"])]
                    if self.is_authenticated()
                    else None
                ),
            )
            self._router.add_api_route(
                "/anthropic/v1/models/{model_id}",
                self.anthropic_get_model,
                methods=["GET"],
                dependencies=(
                    [Security(self._auth_service, scopes=["models:list"])]
                    if self.is_authenticated()
                    else None
                ),
            )
        self._router.add_api_route(
            "/v1/embeddings",
            self.create_embedding,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/convert_ids_to_tokens",
            self.convert_ids_to_tokens,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/rerank",
            self.rerank,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/audio/transcriptions",
            self.create_transcriptions,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/audio/translations",
            self.create_translations,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/audio/speech",
            self.create_speech,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/requests/{request_id}/progress",
            self.get_progress,
            methods=["get"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/images/generations",
            self.create_images,
            methods=["POST"],
            response_model=ImageList,
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/images/variations",
            self.create_variations,
            methods=["POST"],
            response_model=ImageList,
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/images/inpainting",
            self.create_inpainting,
            methods=["POST"],
            response_model=ImageList,
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/images/ocr",
            self.create_ocr,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/images/edits",
            self.create_image_edits,
            methods=["POST"],
            response_model=ImageList,
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )

        # SD WebUI API
        self._router.add_api_route(
            "/sdapi/v1/options",
            self.sdapi_options,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/sdapi/v1/sd-models",
            self.sdapi_sd_models,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/sdapi/v1/samplers",
            self.sdapi_samplers,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/sdapi/v1/txt2img",
            self.sdapi_txt2img,
            methods=["POST"],
            response_model=SDAPIResult,
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/sdapi/v1/img2img",
            self.sdapi_img2img,
            methods=["POST"],
            response_model=SDAPIResult,
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/video/generations",
            self.create_videos,
            methods=["POST"],
            response_model=VideoList,
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/video/generations/image",
            self.create_videos_from_images,
            methods=["POST"],
            response_model=VideoList,
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/video/generations/flf",
            self.create_videos_from_first_last_frame,
            methods=["POST"],
            response_model=VideoList,
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/chat/completions",
            self.create_chat_completion,
            methods=["POST"],
            response_model=ChatCompletion,
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/flexible/infers",
            self.create_flexible_infer,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:read"])]
                if self.is_authenticated()
                else None
            ),
        )

        # for custom models
        self._router.add_api_route(
            "/v1/model_registrations/{model_type}",
            self.register_model,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:register"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/model_registrations/{model_type}/{model_name}",
            self.unregister_model,
            methods=["DELETE"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:unregister"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/model_registrations/{model_type}",
            self.list_model_registrations,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:list"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/models/update_type",
            self.update_model_type,
            methods=["POST"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:add"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/model_registrations/{model_type}/{model_name}",
            self.get_model_registrations,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["models:list"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/cache/models",
            self.list_cached_models,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["cache:list"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/cache/models/files",
            self.list_model_files,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["cache:list"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/cache/models",
            self.confirm_and_remove_model,
            methods=["DELETE"],
            dependencies=(
                [Security(self._auth_service, scopes=["cache:delete"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/virtualenvs",
            self.list_virtual_envs,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["virtualenv:list"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/virtualenvs",
            self.remove_virtual_env,
            methods=["DELETE"],
            dependencies=(
                [Security(self._auth_service, scopes=["virtualenv:delete"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/workers",
            self.get_workers_info,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["admin"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/supervisor",
            self.get_supervisor_info,
            methods=["GET"],
            dependencies=(
                [Security(self._auth_service, scopes=["admin"])]
                if self.is_authenticated()
                else None
            ),
        )
        self._router.add_api_route(
            "/v1/clusters",
            self.abort_cluster,
            methods=["DELETE"],
            dependencies=(
                [Security(self._auth_service, scopes=["admin"])]
                if self.is_authenticated()
                else None
            ),
        )

        if XINFERENCE_DISABLE_METRICS:
            logger.info(
                "Supervisor metrics is disabled due to the environment XINFERENCE_DISABLE_METRICS=1"
            )
            self._app.include_router(self._router)
        else:
            # Clear the global Registry for the MetricsMiddleware, or
            # the MetricsMiddleware will register duplicated metrics if the port
            # conflict (This serve method run more than once).
            REGISTRY.clear()
            self._app.add_middleware(MetricsMiddleware)
            self._app.include_router(self._router)
            self._app.add_route("/metrics", metrics)

        # Check all the routes returns Response.
        # This is to avoid `jsonable_encoder` performance issue:
        # https://github.com/xorbitsai/inference/issues/647
        invalid_routes = []
        try:
            for router in self._router.routes:
                return_annotation = router.endpoint.__annotations__.get("return")
                if not inspect.isclass(return_annotation) or not issubclass(
                    return_annotation, Response
                ):
                    invalid_routes.append(
                        (router.path, router.endpoint, return_annotation)
                    )
        except Exception:
            pass  # In case that some Python version does not have __annotations__
        if invalid_routes:
            raise Exception(
                f"The return value type of the following routes is not Response:\n"
                f"{pprint.pformat(invalid_routes)}"
            )

        class SPAStaticFiles(StaticFiles):
            async def get_response(self, path: str, scope):
                response = await super().get_response(path, scope)
                if response.status_code == 404:
                    response = await super().get_response(".", scope)
                return response

        try:
            package_file_path = __import__("xinference").__file__
            assert package_file_path is not None
            lib_location = os.path.abspath(os.path.dirname(package_file_path))
            ui_location = os.path.join(lib_location, "ui/web/ui/build/")
        except ImportError as e:
            raise ImportError(f"Xinference is imported incorrectly: {e}")

        if os.path.exists(ui_location):

            @self._app.get("/")
            def read_main():
                response = RedirectResponse(url="/ui/")
                return response

            self._app.mount(
                "/ui/",
                SPAStaticFiles(directory=ui_location, html=True),
            )
        else:
            warnings.warn(
                f"""
            Xinference ui is not built at expected directory: {ui_location}
            To resolve this warning, navigate to {os.path.join(lib_location, "ui/web/ui/")}
            And build the Xinference ui by running "npm run build"
            """
            )

        config = Config(
            app=self._app, host=self._host, port=self._port, log_config=logging_conf
        )
        server = Server(config)
        server.run()

    async def _get_builtin_prompts(self) -> JSONResponse:
        """
        For internal usage
        """
        try:
            data = await (await self._get_supervisor_ref()).get_builtin_prompts()
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_builtin_families(self) -> JSONResponse:
        """
        For internal usage
        """
        try:
            data = await (await self._get_supervisor_ref()).get_builtin_families()
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def build_llm_registration_from_config(
        self, request: Request
    ) -> JSONResponse:
        try:
            body = AutoConfigLLMRequest.parse_obj(await request.json())
            from ..model.llm.config_parser import (
                build_llm_registration_from_local_config,
            )

            data = build_llm_registration_from_local_config(
                model_path=body.model_path,
                model_family=body.model_family,
            )
            return JSONResponse(content=data)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_devices_count(self) -> JSONResponse:
        """
        For internal usage
        """
        try:
            data = await (await self._get_supervisor_ref()).get_devices_count()
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_status(self) -> JSONResponse:
        try:
            data = await (await self._get_supervisor_ref()).get_status()
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def list_models(self) -> JSONResponse:
        try:
            models = await (await self._get_supervisor_ref()).list_models()

            model_list = []
            for model_id, model_info in models.items():
                model_list.append(
                    {
                        "id": model_id,
                        "object": "model",
                        "created": 0,
                        "owned_by": "xinference",
                        **model_info,
                    }
                )
            response = {"object": "list", "data": model_list}

            return JSONResponse(content=response)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def anthropic_list_models(self) -> JSONResponse:
        """Anthropic-compatible models endpoint"""
        try:

            # Get running models from xinference
            running_models = await (await self._get_supervisor_ref()).list_models()

            # For backward compatibility with tests, only return running models by default
            model_list = []

            # Add running models to the list
            for model_id, model_info in running_models.items():
                anthropic_model = {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "display_name": model_info.get("model_name", model_id),
                    "type": model_info.get("model_type", "model"),
                    "max_tokens": model_info.get("context_length", 4096),
                }
                model_list.append(anthropic_model)

            return JSONResponse(content=model_list)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def anthropic_get_model(self, model_id: str) -> JSONResponse:
        """Anthropic-compatible model retrieval endpoint"""
        try:
            models = await (await self._get_supervisor_ref()).list_models()

            model_info = models[model_id]

            # Convert to Anthropic format
            anthropic_model = {
                "id": model_id,  # Return the original requested ID
                "object": "model",
                "created": 0,
                "display_name": model_info.get("model_name", model_id),
                "type": model_info.get("model_type", "model"),
                "max_tokens": model_info.get("context_length", 4096),
                **model_info,
            }

            return JSONResponse(content=anthropic_model)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def describe_model(self, model_uid: str) -> JSONResponse:
        try:
            data = await (await self._get_supervisor_ref()).describe_model(model_uid)
            return JSONResponse(content=data)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))

        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def launch_model(
        self, request: Request, wait_ready: bool = Query(True)
    ) -> JSONResponse:
        payload = await request.json()
        model_uid = payload.get("model_uid")
        model_name = payload.get("model_name")
        model_engine = payload.get("model_engine")
        model_size_in_billions = payload.get("model_size_in_billions")
        model_format = payload.get("model_format")
        quantization = payload.get("quantization")
        model_type = payload.get("model_type", "LLM")
        replica = payload.get("replica", 1)
        n_gpu = payload.get("n_gpu", "auto")
        request_limits = payload.get("request_limits", None)
        peft_model_config = payload.get("peft_model_config", None)
        worker_ip = payload.get("worker_ip", None)
        gpu_idx = payload.get("gpu_idx", None)
        download_hub = payload.get("download_hub", None)
        model_path = payload.get("model_path", None)
        enable_virtual_env = payload.get("enable_virtual_env", None)
        virtual_env_packages = payload.get("virtual_env_packages", None)
        envs = payload.get("envs", None)

        exclude_keys = {
            "model_uid",
            "model_name",
            "model_engine",
            "model_size_in_billions",
            "model_format",
            "quantization",
            "model_type",
            "replica",
            "n_gpu",
            "request_limits",
            "peft_model_config",
            "worker_ip",
            "gpu_idx",
            "download_hub",
            "model_path",
            "enable_virtual_env",
            "virtual_env_packages",
            "envs",
        }

        kwargs = {
            key: value for key, value in payload.items() if key not in exclude_keys
        }

        if not model_name:
            raise HTTPException(
                status_code=400,
                detail="Invalid input. Please specify the `model_name` field.",
            )
        if not model_engine and model_type == "LLM":
            raise HTTPException(
                status_code=400,
                detail="Invalid input. Please specify the `model_engine` field.",
            )

        if isinstance(gpu_idx, int):
            gpu_idx = [gpu_idx]
        if gpu_idx:
            if len(gpu_idx) % replica:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid input. Allocated gpu must be a multiple of replica.",
                )

        if peft_model_config is not None:
            peft_model_config = PeftModelConfig.from_dict(peft_model_config)
        else:
            peft_model_config = None

        try:
            model_uid = await (await self._get_supervisor_ref()).launch_builtin_model(
                model_uid=model_uid,
                model_name=model_name,
                model_engine=model_engine,
                model_size_in_billions=model_size_in_billions,
                model_format=model_format,
                quantization=quantization,
                model_type=model_type,
                replica=replica,
                n_gpu=n_gpu,
                request_limits=request_limits,
                wait_ready=wait_ready,
                peft_model_config=peft_model_config,
                worker_ip=worker_ip,
                gpu_idx=gpu_idx,
                download_hub=download_hub,
                model_path=model_path,
                enable_virtual_env=enable_virtual_env,
                virtual_env_packages=virtual_env_packages,
                envs=envs,
                **kwargs,
            )
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except RuntimeError as re:
            logger.error(str(re), exc_info=True)
            raise HTTPException(status_code=503, detail=str(re))
        except asyncio.CancelledError as ce:
            # cancelled by user
            logger.error(str(ce), exc_info=True)
            raise HTTPException(status_code=499, detail=str(ce))
        except Exception as e:
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        return JSONResponse(content={"model_uid": model_uid})

    async def get_instance_info(
        self,
        model_name: Optional[str] = Query(None),
        model_uid: Optional[str] = Query(None),
    ) -> JSONResponse:
        try:
            infos = await (await self._get_supervisor_ref()).get_instance_info(
                model_name, model_uid
            )
        except Exception as e:
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content=infos)

    async def get_model_replicas(self, model_uid: str) -> JSONResponse:
        """Get detailed status of all replicas for a model"""
        try:
            replicas = await (await self._get_supervisor_ref()).get_replica_statuses(
                model_uid
            )
            return JSONResponse(content=replicas)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_launch_model_progress(self, model_uid: str) -> JSONResponse:
        try:
            progress = await (
                await self._get_supervisor_ref()
            ).get_launch_builtin_model_progress(model_uid)
        except Exception as e:
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content={"progress": progress})

    async def cancel_launch_model(self, model_uid: str) -> JSONResponse:
        try:
            await (await self._get_supervisor_ref()).cancel_launch_builtin_model(
                model_uid
            )
        except Exception as e:
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content=None)

    async def launch_model_by_version(
        self, request: Request, wait_ready: bool = Query(True)
    ) -> JSONResponse:
        payload = await request.json()
        model_uid = payload.get("model_uid")
        model_engine = payload.get("model_engine")
        model_type = payload.get("model_type")
        model_version = payload.get("model_version")
        replica = payload.get("replica", 1)
        n_gpu = payload.get("n_gpu", "auto")

        try:
            model_uid = await (
                await self._get_supervisor_ref()
            ).launch_model_by_version(
                model_uid=model_uid,
                model_engine=model_engine,
                model_type=model_type,
                model_version=model_version,
                replica=replica,
                n_gpu=n_gpu,
                wait_ready=wait_ready,
            )
        except Exception as e:
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content={"model_uid": model_uid})

    async def get_model_versions(
        self, model_type: str, model_name: str
    ) -> JSONResponse:
        try:
            content = await (await self._get_supervisor_ref()).get_model_versions(
                model_type, model_name
            )
            return JSONResponse(content=content)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def build_gradio_interface(
        self, model_uid: str, request: Request
    ) -> JSONResponse:
        """
        Separate build_interface with launch_model
        build_interface requires RESTful Client for API calls
        but calling API in async function does not return
        """
        payload = await request.json()
        body = BuildGradioInterfaceRequest.parse_obj(payload)
        assert self._app is not None
        assert body.model_type == "LLM"

        from ..ui.gradio.chat_interface import GradioInterface

        try:
            access_token = request.headers.get("Authorization")
            internal_host = "localhost" if self._host == "0.0.0.0" else self._host
            interface = GradioInterface(
                endpoint=f"http://{internal_host}:{self._port}",
                model_uid=model_uid,
                model_name=body.model_name,
                model_size_in_billions=body.model_size_in_billions,
                model_type=body.model_type,
                model_format=body.model_format,
                quantization=body.quantization,
                context_length=body.context_length,
                model_ability=body.model_ability,
                model_description=body.model_description,
                model_lang=body.model_lang,
                access_token=access_token,
            ).build()
            gr.mount_gradio_app(self._app, interface, f"/{model_uid}")
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))

        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        return JSONResponse(content={"model_uid": model_uid})

    async def build_gradio_media_interface(
        self, model_uid: str, request: Request
    ) -> JSONResponse:
        """
        Build a Gradio interface for image processing models.
        """
        payload = await request.json()
        body = BuildGradioMediaInterfaceRequest.parse_obj(payload)
        assert self._app is not None
        assert body.model_type in ("image", "video", "audio")

        from ..ui.gradio.media_interface import MediaInterface

        try:
            access_token = request.headers.get("Authorization")
            internal_host = "localhost" if self._host == "0.0.0.0" else self._host
            interface = MediaInterface(
                endpoint=f"http://{internal_host}:{self._port}",
                model_uid=model_uid,
                model_family=body.model_family,
                model_name=body.model_name,
                model_id=body.model_id,
                model_revision=body.model_revision,
                controlnet=body.controlnet,
                access_token=access_token,
                model_ability=body.model_ability,
                model_type=body.model_type,
            ).build()

            gr.mount_gradio_app(self._app, interface, f"/{model_uid}")
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))

        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        return JSONResponse(content={"model_uid": model_uid})

    async def terminate_model(self, model_uid: str) -> JSONResponse:
        try:
            assert self._app is not None
            await (await self._get_supervisor_ref()).terminate_model(model_uid)
            self._app.router.routes = [
                route
                for route in self._app.router.routes
                if not (
                    hasattr(route, "path")
                    and isinstance(route.path, str)
                    and route.path == "/" + model_uid
                )
            ]
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content=None)

    async def get_address(self) -> JSONResponse:
        return JSONResponse(content=self._supervisor_address)

    async def _get_model_last_error(self, replica_model_uid: bytes, e: Exception):
        if not isinstance(e, xo.ServerClosed):
            return e
        try:
            model_status = await (await self._get_supervisor_ref()).get_model_status(
                replica_model_uid.decode("utf-8")
            )
            if model_status is not None and model_status.last_error:
                return Exception(model_status.last_error)
        except Exception as ex:
            return ex
        return e

    async def create_completion(self, request: Request) -> Response:
        raw_body = await request.json()
        body = CreateCompletionRequest.parse_obj(raw_body)
        exclude = {
            "prompt",
            "model",
            "n",
            "best_of",
            "logit_bias",
            "logit_bias_type",
            "user",
        }
        raw_kwargs = {k: v for k, v in raw_body.items() if k not in exclude}
        kwargs = body.dict(exclude_unset=True, exclude=exclude)

        # guided_decoding params
        kwargs.update(self.extract_guided_params(raw_body=raw_body))

        # TODO: Decide if this default value override is necessary #1061
        if body.max_tokens is None:
            kwargs["max_tokens"] = max_tokens_field.default

        if body.logit_bias is not None:
            raise HTTPException(status_code=501, detail="Not implemented")

        model_uid = body.model

        try:
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))

        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        if body.stream:

            async def stream_results():
                iterator = None
                try:
                    try:
                        iterator = await model.generate(
                            body.prompt, kwargs, raw_params=raw_kwargs
                        )
                    except RuntimeError as re:
                        self.handle_request_limit_error(re)
                    async for item in iterator:
                        yield item
                except asyncio.CancelledError:
                    logger.info(
                        f"Disconnected from client (via refresh/close) {request.client} during generate."
                    )
                    return
                except Exception as ex:
                    ex = await self._get_model_last_error(model.uid, ex)
                    logger.exception("Completion stream got an error: %s", ex)
                    await self._report_error_event(model_uid, str(ex))
                    # https://github.com/openai/openai-python/blob/e0aafc6c1a45334ac889fe3e54957d309c3af93f/src/openai/_streaming.py#L107
                    yield dict(data=json.dumps({"error": str(ex)}))
                    return
                finally:
                    await model.decrease_serve_count()

            return EventSourceResponse(
                stream_results(), ping=XINFERENCE_SSE_PING_ATTEMPTS_SECONDS
            )
        else:
            try:
                data = await model.generate(body.prompt, kwargs, raw_params=raw_kwargs)
                return Response(data, media_type="application/json")
            except Exception as e:
                e = await self._get_model_last_error(model.uid, e)
                logger.error(e, exc_info=True)
                await self._report_error_event(model_uid, str(e))
                self.handle_request_limit_error(e)
                raise HTTPException(status_code=500, detail=str(e))

    async def create_message(self, request: Request) -> Response:
        raw_body = await request.json()
        body = CreateMessage.parse_obj(raw_body)

        exclude = {
            "model",
            "messages",
            "stream",
            "stop_sequences",
            "metadata",
            "tool_choice",
            "tools",
        }
        raw_kwargs = {k: v for k, v in raw_body.items() if k not in exclude}
        kwargs = body.dict(exclude_unset=True, exclude=exclude)

        # guided_decoding params
        kwargs.update(self.extract_guided_params(raw_body=raw_body))

        # TODO: Decide if this default value override is necessary #1061
        if body.max_tokens is None:
            kwargs["max_tokens"] = max_tokens_field.default

        messages = body.messages and list(body.messages)

        if not messages or messages[-1].get("role") not in ["user", "assistant"]:
            raise HTTPException(
                status_code=400, detail="Invalid input. Please specify the prompt."
            )

        # Handle tools parameter
        if hasattr(body, "tools") and body.tools:
            kwargs["tools"] = body.tools

        # Handle tool_choice parameter
        if hasattr(body, "tool_choice") and body.tool_choice:
            kwargs["tool_choice"] = body.tool_choice

        # Get model mapping
        try:
            running_models = await (await self._get_supervisor_ref()).list_models()
        except Exception as e:
            logger.error(f"Failed to get model mapping: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to get model mapping")

        if not running_models:
            raise HTTPException(
                status_code=400,
                detail=f"No running models available. Please start a model in xinference first.",
            )

        requested_model_id = body.model
        if "claude" in requested_model_id:
            requested_model_id = list(running_models.keys())[0]

        if requested_model_id not in running_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{requested_model_id}' is not available. Available models: {list(running_models.keys())}",
            )
        else:
            model_uid = requested_model_id

        try:
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        if body.stream:

            async def stream_results():
                iterator = None
                try:
                    try:
                        iterator = await model.chat(
                            messages, kwargs, raw_params=raw_kwargs
                        )
                    except RuntimeError as re:
                        self.handle_request_limit_error(re)

                    # Check if iterator is actually an async iterator
                    if hasattr(iterator, "__aiter__"):
                        async for item in iterator:
                            yield item
                    elif isinstance(iterator, (str, bytes)):
                        # Handle case where chat returns bytes/string instead of iterator
                        if isinstance(iterator, bytes):
                            try:
                                content = iterator.decode("utf-8")
                            except UnicodeDecodeError:
                                content = str(iterator)
                        else:
                            content = iterator
                        yield dict(data=json.dumps({"content": content}))
                    else:
                        # Fallback: try to iterate normally
                        try:
                            for item in iterator:
                                yield item
                        except TypeError:
                            # If not iterable, yield as single result
                            yield dict(data=json.dumps({"content": str(iterator)}))

                    yield "[DONE]"
                except asyncio.CancelledError:
                    logger.info(
                        f"Disconnected from client (via refresh/close) {request.client} during chat."
                    )
                    return
                except Exception as ex:
                    ex = await self._get_model_last_error(model.uid, ex)
                    logger.exception("Message stream got an error: %s", ex)
                    await self._report_error_event(model_uid, str(ex))
                    yield dict(data=json.dumps({"error": str(ex)}))
                    return
                finally:
                    await model.decrease_serve_count()

            return EventSourceResponse(
                stream_results(), ping=XINFERENCE_SSE_PING_ATTEMPTS_SECONDS
            )
        else:
            try:
                data = await model.chat(messages, kwargs, raw_params=raw_kwargs)
                # Convert OpenAI format to Anthropic format
                openai_response = json.loads(data)
                anthropic_response = self._convert_openai_to_anthropic(
                    openai_response, body.model
                )
                return Response(
                    json.dumps(anthropic_response), media_type="application/json"
                )
            except Exception as e:
                e = await self._get_model_last_error(model.uid, e)
                logger.error(e, exc_info=True)
                await self._report_error_event(model_uid, str(e))
                self.handle_request_limit_error(e)
                raise HTTPException(status_code=500, detail=str(e))

    async def create_embedding(self, request: Request) -> Response:
        payload = await request.json()
        body = CreateEmbeddingRequest.parse_obj(payload)
        model_uid = body.model
        exclude = {
            "model",
            "input",
            "user",
            "encoding_format",
        }
        kwargs = {key: value for key, value in payload.items() if key not in exclude}

        try:
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            kwargs["model_uid"] = model_uid
            embedding = await model.create_embedding(body.input, **kwargs)
            return Response(embedding, media_type="application/json")
        except Exception as e:
            e = await self._get_model_last_error(model.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def convert_ids_to_tokens(self, request: Request) -> Response:
        payload = await request.json()
        body = CreateEmbeddingRequest.parse_obj(payload)
        model_uid = body.model
        exclude = {
            "model",
            "input",
            "user",
        }
        kwargs = {key: value for key, value in payload.items() if key not in exclude}

        try:
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            decoded_texts = await model.convert_ids_to_tokens(body.input, **kwargs)
            return Response(decoded_texts, media_type="application/json")
        except Exception as e:
            e = await self._get_model_last_error(model.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def rerank(self, request: Request) -> Response:
        payload = await request.json()
        body = RerankRequest.parse_obj(payload)
        model_uid = body.model

        try:
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            if body.kwargs is not None:
                parsed_kwargs = json.loads(body.kwargs)
            else:
                parsed_kwargs = {}
            scores = await model.rerank(
                body.documents,
                body.query,
                top_n=body.top_n,
                max_chunks_per_doc=body.max_chunks_per_doc,
                return_documents=body.return_documents,
                return_len=body.return_len,
                **parsed_kwargs,
            )
            return Response(scores, media_type="application/json")
        except Exception as e:
            e = await self._get_model_last_error(model.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_transcriptions(
        self,
        request: Request,
        model: str = Form(...),
        file: UploadFile = File(media_type="application/octet-stream"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(0),
        kwargs: Optional[str] = Form(None),
    ) -> Response:
        form = await request.form()
        timestamp_granularities = form.get("timestamp_granularities[]")
        if timestamp_granularities:
            timestamp_granularities = [timestamp_granularities]
        model_uid = model
        try:
            model_ref = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            if kwargs is not None:
                parsed_kwargs = json.loads(kwargs)
            else:
                parsed_kwargs = {}
            transcription = await model_ref.transcriptions(
                audio=await file.read(),
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timestamp_granularities=timestamp_granularities,
                **parsed_kwargs,
            )
            return Response(content=transcription, media_type="application/json")
        except Exception as e:
            e = await self._get_model_last_error(model_ref.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_translations(
        self,
        request: Request,
        model: str = Form(...),
        file: UploadFile = File(media_type="application/octet-stream"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(0),
        kwargs: Optional[str] = Form(None),
    ) -> Response:
        form = await request.form()
        timestamp_granularities = form.get("timestamp_granularities[]")
        if timestamp_granularities:
            timestamp_granularities = [timestamp_granularities]
        model_uid = model
        try:
            model_ref = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            if kwargs is not None:
                parsed_kwargs = json.loads(kwargs)
            else:
                parsed_kwargs = {}
            translation = await model_ref.translations(
                audio=await file.read(),
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timestamp_granularities=timestamp_granularities,
                **parsed_kwargs,
            )
            return Response(content=translation, media_type="application/json")
        except Exception as e:
            e = await self._get_model_last_error(model_ref.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_speech(
        self,
        request: Request,
        prompt_speech: Optional[UploadFile] = File(
            None, media_type="application/octet-stream"
        ),
        prompt_latent: Optional[UploadFile] = File(
            None, media_type="application/octet-stream"
        ),
    ) -> Response:
        if prompt_speech or prompt_latent:
            f = await request.form()
        else:
            f = await request.json()
        body = SpeechRequest.parse_obj(f)
        model_uid = body.model
        try:
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            if body.kwargs is not None:
                parsed_kwargs = json.loads(body.kwargs)
            else:
                parsed_kwargs = {}
            if prompt_speech is not None:
                parsed_kwargs["prompt_speech"] = await prompt_speech.read()
            if prompt_latent is not None:
                parsed_kwargs["prompt_latent"] = await prompt_latent.read()
            out = await model.speech(
                input=body.input,
                voice=body.voice,
                response_format=body.response_format,
                speed=body.speed,
                stream=body.stream,
                **parsed_kwargs,
            )
            if body.stream:

                async def stream_results():
                    try:
                        async for item in out:
                            yield item
                    finally:
                        await model.decrease_serve_count()

                return EventSourceResponse(
                    media_type="application/octet-stream",
                    content=stream_results(),
                    ping=XINFERENCE_SSE_PING_ATTEMPTS_SECONDS,
                )
            else:
                return Response(media_type="application/octet-stream", content=out)
        except Exception as e:
            e = await self._get_model_last_error(model.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_progress(self, request_id: str) -> JSONResponse:
        try:
            supervisor_ref = await self._get_supervisor_ref()
            result = {"progress": await supervisor_ref.get_progress(request_id)}
            return JSONResponse(content=result)
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_images(self, request: Request) -> Response:
        body = TextToImageRequest.parse_obj(await request.json())
        model_uid = body.model
        try:
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        request_id = None
        try:
            kwargs = json.loads(body.kwargs) if body.kwargs else {}
            request_id = kwargs.get("request_id")
            self._add_running_task(request_id)
            image_list = await model.text_to_image(
                prompt=body.prompt,
                n=body.n,
                size=body.size,
                response_format=body.response_format,
                **kwargs,
            )
            return Response(content=image_list, media_type="application/json")
        except asyncio.CancelledError:
            err_str = f"The request has been cancelled: {request_id}"
            logger.error(err_str)
            await self._report_error_event(model_uid, err_str)
            raise HTTPException(status_code=409, detail=err_str)
        except Exception as e:
            e = await self._get_model_last_error(model.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def sdapi_options(self, request: Request) -> Response:
        body = SDAPIOptionsRequest.parse_obj(await request.json())
        model_uid = body.sd_model_checkpoint

        try:
            if not model_uid:
                raise ValueError("Unknown model")
            await (await self._get_supervisor_ref()).get_model(model_uid)
            return Response()
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

    async def sdapi_sd_models(self, request: Request) -> Response:
        try:
            models = await (await self._get_supervisor_ref()).list_models()
            sd_models = []
            for model_name, info in models.items():
                if info["model_type"] != "image":
                    continue
                sd_models.append({"model_name": model_name, "config": None})
            return JSONResponse(content=sd_models)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def sdapi_samplers(self, request: Request) -> Response:
        try:
            from ..model.image.stable_diffusion.core import SAMPLING_METHODS

            samplers = [
                {"name": sample_method, "alias": [], "options": {}}
                for sample_method in SAMPLING_METHODS
            ]
            return JSONResponse(content=samplers)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def sdapi_txt2img(self, request: Request) -> Response:
        body = SDAPITxt2imgRequst.parse_obj(await request.json())
        model_uid = body.model or body.override_settings.get("sd_model_checkpoint")

        try:
            if not model_uid:
                raise ValueError("Unknown model")
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            kwargs = dict(body)
            kwargs.update(json.loads(body.kwargs) if body.kwargs else {})
            image_list = await model.txt2img(
                **kwargs,
            )
            return Response(content=image_list, media_type="application/json")
        except Exception as e:
            e = await self._get_model_last_error(model.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def sdapi_img2img(self, request: Request) -> Response:
        body = SDAPIImg2imgRequst.parse_obj(await request.json())
        model_uid = body.model or body.override_settings.get("sd_model_checkpoint")

        try:
            if not model_uid:
                raise ValueError("Unknown model")
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            kwargs = dict(body)
            kwargs.update(json.loads(body.kwargs) if body.kwargs else {})
            image_list = await model.img2img(
                **kwargs,
            )
            return Response(content=image_list, media_type="application/json")
        except Exception as e:
            e = await self._get_model_last_error(model.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_variations(
        self,
        model: str = Form(...),
        image: List[UploadFile] = File(media_type="application/octet-stream"),
        prompt: Optional[Union[str, List[str]]] = Form(None),
        negative_prompt: Optional[Union[str, List[str]]] = Form(None),
        n: Optional[int] = Form(1),
        response_format: Optional[str] = Form("url"),
        size: Optional[str] = Form(None),
        kwargs: Optional[str] = Form(None),
    ) -> Response:
        model_uid = model
        try:
            model_ref = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        request_id = None
        try:
            if kwargs is not None:
                parsed_kwargs = json.loads(kwargs)
            else:
                parsed_kwargs = {}
            request_id = parsed_kwargs.get("request_id")
            self._add_running_task(request_id)

            # Handle single image or multiple images
            if len(image) == 1:
                # Single image
                image_data = Image.open(image[0].file)
            else:
                # Multiple images - convert to list of PIL Images
                image_data = [Image.open(img.file) for img in image]

            image_list = await model_ref.image_to_image(
                image=image_data,
                prompt=prompt,
                negative_prompt=negative_prompt,
                n=n,
                size=size,
                response_format=response_format,
                **parsed_kwargs,
            )
            return Response(content=image_list, media_type="application/json")
        except asyncio.CancelledError:
            err_str = f"The request has been cancelled: {request_id}"
            logger.error(err_str)
            await self._report_error_event(model_uid, err_str)
            raise HTTPException(status_code=409, detail=err_str)
        except Exception as e:
            e = await self._get_model_last_error(model_ref.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_inpainting(
        self,
        model: str = Form(...),
        image: UploadFile = File(media_type="application/octet-stream"),
        mask_image: UploadFile = File(media_type="application/octet-stream"),
        prompt: Optional[Union[str, List[str]]] = Form(None),
        negative_prompt: Optional[Union[str, List[str]]] = Form(None),
        n: Optional[int] = Form(1),
        response_format: Optional[str] = Form("url"),
        size: Optional[str] = Form(None),
        kwargs: Optional[str] = Form(None),
    ) -> Response:
        model_uid = model
        try:
            model_ref = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        request_id = None
        try:
            if kwargs is not None:
                parsed_kwargs = json.loads(kwargs)
            else:
                parsed_kwargs = {}
            request_id = parsed_kwargs.get("request_id")
            self._add_running_task(request_id)
            im = Image.open(image.file)
            mask_im = Image.open(mask_image.file)
            if not size:
                w, h = im.size
                size = f"{w}*{h}"
            image_list = await model_ref.inpainting(
                image=im,
                mask_image=mask_im,
                prompt=prompt,
                negative_prompt=negative_prompt,
                n=n,
                size=size,
                response_format=response_format,
                **parsed_kwargs,
            )
            return Response(content=image_list, media_type="application/json")
        except asyncio.CancelledError:
            err_str = f"The request has been cancelled: {request_id}"
            logger.error(err_str)
            await self._report_error_event(model_uid, err_str)
            raise HTTPException(status_code=409, detail=err_str)
        except Exception as e:
            e = await self._get_model_last_error(model_ref.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_ocr(
        self,
        model: str = Form(...),
        image: UploadFile = File(media_type="application/octet-stream"),
        kwargs: Optional[str] = Form(None),
    ) -> Response:
        model_uid = model
        try:
            model_ref = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        request_id = None
        try:
            if kwargs is not None:
                parsed_kwargs = json.loads(kwargs)
            else:
                parsed_kwargs = {}
            request_id = parsed_kwargs.get("request_id")
            self._add_running_task(request_id)
            im = Image.open(image.file)
            text = await model_ref.ocr(
                image=im,
                **parsed_kwargs,
            )
            return Response(content=text, media_type="text/plain")
        except asyncio.CancelledError:
            err_str = f"The request has been cancelled: {request_id}"
            logger.error(err_str)
            await self._report_error_event(model_uid, err_str)
            raise HTTPException(status_code=409, detail=err_str)
        except Exception as e:
            e = await self._get_model_last_error(model_ref.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_image_edits(
        self,
        request: Request,
        prompt: str = Form(...),
        mask: Optional[UploadFile] = File(None, media_type="application/octet-stream"),
        model: Optional[str] = Form(None),
        n: Optional[int] = Form(1),
        size: Optional[str] = Form("original"),
        response_format: Optional[str] = Form("url"),
        stream: Optional[bool] = Form(False),
    ) -> Response:
        """OpenAI-compatible image edit endpoint."""
        import io

        # Parse multipart form data to handle files
        content_type = request.headers.get("content-type", "")

        if "multipart/form-data" in content_type:
            # Try manual multipart parsing for better duplicate field handling
            try:
                image_files, manual_mask = await self._parse_multipart_manual(request)
                # Use manually parsed mask if available, otherwise keep the original
                if manual_mask is not None:
                    mask = manual_mask
            except Exception as e:
                logger.error(f"Manual parsing failed, falling back to FastAPI: {e}")
                # Fallback to FastAPI form parsing
                form = await request.form()
                multipart_files: dict[str, list] = {}
                for key, value in form.items():
                    if hasattr(value, "filename") and value.filename:
                        if key not in multipart_files:
                            multipart_files[key] = []
                        multipart_files[key].append(value)

                image_files = multipart_files.get("image", [])
                if not image_files:
                    image_files = multipart_files.get("image[]", [])
                if not image_files:
                    image_files = multipart_files.get("images", [])

        else:
            # Fallback to FastAPI form parsing
            form = await request.form()
            fallback_files: dict[str, list] = {}
            for key, value in form.items():
                if hasattr(value, "filename") and value.filename:
                    if key not in fallback_files:
                        fallback_files[key] = []
                    fallback_files[key].append(value)

            image_files = fallback_files.get("image", [])
            if not image_files:
                image_files = fallback_files.get("image[]", [])
            if not image_files:
                image_files = fallback_files.get("images", [])

        all_file_keys = []
        if "multipart/form-data" in content_type:
            all_file_keys = [f"image[] (x{len(image_files)})"] if image_files else []
        else:
            # Fallback to FastAPI form parsing
            form = await request.form()
            debug_files: dict[str, list] = {}
            for key, value in form.items():
                if hasattr(value, "filename") and value.filename:
                    if key not in debug_files:
                        debug_files[key] = []
                    debug_files[key].append(value)

            # Get image files
            image_files = debug_files.get("image", [])
            if not image_files:
                image_files = debug_files.get("image[]", [])
            if not image_files:
                image_files = debug_files.get("images", [])

        logger.info(f"Total image files found: {len(image_files)}")

        if not image_files:
            # Debug: log all received file fields
            logger.warning(
                f"No image files found. Available file fields: {all_file_keys}"
            )
            raise HTTPException(
                status_code=400, detail="At least one image file is required"
            )

        # Validate response format
        if response_format not in ["url", "b64_json"]:
            raise HTTPException(
                status_code=400, detail="response_format must be 'url' or 'b64_json'"
            )

        # Get default model if not specified
        if not model:
            try:
                models = await (await self._get_supervisor_ref()).list_models()
                image_models = [
                    name
                    for name, info in models.items()
                    if info["model_type"] == "image"
                    and info.get("model_ability", [])
                    and (
                        "image2image" in info["model_ability"]
                        or "inpainting" in info["model_ability"]
                    )
                ]
                if not image_models:
                    raise HTTPException(
                        status_code=400, detail="No available image models found"
                    )
                model = image_models[0]
            except Exception as e:
                logger.error(f"Failed to get available models: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500, detail="Failed to get available models"
                )

        model_uid = model
        try:
            model_ref = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        request_id = None
        try:
            self._add_running_task(request_id)

            # Read and process all images (needed for both streaming and non-streaming)
            images = []
            original_filenames = []
            for i, img in enumerate(image_files):
                # Store original filename before processing
                original_filename = (
                    img.filename if hasattr(img, "filename") else f"upload_{i}"
                )
                original_filenames.append(original_filename)

                image_content = await img.read()
                image_file = io.BytesIO(image_content)
                pil_image = Image.open(image_file)

                # Debug: save the received image for inspection
                debug_filename = f"/tmp/received_image_{i}_{pil_image.mode}_{pil_image.size[0]}x{pil_image.size[1]}.png"
                pil_image.save(debug_filename)
                logger.info(f"Saved received image {i} to {debug_filename}")

                # Convert to RGB format to avoid channel mismatch errors
                if pil_image.mode == "RGBA":
                    logger.info(f"Converting RGBA image {i} to RGB")
                    # Create white background for RGBA images
                    background = Image.new("RGB", pil_image.size, (255, 255, 255))
                    background.paste(pil_image, mask=pil_image.split()[3])
                    pil_image = background
                elif pil_image.mode != "RGB":
                    logger.info(f"Converting {pil_image.mode} image {i} to RGB")
                    pil_image = pil_image.convert("RGB")

                # Debug: save the converted image
                converted_filename = f"/tmp/converted_image_{i}_RGB_{pil_image.size[0]}x{pil_image.size[1]}.png"
                pil_image.save(converted_filename)
                logger.info(f"Saved converted image {i} to {converted_filename}")

                images.append(pil_image)

            # Debug: log image summary
            logger.info(f"Processing {len(images)} images:")
            for i, img in enumerate(images):
                logger.info(
                    f"  Image {i}: mode={img.mode}, size={img.size}, filename={original_filenames[i]}"
                )

            # Handle streaming if requested
            if stream:
                return EventSourceResponse(
                    self._stream_image_edit(
                        model_ref,
                        images,  # Pass processed images instead of raw files
                        mask,
                        prompt,
                        (
                            size.replace("x", "*") if size else ""
                        ),  # Convert size format for streaming
                        response_format,
                        n,
                    )
                )

            # Use the first image as primary, others as reference
            primary_image = images[0]
            reference_images = images[1:] if len(images) > 1 else []

            # Prepare model parameters
            # If size is "original", use empty string to let model determine original dimensions
            if size == "original":
                model_size = ""
            else:
                model_size = size.replace("x", "*") if size else ""

            model_params = {
                "prompt": prompt,
                "n": n or 1,
                "size": model_size,
                "response_format": response_format,
                "denoising_strength": 0.75,  # Default strength for image editing
                "reference_images": reference_images,  # Pass reference images
                "negative_prompt": " ",  # Space instead of empty string to prevent filtering
            }

            # Generate the image
            if mask:
                # Use inpainting for masked edits
                mask_content = await mask.read()
                mask_image = Image.open(io.BytesIO(mask_content))
                result = await model_ref.inpainting(
                    image=primary_image,
                    mask_image=mask_image,
                    **model_params,
                )
            else:
                # Use image-to-image for general edits
                result = await model_ref.image_to_image(
                    image=primary_image, **model_params
                )

            # Return the result directly (should be ImageList format)
            return Response(content=result, media_type="application/json")

        except asyncio.CancelledError:
            err_str = f"The request has been cancelled: {request_id or 'unknown'}"
            logger.error(err_str)
            await self._report_error_event(model_uid, err_str)
            raise HTTPException(status_code=409, detail=err_str)
        except Exception as e:
            e = await self._get_model_last_error(model_ref.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def _parse_multipart_manual(self, request: Request):
        """Manually parse multipart form data to handle duplicate field names"""
        import io

        class FileWrapper:
            """Wrapper for BytesIO to add filename and content_type attributes"""

            def __init__(self, data, filename, content_type="application/octet-stream"):
                self._file = io.BytesIO(data)
                self.filename = filename
                self.content_type = content_type

            def read(self, *args, **kwargs):
                return self._file.read(*args, **kwargs)

            def seek(self, *args, **kwargs):
                return self._file.seek(*args, **kwargs)

            def tell(self, *args, **kwargs):
                return self._file.tell(*args, **kwargs)

        from multipart.multipart import parse_options_header

        content_type = request.headers.get("content-type", "")
        if not content_type:
            return [], None

        # Parse content type and boundary
        content_type, options = parse_options_header(content_type.encode("utf-8"))
        if content_type != b"multipart/form-data":
            return [], None

        boundary = options.get(b"boundary")
        if not boundary:
            return [], None

        # Get the raw body
        body = await request.body()

        # Parse multipart data manually
        image_files = []
        mask_file = None
        try:
            # Import multipart parser
            from multipart.multipart import MultipartParser

            # Parse the multipart data
            parser = MultipartParser(
                io.BytesIO(body),
                boundary.decode("utf-8") if isinstance(boundary, bytes) else boundary,
            )

            for part in parser:
                # Check if this part is an image file
                field_name = part.name
                filename = part.filename or ""

                # Look for image fields with different naming conventions
                if field_name in ["image", "image[]", "images"] and filename:
                    # Create a file-like object from the part data
                    file_obj = FileWrapper(
                        part.data,
                        filename,
                        part.content_type or "application/octet-stream",
                    )
                    image_files.append(file_obj)
                elif field_name == "mask" and filename:
                    # Handle mask file
                    mask_file = FileWrapper(
                        part.data,
                        filename,
                        part.content_type or "application/octet-stream",
                    )
                    logger.info(f"Manual multipart parsing found mask file: {filename}")

            logger.info(
                f"Manual multipart parsing found {len(image_files)} image files and mask: {mask_file is not None}"
            )

        except Exception as e:
            logger.error(f"Manual multipart parsing failed: {e}")
            # Return empty list to trigger fallback
            return [], None

        return image_files, mask_file

    async def _stream_image_edit(
        self, model_ref, images, mask, prompt, size, response_format, n
    ):
        """Stream image editing progress and results"""
        import io
        import json
        from datetime import datetime

        try:
            # Send start event
            yield {
                "event": "start",
                "data": json.dumps(
                    {
                        "type": "image_edit_started",
                        "timestamp": datetime.now().isoformat(),
                        "prompt": prompt,
                        "image_count": len(images),
                    }
                ),
            }

            # Images are already processed in the main method, just use them directly
            image_objects = images
            logger.info(f"Streaming: Using {len(image_objects)} pre-processed images")

            # Debug: log streaming image summary
            logger.info(f"Streaming: Processing {len(image_objects)} images:")
            for i, img in enumerate(image_objects):
                logger.info(f"  Streaming Image {i}: mode={img.mode}, size={img.size}")

            # Use the first image as primary, others as reference
            primary_image = image_objects[0]
            reference_images = image_objects[1:] if len(image_objects) > 1 else []

            # Send processing event
            yield {
                "event": "processing",
                "data": json.dumps(
                    {
                        "type": "images_loaded",
                        "timestamp": datetime.now().isoformat(),
                        "primary_image_size": primary_image.size,
                        "reference_images_count": len(reference_images),
                    }
                ),
            }

            # Prepare model parameters
            # If size is "original", use empty string to let model determine original dimensions
            if size == "original":
                model_size = ""
            else:
                model_size = size

            model_params = {
                "prompt": prompt,
                "n": n or 1,
                "size": model_size,
                "response_format": response_format,
                "denoising_strength": 0.75,
                "reference_images": reference_images,
                "negative_prompt": " ",  # Space instead of empty string to prevent filtering
            }

            # Generate the image
            if mask:
                mask_content = await mask.read()
                mask_image = Image.open(io.BytesIO(mask_content))
                yield {
                    "event": "processing",
                    "data": json.dumps(
                        {
                            "type": "mask_loaded",
                            "timestamp": datetime.now().isoformat(),
                            "mask_size": mask_image.size,
                        }
                    ),
                }
                result = await model_ref.inpainting(
                    image=primary_image,
                    mask_image=mask_image,
                    **model_params,
                )
            else:
                yield {
                    "event": "processing",
                    "data": json.dumps(
                        {
                            "type": "starting_generation",
                            "timestamp": datetime.now().isoformat(),
                        }
                    ),
                }
                result = await model_ref.image_to_image(
                    image=primary_image, **model_params
                )

            # Parse the result and send final event in OpenAI format
            result_data = json.loads(result)

            # Send completion event with OpenAI-compatible format
            yield {
                "event": "complete",
                "data": json.dumps(
                    result_data
                ),  # Direct send the result in OpenAI format
            }

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps(
                    {
                        "type": "image_edit_error",
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e),
                    }
                ),
            }

    async def create_flexible_infer(self, request: Request) -> Response:
        payload = await request.json()

        model_uid = payload.get("model")
        args = payload.get("args")

        exclude = {"model", "args"}
        kwargs = {key: value for key, value in payload.items() if key not in exclude}

        try:
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            result = await model.infer(*args, **kwargs)
            return Response(result, media_type="application/json")
        except Exception as e:
            e = await self._get_model_last_error(model.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_videos(self, request: Request) -> Response:
        body = TextToVideoRequest.parse_obj(await request.json())
        model_uid = body.model
        try:
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        request_id = None
        try:
            kwargs = json.loads(body.kwargs) if body.kwargs else {}
            request_id = kwargs.get("request_id")
            self._add_running_task(request_id)
            video_list = await model.text_to_video(
                prompt=body.prompt,
                n=body.n,
                **kwargs,
            )
            return Response(content=video_list, media_type="application/json")
        except asyncio.CancelledError:
            err_str = f"The request has been cancelled: {request_id or 'unknown'}"
            logger.error(err_str)
            await self._report_error_event(model_uid, err_str)
            raise HTTPException(status_code=409, detail=err_str)
        except Exception as e:
            e = await self._get_model_last_error(model.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_videos_from_images(
        self,
        model: str = Form(...),
        image: UploadFile = File(media_type="application/octet-stream"),
        prompt: Optional[Union[str, List[str]]] = Form(None),
        negative_prompt: Optional[Union[str, List[str]]] = Form(None),
        n: Optional[int] = Form(1),
        kwargs: Optional[str] = Form(None),
    ) -> Response:
        model_uid = model
        try:
            model_ref = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        request_id = None
        try:
            if kwargs is not None:
                parsed_kwargs = json.loads(kwargs)
            else:
                parsed_kwargs = {}
            request_id = parsed_kwargs.get("request_id")
            self._add_running_task(request_id)
            video_list = await model_ref.image_to_video(
                image=Image.open(image.file),
                prompt=prompt,
                negative_prompt=negative_prompt,
                n=n,
                **parsed_kwargs,
            )
            return Response(content=video_list, media_type="application/json")
        except asyncio.CancelledError:
            err_str = f"The request has been cancelled: {request_id or 'unknown'}"
            logger.error(err_str)
            await self._report_error_event(model_uid, err_str)
            raise HTTPException(status_code=409, detail=err_str)
        except Exception as e:
            e = await self._get_model_last_error(model_ref.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_videos_from_first_last_frame(
        self,
        model: str = Form(...),
        first_frame: UploadFile = File(media_type="application/octet-stream"),
        last_frame: UploadFile = File(media_type="application/octet-stream"),
        prompt: Optional[Union[str, List[str]]] = Form(None),
        negative_prompt: Optional[Union[str, List[str]]] = Form(None),
        n: Optional[int] = Form(1),
        kwargs: Optional[str] = Form(None),
    ) -> Response:
        model_uid = model
        try:
            model_ref = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        request_id = None
        try:
            if kwargs is not None:
                parsed_kwargs = json.loads(kwargs)
            else:
                parsed_kwargs = {}
            request_id = parsed_kwargs.get("request_id")
            self._add_running_task(request_id)
            video_list = await model_ref.flf_to_video(
                first_frame=Image.open(first_frame.file),
                last_frame=Image.open(last_frame.file),
                prompt=prompt,
                negative_prompt=negative_prompt,
                n=n,
                **parsed_kwargs,
            )
            return Response(content=video_list, media_type="application/json")
        except asyncio.CancelledError:
            err_str = f"The request has been cancelled: {request_id or 'unknown'}"
            logger.error(err_str)
            await self._report_error_event(model_uid, err_str)
            raise HTTPException(status_code=409, detail=err_str)
        except Exception as e:
            e = await self._get_model_last_error(model_ref.uid, e)
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            self.handle_request_limit_error(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_chat_completion(self, request: Request) -> Response:
        raw_body = await request.json()
        body = CreateChatCompletion.parse_obj(raw_body)
        exclude = {
            "prompt",
            "model",
            "n",
            "messages",
            "logit_bias",
            "logit_bias_type",
            "user",
            "max_completion_tokens",
        }

        raw_kwargs = {k: v for k, v in raw_body.items() if k not in exclude}
        kwargs = body.dict(exclude_unset=True, exclude=exclude)

        enable_thinking = raw_body.get("enable_thinking")
        if enable_thinking is None:
            extra_body = raw_body.get("extra_body")
            if isinstance(extra_body, dict):
                enable_thinking = extra_body.get("enable_thinking")
        if isinstance(enable_thinking, bool):
            raw_kwargs.pop("enable_thinking", None)
            chat_template_kwargs = raw_kwargs.get("chat_template_kwargs") or {}
            if isinstance(chat_template_kwargs, str):
                try:
                    chat_template_kwargs = json.loads(chat_template_kwargs)
                except json.JSONDecodeError:
                    chat_template_kwargs = {}
            if not isinstance(chat_template_kwargs, dict):
                chat_template_kwargs = {}
            chat_template_kwargs = dict(chat_template_kwargs)
            chat_template_kwargs["enable_thinking"] = enable_thinking
            chat_template_kwargs["thinking"] = enable_thinking
            raw_kwargs["chat_template_kwargs"] = chat_template_kwargs
            kwargs["chat_template_kwargs"] = chat_template_kwargs

        # guided_decoding params
        kwargs.update(self.extract_guided_params(raw_body=raw_body))

        # TODO: Decide if this default value override is necessary #1061
        if body.max_tokens is None:
            kwargs["max_tokens"] = max_tokens_field.default

        if body.max_completion_tokens is not None:
            kwargs["max_tokens"] = body.max_completion_tokens

        if body.logit_bias is not None:
            raise HTTPException(status_code=501, detail="Not implemented")

        messages = body.messages and list(body.messages) or None

        if not messages or messages[-1].get("role") not in ["user", "system", "tool"]:
            raise HTTPException(
                status_code=400, detail="Invalid input. Please specify the prompt."
            )

        has_tool_message = messages[-1].get("role") == "tool"
        model_uid = body.model

        try:
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        try:
            desc = await (await self._get_supervisor_ref()).describe_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            await self._report_error_event(model_uid, str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            await self._report_error_event(model_uid, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        from ..model.llm.utils import (
            GLM4_TOOL_CALL_FAMILY,
            QWEN_TOOL_CALL_FAMILY,
            TOOL_CALL_FAMILY,
        )

        model_family = desc.get("model_family", "")

        if model_family not in TOOL_CALL_FAMILY:
            if body.tools:
                raise HTTPException(
                    status_code=400,
                    detail=f"Only {TOOL_CALL_FAMILY} support tool calls",
                )
            if has_tool_message:
                raise HTTPException(
                    status_code=400,
                    detail=f"Only {TOOL_CALL_FAMILY} support tool messages",
                )
        if body.tools and body.stream:
            is_vllm = await model.is_vllm_backend()
            is_sglang = await model.is_sglang_backend()
            if not (
                ((is_vllm or is_sglang) and model_family in QWEN_TOOL_CALL_FAMILY)
                or (not is_vllm and model_family in GLM4_TOOL_CALL_FAMILY)
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Streaming support for tool calls is available only when using "
                    "Qwen models with vLLM backend or GLM4-chat models without vLLM backend.",
                )
        if "skip_special_tokens" in raw_kwargs and await model.is_vllm_backend():
            kwargs["skip_special_tokens"] = raw_kwargs["skip_special_tokens"]
        if body.stream:

            async def stream_results():
                iterator = None
                try:
                    try:
                        iterator = await model.chat(
                            messages,
                            kwargs,
                            raw_params=raw_kwargs,
                        )
                    except RuntimeError as re:
                        await self._report_error_event(model_uid, str(re))
                        self.handle_request_limit_error(re)
                    async for item in iterator:
                        yield item
                    yield "[DONE]"
                # Note that asyncio.CancelledError does not inherit from Exception.
                # When the user uses ctrl+c to cancel the streaming chat, asyncio.CancelledError would be triggered.
                # See https://github.com/sysid/sse-starlette/blob/main/examples/example.py#L48
                except asyncio.CancelledError:
                    logger.info(
                        f"Disconnected from client (via refresh/close) {request.client} during chat."
                    )
                    # See https://github.com/sysid/sse-starlette/blob/main/examples/error_handling.py#L13
                    # Use return to stop the generator from continuing.
                    # TODO: Cannot yield here. Yield here would leads to error for the next streaming request.
                    return
                except Exception as ex:
                    ex = await self._get_model_last_error(model.uid, ex)
                    logger.exception("Chat completion stream got an error: %s", ex)
                    await self._report_error_event(model_uid, str(ex))
                    # https://github.com/openai/openai-python/blob/e0aafc6c1a45334ac889fe3e54957d309c3af93f/src/openai/_streaming.py#L107
                    yield dict(data=json.dumps({"error": str(ex)}))
                    return
                finally:
                    await model.decrease_serve_count()

            return EventSourceResponse(
                stream_results(), ping=XINFERENCE_SSE_PING_ATTEMPTS_SECONDS
            )
        else:
            try:
                data = await model.chat(
                    messages,
                    kwargs,
                    raw_params=raw_kwargs,
                )
                return Response(content=data, media_type="application/json")
            except Exception as e:
                e = await self._get_model_last_error(model.uid, e)
                logger.error(e, exc_info=True)
                await self._report_error_event(model_uid, str(e))
                self.handle_request_limit_error(e)
                raise HTTPException(status_code=500, detail=str(e))

    async def query_engines_by_model_name(
        self, request: Request, model_name: str, model_type: Optional[str] = None
    ) -> JSONResponse:
        try:
            model_type = model_type or request.path_params.get("model_type", "LLM")
            enable_virtual_env = request.query_params.get("enable_virtual_env")
            if enable_virtual_env is not None:
                enable_virtual_env = enable_virtual_env.lower() in ("1", "true", "yes")
            content = await (
                await self._get_supervisor_ref()
            ).query_engines_by_model_name(
                model_name,
                model_type=model_type,
                enable_virtual_env=enable_virtual_env,
            )
            return JSONResponse(content=content)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def register_model(self, model_type: str, request: Request) -> JSONResponse:
        body = RegisterModelRequest.parse_obj(await request.json())
        model = body.model
        worker_ip = body.worker_ip
        persist = body.persist

        try:
            await (await self._get_supervisor_ref()).register_model(
                model_type, model, persist, worker_ip
            )
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content=None)

    async def unregister_model(self, model_type: str, model_name: str) -> JSONResponse:
        try:
            await (await self._get_supervisor_ref()).unregister_model(
                model_type, model_name
            )
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content=None)

    async def update_model_type(self, request: Request) -> JSONResponse:
        try:
            # Parse request
            raw_json = await request.json()
            body = UpdateModelRequest.parse_obj(raw_json)
            model_type = body.model_type

            # Get supervisor reference
            supervisor_ref = await self._get_supervisor_ref()

            # Call supervisor with model_type
            await supervisor_ref.update_model_type(model_type)

        except ValueError as re:
            logger.error(f"ValueError in update_model_type API: {re}", exc_info=True)
            logger.error(f"ValueError details: {type(re).__name__}: {re}")
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(
                f"Unexpected error in update_model_type API: {e}", exc_info=True
            )
            logger.error(f"Error details: {type(e).__name__}: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

        response_data = {
            "data": {
                "model_type": model_type,
                "message": f"Successfully updated model type: {model_type}",
            }
        }

        return JSONResponse(content=response_data)

    async def list_model_registrations(
        self, model_type: str, detailed: bool = Query(False)
    ) -> JSONResponse:
        try:
            data = await (await self._get_supervisor_ref()).list_model_registrations(
                model_type, detailed=detailed
            )
            # Remove duplicate model names.
            model_names = set()
            final_data = []
            for item in data:
                if item["model_name"] not in model_names:
                    model_names.add(item["model_name"])
                    final_data.append(item)
            return JSONResponse(content=final_data)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_model_registrations(
        self, model_type: str, model_name: str
    ) -> JSONResponse:
        try:
            data = await (await self._get_supervisor_ref()).get_model_registration(
                model_type, model_name
            )
            return JSONResponse(content=data)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def list_cached_models(
        self, model_name: str = Query(None), worker_ip: str = Query(None)
    ) -> JSONResponse:
        try:
            data = await (await self._get_supervisor_ref()).list_cached_models(
                model_name, worker_ip
            )
            resp = {
                "list": data,
            }
            return JSONResponse(content=resp)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_model_events(self, model_uid: str) -> JSONResponse:
        try:
            event_collector_ref = await self._get_event_collector_ref()
            events = await event_collector_ref.get_model_events(model_uid)
            return JSONResponse(content=events)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def abort_request(
        self, request: Request, model_uid: str, request_id: str
    ) -> JSONResponse:
        try:
            payload = await request.json()
            block_duration = payload.get(
                "block_duration", XINFERENCE_DEFAULT_CANCEL_BLOCK_DURATION
            )
            logger.info(
                "Abort request with model uid: %s, request id: %s, block duration: %s",
                model_uid,
                request_id,
                block_duration,
            )
            supervisor_ref = await self._get_supervisor_ref()
            res = await supervisor_ref.abort_request(
                model_uid, request_id, block_duration
            )
            self._cancel_running_task(request_id, block_duration)
            return JSONResponse(content=res)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def list_vllm_supported_model_families(self) -> JSONResponse:
        try:
            from ..model.llm.vllm.core import (
                VLLM_SUPPORTED_CHAT_MODELS,
                VLLM_SUPPORTED_MODELS,
            )

            data = {
                "chat": VLLM_SUPPORTED_CHAT_MODELS,
                "generate": VLLM_SUPPORTED_MODELS,
            }
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_cluster_device_info(
        self, detailed: bool = Query(False)
    ) -> JSONResponse:
        try:
            data = await (await self._get_supervisor_ref()).get_cluster_device_info(
                detailed=detailed
            )
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_cluster_version(self) -> JSONResponse:
        try:
            data = get_versions()
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def list_model_files(
        self, model_version: str = Query(None), worker_ip: str = Query(None)
    ) -> JSONResponse:
        try:
            data = await (await self._get_supervisor_ref()).list_deletable_models(
                model_version, worker_ip
            )
            response = {
                "model_version": model_version,
                "worker_ip": worker_ip,
                "paths": data,
            }
            return JSONResponse(content=response)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def confirm_and_remove_model(
        self, model_version: str = Query(None), worker_ip: str = Query(None)
    ) -> JSONResponse:
        try:
            res = await (await self._get_supervisor_ref()).confirm_and_remove_model(
                model_version=model_version, worker_ip=worker_ip
            )
            return JSONResponse(content={"result": res})
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def list_virtual_envs(
        self,
        model_name: str = Query(None),
        model_engine: str = Query(None),
        worker_ip: str = Query(None),
    ) -> JSONResponse:
        """List all virtual environments or filter by model name."""
        try:
            data = await (await self._get_supervisor_ref()).list_virtual_envs(
                model_name, model_engine, worker_ip
            )
            resp = {
                "list": data,
            }
            return JSONResponse(content=resp)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def remove_virtual_env(
        self,
        model_name: str = Query(None),
        model_engine: str = Query(None),
        python_version: str = Query(None),
        worker_ip: str = Query(None),
    ) -> JSONResponse:
        """Remove a virtual environment for a specific model."""
        if not model_name:
            raise HTTPException(
                status_code=400, detail="model_name parameter is required"
            )

        try:
            res = await (await self._get_supervisor_ref()).remove_virtual_env(
                model_name=model_name,
                model_engine=model_engine,
                python_version=python_version,
                worker_ip=worker_ip,
            )
            return JSONResponse(content={"result": res})
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_workers_info(self) -> JSONResponse:
        try:
            res = await (await self._get_supervisor_ref()).get_workers_info()
            return JSONResponse(content=res)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_supervisor_info(self) -> JSONResponse:
        try:
            res = await (await self._get_supervisor_ref()).get_supervisor_info()
            return res
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def abort_cluster(self) -> JSONResponse:
        import os
        import signal

        try:
            res = await (await self._get_supervisor_ref()).abort_cluster()
            os.kill(os.getpid(), signal.SIGINT)
            return JSONResponse(content={"result": res})
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    def extract_guided_params(raw_body: dict) -> dict:
        kwargs = {}
        raw_extra_body: dict = raw_body.get("extra_body")  # type: ignore
        if raw_body.get("guided_json"):
            kwargs["guided_json"] = raw_body.get("guided_json")
        if raw_body.get("guided_regex") is not None:
            kwargs["guided_regex"] = raw_body.get("guided_regex")
        if raw_body.get("guided_choice") is not None:
            kwargs["guided_choice"] = raw_body.get("guided_choice")
        if raw_body.get("guided_grammar") is not None:
            kwargs["guided_grammar"] = raw_body.get("guided_grammar")
        if raw_body.get("guided_json_object") is not None:
            kwargs["guided_json_object"] = raw_body.get("guided_json_object")
        if raw_body.get("guided_decoding_backend") is not None:
            kwargs["guided_decoding_backend"] = raw_body.get("guided_decoding_backend")
        if raw_body.get("guided_whitespace_pattern") is not None:
            kwargs["guided_whitespace_pattern"] = raw_body.get(
                "guided_whitespace_pattern"
            )
        # Parse OpenAI extra_body
        if raw_extra_body is not None:
            if raw_extra_body.get("guided_json"):
                kwargs["guided_json"] = raw_extra_body.get("guided_json")
            if raw_extra_body.get("guided_regex") is not None:
                kwargs["guided_regex"] = raw_extra_body.get("guided_regex")
            if raw_extra_body.get("guided_choice") is not None:
                kwargs["guided_choice"] = raw_extra_body.get("guided_choice")
            if raw_extra_body.get("guided_grammar") is not None:
                kwargs["guided_grammar"] = raw_extra_body.get("guided_grammar")
            if raw_extra_body.get("guided_json_object") is not None:
                kwargs["guided_json_object"] = raw_extra_body.get("guided_json_object")
            if raw_extra_body.get("guided_decoding_backend") is not None:
                kwargs["guided_decoding_backend"] = raw_extra_body.get(
                    "guided_decoding_backend"
                )
            if raw_extra_body.get("guided_whitespace_pattern") is not None:
                kwargs["guided_whitespace_pattern"] = raw_extra_body.get(
                    "guided_whitespace_pattern"
                )
            if raw_extra_body.get("platform") is not None:
                kwargs["platform"] = raw_extra_body.get("platform")
            if raw_extra_body.get("format") is not None:
                kwargs["format"] = raw_extra_body.get("format")

        return kwargs

    def _convert_openai_to_anthropic(self, openai_response: dict, model: str) -> dict:
        """
        Convert OpenAI response format to Anthropic response format.

        Args:
            openai_response: OpenAI format response
            model: Model name

        Returns:
            Anthropic format response
        """

        # Extract content and tool calls from OpenAI response
        content_blocks = []
        stop_reason = "stop"

        if "choices" in openai_response and len(openai_response["choices"]) > 0:
            choice = openai_response["choices"][0]
            message = choice.get("message", {})

            # Handle content text
            content = message.get("content", "")
            if content:
                if isinstance(content, str):
                    # If content is a string, use it directly
                    content_blocks.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    # If content is a list, extract text from each content block
                    for content_block in content:
                        if isinstance(content_block, dict):
                            if content_block.get("type") == "text":
                                text = content_block.get("text", "")
                                if text:
                                    content_blocks.append(
                                        {"type": "text", "text": text}
                                    )
                            elif "text" in content_block:
                                # Handle different content block format
                                text = content_block.get("text", "")
                                if text:
                                    content_blocks.append(
                                        {"type": "text", "text": text}
                                    )

            # Handle tool calls
            tool_calls = message.get("tool_calls", [])
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                arguments = function.get("arguments", "{}")
                try:
                    input_data = json.loads(arguments)
                except json.JSONDecodeError:
                    input_data = {}
                tool_use_block = {
                    "type": "tool_use",
                    "cache_control": {"type": "ephemeral"},
                    "id": tool_call.get("id", str(uuid.uuid4())),
                    "name": function.get("name", ""),
                    "input": input_data,
                }
                content_blocks.append(tool_use_block)

            # Set stop reason based on finish reason
            finish_reason = choice.get("finish_reason", "stop")
            if finish_reason == "tool_calls":
                stop_reason = "tool_use"

        # Build Anthropic response
        anthropic_response = {
            "id": str(uuid.uuid4()),
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": openai_response.get("usage", {}).get(
                    "prompt_tokens", 0
                ),
                "output_tokens": openai_response.get("usage", {}).get(
                    "completion_tokens", 0
                ),
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }

        return anthropic_response


def run(
    supervisor_address: str,
    host: str,
    port: int,
    logging_conf: Optional[dict] = None,
    auth_config_file: Optional[str] = None,
):
    logger.info(f"Starting Xinference at endpoint: http://{host}:{port}")
    try:
        api = RESTfulAPI(
            supervisor_address=supervisor_address,
            host=host,
            port=port,
            auth_config_file=auth_config_file,
        )
        api.serve(logging_conf=logging_conf)
    except SystemExit:
        logger.warning("Failed to create socket with port %d", port)
        # compare the reference to differentiate between the cases where the user specify the
        # default port and the user does not specify the port.
        if port is XINFERENCE_DEFAULT_ENDPOINT_PORT:
            port = get_next_port()
            logger.info(f"Found available port: {port}")
            logger.info(f"Starting Xinference at endpoint: http://{host}:{port}")
            api = RESTfulAPI(
                supervisor_address=supervisor_address,
                host=host,
                port=port,
                auth_config_file=auth_config_file,
            )
            api.serve(logging_conf=logging_conf)
        else:
            raise


def run_in_subprocess(
    supervisor_address: str,
    host: str,
    port: int,
    logging_conf: Optional[dict] = None,
    auth_config_file: Optional[str] = None,
) -> multiprocessing.Process:
    p = multiprocessing.Process(
        target=run,
        args=(supervisor_address, host, port, logging_conf, auth_config_file),
    )
    p.daemon = True
    p.start()
    return p
