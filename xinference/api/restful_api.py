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

import asyncio
import inspect
import json
import logging
import multiprocessing
import os
import pprint
import sys
import warnings
from typing import Any, Dict, List, Literal, Optional, Union

import gradio as gr
import xoscar as xo
from fastapi import (
    APIRouter,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.responses import JSONResponse as StarletteJSONResponse
from starlette.responses import RedirectResponse
from typing_extensions import TypedDict
from uvicorn import Config, Server
from xoscar.utils import get_next_port

from ..constants import XINFERENCE_DEFAULT_ENDPOINT_PORT
from ..core.supervisor import SupervisorActor
from ..core.utils import json_dumps
from ..fields import (
    frequency_penalty_field,
    max_tokens_field,
    mirostat_eta_field,
    mirostat_mode_field,
    mirostat_tau_field,
    presence_penalty_field,
    repeat_penalty_field,
    stop_field,
    stream_field,
    temperature_field,
    top_k_field,
    top_p_field,
)
from ..types import (
    ChatCompletion,
    Completion,
    CreateChatCompletion,
    CreateCompletion,
    ImageList,
)

logger = logging.getLogger(__name__)


class JSONResponse(StarletteJSONResponse):
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
    input: Union[str, List[str]] = Field(description="The input to embed.")
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
    max_chunks_per_doc: Optional[int] = None


class TextToImageRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]] = Field(description="The input to embed.")
    n: Optional[int] = 1
    response_format: Optional[str] = "url"
    size: Optional[str] = "1024*1024"
    user: Optional[str] = None


class RegisterModelRequest(BaseModel):
    model: str
    persist: bool


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


class RESTfulAPI:
    def __init__(self, supervisor_address: str, host: str, port: int):
        super().__init__()
        self._supervisor_address = supervisor_address
        self._host = host
        self._port = port
        self._supervisor_ref = None
        self._router = APIRouter()
        self._app = FastAPI()

    @staticmethod
    def handle_request_limit_error(e: Exception):
        if "Rate limit reached" in str(e):
            raise HTTPException(status_code=429, detail=str(e))

    async def _get_supervisor_ref(self) -> xo.ActorRefType[SupervisorActor]:
        if self._supervisor_ref is None:
            self._supervisor_ref = await xo.actor_ref(
                address=self._supervisor_address, uid=SupervisorActor.uid()
            )
        return self._supervisor_ref

    def serve(self, logging_conf: Optional[dict] = None):
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._router.add_api_route("/status", self.get_status, methods=["GET"])
        self._router.add_api_route("/v1/models", self.list_models, methods=["GET"])
        self._router.add_api_route(
            "/v1/models/{model_uid}", self.describe_model, methods=["GET"]
        )
        self._router.add_api_route("/v1/models", self.launch_model, methods=["POST"])
        self._router.add_api_route(
            "/experimental/speculative_llms",
            self.launch_speculative_llm,
            methods=["POST"],
        )
        self._router.add_api_route(
            "/v1/models/{model_uid}", self.terminate_model, methods=["DELETE"]
        )
        self._router.add_api_route("/v1/address", self.get_address, methods=["GET"])
        self._router.add_api_route(
            "/v1/completions",
            self.create_completion,
            methods=["POST"],
            response_model=Completion,
        )
        self._router.add_api_route(
            "/v1/embeddings",
            self.create_embedding,
            methods=["POST"],
        )
        self._router.add_api_route(
            "/v1/rerank",
            self.rerank,
            methods=["POST"],
        )
        self._router.add_api_route(
            "/v1/images/generations",
            self.create_images,
            methods=["POST"],
            response_model=ImageList,
        )
        self._router.add_api_route(
            "/v1/images/variations",
            self.create_variations,
            methods=["POST"],
            response_model=ImageList,
        )
        self._router.add_api_route(
            "/v1/chat/completions",
            self.create_chat_completion,
            methods=["POST"],
            response_model=ChatCompletion,
        )

        # for custom models
        self._router.add_api_route(
            "/v1/model_registrations/{model_type}",
            self.register_model,
            methods=["POST"],
        )
        self._router.add_api_route(
            "/v1/model_registrations/{model_type}/{model_name}",
            self.unregister_model,
            methods=["DELETE"],
        )
        self._router.add_api_route(
            "/v1/model_registrations/{model_type}",
            self.list_model_registrations,
            methods=["GET"],
        )
        self._router.add_api_route(
            "/v1/model_registrations/{model_type}/{model_name}",
            self.get_model_registrations,
            methods=["GET"],
        )

        self._router.add_api_route(
            "/v1/ui/{model_uid}", self.build_gradio_interface, methods=["POST"]
        )

        self._app.include_router(self._router)

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
            ui_location = os.path.join(lib_location, "web/ui/build/")
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
            To resolve this warning, navigate to {os.path.join(lib_location, "web/ui/")}
            And build the Xinference ui by running "npm run build"
            """
            )

        config = Config(
            app=self._app, host=self._host, port=self._port, log_config=logging_conf
        )
        server = Server(config)
        server.run()

    async def get_status(self) -> JSONResponse:
        try:
            data = await (await self._get_supervisor_ref()).get_status()
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def list_models(self) -> JSONResponse:
        try:
            data = await (await self._get_supervisor_ref()).list_models()
            return JSONResponse(content=data)
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

    async def launch_speculative_llm(self, request: Request) -> JSONResponse:
        payload = await request.json()
        model_uid = payload.get("model_uid")
        model_name = payload.get("model_name")
        model_size_in_billions = payload.get("model_size_in_billions")
        quantization = payload.get("quantization")
        draft_model_name = payload.get("draft_model_name")
        draft_model_size_in_billions = payload.get("draft_model_size_in_billions")
        draft_quantization = payload.get("draft_quantization")
        n_gpu = payload.get("n_gpu", "auto")

        if model_uid is None or model_uid is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid input. Please specify the model UID and the model name",
            )

        try:
            model_uid = await (await self._get_supervisor_ref()).launch_speculative_llm(
                model_uid=model_uid,
                model_name=model_name,
                model_size_in_billions=model_size_in_billions,
                quantization=quantization,
                draft_model_name=draft_model_name,
                draft_model_size_in_billions=draft_model_size_in_billions,
                draft_quantization=draft_quantization,
                n_gpu=n_gpu,
            )

        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except RuntimeError as re:
            logger.error(str(re), exc_info=True)
            raise HTTPException(status_code=503, detail=str(re))
        except Exception as e:
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        return JSONResponse(content={"model_uid": model_uid})

    async def launch_model(self, request: Request) -> JSONResponse:
        payload = await request.json()
        model_uid = payload.get("model_uid")
        model_name = payload.get("model_name")
        model_size_in_billions = payload.get("model_size_in_billions")
        model_format = payload.get("model_format")
        quantization = payload.get("quantization")
        model_type = payload.get("model_type")
        replica = payload.get("replica", 1)
        n_gpu = payload.get("n_gpu", "auto")
        request_limits = payload.get("request_limits", None)

        exclude_keys = {
            "model_uid",
            "model_name",
            "model_size_in_billions",
            "model_format",
            "quantization",
            "model_type",
            "replica",
            "n_gpu",
            "request_limits",
        }

        kwargs = {
            key: value for key, value in payload.items() if key not in exclude_keys
        }

        if model_uid is None or model_uid is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid input. Please specify the model UID and the model name",
            )

        try:
            model_uid = await (await self._get_supervisor_ref()).launch_builtin_model(
                model_uid=model_uid,
                model_name=model_name,
                model_size_in_billions=model_size_in_billions,
                model_format=model_format,
                quantization=quantization,
                model_type=model_type,
                replica=replica,
                n_gpu=n_gpu,
                request_limits=request_limits,
                **kwargs,
            )

        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except RuntimeError as re:
            logger.error(str(re), exc_info=True)
            raise HTTPException(status_code=503, detail=str(re))
        except Exception as e:
            logger.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        return JSONResponse(content={"model_uid": model_uid})

    async def build_gradio_interface(
        self, model_uid: str, body: BuildGradioInterfaceRequest
    ) -> JSONResponse:
        """
        Separate build_interface with launch_model
        build_interface requires RESTful Client for API calls
        but calling API in async function does not return
        """
        assert self._app is not None
        assert body.model_type == "LLM"

        # asyncio.Lock() behaves differently in 3.9 than 3.10+
        # A event loop is required in 3.9 but not 3.10+
        if sys.version_info < (3, 10):
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                warnings.warn(
                    "asyncio.Lock() requires an event loop in Python 3.9"
                    + "a placeholder event loop has been created"
                )
                asyncio.set_event_loop(asyncio.new_event_loop())

        from ..core.chat_interface import LLMInterface

        try:
            internal_host = "localhost" if self._host == "0.0.0.0" else self._host
            interface = LLMInterface(
                endpoint=f"http://{internal_host}:{self._port}",
                model_uid=model_uid,
                model_name=body.model_name,
                model_size_in_billions=body.model_size_in_billions,
                model_format=body.model_format,
                quantization=body.quantization,
                context_length=body.context_length,
                model_ability=body.model_ability,
                model_description=body.model_description,
                model_lang=body.model_lang,
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

    async def create_completion(
        self, request: Request, body: CreateCompletionRequest
    ) -> Response:
        exclude = {
            "prompt",
            "model",
            "n",
            "best_of",
            "logit_bias",
            "logit_bias_type",
            "user",
        }
        kwargs = body.dict(exclude_unset=True, exclude=exclude)

        if body.logit_bias is not None:
            raise HTTPException(status_code=501, detail="Not implemented")

        model_uid = body.model

        try:
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))

        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        if body.stream:

            async def stream_results():
                iterator = None
                try:
                    try:
                        iterator = await model.generate(body.prompt, kwargs)
                    except RuntimeError as re:
                        self.handle_request_limit_error(re)
                    async for item in iterator:
                        yield dict(data=json.dumps(item))
                except Exception as ex:
                    if iterator is not None:
                        await iterator.destroy()
                    logger.exception("Completion stream got an error: %s", ex)
                    # https://github.com/openai/openai-python/blob/e0aafc6c1a45334ac889fe3e54957d309c3af93f/src/openai/_streaming.py#L107
                    yield dict(data=json.dumps({"error": str(ex)}))

            return EventSourceResponse(stream_results())
        else:
            try:
                data = await model.generate(body.prompt, kwargs)
                return JSONResponse(content=data)
            except Exception as e:
                logger.error(e, exc_info=True)
                self.handle_request_limit_error(e)
                raise HTTPException(status_code=500, detail=str(e))

    async def create_embedding(self, request: CreateEmbeddingRequest) -> Response:
        model_uid = request.model

        try:
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        try:
            embedding = await model.create_embedding(request.input)
            return Response(embedding, media_type="application/json")
        except RuntimeError as re:
            logger.error(re, exc_info=True)
            self.handle_request_limit_error(re)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def rerank(self, request: RerankRequest) -> Response:
        model_uid = request.model
        try:
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        try:
            scores = await model.rerank(
                request.documents,
                request.query,
                top_n=request.top_n,
                max_chunks_per_doc=request.max_chunks_per_doc,
                return_documents=request.return_documents,
            )
            return Response(scores, media_type="application/json")
        except RuntimeError as re:
            logger.error(re, exc_info=True)
            self.handle_request_limit_error(re)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_images(self, request: TextToImageRequest) -> JSONResponse:
        model_uid = request.model
        try:
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        try:
            image_list = await model.text_to_image(
                request.prompt, request.n, request.size, request.response_format
            )
            return JSONResponse(content=image_list)
        except RuntimeError as re:
            logger.error(re, exc_info=True)
            self.handle_request_limit_error(re)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_variations(
        self,
        model: str = Form(...),
        image: UploadFile = File(media_type="application/octet-stream"),
        prompt: Optional[Union[str, List[str]]] = Form(None),
        negative_prompt: Optional[Union[str, List[str]]] = Form(None),
        n: Optional[int] = Form(1),
        response_format: Optional[str] = Form("url"),
        size: Optional[str] = Form("1024*1024"),
        kwargs: Optional[str] = Form(None),
    ) -> JSONResponse:
        model_uid = model
        try:
            model_ref = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        try:
            if kwargs is not None:
                kwargs = json.loads(kwargs)
            image_list = await model_ref.image_to_image(
                image=Image.open(image.file),
                prompt=prompt,
                negative_prompt=negative_prompt,
                n=n,
                size=size,
                response_format=response_format,
                **kwargs,
            )
            return JSONResponse(content=image_list)
        except RuntimeError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_chat_completion(
        self,
        request: Request,
        body: CreateChatCompletion,
    ) -> Response:
        exclude = {
            "prompt",
            "model",
            "n",
            "messages",
            "logit_bias",
            "logit_bias_type",
            "user",
        }
        kwargs = body.dict(exclude_unset=True, exclude=exclude)

        if body.logit_bias is not None:
            raise HTTPException(status_code=501, detail="Not implemented")

        if (
            not body.messages
            or body.messages[-1].get("role") != "user"
            or not body.messages[-1].get("content")
        ):
            raise HTTPException(
                status_code=400, detail="Invalid input. Please specify the prompt."
            )

        system_messages = []
        non_system_messages = []
        for msg in body.messages:
            if msg["role"] == "system":
                system_messages.append(msg)
            else:
                non_system_messages.append(msg)

        if len(system_messages) > 1:
            raise HTTPException(
                status_code=400, detail="Multiple system messages are not supported."
            )
        if len(system_messages) == 1 and body.messages[0]["role"] != "system":
            raise HTTPException(
                status_code=400, detail="System message should be the first one."
            )
        assert non_system_messages

        prompt = non_system_messages[-1]["content"]
        system_prompt = system_messages[0]["content"] if system_messages else None
        chat_history = non_system_messages[:-1]  # exclude the prompt

        model_uid = body.model

        try:
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        try:
            desc = await (await self._get_supervisor_ref()).describe_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        is_chatglm_ggml = desc.get(
            "model_format"
        ) == "ggmlv3" and "chatglm" in desc.get("model_name", "")

        is_qwen = desc.get("model_format") == "ggmlv3" and "qwen" in desc.get(
            "model_name", ""
        )

        if (is_chatglm_ggml or is_qwen) and system_prompt is not None:
            raise HTTPException(
                status_code=400, detail="ChatGLM ggml does not have system prompt"
            )
        if is_chatglm_ggml and body.tools and body.stream:
            raise HTTPException(
                status_code=400, detail="ChatGLM tool does not support stream"
            )

        if body.stream:

            async def stream_results():
                iterator = None
                try:
                    try:
                        if is_chatglm_ggml or is_qwen:
                            iterator = await model.chat(prompt, chat_history, kwargs)
                        else:
                            iterator = await model.chat(
                                prompt, system_prompt, chat_history, kwargs
                            )
                    except RuntimeError as re:
                        self.handle_request_limit_error(re)
                    async for item in iterator:
                        yield dict(data=json.dumps(item))
                except Exception as ex:
                    if iterator is not None:
                        await iterator.destroy()
                    logger.exception("Chat completion stream got an error: %s", ex)
                    # https://github.com/openai/openai-python/blob/e0aafc6c1a45334ac889fe3e54957d309c3af93f/src/openai/_streaming.py#L107
                    yield dict(data=json.dumps({"error": str(ex)}))

            return EventSourceResponse(stream_results())
        else:
            try:
                if is_chatglm_ggml or is_qwen:
                    data = await model.chat(prompt, chat_history, kwargs)
                else:
                    data = await model.chat(prompt, system_prompt, chat_history, kwargs)
                return JSONResponse(content=data)
            except Exception as e:
                logger.error(e, exc_info=True)
                self.handle_request_limit_error(e)
                raise HTTPException(status_code=500, detail=str(e))

    async def register_model(
        self, model_type: str, request: RegisterModelRequest
    ) -> JSONResponse:
        model = request.model
        persist = request.persist

        try:
            await (await self._get_supervisor_ref()).register_model(
                model_type, model, persist
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

    async def list_model_registrations(
        self, model_type: str, detailed: bool = Query(False)
    ) -> JSONResponse:
        try:
            data = await (await self._get_supervisor_ref()).list_model_registrations(
                model_type, detailed=detailed
            )
            return JSONResponse(content=data)
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


def run(
    supervisor_address: str, host: str, port: int, logging_conf: Optional[dict] = None
):
    logger.info(f"Starting Xinference at endpoint: http://{host}:{port}")
    try:
        api = RESTfulAPI(supervisor_address=supervisor_address, host=host, port=port)
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
                supervisor_address=supervisor_address, host=host, port=port
            )
            api.serve(logging_conf=logging_conf)
        else:
            raise


def run_in_subprocess(
    supervisor_address: str, host: str, port: int, logging_conf: Optional[dict] = None
) -> multiprocessing.Process:
    p = multiprocessing.Process(
        target=run, args=(supervisor_address, host, port, logging_conf)
    )
    p.daemon = True
    p.start()
    return p
