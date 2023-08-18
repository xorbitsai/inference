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
import json
import logging
import os
import socket
import sys
import threading
import warnings
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union

import anyio
import gradio as gr
import xoscar as xo
from anyio.streams.memory import MemoryObjectSendStream
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.responses import RedirectResponse
from typing_extensions import NotRequired, TypedDict
from uvicorn import Config, Server

from ..types import ChatCompletion, Completion, Embedding
from .supervisor import SupervisorActor

logger = logging.getLogger(__name__)

max_tokens_field = Field(
    default=128, ge=1, le=32768, description="The maximum number of tokens to generate."
)

temperature_field = Field(
    default=0.8,
    ge=0.0,
    le=2.0,
    description="Adjust the randomness of the generated text.\n\n"
    + "Temperature is a hyperparameter that controls the randomness of the generated text. It affects the probability distribution of the model's output tokens. A higher temperature (e.g., 1.5) makes the output more random and creative, while a lower temperature (e.g., 0.5) makes the output more focused, deterministic, and conservative. The default value is 0.8, which provides a balance between randomness and determinism. At the extreme, a temperature of 0 will always pick the most likely next token, leading to identical outputs in each run.",
)

top_p_field = Field(
    default=0.95,
    ge=0.0,
    le=1.0,
    description="Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P.\n\n"
    + "Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least p. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top_p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.",
)

stop_field = Field(
    default=None,
    description="A list of tokens at which to stop generation. If None, no stop tokens are used.",
)

stream_field = Field(
    default=False,
    description="Whether to stream the results as they are generated. Useful for chatbots.",
)

top_k_field = Field(
    default=40,
    ge=0,
    description="Limit the next token selection to the K most probable tokens.\n\n"
    + "Top-k sampling is a text generation method that selects the next token only from the top k most likely tokens predicted by the model. It helps reduce the risk of generating low-probability or nonsensical tokens, but it may also limit the diversity of the output. A higher value for top_k (e.g., 100) will consider more tokens and lead to more diverse text, while a lower value (e.g., 10) will focus on the most probable tokens and generate more conservative text.",
)

repetition_penalty_field = Field(
    default=1.1,
    ge=0.0,
    description="A penalty applied to each token that is already generated. This helps prevent the model from repeating itself.\n\n"
    + "Repeat penalty is a hyperparameter used to penalize the repetition of token sequences during text generation. It helps prevent the model from generating repetitive or monotonous text. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient.",
)

presence_penalty_field = Field(
    default=0.0,
    ge=-2.0,
    le=2.0,
    description="Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
)

frequency_penalty_field = Field(
    default=0.0,
    ge=-2.0,
    le=2.0,
    description="Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
)

mirostat_mode_field = Field(
    default=0,
    ge=0,
    le=2,
    description="Enable Mirostat constant-perplexity algorithm of the specified version (1 or 2; 0 = disabled)",
)

mirostat_tau_field = Field(
    default=5.0,
    ge=0.0,
    le=10.0,
    description="Mirostat target entropy, i.e. the target perplexity - lower values produce focused and coherent text, larger values produce more diverse and less coherent text",
)

mirostat_eta_field = Field(
    default=0.1, ge=0.001, le=1.0, description="Mirostat learning rate"
)


class CreateCompletionRequest(BaseModel):
    prompt: str
    suffix: Optional[str] = Field(None)
    max_tokens: int = max_tokens_field
    temperature: float = temperature_field
    top_p: float = top_p_field
    mirostat_mode: int = mirostat_mode_field
    mirostat_tau: float = mirostat_tau_field
    mirostat_eta: float = mirostat_eta_field
    echo: bool = Field(
        default=False,
        description="Whether to echo the prompt in the generated text. Useful for chatbots.",
    )
    stop: Optional[Union[str, List[str]]] = stop_field
    stream: bool = stream_field
    logprobs: Optional[int] = Field(
        default=None,
        ge=0,
        description="The number of logprobs to generate. If None, no logprobs are generated.",
    )
    presence_penalty: Optional[float] = presence_penalty_field
    frequency_penalty: Optional[float] = frequency_penalty_field
    logit_bias: Optional[Dict[str, float]] = Field(None)

    model: str
    n: Optional[int] = 1
    best_of: Optional[int] = 1
    user: Optional[str] = Field(None)

    # llama.cpp specific parameters
    top_k: int = top_k_field
    repetition_penalty: float = repetition_penalty_field
    logit_bias_type: Optional[Literal["input_ids", "tokens"]] = Field(None)

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


class ChatCompletionRequestMessage(TypedDict):
    role: Literal["assistant", "user", "system"]
    content: str
    user: NotRequired[str]


class CreateChatCompletionRequest(BaseModel):
    messages: List[ChatCompletionRequestMessage] = Field(
        default=[], description="A list of messages to generate completions for."
    )
    max_tokens: int = max_tokens_field
    temperature: float = temperature_field
    top_p: float = top_p_field
    mirostat_mode: int = mirostat_mode_field
    mirostat_tau: float = mirostat_tau_field
    mirostat_eta: float = mirostat_eta_field
    stop: Optional[Union[str, List[str]]] = stop_field
    stream: bool = stream_field
    presence_penalty: Optional[float] = presence_penalty_field
    frequency_penalty: Optional[float] = frequency_penalty_field
    logit_bias: Optional[Dict[str, float]] = Field(None)

    model: str
    n: Optional[int] = 1
    user: Optional[str] = Field(None)

    # llama.cpp specific parameters
    top_k: int = top_k_field
    repetition_penalty: float = repetition_penalty_field
    logit_bias_type: Optional[Literal["input_ids", "tokens"]] = Field(None)

    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    {"role": "system", "content": "you are a helpful AI assistant"},
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi what can I help you?"},
                ]
            }
        }


class RegisterModelRequest(BaseModel):
    model: str
    persist: bool


class RESTfulAPIActor(xo.Actor):
    def __init__(self, sockets: List[socket.socket], endpoint: str):
        super().__init__()
        self._supervisor_ref: xo.ActorRefType["SupervisorActor"]
        self._sockets = sockets
        self._endpoint = endpoint
        self._router = None
        self._app = None

    @classmethod
    def uid(cls) -> str:
        return "RESTfulAPI"

    async def __post_create__(self):
        self._supervisor_ref = await xo.actor_ref(
            address=self.address, uid=SupervisorActor.uid()
        )

    def serve(self):
        self._app = FastAPI()
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._router = APIRouter()
        self._router.add_api_route("/v1/models", self.list_models, methods=["GET"])
        self._router.add_api_route(
            "/v1/models/{model_uid}", self.describe_model, methods=["GET"]
        )
        self._router.add_api_route("/v1/models", self.launch_model, methods=["POST"])
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
            response_model=Embedding,
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
            "/v1/ui/{model_uid}", self.build_interface, methods=["POST"]
        )

        self._app.include_router(self._router)

        class SPAStaticFiles(StaticFiles):
            async def get_response(self, path: str, scope):
                response = await super().get_response(path, scope)
                if response.status_code == 404:
                    response = await super().get_response(".", scope)
                return response

        try:
            lib_location = os.path.abspath(
                os.path.dirname(__import__("xinference").__file__)
            )
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

        # run uvicorn in another daemon thread.
        config = Config(app=self._app, log_level="critical")
        server = Server(config)

        def _serve():
            httpx_logger = logging.getLogger("httpx")
            httpx_logger.setLevel(logging.CRITICAL)
            server.run(self._sockets)

        server_thread = threading.Thread(target=_serve, daemon=True)
        server_thread.start()

    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        try:
            return await self._supervisor_ref.list_models()
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def describe_model(self, model_uid: str) -> Dict[str, Any]:
        try:
            return await self._supervisor_ref.describe_model(model_uid)

        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))

        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def launch_model(self, request: Request) -> JSONResponse:
        payload = await request.json()
        model_uid = payload.get("model_uid")
        model_name = payload.get("model_name")
        model_size_in_billions = payload.get("model_size_in_billions")
        model_format = payload.get("model_format")
        quantization = payload.get("quantization")

        exclude_keys = {
            "model_uid",
            "model_name",
            "model_size_in_billions",
            "model_format",
            "quantization",
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
            await self._supervisor_ref.launch_builtin_model(
                model_uid=model_uid,
                model_name=model_name,
                model_size_in_billions=model_size_in_billions,
                model_format=model_format,
                quantization=quantization,
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

    def build_interface(self, model_uid: str):
        """
        Separate build_interface with launch_model
        build_interface requires RESTful Client for API calls
        but calling API in async function does not return
        """
        assert self._app is not None
        assert self._endpoint is not None

        from .chat_interface import LLMInterface

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

        try:
            interface = LLMInterface(self._endpoint, model_uid)
            gr.mount_gradio_app(self._app, interface.build(), f"/{model_uid}")
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))

        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        return JSONResponse(content={"model_uid": model_uid})

    async def terminate_model(self, model_uid: str):
        try:
            assert self._app is not None
            await self._supervisor_ref.terminate_model(model_uid)
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

    async def get_address(self):
        return self.address

    async def create_completion(self, request: Request, body: CreateCompletionRequest):
        exclude = {
            "prompt",
            "model",
            "n",
            "best_of",
            "logit_bias",
            "logit_bias_type",
            "user",
        }
        kwargs = body.dict(exclude=exclude)

        if body.logit_bias is not None:
            raise HTTPException(status_code=501, detail="Not implemented")

        model_uid = body.model

        try:
            model = await self._supervisor_ref.get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))

        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        if body.stream:
            # create a pair of memory object streams
            send_chan, recv_chan = anyio.create_memory_object_stream(10)

            async def event_publisher(inner_send_chan: MemoryObjectSendStream):
                async with inner_send_chan:
                    try:
                        iterator = await model.generate(body.prompt, kwargs)
                        async for chunk in iterator:
                            await inner_send_chan.send(dict(data=json.dumps(chunk)))
                            if await request.is_disconnected():
                                raise anyio.get_cancelled_exc_class()()
                    except anyio.get_cancelled_exc_class() as e:
                        logger.warning("disconnected")
                        with anyio.move_on_after(1, shield=True):
                            logger.warning(
                                f"Disconnected from client (via refresh/close) {request.client}"
                            )
                            await inner_send_chan.send(dict(closing=True))
                            raise e
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=str(e))

            return EventSourceResponse(
                recv_chan, data_sender_callable=partial(event_publisher, send_chan)
            )

        else:
            try:
                return await model.generate(body.prompt, kwargs)
            except Exception as e:
                logger.error(e, exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

    async def create_embedding(self, request: CreateEmbeddingRequest):
        model_uid = request.model

        try:
            model = await self._supervisor_ref.get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        input = request.input

        try:
            embedding = await model.create_embedding(input)
            return embedding
        except RuntimeError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_chat_completion(
        self,
        request: Request,
        body: CreateChatCompletionRequest,
    ):
        exclude = {
            "n",
            "model",
            "messages",
            "logit_bias",
            "logit_bias_type",
            "user",
        }
        kwargs = body.dict(exclude=exclude)

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

        prompt = body.messages[-1]["content"]

        system_prompt = next(
            (msg["content"] for msg in body.messages if msg["role"] == "system"), None
        )

        chat_history = body.messages[:-1]  # exclude the prompt

        model_uid = body.model

        try:
            model = await self._supervisor_ref.get_model(model_uid)

        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        try:
            desc = await self._supervisor_ref.describe_model(model_uid)

        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))

        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        is_chatglm_ggml = desc.get(
            "model_format"
        ) == "ggmlv3" and "chatglm" in desc.get("model_name", "")

        if is_chatglm_ggml and system_prompt is not None:
            raise HTTPException(
                status_code=400, detail="ChatGLM ggml does not have system prompt"
            )

        if body.stream:
            # create a pair of memory object streams
            send_chan, recv_chan = anyio.create_memory_object_stream(10)

            async def event_publisher(inner_send_chan: MemoryObjectSendStream):
                async with inner_send_chan:
                    try:
                        if is_chatglm_ggml:
                            iterator = await model.chat(prompt, chat_history, kwargs)
                        else:
                            iterator = await model.chat(
                                prompt, system_prompt, chat_history, kwargs
                            )
                        async for chunk in iterator:
                            await inner_send_chan.send(dict(data=json.dumps(chunk)))
                            if await request.is_disconnected():
                                raise anyio.get_cancelled_exc_class()()
                    except anyio.get_cancelled_exc_class() as e:
                        logger.warning("disconnected")
                        with anyio.move_on_after(1, shield=True):
                            logger.warning(
                                f"Disconnected from client (via refresh/close) {request.client}"
                            )
                            await inner_send_chan.send(dict(closing=True))
                            raise e
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=str(e))

            return EventSourceResponse(
                recv_chan, data_sender_callable=partial(event_publisher, send_chan)
            )

        else:
            try:
                if is_chatglm_ggml:
                    return await model.chat(prompt, chat_history, kwargs)
                else:
                    return await model.chat(prompt, system_prompt, chat_history, kwargs)
            except Exception as e:
                logger.error(e, exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

    async def register_model(self, model_type: str, request: RegisterModelRequest):
        model = request.model
        persist = request.persist

        try:
            await self._supervisor_ref.register_model(model_type, model, persist)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def unregister_model(self, model_type: str, model_name: str):
        try:
            await self._supervisor_ref.unregister_model(model_type, model_name)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def list_model_registrations(self, model_type: str) -> List[Dict[str, Any]]:
        try:
            return await self._supervisor_ref.list_model_registrations(model_type)
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_model_registrations(
        self, model_type: str, model_name: str
    ) -> Dict[str, Any]:
        try:
            return await self._supervisor_ref.get_model_registration(
                model_type, model_name
            )
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
