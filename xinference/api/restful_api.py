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
import multiprocessing
import os
import sys
import warnings
from typing import Any, Dict, List, Literal, Optional, Union

import gradio as gr
import xoscar as xo
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from starlette.responses import RedirectResponse
from typing_extensions import NotRequired, TypedDict
from uvicorn import Config, Server
from xoscar.utils import get_next_port

from ..constants import XINFERENCE_DEFAULT_ENDPOINT_PORT
from ..core.supervisor import SupervisorActor
from ..types import ChatCompletion, Completion, Embedding, ImageList

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
    default=1.0,
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
    grammar: Optional[str] = Field(None)

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


class TextToImageRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]] = Field(description="The input to embed.")
    n: Optional[int] = 1
    response_format: Optional[str] = "url"
    size: Optional[str] = "1024*1024"
    user: Optional[str] = None


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
    grammar: Optional[str] = Field(None)

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
            response_model=Embedding,
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

    async def get_status(self) -> Dict[str, Any]:
        try:
            return await (await self._get_supervisor_ref()).get_status()
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        try:
            return await (await self._get_supervisor_ref()).list_models()
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def describe_model(self, model_uid: str) -> Dict[str, Any]:
        try:
            return await (await self._get_supervisor_ref()).describe_model(model_uid)

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
            assert self._supervisor_ref is not None
            model_uid = await self._supervisor_ref.launch_speculative_llm(
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

        exclude_keys = {
            "model_uid",
            "model_name",
            "model_size_in_billions",
            "model_format",
            "quantization",
            "model_type",
            "replica",
            "n_gpu",
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
    ):
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

    async def terminate_model(self, model_uid: str):
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

    async def get_address(self):
        return self._supervisor_address

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
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))

        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        if body.stream:

            async def stream_results():
                try:
                    iterator = await model.generate(body.prompt, kwargs)
                    async for item in iterator:
                        yield f"data: {json.dumps(item)}\n"
                except Exception as ex:
                    logger.exception("Completion stream got an error: %s", ex)
                    yield f"error: {ex}"

            # The Content-Type: text/event-stream header is required for openai stream.
            return StreamingResponse(
                stream_results(), headers={"Content-Type": "text/event-stream"}
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
            model = await (await self._get_supervisor_ref()).get_model(model_uid)
        except ValueError as ve:
            logger.error(str(ve), exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        try:
            embedding = await model.create_embedding(request.input)
            return embedding
        except RuntimeError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def create_images(self, request: TextToImageRequest):
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
            return image_list
        except RuntimeError as re:
            logger.error(re, exc_info=True)
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
    ):
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
            return image_list
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
            or (body.messages[-1].get("role") != "user" and body.messages[-1].get("role") != "system")
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

        if is_chatglm_ggml and system_prompt is not None:
            raise HTTPException(
                status_code=400, detail="ChatGLM ggml does not have system prompt"
            )

        if body.stream:

            async def stream_results():
                try:
                    if is_chatglm_ggml:
                        iterator = await model.chat(prompt, chat_history, kwargs)
                    else:
                        iterator = await model.chat(
                            prompt, system_prompt, chat_history, kwargs
                        )
                    async for item in iterator:
                        yield f"data: {json.dumps(item)}\n"
                except Exception as ex:
                    logger.exception("Chat completion stream got an error: %s", ex)
                    yield f"error: {ex}"

            # The Content-Type: text/event-stream header is required for openai stream.
            return StreamingResponse(
                stream_results(), headers={"Content-Type": "text/event-stream"}
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
            await (await self._get_supervisor_ref()).register_model(
                model_type, model, persist
            )
        except ValueError as re:
            logger.error(re, exc_info=True)
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def unregister_model(self, model_type: str, model_name: str):
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

    async def list_model_registrations(self, model_type: str) -> List[Dict[str, Any]]:
        try:
            return await (await self._get_supervisor_ref()).list_model_registrations(
                model_type
            )
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
            return await (await self._get_supervisor_ref()).get_model_registration(
                model_type, model_name
            )
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
