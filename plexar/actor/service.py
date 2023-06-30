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

from logging import getLogger
from typing import Callable, Dict, List, Optional, Union, Literal

import xoscar as xo

from plexar.actor import ModelActor
from plexar.model import ModelSpec

from fastapi import FastAPI, APIRouter, Request
from fastapi import HTTPException
import asyncio
from uvicorn import Config, Server
import uuid
import llama_cpp
from pydantic import BaseModel, BaseSettings, Field, create_model_from_typeddict

max_tokens_field = Field(
    default=16, ge=1, le=2048, description="The maximum number of tokens to generate."
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

repeat_penalty_field = Field(
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
    description="Enable Mirostat constant-perplexity algorithm of the specified version (1 or 2; 0 = disabled)"
)

mirostat_tau_field = Field(
    default=5.0,
    ge=0.0,
    le=10.0,
    description="Mirostat target entropy, i.e. the target perplexity - lower values produce focused and coherent text, larger values produce more diverse and less coherent text"
)

mirostat_eta_field = Field(
    default=0.1,
    ge=0.001,
    le=1.0,
    description="Mirostat learning rate"
)


# CreateComplete request
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
    logprobs: Optional[int] = Field(None)

    model: str
    n: Optional[int] = 1
    best_of: Optional[int] = 1
    user: Optional[str] = Field(None)

    # llama.cpp specific parameters
    top_k: int = top_k_field
    repeat_penalty: float = repeat_penalty_field
    logit_bias_type: Optional[Literal["input_ids", "tokens"]] = Field(None)

    class Config:
        schema_extra = {
            "example": {
                "prompt": "\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n",
                "stop": ["\n", "###"],
            }
        }

logger = getLogger(__name__)


def log(func: Callable):
    # TODO: support non-async function
    import time
    from functools import wraps

    @wraps(func)
    async def wrapped(*args, **kwargs):
        logger.debug(f"Enter {func.__name__}, args: {args}, kwargs: {kwargs}")
        start = time.time()
        ret = await func(*args, **kwargs)
        logger.debug(
            f"Leave {func.__name__}, elapsed time: {int(time.time() - start)} ms"
        )
        return ret

    return wrapped


class ControllerActor(xo.Actor):
    def __init__(self):
        super().__init__()
        self._worker_address_to_worker: Dict[str, xo.ActorRefType[WorkerActor]] = {}
        self._model_uid_to_worker: Dict[str, xo.ActorRefType[WorkerActor]] = {}

    @classmethod
    def uid(cls) -> str:
        return "plexar_controller"

    async def _choose_worker(self) -> xo.ActorRefType["WorkerActor"]:
        # TODO: better allocation strategy.
        min_running_model_count = None
        target_worker = None
        for worker in self._worker_address_to_worker.values():
            running_model_count = await worker.get_model_count()
            if (
                min_running_model_count is None
                or running_model_count < min_running_model_count
            ):
                min_running_model_count = running_model_count
                target_worker = worker

            return target_worker

        raise RuntimeError("TODO")
    
    async def get_worker(self, model_uid: str) -> xo.ActorRefType["WorkerActor"]:
        if model_uid not in self._model_uid_to_worker:
            raise Exception(f"Worker for model UID '{model_uid}' not found.")
        return self._model_uid_to_worker[model_uid]

    @log
    async def launch_builtin_model(
        self,
        model_uid: str,
        model_name: str,
        model_size_in_billions: Optional[int],
        model_format: Optional[str],
        quantization: Optional[str],
        **kwargs,
    ) -> xo.ActorRefType["ModelActor"]:
        assert model_uid not in self._model_uid_to_worker

        worker_ref = await self._choose_worker()
        model_ref = await worker_ref.launch_builtin_model(
            model_uid=model_uid,
            model_name=model_name,
            model_size_in_billions=model_size_in_billions,
            model_format=model_format,
            quantization=quantization,
            **kwargs,
        )
        self._model_uid_to_worker[model_uid] = worker_ref

        return model_ref

    @log
    async def terminate_model(self, model_uid: str):
        assert model_uid in self._model_uid_to_worker

        worker_ref = self._model_uid_to_worker[model_uid]
        await worker_ref.terminate_model(model_uid=model_uid)
        del self._model_uid_to_worker[model_uid]

    @log
    async def get_model(self, model_uid: str):
        assert model_uid in self._model_uid_to_worker

        worker_ref = self._model_uid_to_worker[model_uid]
        return await worker_ref.get_model(model_uid=model_uid)

    @log
    async def list_models(self) -> List[tuple[str, ModelSpec]]:
        ret = []
        for worker in self._worker_address_to_worker.values():
            ret.extend(await worker.list_models())
        return ret

    @log
    async def add_worker(self, worker_address: str):
        assert worker_address not in self._worker_address_to_worker

        worker_ref = await xo.actor_ref(address=worker_address, uid=WorkerActor.uid())
        self._worker_address_to_worker[worker_address] = worker_ref


class RESTAPIActor(xo.Actor):
    def __init__(self, host:str, port:int):
        super().__init__()
        self._controller_ref = None
        app = FastAPI()
        self.router = APIRouter()
        self.router.add_api_route("/v1/models", self.list_models, methods=["GET"])
        self.router.add_api_route("/v1/models", self.launch_model, methods=["POST"])
        self.router.add_api_route("/v1/models/{model_uid}", self.terminate_model, methods=["DELETE"])
        self.router.add_api_route("/v1/completions", self.create_completion, methods=["POST"])
        # self.router.add_api_route("/v1/embeddings", self.create_embedding, methods=["POST"])
        # self.router.add_api_route("/v1/chat/completions", self.create_chat_completion, methods=["POST"])
        app.include_router(self.router)

        # uvicorn
        loop = asyncio.get_event_loop()
        config = Config(app=app, loop=loop, host=host, port=port)
        server = Server(config)
        loop.create_task(server.serve())
    
    async def __post_create__(self):
        self._controller_ref = await xo.actor_ref(
            address=self.address, uid=ControllerActor.uid()
        )
    
    def gen_model_uid(self) -> str:
        # generate a time-based uuid.
        return str(uuid.uuid1())

    async def list_models(self) -> List[str]:
        models = await self._controller_ref.list_models()
        return [model_uid for model_uid, _ in models]

    async def launch_model(self, request: Request) -> str:
        payload = await request.json()
        model_name = payload.get('model_name')
        model_size_in_billions = payload.get('model_size_in_billions')
        model_format = payload.get('model_format')
        quantization = payload.get('quantization')
        kwargs = payload.get('kwargs', {}) or {}

        model_uid = self.gen_model_uid()

        await self._controller_ref.launch_builtin_model(
            model_uid=model_uid,
            model_name=model_name,
            model_size_in_billions=model_size_in_billions,
            model_format=model_format,
            quantization=quantization,
            **kwargs
        )
        return model_uid

    async def terminate_model(self, model_uid: str):
        await self._controller_ref.terminate_model(model_uid)
        return {"message": "Model terminated successfully."}

    async def create_completion(self, 
        # request: Request,
        body: CreateCompletionRequest
    ):

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
            kwargs['logits_processor'] = llama_cpp.LogitsProcessorList([
                make_logit_bias_processor(llama, body.logit_bias, body.logit_bias_type),
        ])

        model_uid = body.model
        prompt = body.prompt
        worker_ref = await self._controller_ref.get_worker(model_uid)
        model = await worker_ref.get_model(model_uid)
        
        if body.stream: #TODO
            import sys
            async for c in await model.generate(body.prompt, kwargs):
                sys.stdout.write(c['text'])
        else:
            return await model.generate(body.prompt, kwargs)

    

class WorkerActor(xo.Actor):
    def __init__(self, controller_address: str):
        super().__init__()
        self._controller_address = controller_address
        self._model_uid_to_model: Dict[str, xo.ActorRefType["ModelActor"]] = {}
        self._model_uid_to_model_spec: Dict[str, ModelSpec] = {}

    @classmethod
    def uid(cls) -> str:
        return "plexar_worker"

    async def __post_create__(self):
        controller_ref: xo.ActorRefType["ControllerActor"] = await xo.actor_ref(
            address=self._controller_address, uid=ControllerActor.uid()
        )
        await controller_ref.add_worker(self.address)

    async def get_model_count(self) -> int:
        return len(self._model_uid_to_model)

    @log
    async def launch_builtin_model(
        self,
        model_uid: str,
        model_name: str,
        model_size_in_billions: Optional[int],
        model_format: Optional[str],
        quantization: Optional[str],
        **kwargs,
    ) -> xo.ActorRefType["ModelActor"]:
        assert model_uid not in self._model_uid_to_model

        from plexar.model import MODEL_FAMILIES

        for model_family in MODEL_FAMILIES:
            model_spec = model_family.match(
                model_name=model_name,
                model_format=model_format,
                model_size_in_billions=model_size_in_billions,
                quantization=quantization,
            )

            if model_spec is None:
                continue

            cls = model_family.cls
            save_path = model_family.cache(
                model_spec.model_size_in_billions, model_spec.quantization
            )
            model = cls(save_path, kwargs)
            model_ref = await xo.create_actor(
                ModelActor, address=self.address, uid=model_uid, model=model
            )
            self._model_uid_to_model[model_uid] = model_ref
            self._model_uid_to_model_spec[model_uid] = model_spec
            return model_ref

        raise ValueError(
            f"Model not found, name: {model_name}, format: {model_format},"
            f" size: {model_size_in_billions}, quantization: {quantization}"
        )

    @log
    async def terminate_model(self, model_uid: str):
        assert model_uid in self._model_uid_to_model

        model_ref = self._model_uid_to_model[model_uid]
        await xo.destroy_actor(model_ref)
        del self._model_uid_to_model[model_uid]
        del self._model_uid_to_model_spec[model_uid]

    @log
    async def list_models(self) -> List[tuple[str, ModelSpec]]:
        return list(self._model_uid_to_model_spec.items())

    @log
    async def get_model(self, model_uid: str) -> xo.ActorRefType["ModelActor"]:
        assert model_uid in self._model_uid_to_model

        return self._model_uid_to_model[model_uid]