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
from typing import Callable, Dict, List, Optional

import xoscar as xo

from plexar.actor import ModelActor
from plexar.model import ModelSpec

from fastapi import FastAPI, APIRouter, Request
from fastapi import HTTPException
import asyncio
from uvicorn import Config, Server
import uuid

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

    @log
    async def launch_builtin_model(
        self,
        model_uid: str,
        model_name: str,
        n_parameters_in_billions: Optional[int] = None,
        fmt: Optional[str] = None,
        quantization: Optional[str] = None,
        **kwargs,
    ) -> xo.ActorRefType["ModelActor"]:
        assert model_uid not in self._model_uid_to_worker

        worker_ref = await self._choose_worker()
        model_ref = await worker_ref.launch_builtin_model(
            model_uid=model_uid,
            model_name=model_name,
            n_parameters_in_billions=n_parameters_in_billions,
            fmt=fmt,
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
        self.router.add_api_route("/models", self.list_models, methods=["GET"])
        self.router.add_api_route("/models", self.launch_model, methods=["POST"])
        self.router.add_api_route("/models/{model_uid}", self.terminate_model, methods=["DELETE"])
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
        n_parameters_in_billions = payload.get('n_parameters_in_billions')
        fmt = payload.get('fmt')
        quantization = payload.get('quantization')
        kwargs = payload.get('kwargs', {}) or {}

        model_uid = self.gen_model_uid()

        await self._controller_ref.launch_builtin_model(
            model_uid=model_uid,
            model_name=model_name,
            n_parameters_in_billions=n_parameters_in_billions,
            fmt=fmt,
            quantization=quantization,
            **kwargs
        )
        return model_uid

    async def terminate_model(self, model_uid: str):
        await self._controller_ref.terminate_model(model_uid)
        return {"message": "Model terminated successfully."}


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
        n_parameters_in_billions: Optional[int],
        fmt: Optional[str],
        quantization: Optional[str],
        **kwargs,
    ) -> xo.ActorRefType["ModelActor"]:
        assert model_uid not in self._model_uid_to_model

        from plexar.model import MODEL_SPECS

        for model_spec in MODEL_SPECS:
            if model_spec.match(
                model_name, n_parameters_in_billions, fmt, quantization
            ):
                logger.debug(f"Matched model spec for %s: %s", model_uid, model_spec)

                model_cls = model_spec.cls
                assert model_cls is not None

                save_path = model_spec.cache()
                model = model_cls(save_path, kwargs)
                model_ref = await xo.create_actor(
                    ModelActor, address=self.address, uid=model_uid, model=model
                )
                self._model_uid_to_model[model_uid] = model_ref
                self._model_uid_to_model_spec[model_uid] = model_spec
                return model_ref
        available_models = "\n".join(model_spec.name for model_spec in MODEL_SPECS)
        raise ValueError(
            f"\nModel {model_name} can't match any model, here are available models:\n"
            f"{available_models}"
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