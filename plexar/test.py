import time

import uvicorn
from fastapi import FastAPI, APIRouter
from uvicorn import Config, Server

from plexar.actor import ModelActor

import xoscar as xo
import asyncio
from typing import TypeVar

from plexar.model.llm.vicuna import VicunaUncensoredGgml


async def main_():
    from plexar.client import Client
    client = Client("localhost:9999")
    baichuan_uid = client.launch_model("baichuan")
    baichuan_ref = client.get_model(baichuan_uid)
    await baichuan_ref.generate('once opon a time, there was a very old computer.',
                                {'max_tokens': 512, 'stream': False})

# class WebActor(xo.Actor):
#     def __init__(self):
#         super().__init__()
#         # fastapi.
#         app = FastAPI()
#         self.router = APIRouter()
#         self.router.add_api_route("/", self.hello, methods=["GET"])
#         app.include_router(self.router)

#         # uvicorn.
#         loop = asyncio.get_event_loop()
#         config = Config(app=app, loop=loop, host="0.0.0.0", port=8000)
#         server = Server(config)
#         loop.create_task(server.serve())

#     def hello(self):
#         return {"Hello": "World"}


# class WebActor(xo.Actor):
#     def __init__(self):
#         super().__init__()
#         # fastapi.
#         self._app = FastAPI()
#         # self.router = APIRouter()
#         # self.router.add_api_route("/", self.hello, methods=["GET"])
#         # app.include_router(self.router)

#         # uvicorn.
#         loop = asyncio.get_event_loop()
#         config = Config(app=app, loop=loop, host="0.0.0.0", port=8000)
#         server = Server(config)
#         loop.create_task(server.serve())

#     @self._app.get("/")
#     def hello():
#         return {"Hello": "World"}





# async def main_():
#     pool = await xo.create_actor_pool(address="localhost:9999", n_process=0)
#     await xo.create_actor(WebActor, address="localhost:9999", uid="web")
#     await pool.join()


# if __name__ == '__main__':
#     pool = asyncio.get_event_loop()
#     pool.run_until_complete(main_())


loop = asyncio.get_event_loop()
loop.run_until_complete(main_())

# loop.close()


