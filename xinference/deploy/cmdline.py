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
import logging

import click

from .. import __version__
from ..client import AsyncClient
from ..constants import (
    XINFERENCE_DEFAULT_HOST,
    XINFERENCE_DEFAULT_SUPERVISOR_PORT,
    XINFERENCE_DEFAULT_WORKER_PORT,
)


@click.group(name="xinference")
@click.version_option(__version__, "--version", "-v")
def cli():
    pass


@cli.command()
@click.option(
    "--address",
    "-a",
    default=f"{XINFERENCE_DEFAULT_HOST}:{XINFERENCE_DEFAULT_SUPERVISOR_PORT}",
    type=str,
)
@click.option("--log-level", default="INFO", type=str)
@click.option("--host", "-h", default=None, type=str)
@click.option("--port", "-p", default=None, type=int)
def supervisor(
    address: str,
    log_level: str,
    host: str,
    port: str,
):
    from ..deploy.supervisor import main

    if log_level:
        logging.basicConfig(level=logging.getLevelName(log_level.upper()))

    main(address=address, host=host, port=port)


@cli.command()
@click.option(
    "--address",
    "-a",
    default=f"{XINFERENCE_DEFAULT_HOST}:{XINFERENCE_DEFAULT_WORKER_PORT}",
    type=str,
)
@click.option(
    "--supervisor-address",
    default=f"{XINFERENCE_DEFAULT_HOST}:{XINFERENCE_DEFAULT_SUPERVISOR_PORT}",
    type=str,
)
@click.option("--log-level", default="INFO", type=str)
def worker(address: str, supervisor_address: str, log_level: str):
    from ..deploy.worker import main

    if log_level:
        logging.basicConfig(level=logging.getLevelName(log_level.upper()))

    main(address=address, supervisor_address=supervisor_address)


@cli.group()
def model():
    pass


@model.command("list")
def model_list():
    import sys

    from tabulate import tabulate

    from ..model import MODEL_FAMILIES

    table = []
    for model_family in MODEL_FAMILIES:
        table.append(
            [
                model_family.model_name,
                model_family.model_format,
                model_family.model_sizes_in_billions,
                model_family.quantizations,
            ]
        )

    print(
        tabulate(
            table, headers=["Name", "Format", "Size (in billions)", "Quantization"]
        ),
        file=sys.stderr,
    )


@model.command("launch")
@click.option("--name", "-n", type=str)
@click.option("--size-in-billions", "-s", default=None, type=int)
@click.option("--model-format", "-f", default=None, type=str)
@click.option("--quantization", "-q", default=None, type=str)
@click.option("--host", "-h", default=None, type=str)
@click.option("--port", "-p", default=None, type=int)
def model_launch(
    name: str,
    size_in_billions: int,
    model_format: str,
    quantization: str,
    host: str,
    port: str,
):
    address = f"{XINFERENCE_DEFAULT_HOST}:{XINFERENCE_DEFAULT_SUPERVISOR_PORT}"

    from .local import main

    main(
        address=address,
        model_name=name,
        size_in_billions=size_in_billions,
        model_format=model_format,
        quantization=quantization,
        host=host,
        port=port,
    )


@model.command("generate")
@click.option(
    "--address",
    "-a",
    default=f"{XINFERENCE_DEFAULT_HOST}:{XINFERENCE_DEFAULT_SUPERVISOR_PORT}",
    type=str,
)
@click.option("--model-uid", type=str)
@click.option("--prompt", type=str)
def model_generate(address: str, model_uid: str, prompt: str):
    async def generate_internal():
        # async tasks generating text.
        async_client = AsyncClient(supervisor_address=address)
        model_ref = await async_client.get_model(model_uid)
        async for completion_chunk in await model_ref.generate(
            prompt, {"stream": True}
        ):
            print(completion_chunk["choices"][0]["text"], end="", flush=True)

    loop = asyncio.get_event_loop()
    coro = generate_internal()

    if loop.is_running():
        # for testing.
        from ..isolation import Isolation

        isolation = Isolation(asyncio.new_event_loop(), threaded=True)
        isolation.start()
        isolation.call(coro)
    else:
        task = loop.create_task(coro)
        try:
            loop.run_until_complete(task)
        except KeyboardInterrupt:
            task.cancel()
            loop.run_until_complete(task)
            # avoid displaying exception-unhandled warnings
            task.exception()


@model.command("chat")
@click.option(
    "--address",
    "-a",
    default=f"{XINFERENCE_DEFAULT_HOST}:{XINFERENCE_DEFAULT_SUPERVISOR_PORT}",
    type=str,
)
@click.option("--model-uid", required=True, type=str)
def model_chat(address: str, model_uid: str):
    async def chat_internal():
        # async tasks generating text.
        async_client = AsyncClient(supervisor_address=address)
        model_ref = await async_client.get_model(model_uid)
        chat_history = []
        while True:
            prompt = input("\nUser: ")
            response = []
            if prompt == "exit" or prompt == "e":
                break
            chat_history.append({"role": "user", "content": prompt})
            print("Assistant:", end="")
            async for completion_chunk in await model_ref.chat(
                prompt, chat_history=chat_history, generate_config={"stream": True}
            ):
                delta = completion_chunk["choices"][0]["delta"]
                if "content" not in delta:
                    continue
                else:
                    print(delta["content"], end="", flush=True)
                    response.append(delta["content"])
            chat_history.append({"role": "assistant", "content": response})

    loop = asyncio.get_event_loop()
    coro = chat_internal()

    if loop.is_running():
        # for testing.
        from ..isolation import Isolation

        isolation = Isolation(asyncio.new_event_loop(), threaded=True)
        isolation.start()
        isolation.call(coro)
    else:
        task = loop.create_task(coro)
        try:
            loop.run_until_complete(task)
        except KeyboardInterrupt:
            task.cancel()
            loop.run_until_complete(task)
            # avoid displaying exception-unhandled warnings
            task.exception()

    print("Thank You For Chatting With Me, Have a Nice Day!")


if __name__ == "__main__":
    cli()
