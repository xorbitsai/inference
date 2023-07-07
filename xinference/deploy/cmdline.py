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
from xoscar.utils import get_next_port

from .. import __version__
from ..client import RESTfulClient
from ..constants import XINFERENCE_DEFAULT_ENDPOINT_PORT, XINFERENCE_DEFAULT_HOST


@click.group(invoke_without_command=True, name="xinference")
@click.pass_context
@click.version_option(__version__, "--version", "-v")
@click.option("--log-level", default="INFO", type=str)
@click.option("--host", "-H", default=XINFERENCE_DEFAULT_HOST, type=str)
@click.option("--port", "-p", default=XINFERENCE_DEFAULT_ENDPOINT_PORT, type=int)
def cli(
    ctx,
    log_level: str,
    host: str,
    port: str,
):
    if ctx.invoked_subcommand is None:
        from .local import main

        if log_level:
            logging.basicConfig(level=logging.getLevelName(log_level.upper()))

        address = f"{host}:{get_next_port()}"
        main(
            address=address,
            model_name=None,
            size_in_billions=None,
            model_format=None,
            quantization=None,
            host=host,
            port=port,
            use_launched_model=False,
        )


@click.command()
@click.option("--log-level", default="INFO", type=str)
@click.option("--host", "-H", default=XINFERENCE_DEFAULT_HOST, type=str)
@click.option("--port", "-p", default=XINFERENCE_DEFAULT_ENDPOINT_PORT, type=int)
def supervisor(
    log_level: str,
    host: str,
    port: str,
):
    from ..deploy.supervisor import main

    if log_level:
        logging.basicConfig(level=logging.getLevelName(log_level.upper()))

    address = f"{host}:{get_next_port()}"
    main(address=address, host=host, port=port)


@click.command()
@click.option("--log-level", default="INFO", type=str)
@click.option(
    "--endpoint",
    "-e",
    default=f"http://{XINFERENCE_DEFAULT_HOST}:{XINFERENCE_DEFAULT_ENDPOINT_PORT}",
    type=str,
)
@click.option("--host", "-H", default=XINFERENCE_DEFAULT_HOST, type=str)
def worker(log_level: str, endpoint: str, host: str):
    from ..deploy.worker import main

    if log_level:
        logging.basicConfig(level=logging.getLevelName(log_level.upper()))

    client = RESTfulClient(base_url=endpoint)
    supervisor_internal_addr = client.get_supervisor_internal_address()

    address = f"{host}:{get_next_port()}"
    main(address=address, supervisor_address=supervisor_internal_addr)


@cli.command("launch")
@click.option(
    "--endpoint",
    "-e",
    default=f"http://{XINFERENCE_DEFAULT_HOST}:{XINFERENCE_DEFAULT_ENDPOINT_PORT}",
    type=str,
)
@click.option("--model-name", "-n", type=str)
@click.option("--size-in-billions", "-s", default=None, type=int)
@click.option("--model-format", "-f", default=None, type=str)
@click.option("--quantization", "-q", default=None, type=str)
def model_launch(
    endpoint: str,
    model_name: str,
    size_in_billions: int,
    model_format: str,
    quantization: str,
):
    client = RESTfulClient(base_url=endpoint)
    model_uid = client.launch_model(
        model_name=model_name,
        model_size_in_billions=size_in_billions,
        model_format=model_format,
        quantization=quantization,
    )

    print(f"Model uid: {model_uid}")


@cli.command("list")
@click.option(
    "--endpoint",
    "-e",
    default=f"http://{XINFERENCE_DEFAULT_HOST}:{XINFERENCE_DEFAULT_ENDPOINT_PORT}",
    type=str,
)
@click.option("--all", is_flag=True)
def model_list(endpoint: str, all: bool):
    import sys

    from tabulate import tabulate

    from ..model import MODEL_FAMILIES

    if all:
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
    else:
        client = RESTfulClient(base_url=endpoint)
        models = client.list_models()
        print(
            tabulate(
                models,
                headers=[
                    "ModelUid",
                    "Name",
                    "Format",
                    "Size (in billions)",
                    "Quantization",
                ],
            ),
            file=sys.stderr,
        )


@cli.command("terminate")
@click.option(
    "--endpoint",
    "-e",
    default=f"http://{XINFERENCE_DEFAULT_HOST}:{XINFERENCE_DEFAULT_ENDPOINT_PORT}",
    type=str,
)
@click.option("--model-uid", type=str)
def model_terminate(
    endpoint: str,
    model_uid: str,
):
    client = RESTfulClient(base_url=endpoint)
    client.terminate_model(model_uid=model_uid)


@cli.command("generate")
@click.option(
    "--endpoint",
    "-e",
    default=f"http://{XINFERENCE_DEFAULT_HOST}:{XINFERENCE_DEFAULT_ENDPOINT_PORT}",
    type=str,
)
@click.option("--model-uid", type=str)
@click.option("--prompt", type=str)
def model_generate(endpoint: str, model_uid: str, prompt: str):
    async def generate_internal():
        # async tasks generating text.
        client = RESTfulClient(base_url=endpoint)
        async for completion_chunk in await client.generate(
            model_uid, prompt, stream=True
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


@cli.command("chat")
@click.option(
    "--endpoint",
    "-e",
    default=f"http://{XINFERENCE_DEFAULT_HOST}:{XINFERENCE_DEFAULT_ENDPOINT_PORT}",
    type=str,
)
@click.option("--model-uid", required=True, type=str)
def model_chat(endpoint: str, model_uid: str):
    async def chat_internal():
        # async tasks generating text.
        client = RESTfulClient(base_url=endpoint)
        chat_history = []
        while True:
            prompt = input("\nUser: ")
            response = []
            if prompt == "exit" or prompt == "e":
                break
            chat_history.append({"role": "user", "content": prompt})
            print("Assistant:", end="")
            async for completion_chunk in await client.chat(
                model_uid,
                prompt,
                chat_history=chat_history,
                generate_config={"stream": True},
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
