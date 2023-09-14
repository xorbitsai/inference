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
import configparser
import logging
import os
import sys
from typing import List, Optional, Union

import click
from xoscar.utils import get_next_port

from .. import __version__
from ..client import (
    Client,
    RESTfulChatglmCppChatModelHandle,
    RESTfulChatModelHandle,
    RESTfulClient,
    RESTfulGenerateModelHandle,
)
from ..constants import (
    XINFERENCE_DEFAULT_DISTRIBUTED_HOST,
    XINFERENCE_DEFAULT_ENDPOINT_PORT,
    XINFERENCE_DEFAULT_LOCAL_HOST,
    XINFERENCE_ENV_ENDPOINT,
)
from ..isolation import Isolation
from ..types import ChatCompletionMessage

try:
    # provide elaborate line editing and history features.
    # https://docs.python.org/3/library/functions.html#input
    import readline  # noqa: F401
except ImportError:
    pass


def get_config_string(log_level: str) -> str:
    return f"""[loggers]
keys=root

[handlers]
keys=stream_handler

[formatters]
keys=formatter

[logger_root]
level={log_level.upper()}
handlers=stream_handler

[handler_stream_handler]
class=StreamHandler
formatter=formatter
level={log_level.upper()}
args=(sys.stderr,)

[formatter_formatter]
format=%(asctime)s %(name)-12s %(process)d %(levelname)-8s %(message)s
"""


def get_endpoint(endpoint: Optional[str]) -> str:
    # user didn't specify the endpoint.
    if endpoint is None:
        if XINFERENCE_ENV_ENDPOINT in os.environ:
            return os.environ[XINFERENCE_ENV_ENDPOINT]
        else:
            default_endpoint = f"http://{XINFERENCE_DEFAULT_LOCAL_HOST}:{XINFERENCE_DEFAULT_ENDPOINT_PORT}"
            return default_endpoint
    else:
        return endpoint


@click.group(
    invoke_without_command=True,
    name="xinference",
    help="Xinference command-line interface for serving and deploying models.",
)
@click.pass_context
@click.version_option(
    __version__,
    "--version",
    "-v",
    help="Show the current version of the Xinference tool.",
)
@click.option(
    "--log-level",
    default="INFO",
    type=str,
    help="""Set the logger level. Options listed from most log to least log are:
              DEBUG > INFO > WARNING > ERROR > CRITICAL (Default level is INFO)""",
)
@click.option(
    "--host",
    "-H",
    default=XINFERENCE_DEFAULT_LOCAL_HOST,
    type=str,
    help="Specify the host address for the Xinference server.",
)
@click.option(
    "--port",
    "-p",
    default=XINFERENCE_DEFAULT_ENDPOINT_PORT,
    type=int,
    help="Specify the port number for the Xinference server.",
)
def cli(
    ctx,
    log_level: str,
    host: str,
    port: str,
):
    if ctx.invoked_subcommand is None:
        from .local import main

        logging_conf = configparser.RawConfigParser()
        logger_config_string = get_config_string(log_level)
        logging_conf.read_string(logger_config_string)
        logging.config.fileConfig(logging_conf)  # type: ignore

        address = f"{host}:{get_next_port()}"

        main(
            address=address,
            host=host,
            port=port,
            logging_conf=logging_conf,
        )


@click.command(
    help="Starts an Xinference supervisor to control and monitor the worker actors."
)
@click.option(
    "--log-level",
    default="INFO",
    type=str,
    help="""Set the logger level for the supervisor. Options listed from most log to least log are:
              DEBUG > INFO > WARNING > ERROR > CRITICAL (Default level is INFO)""",
)
@click.option(
    "--host",
    "-H",
    default=XINFERENCE_DEFAULT_DISTRIBUTED_HOST,
    type=str,
    help="Specify the host address for the supervisor.",
)
@click.option(
    "--port",
    "-p",
    default=XINFERENCE_DEFAULT_ENDPOINT_PORT,
    type=int,
    help="Specify the port number for the supervisor.",
)
def supervisor(
    log_level: str,
    host: str,
    port: str,
):
    from ..deploy.supervisor import main

    if log_level:
        logging.basicConfig(level=logging.getLevelName(log_level.upper()))
    logging_conf = dict(level=log_level.upper())

    address = f"{host}:{get_next_port()}"

    main(address=address, host=host, port=port, logging_conf=logging_conf)


@click.command(
    help="Starts an Xinference worker to execute tasks assigned by the supervisor in a distributed setup."
)
@click.option(
    "--log-level",
    default="INFO",
    type=str,
    help="""Set the logger level for the worker. Options listed from most log to least log are:
              DEBUG > INFO > WARNING > ERROR > CRITICAL (Default level is INFO)""",
)
@click.option("--endpoint", "-e", type=str, help="Xinference endpoint.")
@click.option(
    "--host",
    "-H",
    default=XINFERENCE_DEFAULT_DISTRIBUTED_HOST,
    type=str,
    help="Specify the host address for the worker.",
)
def worker(log_level: str, endpoint: Optional[str], host: str):
    from ..deploy.worker import main

    logging_conf = configparser.RawConfigParser()
    logger_config_string = get_config_string(log_level)
    logging_conf.read_string(logger_config_string)
    logging.config.fileConfig(logging_conf)  # type: ignore

    endpoint = get_endpoint(endpoint)

    client = RESTfulClient(base_url=endpoint)
    supervisor_internal_addr = client._get_supervisor_internal_address()

    address = f"{host}:{get_next_port()}"
    main(
        address=address,
        supervisor_address=supervisor_internal_addr,
        logging_conf=logging_conf,
    )


@cli.command("register", help="Registers a new model with Xinference for deployment.")
@click.option("--endpoint", "-e", type=str, help="Xinference endpoint.")
@click.option(
    "--model-type",
    "-t",
    default="LLM",
    type=str,
    help="Type of model to register (default is 'LLM').",
)
@click.option("--file", "-f", type=str, help="Path to the model configuration file.")
@click.option(
    "--persist",
    "-p",
    is_flag=True,
    help="Persist the model configuration to the filesystem, retains the model registration after server restarts.",
)
def register_model(
    endpoint: Optional[str],
    model_type: str,
    file: str,
    persist: bool,
):
    endpoint = get_endpoint(endpoint)
    with open(file) as fd:
        model = fd.read()

    client = RESTfulClient(base_url=endpoint)
    client.register_model(
        model_type=model_type,
        model=model,
        persist=persist,
    )


@cli.command(
    "unregister",
    help="Unregisters a model from Xinference, removing it from deployment.",
)
@click.option("--endpoint", "-e", type=str, help="Xinference endpoint.")
@click.option(
    "--model-type",
    "-t",
    default="LLM",
    type=str,
    help="Type of model to unregister (default is 'LLM').",
)
@click.option("--model-name", "-n", type=str, help="Name of the model to unregister.")
def unregister_model(
    endpoint: Optional[str],
    model_type: str,
    model_name: str,
):
    endpoint = get_endpoint(endpoint)

    client = RESTfulClient(base_url=endpoint)
    client.unregister_model(
        model_type=model_type,
        model_name=model_name,
    )


@cli.command("registrations", help="Lists all registered models in Xinference.")
@click.option(
    "--endpoint",
    "-e",
    type=str,
    help="Xinference endpoint.",
)
@click.option(
    "--model-type",
    "-t",
    default="LLM",
    type=str,
    help="Filter by model type (default is 'LLM').",
)
def list_model_registrations(
    endpoint: Optional[str],
    model_type: str,
):
    from tabulate import tabulate

    endpoint = get_endpoint(endpoint)

    client = RESTfulClient(base_url=endpoint)
    registrations = client.list_model_registrations(model_type=model_type)

    table = []
    for registration in registrations:
        model_name = registration["model_name"]
        model_family = client.get_model_registration(model_type, model_name)
        table.append(
            [
                model_type,
                model_family["model_name"],
                model_family["model_lang"],
                model_family["model_ability"],
                registration["is_builtin"],
            ]
        )
    print(
        tabulate(table, headers=["Type", "Name", "Language", "Ability", "Is-built-in"]),
        file=sys.stderr,
    )


@cli.command(
    "launch",
    help="Launch a model with the Xinference framework with the given parameters.",
)
@click.option(
    "--endpoint",
    "-e",
    type=str,
    help="Xinference endpoint.",
)
@click.option(
    "--model-name",
    "-n",
    type=str,
    required=True,
    help="Provide the name of the model to be launched.",
)
@click.option(
    "--model-type",
    "-t",
    type=str,
    default="LLM",
    help="Specify type of model, LLM as default.",
)
@click.option(
    "--size-in-billions",
    "-s",
    default=None,
    type=int,
    help="Specify the model size in billions of parameters.",
)
@click.option(
    "--model-format",
    "-f",
    default=None,
    type=str,
    help="Specify the format of the model, e.g. pytorch, ggmlv3, etc.",
)
@click.option(
    "--quantization",
    "-q",
    default=None,
    type=str,
    help="Define the quantization settings for the model.",
)
@click.option(
    "--replica",
    "-r",
    default=1,
    type=int,
    help="The replica count of the model, default is 1.",
)
@click.option(
    "--n-gpu",
    "-g",
    default="auto",
    type=str,
    help='The number of GPUs used by the model, default is "auto".',
)
def model_launch(
    endpoint: Optional[str],
    model_name: str,
    model_type: str,
    size_in_billions: int,
    model_format: str,
    quantization: str,
    replica: int,
    n_gpu: str,
):
    if n_gpu.lower() == "none":
        _n_gpu: Optional[Union[int, str]] = None
    elif n_gpu == "auto":
        _n_gpu = n_gpu
    else:
        _n_gpu = int(n_gpu)

    endpoint = get_endpoint(endpoint)

    client = RESTfulClient(base_url=endpoint)
    model_uid = client.launch_model(
        model_name=model_name,
        model_type=model_type,
        model_size_in_billions=size_in_billions,
        model_format=model_format,
        quantization=quantization,
        replica=replica,
        n_gpu=_n_gpu,
    )

    print(f"Model uid: {model_uid}", file=sys.stderr)


@cli.command(
    "list",
    help="List all running models in Xinference.",
)
@click.option(
    "--endpoint",
    "-e",
    type=str,
    help="Xinference endpoint.",
)
def model_list(endpoint: Optional[str]):
    from tabulate import tabulate

    endpoint = get_endpoint(endpoint)
    client = RESTfulClient(base_url=endpoint)

    llm_table = []
    embedding_table = []
    models = client.list_models()
    for model_uid, model_spec in models.items():
        if model_spec["model_type"] == "LLM":
            llm_table.append(
                [
                    model_uid,
                    model_spec["model_type"],
                    model_spec["model_name"],
                    model_spec["model_format"],
                    model_spec["model_size_in_billions"],
                    model_spec["quantization"],
                ]
            )
        elif model_spec["model_type"] == "embedding":
            embedding_table.append(
                [
                    model_uid,
                    model_spec["model_type"],
                    model_spec["model_name"],
                    model_spec["dimensions"],
                ]
            )
    if llm_table:
        print(
            tabulate(
                llm_table,
                headers=[
                    "UID",
                    "Type",
                    "Name",
                    "Format",
                    "Size (in billions)",
                    "Quantization",
                ],
            ),
            file=sys.stderr,
        )
    if embedding_table:
        print(
            tabulate(
                embedding_table,
                headers=[
                    "UID",
                    "Type",
                    "Name",
                    "Dimensions",
                ],
            ),
            file=sys.stderr,
        )


@cli.command(
    "terminate",
    help="Terminate a deployed model through unique identifier (UID) of the model.",
)
@click.option(
    "--endpoint",
    "-e",
    type=str,
    help="Xinference endpoint.",
)
@click.option(
    "--model-uid",
    type=str,
    required=True,
    help="The unique identifier (UID) of the model.",
)
def model_terminate(
    endpoint: Optional[str],
    model_uid: str,
):
    endpoint = get_endpoint(endpoint)

    client = RESTfulClient(base_url=endpoint)
    client.terminate_model(model_uid=model_uid)


@cli.command("generate", help="Generate text using a running LLM.")
@click.option("--endpoint", "-e", type=str, help="Xinference endpoint.")
@click.option(
    "--model-uid",
    type=str,
    help="The unique identifier (UID) of the model.",
)
@click.option(
    "--max_tokens",
    default=512,
    type=int,
    help="Maximum number of tokens in the generated text (default is 512).",
)
@click.option(
    "--stream",
    default=True,
    type=bool,
    help="Whether to stream the generated text. Use 'True' for streaming (default is True).",
)
def model_generate(
    endpoint: Optional[str],
    model_uid: str,
    max_tokens: int,
    stream: bool,
):
    endpoint = get_endpoint(endpoint)
    if stream:
        # TODO: when stream=True, RestfulClient cannot generate words one by one.
        # So use Client in temporary. The implementation needs to be changed to
        # RestfulClient in the future.
        async def generate_internal():
            while True:
                # the prompt will be written to stdout.
                # https://docs.python.org/3.10/library/functions.html#input
                prompt = input("Prompt: ")
                if prompt == "":
                    break
                print(f"Completion: {prompt}", end="", file=sys.stdout)
                async for chunk in model.generate(
                    prompt=prompt,
                    generate_config={"stream": stream, "max_tokens": max_tokens},
                ):
                    choice = chunk["choices"][0]
                    if "text" not in choice:
                        continue
                    else:
                        print(choice["text"], end="", flush=True, file=sys.stdout)
                print("", file=sys.stdout)

        client = Client(endpoint=endpoint)
        model = client.get_model(model_uid=model_uid)

        loop = asyncio.get_event_loop()
        coro = generate_internal()

        if loop.is_running():
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
    else:
        restful_client = RESTfulClient(base_url=endpoint)
        restful_model = restful_client.get_model(model_uid=model_uid)
        if not isinstance(
            restful_model, (RESTfulChatModelHandle, RESTfulGenerateModelHandle)
        ):
            raise ValueError(f"model {model_uid} has no generate method")

        while True:
            prompt = input("User: ")
            if prompt == "":
                break
            print(f"Assistant: {prompt}", end="", file=sys.stdout)
            response = restful_model.generate(
                prompt=prompt,
                generate_config={"stream": stream, "max_tokens": max_tokens},
            )
            if not isinstance(response, dict):
                raise ValueError("generate result is not valid")
            print(f"{response['choices'][0]['text']}\n", file=sys.stdout)


@cli.command("chat", help="Chat with a running LLM.")
@click.option("--endpoint", "-e", type=str, help="Xinference endpoint.")
@click.option("--model-uid", type=str, help="The unique identifier (UID) of the model.")
@click.option(
    "--max_tokens",
    default=512,
    type=int,
    help="Maximum number of tokens in each message (default is 512).",
)
@click.option(
    "--stream",
    default=True,
    type=bool,
    help="Whether to stream the chat messages. Use 'True' for streaming (default is True).",
)
def model_chat(
    endpoint: Optional[str],
    model_uid: str,
    max_tokens: int,
    stream: bool,
):
    # TODO: chat model roles may not be user and assistant.
    endpoint = get_endpoint(endpoint)
    chat_history: "List[ChatCompletionMessage]" = []
    if stream:
        # TODO: when stream=True, RestfulClient cannot generate words one by one.
        # So use Client in temporary. The implementation needs to be changed to
        # RestfulClient in the future.
        async def chat_internal():
            while True:
                # the prompt will be written to stdout.
                # https://docs.python.org/3.10/library/functions.html#input
                prompt = input("User: ")
                if prompt == "":
                    break
                print("Assistant: ", end="", file=sys.stdout)
                response_content = ""
                async for chunk in model.chat(
                    prompt=prompt,
                    chat_history=chat_history,
                    generate_config={"stream": stream, "max_tokens": max_tokens},
                ):
                    delta = chunk["choices"][0]["delta"]
                    if "content" not in delta:
                        continue
                    else:
                        response_content += delta["content"]
                        print(delta["content"], end="", flush=True, file=sys.stdout)
                print("", file=sys.stdout)
                chat_history.append(ChatCompletionMessage(role="user", content=prompt))
                chat_history.append(
                    ChatCompletionMessage(role="assistant", content=response_content)
                )

        client = Client(endpoint=endpoint)
        model = client.get_model(model_uid=model_uid)

        loop = asyncio.get_event_loop()
        coro = chat_internal()

        if loop.is_running():
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
    else:
        restful_client = RESTfulClient(base_url=endpoint)
        restful_model = restful_client.get_model(model_uid=model_uid)
        if not isinstance(
            restful_model, (RESTfulChatModelHandle, RESTfulChatglmCppChatModelHandle)
        ):
            raise ValueError(f"model {model_uid} has no chat method")

        while True:
            prompt = input("User: ")
            if prompt == "":
                break
            chat_history.append(ChatCompletionMessage(role="user", content=prompt))
            print("Assistant: ", end="", file=sys.stdout)
            response = restful_model.chat(
                prompt=prompt,
                chat_history=chat_history,
                generate_config={"stream": stream, "max_tokens": max_tokens},
            )
            if not isinstance(response, dict):
                raise ValueError("chat result is not valid")
            response_content = response["choices"][0]["message"]["content"]
            print(f"{response_content}\n", file=sys.stdout)
            chat_history.append(
                ChatCompletionMessage(role="assistant", content=response_content)
            )


if __name__ == "__main__":
    cli()
