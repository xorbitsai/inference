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
import configparser
import logging
import os
import sys
from typing import Optional

import click
from xoscar.utils import get_next_port

from .. import __version__
from ..client import RESTfulClient
from ..constants import (
    XINFERENCE_DEFAULT_DISTRIBUTED_HOST,
    XINFERENCE_DEFAULT_ENDPOINT_PORT,
    XINFERENCE_DEFAULT_LOCAL_HOST,
    XINFERENCE_ENV_ENDPOINT,
)


def get_config_string(log_level: str) -> str:
    return f"""
        [loggers]
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


@click.group(invoke_without_command=True, name="xinference")
@click.pass_context
@click.version_option(__version__, "--version", "-v")
@click.option("--log-level", default="INFO", type=str)
@click.option("--host", "-H", default=XINFERENCE_DEFAULT_LOCAL_HOST, type=str)
@click.option("--port", "-p", default=XINFERENCE_DEFAULT_ENDPOINT_PORT, type=int)
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


@click.command()
@click.option("--log-level", default="INFO", type=str)
@click.option("--host", "-H", default=XINFERENCE_DEFAULT_DISTRIBUTED_HOST, type=str)
@click.option("--port", "-p", default=XINFERENCE_DEFAULT_ENDPOINT_PORT, type=int)
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


@click.command()
@click.option("--log-level", default="INFO", type=str)
@click.option(
    "--endpoint",
    "-e",
    type=str,
)
@click.option("--host", "-H", default=XINFERENCE_DEFAULT_DISTRIBUTED_HOST, type=str)
def worker(log_level: str, endpoint: Optional[str], host: str):
    from ..deploy.worker import main

    logging_conf = configparser.RawConfigParser()
    logger_config_string = get_config_string(log_level)
    logging_conf.read_string(logger_config_string)
    logging.config.fileConfig(level=logging.getLevelName(log_level.upper()))  # type: ignore

    endpoint = get_endpoint(endpoint)

    client = RESTfulClient(base_url=endpoint)
    supervisor_internal_addr = client._get_supervisor_internal_address()

    address = f"{host}:{get_next_port()}"
    main(
        address=address,
        supervisor_address=supervisor_internal_addr,
        logging_conf=logging_conf,
    )


@cli.command("launch")
@click.option(
    "--endpoint",
    "-e",
    type=str,
)
@click.option("--model-name", "-n", type=str)
@click.option("--size-in-billions", "-s", default=None, type=int)
@click.option("--model-format", "-f", default=None, type=str)
@click.option("--quantization", "-q", default=None, type=str)
@click.option("--model-uid", "-i", default=None, type=str)
def model_launch(
    endpoint: Optional[str],
    model_name: str,
    size_in_billions: int,
    model_format: str,
    quantization: str,
    model_uid: str,
):
    endpoint = get_endpoint(endpoint)

    client = RESTfulClient(base_url=endpoint)
    model_uid = client.launch_model(
        model_name=model_name,
        model_size_in_billions=size_in_billions,
        model_format=model_format,
        quantization=quantization,
        model_uid=model_uid,
    )

    print(f"Model uid: {model_uid}", file=sys.stderr)


@cli.command("list")
@click.option(
    "--endpoint",
    "-e",
    type=str,
)
@click.option("--all", is_flag=True)
def model_list(endpoint: Optional[str], all: bool):
    from tabulate import tabulate

    # TODO: get from the supervisor
    from ..model.llm import BUILTIN_LLM_FAMILIES

    endpoint = get_endpoint(endpoint)

    table = []
    if all:
        for model_family in BUILTIN_LLM_FAMILIES:
            table.append(
                [
                    model_family.model_name,
                    model_family.model_lang,
                    model_family.model_ability,
                ]
            )

        print(
            tabulate(table, headers=["Name", "Language", "Ability"]),
            file=sys.stderr,
        )
    else:
        client = RESTfulClient(base_url=endpoint)
        models = client.list_models()
        for model_uid, model_spec in models.items():
            table.append(
                [
                    model_uid,
                    model_spec["model_name"],
                    model_spec["model_format"],
                    model_spec["model_size_in_billions"],
                    model_spec["quantization"],
                ]
            )
        print(
            tabulate(
                table,
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
    type=str,
)
@click.option("--model-uid", type=str)
def model_terminate(
    endpoint: Optional[str],
    model_uid: str,
):
    endpoint = get_endpoint(endpoint)

    client = RESTfulClient(base_url=endpoint)
    client.terminate_model(model_uid=model_uid)


if __name__ == "__main__":
    cli()
