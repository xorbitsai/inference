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

import logging

import click

from .. import __version__
from ..constants import (
    PLEXAR_DEFAULT_CONTROLLER_PORT,
    PLEXAR_DEFAULT_HOST,
    PLEXAR_DEFAULT_WORKER_PORT,
)


@click.group(name="plexar")
@click.version_option(__version__, "--version", "-v")
def cli():
    pass


@cli.command()
@click.option(
    "--address",
    "-a",
    default=f"{PLEXAR_DEFAULT_HOST}:{PLEXAR_DEFAULT_CONTROLLER_PORT}",
    type=str,
)
@click.option("--log-level", default="INFO", type=str)
@click.option("--share", is_flag=True)
@click.option("--host", "-h", default=None, type=str)
@click.option("--port", "-p", default=None, type=int)
def controller(
    address: str,
    log_level: str,
    share: bool,
    host: str,
    port: str,
):
    from ..deploy.controller import main

    if log_level:
        logging.basicConfig(level=logging.getLevelName(log_level.upper()))

    main(address=address, share=share, host=host, port=port)


@cli.command()
@click.option(
    "--address",
    "-a",
    default=f"{PLEXAR_DEFAULT_HOST}:{PLEXAR_DEFAULT_WORKER_PORT}",
    type=str,
)
@click.option(
    "--controller-address",
    default=f"{PLEXAR_DEFAULT_HOST}:{PLEXAR_DEFAULT_CONTROLLER_PORT}",
    type=str,
)
@click.option("--log-level", default="INFO", type=str)
def worker(address: str, controller_address: str, log_level: str):
    from ..deploy.worker import main

    if log_level:
        logging.basicConfig(level=logging.getLevelName(log_level.upper()))

    main(address=address, controller_address=controller_address)


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
@click.option("--share", is_flag=True)
@click.option("--host", "-h", default=None, type=str)
@click.option("--port", "-p", default=None, type=int)
def model_launch(
    name: str,
    size_in_billions: int,
    model_format: str,
    quantization: str,
    share: bool,
    host: str,
    port: str,
):
    address = f"{PLEXAR_DEFAULT_HOST}:{PLEXAR_DEFAULT_CONTROLLER_PORT}"

    from .local import main

    main(
        address=address,
        model_name=name,
        size_in_billions=size_in_billions,
        model_format=model_format,
        quantization=quantization,
        share=share,
        host=host,
        port=port,
    )


if __name__ == "__main__":
    cli()