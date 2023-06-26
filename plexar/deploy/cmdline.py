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

import click

from plexar.actor import ModelActor

from .. import __version__


@click.group(name="plexar")
@click.version_option(__version__, "--version", "-v")
def cli():
    pass


@cli.group()
def model():
    pass


@model.command("list")
def model_list():
    raise NotImplemented


@model.command("launch")
@click.option("--path", "-p")
def model_launch(path):
    import asyncio
    import sys

    import xoscar as xo

    from plexar.model.llm.vicuna import VicunaUncensoredGgml

    async def _run():
        await xo.create_actor_pool(address="localhost:9999", n_process=1)

        vu = VicunaUncensoredGgml(model_path=path)
        vu_ref = await xo.create_actor(
            ModelActor, address="localhost:9999", uid="vu", model=vu
        )

        while True:
            i = input("User:\n")
            if i == "exit":
                break

            print(f"Assistant:")
            length = 0
            async for chunk in await vu_ref.chat(i):
                sys.stdout.write(chunk["text"])
                sys.stdout.flush()
                length += len(chunk["text"])
                if length >= 80:
                    print()
                    length = 0
            print()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(_run())
    loop.close()


if __name__ == "__main__":
    cli()
