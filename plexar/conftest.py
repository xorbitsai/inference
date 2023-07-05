# import xoscar as xo
# import pytest
#
# from .client import Client
# from .core.service import SupervisorActor, WorkerActor
#
#
import pytest_asyncio
import xoscar as xo

from plexar.core.service import SupervisorActor, WorkerActor


@pytest_asyncio.fixture
async def setup():
    address = "127.0.0.1:9998"
    pool = await xo.create_actor_pool(address, n_process=0)
    await xo.create_actor(
        SupervisorActor, address=pool.external_address, uid=SupervisorActor.uid()
    )
    await xo.create_actor(
        WorkerActor,
        address=address,
        uid=WorkerActor.uid(),
        supervisor_address=address,
    )  # worker
    yield pool
