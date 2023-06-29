async def start_controller_components(address: str, share: bool, host: str, port: int):
    await xo.create_actor(ControllerActor, address=address, uid=ControllerActor.uid())
    rest_ref = await xo.create_actor(RESTAPIActor, address=address, uid="restful", host="0.0.0.0", port=8000)
    gradio = await xo.create_actor(
        GradioActor,
        xoscar_endpoint=address,
        share=share,
        host=host,
        port=port,
        address=address,
        uid=GradioActor.default_uid(),
    )
    await gradio.launch()


async def _start_controller(address: str, host: str, port: int, share: bool):
    from .utils import create_actor_pool

    pool = await create_actor_pool(address=address, n_process=0)
    await start_controller_components(
        address=address, host=host, port=port, share=share
    )
    await pool.join()


def main(*args, **kwargs):
    loop = asyncio.get_event_loop()
    task = loop.create_task(_start_controller(*args, *kwargs))

    try:
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        task.cancel()
        loop.run_until_complete(task)
        # avoid displaying exception-unhandled warnings
        task.exception()