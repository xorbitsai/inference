# Copyright 2022-2026 XProbe Inc.
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
import os
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import xoscar as xo
from vllm import envs
from vllm.executor.executor_base import DistributedExecutorBase
from vllm.utils import _run_task_with_lock, get_distributed_init_method
from vllm.worker.worker_base import WorkerWrapperBase
from xoscar.utils import get_next_port

try:
    from vllm.v1.executor.abstract import Executor as ExecutorV1
except ImportError:
    ExecutorV1 = None

from ....isolation import Isolation

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.sampler import SamplerOutput
    from vllm.sequence import ExecuteModelRequest

logger = logging.getLogger(__name__)

DEBUG_EXECUTOR = bool(int(os.getenv("XINFERENCE_DEBUG_VLLM_EXECUTOR", "0")))


class WorkerActor(xo.StatelessActor):
    def __init__(self, vllm_config: "VllmConfig", rpc_rank: int = 0, **kwargs):
        super().__init__(**kwargs)
        self._worker = WorkerWrapperBase(vllm_config, rpc_rank=rpc_rank)

    async def __post_create__(self):
        try:
            # Change process title for model
            import setproctitle

            setproctitle.setproctitle(f"Xinf vLLM worker: {self._worker.rpc_rank}")
        except ImportError:
            pass

    def __getattr__(self, item):
        return getattr(self._worker, item)

    @classmethod
    def gen_uid(cls, rank):
        return f"VllmWorker_{rank}"

    def execute_method(self, method: Union[str, Callable], *args, **kwargs):
        if DEBUG_EXECUTOR:
            # NOTE: too many logs, but useful for debug
            logger.debug(
                "Calling method %s in vllm worker %s, args: %s, kwargs: %s",
                method,
                self.uid,
                args,
                kwargs,
            )
        if isinstance(method, str):
            return getattr(self._worker, method)(*args, **kwargs)
        else:
            return method(self._worker, *args, **kwargs)


class WorkerWrapper:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        worker_actor_ref: xo.ActorRefType[WorkerActor],
    ):
        self._loop = loop
        self._worker_actor_ref = worker_actor_ref

    def execute_method(self, method: Union[str, Callable], *args, **kwargs):
        coro = self._worker_actor_ref.execute_method(method, *args, **kwargs)
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def execute_method_async(self, method: Union[str, Callable], *args, **kwargs):
        return await self._worker_actor_ref.execute_method(method, *args, **kwargs)

    def kill(self):
        coro = xo.destroy_actor(self._worker_actor_ref)
        return asyncio.run_coroutine_threadsafe(coro, self._loop)


class XinferenceDistributedExecutor(DistributedExecutorBase):
    """Xoscar based distributed executor"""

    uses_ray: bool = False
    _loop: asyncio.AbstractEventLoop
    _pool_addresses: List[str]
    _n_worker: int

    def __init__(
        self,
        vllm_config: "VllmConfig",
        pool_addresses: List[str],
        n_worker: int,
        loop: asyncio.AbstractEventLoop,
        *args,
        **kwargs,
    ):
        self._pool_addresses = pool_addresses
        self._loop = loop
        self._n_worker = n_worker
        self._is_shutdown = False
        super().__init__(vllm_config, *args, **kwargs)

    def _create_workers(self, refs: xo.ActorRefType[WorkerActor]) -> None:
        self.driver_worker: Optional[WorkerActor] = None
        # The remaining workers are Xoscar actors
        self.workers: List[WorkerWrapper] = []

        self.workers = [WorkerWrapper(self._loop, ref) for ref in refs[1:]]

        # driver worker only for vllm v0
        self.driver_worker = WorkerActor(self.vllm_config, rpc_rank=0)

        def driver_execute_method(*args, **kwargs):
            func = partial(self.driver_worker.execute_method, *args, **kwargs)
            return self._loop.run_in_executor(None, func)

        self.driver_exec_method = driver_execute_method

    def _init_executor(self) -> None:
        # Create the parallel GPU workers.
        world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size

        assert (
            self._pool_addresses and len(self._pool_addresses) == world_size
        ), f"Pool addresses(#{len(self._pool_addresses or [])} must be equal to worldsize(#{world_size})"

        futures = []
        for rank in range(world_size):
            coro = xo.create_actor(
                WorkerActor,
                self.vllm_config,
                rpc_rank=rank,
                address=self._pool_addresses[rank],
                uid=WorkerActor.gen_uid(rank),
            )
            futures.append(asyncio.run_coroutine_threadsafe(coro, self._loop))
        refs: List[xo.ActorRefType[WorkerActor]] = [fut.result() for fut in futures]

        # create workers
        self._create_workers(refs)

        # Set environment variables for the driver and workers.
        all_args_to_update_environment_variables: List[Dict[str, str]] = [
            dict() for _ in range(world_size)
        ]

        for args in all_args_to_update_environment_variables:
            # some carry-over env vars from the driver
            # TODO: refactor platform-specific env vars
            for name in [
                "VLLM_ATTENTION_BACKEND",
                "TPU_CHIPS_PER_HOST_BOUNDS",
                "TPU_HOST_BOUNDS",
                "VLLM_USE_V1",
                "VLLM_TRACE_FUNCTION",
            ]:
                if name in os.environ:
                    args[name] = os.environ[name]

        self._env_vars_for_all_workers = all_args_to_update_environment_variables

        self._run_workers(
            "update_environment_variables", self._env_vars_for_all_workers
        )

        all_kwargs = []
        distributed_init_method = get_distributed_init_method(
            self._pool_addresses[0].split(":", 1)[0], get_next_port()
        )
        for rank in range(world_size):
            local_rank = rank % (world_size // self._n_worker)
            kwargs = dict(
                vllm_config=self.vllm_config,
                local_rank=local_rank,
                rank=rank,
                distributed_init_method=distributed_init_method,
                is_driver_worker=not self.parallel_config
                or (rank % tensor_parallel_size == 0),
            )
            all_kwargs.append(kwargs)
        self._run_workers("init_worker", all_kwargs)
        self._run_workers("init_device")
        self._run_workers(
            "load_model",
            max_concurrent_workers=self.parallel_config.max_parallel_loading_workers,
        )

        # This is the list of workers that are rank 0 of each TP group EXCEPT
        # global rank 0. These are the workers that will broadcast to the
        # rest of the workers.
        self.tp_driver_workers: List[WorkerWrapper] = []
        # This is the list of workers that are not drivers and not the first
        # worker in a TP group. These are the workers that will be
        # broadcasted to.
        self.non_driver_workers: List[WorkerWrapper] = []

        # Enforce rank order for correct rank to return final output.
        for index, worker in enumerate(self.workers):
            # The driver worker is rank 0 and not in self.workers.
            rank = index + 1
            if rank % self.parallel_config.tensor_parallel_size == 0:
                self.tp_driver_workers.append(worker)
            else:
                self.non_driver_workers.append(worker)

        self.pp_locks: Optional[List[asyncio.Lock]] = None

    def _run_workers(
        self,
        method: Union[str, Callable],
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        if max_concurrent_workers:
            raise NotImplementedError("max_concurrent_workers is not supported yet.")

        workers = self.workers
        if async_run_tensor_parallel_workers_only:
            workers = self.non_driver_workers
        worker_outputs = [
            worker.execute_method(method, *args, **kwargs) for worker in workers
        ]

        if async_run_tensor_parallel_workers_only:
            return worker_outputs

        driver_worker_outputs = [
            self.driver_worker.execute_method(method, *args, **kwargs)  # type: ignore
        ]
        return driver_worker_outputs + [output.result() for output in worker_outputs]

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        for result in parallel_worker_tasks:
            result.get()

    def check_health(self) -> None:
        # Assume that the workers are healthy.
        # TODO: check the health by checking if the workers all alive
        return

    def shutdown(self) -> None:
        if self._is_shutdown:
            return

        try:
            self._is_shutdown = True
            futs = [worker.kill() for worker in self.workers]
            _ = [fut.result() for fut in futs]
        except (RuntimeError, ConnectionError, xo.ActorNotExist):
            # event loop closed already, ignore
            # or actor already removed
            pass

    def __del__(self):
        return self.shutdown()

    def _driver_execute_model(
        self, execute_model_req: Optional["ExecuteModelRequest"]
    ) -> Optional[List["SamplerOutput"]]:
        return self.driver_worker.execute_method("execute_model", execute_model_req)  # type: ignore

    async def _driver_execute_model_async(
        self,
        execute_model_req: Optional["ExecuteModelRequest"] = None,
    ) -> List["SamplerOutput"]:
        if not self.tp_driver_workers:
            return await self.driver_exec_method("execute_model", execute_model_req)

        if self.pp_locks is None:
            # This locks each pipeline parallel stage so multiple virtual
            # engines can't execute on the same stage at the same time
            # We create the locks here to avoid creating them in the constructor
            # which uses a different asyncio loop.
            self.pp_locks = [
                asyncio.Lock()
                for _ in range(self.parallel_config.pipeline_parallel_size)
            ]

        tasks = [
            asyncio.create_task(
                _run_task_with_lock(
                    self.driver_exec_method,
                    self.pp_locks[0],
                    "execute_model",
                    execute_model_req,
                )
            )
        ]
        for pp_rank, driver_worker in enumerate(self.tp_driver_workers, start=1):
            tasks.append(
                asyncio.create_task(
                    _run_task_with_lock(
                        driver_worker.execute_method_async,
                        self.pp_locks[pp_rank],
                        "execute_model",
                        execute_model_req,
                    )
                )
            )

        results = await asyncio.gather(*tasks)

        # Only the last PP stage has the final results.
        return results[-1]

    async def _start_worker_execution_loop(self):
        coros = [
            worker.execute_method_async("start_worker_execution_loop")
            for worker in self.non_driver_workers
        ]
        return await asyncio.gather(*coros)


if ExecutorV1:

    class XinferenceDistributedExecutorV1(XinferenceDistributedExecutor, ExecutorV1):
        def __init__(
            self,
            vllm_config: "VllmConfig",
            pool_addresses: List[str],
            n_worker: int,
            *args,
            **kwargs,
        ):
            assert envs.VLLM_USE_V1

            isolation = Isolation(asyncio.new_event_loop())
            isolation.start()
            loop = isolation.loop

            XinferenceDistributedExecutor.__init__(
                self, vllm_config, pool_addresses, n_worker, loop, *args, **kwargs
            )

        def _create_workers(self, refs: xo.ActorRefType[WorkerActor]) -> None:
            self.workers = [WorkerWrapper(self._loop, ref) for ref in refs]

        def execute_model(
            self,
            execute_model_req: "ExecuteModelRequest",
        ) -> List["SamplerOutput"]:
            outputs = self._run_workers("execute_model", execute_model_req)
            return outputs[0]

        def _run_workers(
            self,
            method: Union[str, Callable],
            *args,
            async_run_tensor_parallel_workers_only: bool = False,
            max_concurrent_workers: Optional[int] = None,
            **kwargs,
        ) -> Any:
            if max_concurrent_workers:
                raise NotImplementedError(
                    "max_concurrent_workers is not supported yet."
                )

            workers = self.workers
            if async_run_tensor_parallel_workers_only:
                workers = self.non_driver_workers
            worker_outputs = [
                worker.execute_method(method, *args, **kwargs) for worker in workers
            ]

            if async_run_tensor_parallel_workers_only:
                return worker_outputs

            return [output.result() for output in worker_outputs]
