# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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
import time
from concurrent.futures import Future
from threading import Thread
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import xoscar as xo
from vllm import envs
from vllm.v1.executor.abstract import Executor
from xoscar.utils import get_next_port

from ....isolation import Isolation
from .distributed_worker_actor import WorkerActor
from .utils import get_distributed_init_method

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

logger = logging.getLogger(__name__)
_SHUTDOWN_TIMEOUT_SECONDS = 10.0


class _ExecutorIsolation(Isolation):
    def _run(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            for task in asyncio.all_tasks(self._loop):
                task.cancel()
            # Let cancellation callbacks run without waiting for remote cancel
            # acknowledgements from a rank that may be stuck in NCCL.
            self._loop.run_until_complete(asyncio.sleep(0))
            self._loop.close()

    def stop(self, timeout: float = _SHUTDOWN_TIMEOUT_SECONDS) -> bool:
        if not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        thread = cast(Thread, self._thread)
        thread.join(timeout=timeout)
        return not thread.is_alive()


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


class XinferenceDistributedExecutorV1(Executor):
    """Xoscar based distributed executor"""

    supports_pp: bool = True

    _loop: asyncio.AbstractEventLoop
    _pool_addresses: List[str]
    _n_worker: int

    def __init__(
        self,
        vllm_config: "VllmConfig",
        pool_addresses: List[str],
        n_worker: int,
        *args,
        **kwargs,
    ):
        # XinferenceDistributedExecutorV1
        self._isolation = _ExecutorIsolation(asyncio.new_event_loop())
        self._isolation.start()
        loop = self._isolation.loop

        # XinferenceDistributedExecutor
        self._pool_addresses = pool_addresses
        self._loop = loop
        self._n_worker = n_worker
        self._is_shutdown = False
        self.workers: List[WorkerWrapper] = []

        # DistributedExecutorBase
        self.parallel_worker_tasks: Optional[Union[Any, Awaitable[Any]]] = None

        # Executor
        try:
            Executor.__init__(self, vllm_config, *args, **kwargs)
        except Exception:
            self.shutdown()
            raise

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        return True

    @property
    def max_concurrent_batches(self) -> int:
        # Newer vLLM versions own this setting in VllmConfig. Older V1
        # engines read it from the executor to fill all PP stages.
        configured = getattr(self.vllm_config, "max_concurrent_batches", None)
        if configured is not None:
            return configured
        pp_size = self.parallel_config.pipeline_parallel_size
        if pp_size == 1 and self.scheduler_config.async_scheduling:
            return 2
        return pp_size

    def _init_executor(self) -> None:
        # Create the parallel GPU workers.
        world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size

        if len(self._pool_addresses) != world_size:
            raise ValueError(
                f"Allocated GPU count ({len(self._pool_addresses)}) must equal "
                f"the vLLM world size ({world_size}); set the vLLM world size "
                "to the total allocated GPU count"
            )
        if self._n_worker <= 0 or world_size % self._n_worker != 0:
            raise ValueError(
                f"vLLM world size ({world_size}) must be divisible by "
                f"a positive n_worker ({self._n_worker})"
            )

        futures = []
        for rank in range(world_size):
            coro = xo.create_actor(
                WorkerActor,
                rpc_rank=rank,
                address=self._pool_addresses[rank],
                uid=WorkerActor.gen_uid(rank),
            )
            futures.append(asyncio.run_coroutine_threadsafe(coro, self._loop))
        refs: List[xo.ActorRefType[WorkerActor]] = []
        creation_error = None
        for fut in futures:
            try:
                refs.append(fut.result())
            except Exception as exc:
                if creation_error is None:
                    creation_error = exc

        # create workers
        self._create_workers(refs)
        if creation_error is not None:
            raise creation_error

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
            "update_environment_variables", args=(self._env_vars_for_all_workers,)
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
        self._run_workers("init_worker", args=(all_kwargs,))
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
            rank = index
            if rank == 0:
                continue
            if rank % self.parallel_config.tensor_parallel_size == 0:
                self.tp_driver_workers.append(worker)
            else:
                self.non_driver_workers.append(worker)

        self.pp_locks: Optional[List[asyncio.Lock]] = None

    def _get_output_rank(self) -> int:
        """Get the rank that produces the final output.

        In pipeline parallelism, only the last PP stage produces
        ModelRunnerOutput. The output rank is the first TP worker
        of the last PP stage.
        """
        return (
            self.parallel_config.world_size
            - self.parallel_config.tensor_parallel_size
            * getattr(self.parallel_config, "prefill_context_parallel_size", 1)
        )

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict] = None,
        non_block: bool = False,
    ) -> Union[List[Any], Future]:
        return self._run_workers(
            method, args=args, kwargs=kwargs, timeout=timeout, non_block=non_block
        )

    def execute_model(
        self, scheduler_output: "SchedulerOutput", non_block: bool = False
    ) -> Union["ModelRunnerOutput", None, Future[Union["ModelRunnerOutput", None]]]:
        return self._run_workers(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            output_rank=self._get_output_rank(),
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            aggregate_output=True,
        )

    def sample_tokens(
        self, grammar_output: Optional[Any] = None, non_block: bool = False
    ) -> Any:
        return self._run_workers(
            "sample_tokens",
            args=(grammar_output,),
            non_block=non_block,
            output_rank=self._get_output_rank(),
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            aggregate_output=True,
        )

    def take_draft_token_ids(self) -> Any:
        return self._run_workers(
            "take_draft_token_ids", output_rank=self._get_output_rank()
        )

    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)

    def shutdown(self) -> None:
        if self._is_shutdown:
            return

        self._is_shutdown = True
        deadline = time.monotonic() + _SHUTDOWN_TIMEOUT_SECONDS
        futs = []
        for worker in self.workers:
            try:
                futs.append(worker.kill())
            except Exception:
                logger.debug("Failed to destroy vLLM worker", exc_info=True)
        try:
            for fut in futs:
                try:
                    fut.result(timeout=max(0, deadline - time.monotonic()))
                except Exception:
                    logger.debug("Failed to destroy vLLM worker", exc_info=True)
        finally:
            if not self._isolation.stop(timeout=max(0, deadline - time.monotonic())):
                logger.warning("vLLM executor loop exceeded the shutdown deadline")

    def _create_workers(self, refs: List[xo.ActorRefType[WorkerActor]]) -> None:
        self.workers = [WorkerWrapper(self._loop, ref) for ref in refs]

    def _run_workers(
        self,
        method: Union[str, Callable],
        args: Tuple = (),
        kwargs: Optional[Dict] = None,
        *,
        async_run_tensor_parallel_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        non_block: bool = False,
        output_rank: Optional[int] = None,
        timeout: Optional[float] = None,
        aggregate_output: bool = False,
    ) -> Any:
        if max_concurrent_workers:
            raise NotImplementedError("max_concurrent_workers is not supported yet.")

        workers = self.workers
        if async_run_tensor_parallel_workers_only:
            workers = self.non_driver_workers
        worker_outputs = [
            worker.execute_method(method, *args, **(kwargs or {})) for worker in workers
        ]

        async def collect_outputs():
            # Observe every rank: a failed early PP stage can leave the final
            # stage waiting indefinitely for activation tensors.
            try:
                outputs = await asyncio.wait_for(
                    asyncio.gather(
                        *(asyncio.wrap_future(output) for output in worker_outputs)
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError(f"RPC call to {method} timed out") from exc

            if output_rank is None:
                return outputs
            result = outputs[output_rank]
            if aggregate_output:
                for name in ("kv_output_aggregator", "ec_output_aggregator"):
                    aggregator = getattr(self, name, None)
                    if aggregator is not None:
                        result = aggregator.aggregate(outputs, output_rank=output_rank)
            return result

        result = asyncio.run_coroutine_threadsafe(collect_outputs(), self._loop)
        if async_run_tensor_parallel_workers_only or non_block:
            return result
        return result.result()
