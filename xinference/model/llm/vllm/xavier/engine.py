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
import logging
from typing import Dict, List, Optional, Type, Union

from packaging import version
from vllm import AsyncEngineArgs, EmbeddingRequestOutput, RequestOutput
from vllm import __version__ as VLLM_VERSION
from vllm.config import VllmConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine, _AsyncLLMEngine
from vllm.engine.llm_engine import SchedulerOutputState
from vllm.engine.metrics_types import StatLoggerBase
from vllm.executor.executor_base import ExecutorBase
from vllm.sequence import ExecuteModelRequest
from vllm.usage.usage_lib import UsageContext

from .executor import XavierExecutor
from .scheduler import XavierScheduler

logger = logging.getLogger(__name__)


class XavierInternalEngine(_AsyncLLMEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._xavier_config = kwargs["vllm_config"].xavier_config
        self.scheduler = [
            XavierScheduler(
                self.scheduler_config,
                self.cache_config,
                self.lora_config,
                self.parallel_config.pipeline_parallel_size,
                (
                    self.async_callbacks[v_id]
                    if self.model_config.use_async_output_proc
                    else None
                ),
                xavier_config=self._xavier_config,
                virtual_engine=v_id,
            )
            for v_id in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.output_processor.scheduler = self.scheduler
        self.model_executor.scheduler = self.scheduler

    async def step_async(
        self, virtual_engine: int
    ) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        # these are cached outputs from previous iterations. None if on first
        # iteration
        cached_outputs = self.cached_scheduler_outputs[virtual_engine]
        seq_group_metadata_list = cached_outputs.seq_group_metadata_list
        scheduler_outputs = cached_outputs.scheduler_outputs
        allow_async_output_proc = cached_outputs.allow_async_output_proc

        ctx = self.scheduler_contexts[virtual_engine]

        # Clear outputs for each new scheduler iteration
        ctx.request_outputs.clear()

        # skip the scheduler if there are any remaining steps in the seq groups.
        # This ensures that the scheduler is only called again when the current
        # batch has completed.
        if not self._has_remaining_steps(seq_group_metadata_list):
            # Schedule iteration
            """Xinference Change!!!
            Why copy the entire function code of vllm:
            The purpose here is to modify the way the `schedule` function is invoked to asynchronous calling.
            No other modifications were made elsewhere.
            """
            (
                seq_group_metadata_list,
                scheduler_outputs,
                allow_async_output_proc,
            ) = await self.scheduler[virtual_engine].schedule()

            ctx.seq_group_metadata_list = seq_group_metadata_list
            ctx.scheduler_outputs = scheduler_outputs

            # Maybe switch from async mode to sync mode
            if not allow_async_output_proc and len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)

            if (
                self.scheduler_config.is_multi_step
                and scheduler_outputs.num_lookahead_slots > 0
            ):
                # cache the scheduler outputs for the next iteration if we have
                # lookahead slots
                self._cache_scheduler_outputs_for_multi_step(
                    virtual_engine,
                    seq_group_metadata_list,
                    scheduler_outputs,
                    allow_async_output_proc,
                )

        assert seq_group_metadata_list is not None
        assert scheduler_outputs is not None

        if not scheduler_outputs.is_empty():
            finished_requests_ids = self.scheduler[
                virtual_engine
            ].get_and_reset_finished_requests_ids()

            # Check if we have a cached last_output from the previous iteration.
            # For supporting PP this is probably the best way to pass the
            # sampled_token_ids, as a separate broadcast over all the PP stages
            # will cause one virtual engine's microbatch to block the pipeline.
            last_sampled_token_ids = self._get_last_sampled_token_ids(virtual_engine)

            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                virtual_engine=virtual_engine,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
                finished_requests_ids=finished_requests_ids,
                # We use ExecuteModelRequest to pass the last sampled_token_ids
                # to each of the non-last PP stages for in-place prepare_input.
                last_sampled_token_ids=last_sampled_token_ids,
            )

            if allow_async_output_proc:
                execute_model_req.async_callback = self.async_callbacks[virtual_engine]

            # Execute the model.
            outputs = await self.model_executor.execute_model_async(execute_model_req)

            # we need to do this here so that last step's sampled_token_ids can
            # be passed to the next iteration for PP.
            if self.scheduler_config.is_multi_step:
                self._update_cached_scheduler_output(virtual_engine, outputs)
        else:
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            outputs = []

        # Finish the current step for all the sequence groups.
        if self.scheduler_config.is_multi_step:
            for seq_group in seq_group_metadata_list:
                seq_group.finish_step()

        if not self._has_remaining_steps(seq_group_metadata_list):
            # Clear the cache if we have finished all the steps
            if self.scheduler_config.is_multi_step:
                self.cached_scheduler_outputs[virtual_engine] = SchedulerOutputState()

            # is_first_step_output is True only when the num_steps of all
            # the sequences are 1. When the num_steps > 1,
            # multi_step_model_runner does the first-step output append.
            is_first_step_output: bool = (
                False
                if not seq_group_metadata_list
                else seq_group_metadata_list[0].state.num_steps == 1
            )

            ctx.append_output(
                outputs=outputs,
                seq_group_metadata_list=seq_group_metadata_list,
                scheduler_outputs=scheduler_outputs,
                is_async=allow_async_output_proc,
                is_last_step=True,
                is_first_step_output=is_first_step_output,
            )

            if outputs and allow_async_output_proc:
                assert (
                    len(outputs) == 1
                ), "Async postprocessor expects only a single output set"
                self._advance_to_next_step(
                    outputs[0],
                    seq_group_metadata_list,
                    scheduler_outputs.scheduled_seq_groups,
                )

            if not allow_async_output_proc:
                self._process_model_outputs(ctx=ctx)

                # Log stats.
                self.do_log_stats(scheduler_outputs, outputs)

                # Tracing
                self.do_tracing(scheduler_outputs)

        else:
            # Multi-step case
            return ctx.request_outputs

        if not self.has_unfinished_requests():
            # Drain async postprocessor (if exists)
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            assert len(ctx.output_queue) == 0

        return ctx.request_outputs


class XavierEngine(AsyncLLMEngine):
    _engine_class: Type[_AsyncLLMEngine] = XavierInternalEngine
    _xavier_config: Optional[Dict] = None

    @classmethod
    def _get_executor_cls(cls, engine_config: VllmConfig) -> Type[ExecutorBase]:
        logger.debug(f"Initializing Xavier executor.")
        return XavierExecutor

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        engine_config: Optional[VllmConfig] = None,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        xavier_config: Optional[Dict] = None,
    ) -> "AsyncLLMEngine":
        cls._xavier_config = xavier_config
        if version.parse(VLLM_VERSION) < version.parse("0.8.0"):
            # old vllm
            args = (
                engine_args,
                engine_config,
                start_engine_loop,
                usage_context,
                stat_loggers,
            )
        else:
            args = engine_args, start_engine_loop, usage_context, stat_loggers  # type: ignore
        return super().from_engine_args(*args)

    def __init__(self, *args, **kwargs):
        # set xavier_config to `vllm_config`,
        # because it may be needed everywhere in the vllm internal components
        kwargs["vllm_config"].xavier_config = self._xavier_config
        super().__init__(*args, **kwargs)

    async def init_xavier(self):
        await self.engine.model_executor.init_transfer()
