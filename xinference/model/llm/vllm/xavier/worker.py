from typing import Dict, Optional, Tuple

import torch
from vllm.sequence import ExecuteModelRequest
from vllm.worker.model_runner_base import BroadcastableModelInput
from vllm.worker.worker import Worker
from vllm.worker.worker_base import WorkerInput


class XavierWorker(Worker):
    def prepare_input(
        self, execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> Optional[Tuple[BroadcastableModelInput, WorkerInput, Dict[str, torch.Tensor]]]:
        inputs = super().prepare_input(execute_model_req)
        model_input, worker_input, kwargs = inputs
        # print(f"======prepare_input: kwargs keys: {kwargs.keys()}")
        # print(f"======prepare_input: model_input type: {type(model_input)}")
        tmp = model_input.as_broadcastable_tensor_dict()
        # print(f"======prepare_input: model_input: {tmp.keys()}")
        # print(f"======prepare_input: num_prefills: {tmp['num_prefills']}")
        # print(f"======prepare_input: num_prefill_tokens: {tmp['num_prefill_tokens']}")
        # print(f"======prepare_input: num_decode_tokens: {tmp['num_decode_tokens']}")
        # print(f"======prepare_input: block_tables: {tmp['block_tables']}")
        print(f"======prepare_input: input_positions: {tmp['input_positions']}")
        # print(f"======prepare_input: _cached_prefill_metadata: {tmp['_cached_prefill_metadata']}")
        # print(f"======prepare_input: _cached_decode_metadata: {tmp['_cached_decode_metadata']}")
        # print(
        #     f"======prepare_input: selected_token_indices: {tmp['selected_token_indices']}"
        # )
        # print(f"======prepare_input: slot_mapping: {tmp['slot_mapping']}")
        return inputs
