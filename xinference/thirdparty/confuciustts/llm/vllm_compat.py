import os
import shutil
import tempfile
from typing import Any

import numpy as np


def prepare_vllm_model_dir(vllm_model_dir: str, t2s_ckpt: str) -> str:
    """Expose the T2S checkpoint as model.safetensors for vLLM."""
    weights_link = os.path.join(vllm_model_dir, "model.safetensors")
    link_ok = (
        os.path.lexists(weights_link)
        and os.path.islink(weights_link)
        and os.readlink(weights_link) == t2s_ckpt
    )
    if link_ok:
        return vllm_model_dir

    try:
        if os.path.lexists(weights_link):
            os.remove(weights_link)
        os.symlink(t2s_ckpt, weights_link)
        return vllm_model_dir
    except OSError:
        temp_dir = tempfile.mkdtemp(prefix="confucius_t2s_vllm_")
        for filename in os.listdir(vllm_model_dir):
            source = os.path.join(vllm_model_dir, filename)
            if os.path.isfile(source) and not filename.endswith(".safetensors"):
                shutil.copy2(source, os.path.join(temp_dir, filename))
        shutil.copy2(t2s_ckpt, os.path.join(temp_dir, "model.safetensors"))
        return temp_dir


def correct_confucius_positions(
    runner: Any, scheduler_output: Any, num_scheduled_tokens: np.ndarray
) -> None:
    """Shift positions so Confucius semantic tokens start at position zero."""
    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    num_reqs = runner.input_batch.num_reqs
    req_indices = np.repeat(runner.arange_np[:num_reqs], num_scheduled_tokens)
    positions = runner.positions.np[:total_num_scheduled_tokens]
    offsets = np.array(
        [
            -(len(runner.requests[req_id].prompt_token_ids) - 1)
            for req_id in runner.input_batch.req_ids[:num_reqs]
        ],
        dtype=np.int64,
    )
    np.add(offsets[req_indices], positions, out=positions)
    runner.positions.copy_to_gpu(total_num_scheduled_tokens)
