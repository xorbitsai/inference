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

import os

import torch
from typing_extensions import Literal, Union

DeviceType = Literal["cuda", "mps", "xpu", "npu", "cpu"]
DEVICE_TO_ENV_NAME = {
    "cuda": "CUDA_VISIBLE_DEVICES",
    "npu": "ASCEND_RT_VISIBLE_DEVICES",
}


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_npu_available() -> bool:
    try:
        import torch
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False


def get_available_device() -> DeviceType:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    elif is_xpu_available():
        return "xpu"
    elif is_npu_available():
        return "npu"
    return "cpu"


def is_device_available(device: str) -> bool:
    if device == "cuda":
        return torch.cuda.is_available()
    elif device == "mps":
        return torch.backends.mps.is_available()
    elif device == "xpu":
        return is_xpu_available()
    elif device == "npu":
        return is_npu_available()
    elif device == "cpu":
        return True

    return False


def move_model_to_available_device(model):
    device = get_available_device()

    if device == "cpu":
        return model

    return model.to(device)


def get_device_preferred_dtype(device: str) -> Union[torch.dtype, None]:
    if device == "cpu":
        return torch.float32
    elif device == "cuda" or device == "mps" or device == "npu":
        return torch.float16
    elif device == "xpu":
        return torch.bfloat16

    return None


def is_hf_accelerate_supported(device: str) -> bool:
    return device == "cuda" or device == "xpu" or device == "npu"


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if is_xpu_available():
        torch.xpu.empty_cache()
    if is_npu_available():
        torch.npu.empty_cache()


def get_available_device_env_name():
    return DEVICE_TO_ENV_NAME.get(get_available_device())


def gpu_count():
    if torch.cuda.is_available():
        cuda_visible_devices_env = os.getenv("CUDA_VISIBLE_DEVICES", None)

        if cuda_visible_devices_env is None:
            return torch.cuda.device_count()

        cuda_visible_devices = (
            cuda_visible_devices_env.split(",") if cuda_visible_devices_env else []
        )

        return min(torch.cuda.device_count(), len(cuda_visible_devices))
    elif is_xpu_available():
        return torch.xpu.device_count()
    elif is_npu_available():
        return torch.npu.device_count()
    else:
        return 0
