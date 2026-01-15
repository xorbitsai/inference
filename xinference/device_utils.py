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

from typing import Dict, Literal, Union

import torch

DeviceType = Literal["cuda", "mps", "xpu", "vacc", "npu", "mlu", "musa", "cpu"]
DEVICE_TO_ENV_NAME = {
    "cuda": "CUDA_VISIBLE_DEVICES",
    "npu": "ASCEND_RT_VISIBLE_DEVICES",
    "mlu": "MLU_VISIBLE_DEVICES",
    "vacc": "VACC_VISIBLE_DEVICES",
    "musa": "MUSA_VISIBLE_DEVICES",
}


def is_vacc_available() -> bool:
    try:
        import torch
        import torch_vacc  # noqa: F401

        return torch.vacc.is_available()
    except ImportError:
        return False


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_npu_available() -> bool:
    try:
        import torch
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False


def is_mlu_available() -> bool:
    try:
        import torch
        import torch_mlu  # noqa: F401

        return torch.mlu.is_available()
    except ImportError:
        return False


def is_musa_available() -> bool:
    try:
        import torch
        import torch_musa  # noqa: F401
        import torchada  # noqa: F401

        return torch.musa.is_available()
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
    elif is_mlu_available():
        return "mlu"
    elif is_vacc_available():
        return "vacc"
    elif is_musa_available():
        return "musa"
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
    elif device == "mlu":
        return is_mlu_available()
    elif device == "vacc":
        return is_vacc_available()
    elif device == "musa":
        return is_musa_available()
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
    elif (
        device == "cuda"
        or device == "mps"
        or device == "npu"
        or device == "mlu"
        or device == "vacc"
        or device == "musa"
    ):
        return torch.float16
    elif device == "xpu":
        return torch.bfloat16

    return None


def is_hf_accelerate_supported(device: str) -> bool:
    return (
        device == "cuda"
        or device == "xpu"
        or device == "npu"
        or device == "mlu"
        or device == "musa"
    )


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except RuntimeError as e:
            # Handle known MPS memory management issues in PyTorch 3.13+
            if "invalid low watermark ratio" in str(e):
                # This is a known issue with PyTorch 3.13+ on macOS.
                # We can safely ignore this error as it doesn't affect functionality.
                pass
            else:
                # Re-raise other RuntimeErrors
                raise
    if is_xpu_available():
        torch.xpu.empty_cache()
    if is_npu_available():
        torch.npu.empty_cache()
    if is_mlu_available():
        torch.mlu.empty_cache()
    if is_vacc_available():
        torch.vacc.empty_cache()
    if is_musa_available():
        torch.musa.empty_cache()


def get_available_device_env_name():
    return DEVICE_TO_ENV_NAME.get(get_available_device())


def gpu_count():
    device_module = torch.get_device_module(get_available_device())
    if torch.cuda.is_available() or is_vacc_available() or is_musa_available():
        import os

        visible_devices_env = os.getenv(get_available_device_env_name(), None)
        if visible_devices_env is None:
            return device_module.device_count()

        visible_devices = visible_devices_env.split(",") if visible_devices_env else []

        return min(device_module.device_count(), len(visible_devices))
    elif is_xpu_available() or is_npu_available() or is_mlu_available():
        return device_module.device_count()
    else:
        return 0


def _get_nvidia_gpu_mem_info(gpu_id: int) -> Dict[str, float]:
    from pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
        nvmlDeviceGetUtilizationRates,
    )

    handler = nvmlDeviceGetHandleByIndex(gpu_id)
    gpu_name = nvmlDeviceGetName(handler)
    mem_info = nvmlDeviceGetMemoryInfo(handler)
    utilization = nvmlDeviceGetUtilizationRates(handler)
    return {
        "name": gpu_name,
        "total": mem_info.total,
        "used": mem_info.used,
        "free": mem_info.free,
        "util": utilization.gpu,
    }


def get_nvidia_gpu_info() -> Dict:
    from pynvml import nvmlDeviceGetCount, nvmlInit, nvmlShutdown

    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        res = {}
        for i in range(device_count):
            res[f"gpu-{i}"] = _get_nvidia_gpu_mem_info(i)
        return res
    except Exception:
        # Fall back to torch-based detection when NVML lacks support.
        try:
            import torch

            if torch.cuda.is_available():
                res = {}
                for i in range(torch.cuda.device_count()):
                    res[f"gpu-{i}"] = {
                        "name": torch.cuda.get_device_name(i),
                        "total": 0,
                        "used": 0,
                        "free": 0,
                        "util": 0,
                    }
                return res
        except Exception:
            pass
        # TODO: add log here
        # logger.debug(f"Cannot init nvml. Maybe due to lack of NVIDIA GPUs or incorrect installation of CUDA.")
        return {}
    finally:
        try:
            nvmlShutdown()
        except Exception:
            pass
