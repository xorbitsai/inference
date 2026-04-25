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
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Union, cast

import torch

logger = logging.getLogger(__name__)

DeviceType = Literal[
    "cuda", "mps", "xpu", "vacc", "npu", "mlu", "musa", "gcu", "rocm", "cpu"
]


# ---------------------------------------------------------------------------
# DeviceSpec
# ---------------------------------------------------------------------------


@dataclass
class DeviceSpec:
    name: str
    is_available: Callable[[], bool]
    env_name: Optional[str]
    get_gpu_info_fn: Callable[[], Dict]
    device_count_fn: Callable[[], int]
    empty_cache_fn: Optional[Callable] = None
    preferred_dtype: Optional[torch.dtype] = None
    hf_accelerate: bool = False
    pytorch_device_name: Optional[str] = None
    gpu_count_fn: Optional[Callable[[], int]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_with_visible_devices(env_name: str, device_count: int) -> int:
    env_val = os.getenv(env_name, None)
    if env_val is None:
        return device_count
    visible = env_val.split(",") if env_val else []
    return min(device_count, len(visible))


def _mps_empty_cache():
    try:
        torch.mps.empty_cache()
    except RuntimeError as e:
        if "invalid low watermark ratio" not in str(e):
            raise


def _musa_gpu_count() -> int:
    device_module = torch.get_device_module(torch.device(torch.musa.current_device()))
    count = device_module.device_count()
    env_val = os.getenv("MUSA_VISIBLE_DEVICES", None)
    if env_val is None:
        return count
    visible = env_val.split(",") if env_val else []
    return min(count, len(visible))


def _rocm_gpu_count() -> int:
    count = torch.cuda.device_count()
    env_val = os.getenv("HIP_VISIBLE_DEVICES", None)
    if env_val is None:
        env_val = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if env_val is None:
        return count
    visible = env_val.split(",") if env_val else []
    return min(count, len(visible))


# ---------------------------------------------------------------------------
# Device availability checks
# ---------------------------------------------------------------------------


def is_metax_available() -> bool:
    try:
        from pymxsml import mxSmlGetDeviceCount  # noqa: F401

        return True
    except ImportError:
        return False


def is_rocm_available() -> bool:
    try:
        return torch.cuda.is_available() and torch.version.hip is not None
    except (ImportError, AttributeError):
        return False


def is_cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except ImportError:
        return False


def is_mps_available() -> bool:
    try:
        return torch.backends.mps.is_available()
    except ImportError:
        return False


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False


def is_mlu_available() -> bool:
    try:
        import torch_mlu  # noqa: F401

        return torch.mlu.is_available()
    except ImportError:
        return False


def is_vacc_available() -> bool:
    try:
        import torch_vacc  # noqa: F401

        return torch.vacc.is_available()
    except ImportError:
        return False


def is_musa_available() -> bool:
    try:
        import torch_musa  # noqa: F401
        import torchada  # noqa: F401

        return torch.musa.is_available()
    except ImportError:
        return False


def is_gcu_available() -> bool:
    try:
        import torch_gcu  # noqa: F401

        return torch.gcu.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# GPU info helpers
# ---------------------------------------------------------------------------


def _get_info_by_pynvml(gpu_id: int) -> Dict[str, float]:
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


def _get_info_by_torch(index: int) -> Dict[str, Any]:
    return {
        "name": f"CUDA GPU {index}",
        "total": 1,
        "used": 0,
        "free": 1,
        "util": 0,
    }


def _get_metax_gpu_mem_info(gpu_id: int) -> Dict[str, Union[str, int]]:
    from pymxsml import mxSmlGetDeviceInfo, mxSmlGetMemoryInfo

    info = mxSmlGetDeviceInfo(gpu_id)
    mem = mxSmlGetMemoryInfo(gpu_id)
    total = mem.vramTotal * 1024
    used = mem.vramUse * 1024
    return {
        "name": info.deviceName,
        "total": total,
        "used": used,
        "free": total - used,
        "util": 0,
    }


def _get_rocm_gpu_mem_info(gpu_id: int) -> Dict[str, Union[str, int]]:
    return {
        "name": f"ROCm GPU {gpu_id}",
        "total": 1,
        "used": 0,
        "free": 1,
        "util": 0,
    }


def _get_gcu_mem_info(gpu_id: int) -> Dict[str, float]:
    from pyefml import efmlGetDevInfo, efmlGetDevMem

    dev_info = efmlGetDevInfo(gpu_id)
    dev_mem = efmlGetDevMem(gpu_id)
    return {
        "name": dev_info.productName,
        "total": dev_mem.vramTotal,
        "used": dev_mem.vramUse,
        "free": dev_mem.vramTotal - dev_mem.vramUse,
        "util": 0,
    }


# ---------------------------------------------------------------------------
# GPU info collection
# ---------------------------------------------------------------------------


def _collect_gpu_info_with_env(
    env_name: str,
    device_count_fn: Callable[[], int],
    get_mem_info_fn: Callable[[int], Dict],
    init_fn: Optional[Callable] = None,
    shutdown_fn: Optional[Callable] = None,
) -> Dict:
    try:
        if init_fn:
            init_fn()
        device_count = device_count_fn()
        res = {}

        env_val = os.getenv(env_name, None)

        if not env_val:
            for i in range(device_count):
                res[f"gpu-{i}"] = get_mem_info_fn(i)
        else:
            ids = env_val.split(",")
            for logical_id, id_str in enumerate(ids):
                try:
                    physical_id = int(id_str.strip())
                    if 0 <= physical_id < device_count:
                        res[f"gpu-{logical_id}"] = get_mem_info_fn(physical_id)
                except ValueError:
                    continue

        return res
    except Exception as e:
        logger.error(f"Fail to get GPU info: {e}")
        return {}
    finally:
        if shutdown_fn:
            try:
                shutdown_fn()
            except Exception:
                pass


def get_nvidia_gpu_info() -> Dict[str, Any]:
    try:
        from pynvml import nvmlDeviceGetCount, nvmlInit, nvmlShutdown

        return _collect_gpu_info_with_env(
            "CUDA_VISIBLE_DEVICES",
            nvmlDeviceGetCount,
            _get_info_by_pynvml,
            init_fn=nvmlInit,
            shutdown_fn=nvmlShutdown,
        )
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"Fail to use pynvml with error: {e}, will fallback to torch.")

    # fallback: torch
    try:
        if not torch.cuda.is_available():
            return {}
        return _collect_gpu_info_with_env(
            "CUDA_VISIBLE_DEVICES",
            torch.cuda.device_count,
            _get_info_by_torch,
        )
    except ImportError:
        return {}
    except Exception as e:
        logger.error(f"Fail to use torch with error: {e}, will return empty result.")
        return {}


def get_metax_gpu_info() -> Dict:
    try:
        from pymxsml import mxSmlGetDeviceCount, mxSmlInit

        return _collect_gpu_info_with_env(
            "MACA_VISIBLE_DEVICES",
            mxSmlGetDeviceCount,
            _get_metax_gpu_mem_info,
            init_fn=mxSmlInit,
        )
    except Exception as e:
        logger.error(f"Fail to get Metax GPU info: {e}")
        return {}


def get_rocm_gpu_info() -> Dict:
    try:
        if not is_rocm_available():
            return {}
        count = torch.cuda.device_count()
        res = {}
        for i in range(count):
            res[f"gpu-{i}"] = _get_rocm_gpu_mem_info(i)
        return res
    except Exception:
        return {}


def get_gcu_info() -> Dict:
    from pyefml import efmlGetDevCount, efmlInit, efmlShutdown

    return _collect_gpu_info_with_env(
        "TOPS_VISIBLE_DEVICES",
        efmlGetDevCount,
        _get_gcu_mem_info,
        init_fn=efmlInit,
        shutdown_fn=efmlShutdown,
    )


def get_npu_info() -> Dict:
    import re
    import subprocess

    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        output = result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command `npu-smi info` : {e.stderr}")
        return {}

    lines = output.splitlines()
    all_devices: Dict[str, Any] = {}

    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("|") and ("OK" in line or "Warning" in line):
            parts = line.split("|")
            npu_info = parts[1].strip().split(maxsplit=1)
            if len(npu_info) == 2:
                npu_id = npu_info[0]
                name = npu_info[1]
                next_line = lines[i + 1].strip()
                next_parts = next_line.split("|")
                chip_id = next_parts[1].strip().split()[0]
                pattern = re.compile(r"(\d+)\s*/\s*(\d+)")
                matches = pattern.findall(next_parts[3])
                if matches:
                    used, total = map(int, matches[-1])
                    device_key = f"gpu-{npu_id}-{chip_id}"
                    all_devices[device_key] = {
                        "name": name,
                        "total": total,
                        "used": used,
                        "free": total - used,
                        "util": 0,
                        "npu_id": len(all_devices),
                    }

    ascend_visible_devices_env = os.getenv("ASCEND_RT_VISIBLE_DEVICES", None)
    if not ascend_visible_devices_env:
        return all_devices

    visible_device_ids = set()
    for device_id_str in ascend_visible_devices_env.split(","):
        try:
            visible_device_ids.add(int(device_id_str.strip()))
        except ValueError:
            continue

    res = {}
    for device_key, device_info in all_devices.items():
        if device_info["npu_id"] in visible_device_ids:
            res[device_key] = device_info.copy()
            del res[device_key]["npu_id"]
    return res


# ---------------------------------------------------------------------------
# Device registry (ordered by detection priority)
# ---------------------------------------------------------------------------

DEVICE_REGISTRY: list[DeviceSpec] = [
    # metax and rocm must come before cuda: their is_cuda_available() also returns True
    DeviceSpec(
        name="metax",
        is_available=is_metax_available,
        env_name="MACA_VISIBLE_DEVICES",
        get_gpu_info_fn=get_metax_gpu_info,
        device_count_fn=lambda: torch.cuda.device_count(),
        empty_cache_fn=lambda: torch.cuda.empty_cache(),
        preferred_dtype=torch.float16,
        hf_accelerate=True,
        pytorch_device_name="cuda",
    ),
    DeviceSpec(
        name="rocm",
        is_available=is_rocm_available,
        env_name="HIP_VISIBLE_DEVICES",
        get_gpu_info_fn=get_rocm_gpu_info,
        device_count_fn=lambda: torch.cuda.device_count(),
        empty_cache_fn=lambda: torch.cuda.empty_cache(),
        preferred_dtype=torch.float16,
        gpu_count_fn=_rocm_gpu_count,
    ),
    DeviceSpec(
        name="cuda",
        is_available=is_cuda_available,
        env_name="CUDA_VISIBLE_DEVICES",
        get_gpu_info_fn=get_nvidia_gpu_info,
        device_count_fn=lambda: torch.cuda.device_count(),
        empty_cache_fn=lambda: torch.cuda.empty_cache(),
        preferred_dtype=torch.float16,
        hf_accelerate=True,
    ),
    DeviceSpec(
        name="mps",
        is_available=is_mps_available,
        env_name=None,
        get_gpu_info_fn=lambda: {},
        device_count_fn=lambda: 0,
        empty_cache_fn=_mps_empty_cache,
        preferred_dtype=torch.float16,
    ),
    DeviceSpec(
        name="xpu",
        is_available=is_xpu_available,
        env_name=None,
        get_gpu_info_fn=lambda: {},
        device_count_fn=lambda: torch.xpu.device_count(),
        empty_cache_fn=lambda: torch.xpu.empty_cache(),
        preferred_dtype=torch.bfloat16,
        hf_accelerate=True,
    ),
    DeviceSpec(
        name="npu",
        is_available=is_npu_available,
        env_name="ASCEND_RT_VISIBLE_DEVICES",
        get_gpu_info_fn=get_npu_info,
        device_count_fn=lambda: torch.npu.device_count(),
        empty_cache_fn=lambda: torch.npu.empty_cache(),
        preferred_dtype=torch.float16,
        hf_accelerate=True,
    ),
    DeviceSpec(
        name="mlu",
        is_available=is_mlu_available,
        env_name="MLU_VISIBLE_DEVICES",
        get_gpu_info_fn=lambda: {},
        device_count_fn=lambda: torch.mlu.device_count(),
        empty_cache_fn=lambda: torch.mlu.empty_cache(),
        preferred_dtype=torch.float16,
        hf_accelerate=True,
    ),
    DeviceSpec(
        name="vacc",
        is_available=is_vacc_available,
        env_name="VACC_VISIBLE_DEVICES",
        get_gpu_info_fn=lambda: {},
        device_count_fn=lambda: torch.vacc.device_count(),
        empty_cache_fn=lambda: torch.vacc.empty_cache(),
        preferred_dtype=torch.float16,
    ),
    DeviceSpec(
        name="musa",
        is_available=is_musa_available,
        env_name="MUSA_VISIBLE_DEVICES",
        get_gpu_info_fn=lambda: {},
        device_count_fn=lambda: torch.musa.device_count(),
        empty_cache_fn=lambda: torch.musa.empty_cache(),
        preferred_dtype=torch.float16,
        hf_accelerate=True,
        gpu_count_fn=_musa_gpu_count,
    ),
    DeviceSpec(
        name="gcu",
        is_available=is_gcu_available,
        env_name="TOPS_VISIBLE_DEVICES",
        get_gpu_info_fn=get_gcu_info,
        device_count_fn=lambda: torch.gcu.device_count(),
        empty_cache_fn=lambda: torch.gcu.empty_cache(),
        preferred_dtype=torch.float16,
        hf_accelerate=True,
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _find_device() -> Optional[DeviceSpec]:
    for spec in DEVICE_REGISTRY:
        if spec.is_available():
            return spec
    return None


def get_available_device() -> DeviceType:
    spec = _find_device()
    if spec is None:
        return "cpu"
    return cast(DeviceType, spec.pytorch_device_name or spec.name)


def is_device_available(device: str) -> bool:
    if device == "cpu":
        return True
    return any(s.name == device and s.is_available() for s in DEVICE_REGISTRY)


def move_model_to_available_device(model):
    device = get_available_device()
    if device == "cpu":
        return model
    return model.to(device)


def get_device_preferred_dtype(device: str) -> Union[torch.dtype, None]:
    if device == "cpu":
        return torch.float32
    for spec in DEVICE_REGISTRY:
        if spec.name == device:
            return spec.preferred_dtype
    return None


def is_hf_accelerate_supported(device: str) -> bool:
    return any(s.name == device and s.hf_accelerate for s in DEVICE_REGISTRY)


def empty_cache():
    for spec in DEVICE_REGISTRY:
        if spec.is_available() and spec.empty_cache_fn:
            spec.empty_cache_fn()


def get_available_device_env_name():
    spec = _find_device()
    if spec is None:
        return None
    return spec.env_name


def gpu_count():
    spec = _find_device()
    if spec is None:
        return 0
    if spec.gpu_count_fn:
        return spec.gpu_count_fn()
    count = spec.device_count_fn()
    if spec.env_name is None:
        return count
    return _count_with_visible_devices(spec.env_name, count)


def get_gpu_info() -> Dict:
    spec = _find_device()
    if spec is None:
        return {}
    return spec.get_gpu_info_fn()
