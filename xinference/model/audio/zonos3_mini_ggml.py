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

import os

from .zonos3_mini_native import Zonos3MiniNativeModel


class GGMLZonos3MiniModel(Zonos3MiniNativeModel):
    server_executable = "zonos3-server"
    server_path_env = "XINFERENCE_ZONOS3_GGML_SERVER_PATH"
    server_install_hint = "Build zonos3-server from Zyphra/ZONOS3-Mini (servers/ggml)."

    def _command(self, port: int) -> list[str]:
        command = super()._command(port)
        device = self._device
        if device is None:
            visible_gpu = os.environ.get(
                "CUDA_VISIBLE_DEVICES", os.environ.get("HIP_VISIBLE_DEVICES")
            )
            if visible_gpu is not None:
                device = "cpu" if visible_gpu in {"", "-1"} else "gpu"
        if device == "cpu":
            command.append("--cpu")
        elif device is not None:
            if device.split(":", 1)[0] not in {"cuda", "gpu", "metal", "mps", "vulkan"}:
                raise ValueError(f"Unsupported ZONOS3-Mini GGML device: {device!r}.")
            command.append("--gpu")
        return command
