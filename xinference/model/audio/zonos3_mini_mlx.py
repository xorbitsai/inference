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

from .zonos3_mini_native import Zonos3MiniNativeModel


class MLXZonos3MiniModel(Zonos3MiniNativeModel):
    """Run the official ZONOS3 MLX server on Apple Silicon."""

    server_executable = "zonos3-mlx-server"
    server_path_env = "XINFERENCE_ZONOS3_MLX_SERVER_PATH"
    server_install_hint = (
        "Clone Zyphra/ZONOS3-Mini with its submodules, run "
        "`uv sync --locked --package zonos3-mlx`, and point "
        "XINFERENCE_ZONOS3_MLX_SERVER_PATH at .venv/bin/zonos3-mlx-server."
    )
