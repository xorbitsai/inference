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
"""Build the Xinference Web UI.

Builds the Next.js static export under ``frontend/`` and stages it at
``xinference/ui/web/dist`` so it ships inside the wheel/sdist and the
backend serves it without a Node runtime.

Run directly (``python build_web.py``) or let the in-tree build backend
(``build_backend.py``) invoke it automatically during wheel/sdist/editable
builds. Set ``NO_WEB_UI=1`` to skip the build.
"""

import os
import shutil
import subprocess
import warnings

repo_root = os.path.dirname(os.path.abspath(__file__))

_web_src_path = os.path.join(repo_root, "frontend")
_web_dest_path = os.path.join(
    repo_root, "xinference", "ui", "web", "dist", "index.html"
)
_commands = [
    ["npm", "ci"],
    ["npm", "run", "build"],
]


def build_web():
    if os.environ.get("NO_WEB_UI", "0").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    ):
        return

    npm_path = shutil.which("npm")
    if npm_path is None:
        warnings.warn("Cannot find NPM, may affect displaying Xinference web UI")
        return

    npm_env = os.environ.copy()
    npm_env.setdefault("npm_config_cache", os.path.join(_web_src_path, ".npm-cache"))
    npm_env.setdefault("npm_config_logs_dir", os.path.join(_web_src_path, ".npm-logs"))
    for cmd in _commands:
        cmd = [npm_path if c == "npm" else c for c in cmd]
        proc_result = subprocess.run(cmd, cwd=_web_src_path, env=npm_env)
        if proc_result.returncode != 0:
            warnings.warn(f'Failed when running `{" ".join(cmd)}`')
            return
    # `npm run build` stages the static export at
    # xinference/ui/web/dist via its postbuild hook.
    assert os.path.exists(_web_dest_path)


if __name__ == "__main__":
    build_web()
