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

import os

# Configure MPS memory management to avoid "invalid low watermark ratio" error in PyTorch 3.13+
if os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") is None:
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "1.0"
if os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO") is None:
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.2"

from . import _version

__version__ = _version.get_versions()["version"]


try:
    import intel_extension_for_pytorch  # noqa: F401
except:
    pass


def _install():
    from xoscar.backends.router import Router

    default_router = Router.get_instance_or_empty()
    Router.set_instance(default_router)


_install()
del _install
