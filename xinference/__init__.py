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

from . import _version

__version__ = _version.get_versions()["version"]


try:
    import intel_extension_for_pytorch  # noqa: F401
except:
    pass


def _install():
    from xoscar.backends.router import Router

    from .model import _install as install_model

    default_router = Router.get_instance_or_empty()
    Router.set_instance(default_router)

    install_model()


_install()
del _install
