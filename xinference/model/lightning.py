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

from typing import Any

from ..core.exceptions import InvalidModelRequestError


def resolve_request_lightning(value: Any, adapter_loaded: bool) -> bool:
    if value is None:
        return adapter_loaded
    if not isinstance(value, bool):
        raise InvalidModelRequestError("use_lightning must be a boolean")
    if value and not adapter_loaded:
        raise InvalidModelRequestError(
            "Launch the model with a Lightning adapter before using use_lightning=true"
        )
    return value
