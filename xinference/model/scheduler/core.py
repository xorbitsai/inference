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

from enum import Enum


class AbortRequestMessage(Enum):
    NOT_FOUND = 1
    DONE = 2
    NO_OP = 3


# Streaming flags
XINFERENCE_STREAMING_DONE_FLAG = "<XINFERENCE_STREAMING_DONE>"
XINFERENCE_STREAMING_ERROR_FLAG = "<XINFERENCE_STREAMING_ERROR>"
XINFERENCE_STREAMING_ABORT_FLAG = "<XINFERENCE_STREAMING_ABORT>"
XINFERENCE_NON_STREAMING_ABORT_FLAG = "<XINFERENCE_NON_STREAMING_ABORT>"
