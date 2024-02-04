# Copyright 2022-2024 XProbe Inc.
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
from pydantic.version import VERSION as PYDANTIC_VERSION

PYDANTIC_V2 = PYDANTIC_VERSION.startswith("2.")


if PYDANTIC_V2:
    from pydantic.v1 import (  # noqa: F401
        BaseModel,
        Field,
        Protocol,
        ValidationError,
        create_model,
        create_model_from_namedtuple,
        create_model_from_typeddict,
        parse_file_as,
        validate_arguments,
        validator,
    )
    from pydantic.v1.error_wrappers import ErrorWrapper  # noqa: F401
    from pydantic.v1.parse import load_str_bytes  # noqa: F401
    from pydantic.v1.types import StrBytes  # noqa: F401
    from pydantic.v1.utils import ROOT_KEY  # noqa: F401
else:
    from pydantic import (  # noqa: F401
        BaseModel,
        Field,
        Protocol,
        ValidationError,
        create_model,
        create_model_from_namedtuple,
        create_model_from_typeddict,
        parse_file_as,
        validate_arguments,
        validator,
    )
    from pydantic.error_wrappers import ErrorWrapper  # noqa: F401
    from pydantic.parse import load_str_bytes  # noqa: F401
    from pydantic.types import StrBytes  # noqa: F401
    from pydantic.utils import ROOT_KEY  # noqa: F401
