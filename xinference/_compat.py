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
from typing import Dict, Iterable, List, Literal, Optional, TypeAlias, Union

from pydantic.version import VERSION as PYDANTIC_VERSION
from typing_extensions import Required, TypedDict

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

from openai.types.chat.chat_completion_named_tool_choice_param import (
    ChatCompletionNamedToolChoiceParam,
)
from openai.types.chat.chat_completion_stream_options_param import (
    ChatCompletionStreamOptionsParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.shared_params.response_format_json_object import (
    ResponseFormatJSONObject,
)
from openai.types.shared_params.response_format_text import ResponseFormatText

OpenAIChatCompletionStreamOptionsParam = create_model_from_typeddict(
    ChatCompletionStreamOptionsParam
)
OpenAIChatCompletionToolParam = create_model_from_typeddict(ChatCompletionToolParam)
OpenAIChatCompletionNamedToolChoiceParam = create_model_from_typeddict(
    ChatCompletionNamedToolChoiceParam
)


class JSONSchema(TypedDict, total=False):
    name: Required[str]
    """The name of the response format.

    Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length
    of 64.
    """

    description: str
    """
    A description of what the response format is for, used by the model to determine
    how to respond in the format.
    """

    schema_: Optional[Dict[str, object]] = Field(alias="schema", default=None)  # type: ignore
    """The schema for the response format, described as a JSON Schema object."""

    strict: Optional[bool]
    """Whether to enable strict schema adherence when generating the output.

    If set to true, the model will always follow the exact schema defined in the
    `schema` field. Only a subset of JSON Schema is supported when `strict` is
    `true`. To learn more, read the
    [Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs).
    """


class ResponseFormatJSONSchema(TypedDict, total=False):
    json_schema: Required[JSONSchema]

    type: Required[Literal["json_schema"]]
    """The type of response format being defined: `json_schema`"""


ResponseFormat: TypeAlias = Union[
    ResponseFormatText, ResponseFormatJSONObject, ResponseFormatJSONSchema
]


class CreateChatCompletionOpenAI(BaseModel):
    """
    Comes from source code: https://github.com/openai/openai-python/blob/main/src/openai/types/chat/completion_create_params.py
    """

    messages: List[Dict]
    model: str
    frequency_penalty: Optional[float]
    logit_bias: Optional[Dict[str, int]]
    logprobs: Optional[bool]
    max_tokens: Optional[int]
    n: Optional[int]
    parallel_tool_calls: Optional[bool]
    presence_penalty: Optional[float]
    response_format: Optional[ResponseFormat]
    seed: Optional[int]
    service_tier: Optional[Literal["auto", "default"]]
    stop: Union[Optional[str], List[str]]
    stream_options: Optional[OpenAIChatCompletionStreamOptionsParam]  # type: ignore
    temperature: Optional[float]
    tool_choice: Optional[  # type: ignore
        Union[
            Literal["none", "auto", "required"],
            OpenAIChatCompletionNamedToolChoiceParam,
        ]
    ]
    tools: Optional[Iterable[OpenAIChatCompletionToolParam]]  # type: ignore
    top_logprobs: Optional[int]
    top_p: Optional[float]
    user: Optional[str]
