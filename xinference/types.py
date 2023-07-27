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

from typing import Dict, List, Optional

from typing_extensions import Literal, NotRequired, TypedDict


class EmbeddingUsage(TypedDict):
    prompt_tokens: int
    total_tokens: int


class EmbeddingData(TypedDict):
    index: int
    object: str
    embedding: List[float]


class Embedding(TypedDict):
    object: Literal["list"]
    model: str
    data: List[EmbeddingData]
    usage: EmbeddingUsage


class CompletionLogprobs(TypedDict):
    text_offset: List[int]
    token_logprobs: List[Optional[float]]
    tokens: List[str]
    top_logprobs: List[Optional[Dict[str, float]]]


class CompletionChoice(TypedDict):
    text: str
    index: int
    logprobs: Optional[CompletionLogprobs]
    finish_reason: Optional[str]


class CompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChunk(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]


class Completion(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


class ChatCompletionMessage(TypedDict):
    role: str
    content: str
    user: NotRequired[str]


class ChatCompletionChoice(TypedDict):
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str]


class ChatCompletion(TypedDict):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: CompletionUsage


class ChatCompletionChunkDelta(TypedDict):
    role: NotRequired[str]
    content: NotRequired[str]


class ChatCompletionChunkChoice(TypedDict):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str]


class ChatCompletionChunk(TypedDict):
    id: str
    model: str
    object: Literal["chat.completion.chunk"]
    created: int
    choices: List[ChatCompletionChunkChoice]
