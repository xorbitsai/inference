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

from typing import Any, Callable, Dict, ForwardRef, Iterable, List, Optional, Union

from typing_extensions import Literal, NotRequired, TypedDict

from ._compat import (
    BaseModel,
    create_model,
    create_model_from_typeddict,
    validate_arguments,
)
from .fields import (
    echo_field,
    max_tokens_field,
    none_field,
    repeat_penalty_field,
    stop_field,
    stream_field,
    stream_interval_field,
    stream_option_field,
    temperature_field,
    top_k_field,
    top_p_field,
)


class Image(TypedDict):
    url: Optional[str]
    b64_json: Optional[str]


class ImageList(TypedDict):
    created: int
    data: List[Image]


class SDAPIResult(TypedDict):
    images: List[str]
    parameters: dict
    info: dict


class Video(TypedDict):
    url: Optional[str]
    b64_json: Optional[str]


class VideoList(TypedDict):
    created: int
    data: List[Video]


class EmbeddingUsage(TypedDict):
    prompt_tokens: int
    total_tokens: int


class EmbeddingData(TypedDict):
    index: int
    object: str
    # support sparse embedding
    embedding: Union[List[float], Dict[str, float]]


class Embedding(TypedDict):
    object: Literal["list"]
    model: str
    data: List[EmbeddingData]
    usage: EmbeddingUsage


class Document(TypedDict):
    text: str


class DocumentObj(TypedDict):
    index: int
    relevance_score: float
    document: Optional[Document]


# Cohere API compatibility
class ApiVersion(TypedDict):
    version: str
    is_deprecated: bool
    is_experimental: bool


# Cohere API compatibility
class BilledUnit(TypedDict):
    input_tokens: int
    output_tokens: int
    search_units: int
    classifications: int


class RerankTokens(TypedDict):
    input_tokens: int
    output_tokens: int


class Meta(TypedDict):
    api_version: Optional[ApiVersion]
    billed_units: Optional[BilledUnit]
    tokens: RerankTokens
    warnings: Optional[List[str]]


class Rerank(TypedDict):
    id: str
    results: List[DocumentObj]
    meta: Meta


class CompletionLogprobs(TypedDict):
    text_offset: List[int]
    token_logprobs: List[Optional[float]]
    tokens: List[str]
    top_logprobs: List[Optional[Dict[str, float]]]


class ToolCallFunction(TypedDict):
    name: str
    arguments: str


class ToolCalls(TypedDict):
    id: str
    type: Literal["function"]
    function: ToolCallFunction


class CompletionChoice(TypedDict):
    text: NotRequired[str]
    index: int
    logprobs: Optional[CompletionLogprobs]
    finish_reason: Optional[str]
    tool_calls: NotRequired[List[ToolCalls]]


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
    usage: NotRequired[CompletionUsage]


class Completion(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


class ChatCompletionMessage(TypedDict):
    role: str
    reasoning_content: NotRequired[str]
    content: Optional[str]
    user: NotRequired[str]
    tool_calls: NotRequired[List]


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
    reasoning_content: NotRequired[str]
    content: NotRequired[str]
    tool_calls: NotRequired[List[ToolCalls]]


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
    usage: NotRequired[CompletionUsage]


StoppingCriteria = Callable[[List[int], List[float]], bool]


class StoppingCriteriaList(List[StoppingCriteria]):
    def __call__(self, input_ids: List[int], logits: List[float]) -> bool:
        return any([stopping_criteria(input_ids, logits) for stopping_criteria in self])


LogitsProcessor = Callable[[List[int], List[float]], List[float]]


class LogitsProcessorList(List[LogitsProcessor]):
    def __call__(self, input_ids: List[int], scores: List[float]) -> List[float]:
        for processor in self:
            scores = processor(input_ids, scores)
        return scores


class LlamaCppGenerateConfig(TypedDict, total=False):
    suffix: Optional[str]
    max_tokens: int
    temperature: float
    top_p: float
    logprobs: Optional[int]
    echo: bool
    stop: Optional[Union[str, List[str]]]
    frequency_penalty: float
    presence_penalty: float
    repetition_penalty: float
    top_k: int
    stream: bool
    stream_options: Optional[Union[dict, None]]
    tfs_z: float
    mirostat_mode: int
    mirostat_tau: float
    mirostat_eta: float
    model: Optional[str]
    grammar: Optional[Any]
    stopping_criteria: Optional["StoppingCriteriaList"]
    logits_processor: Optional["LogitsProcessorList"]
    tools: Optional[List[Dict]]


class LlamaCppModelConfig(TypedDict, total=False):
    n_ctx: int
    n_parts: int
    n_gpu_layers: int
    split_mode: int
    main_gpu: int
    seed: int
    f16_kv: bool
    logits_all: bool
    vocab_only: bool
    use_mmap: bool
    use_mlock: bool
    n_threads: Optional[int]
    n_batch: int
    last_n_tokens_size: int
    lora_base: Optional[str]
    lora_path: Optional[str]
    low_vram: bool
    n_gqa: Optional[int]  # (TEMPORARY) must be 8 for llama2 70b
    rms_norm_eps: Optional[float]  # (TEMPORARY)
    verbose: bool


class PytorchGenerateConfig(TypedDict, total=False):
    temperature: float
    repetition_penalty: float
    top_p: float
    top_k: int
    stream: bool
    max_tokens: int
    echo: bool
    stop: Optional[Union[str, List[str]]]
    stop_token_ids: Optional[Union[int, List[int]]]
    stream_interval: int
    model: Optional[str]
    tools: Optional[List[Dict]]
    lora_name: Optional[str]
    stream_options: Optional[Union[dict, None]]
    request_id: Optional[str]


class CogagentGenerateConfig(PytorchGenerateConfig, total=False):
    platform: Optional[Literal["Mac", "WIN", "Mobile"]]
    format: Optional[
        Literal[
            "(Answer in Action-Operation-Sensitive format.)",
            "(Answer in Status-Plan-Action-Operation format.)",
            "(Answer in Status-Action-Operation-Sensitive format.)",
            "(Answer in Status-Action-Operation format.)",
            "(Answer in Action-Operation format.)",
        ]
    ]


class PytorchModelConfig(TypedDict, total=False):
    revision: Optional[str]
    device: str
    gpus: Optional[str]
    num_gpus: int
    max_gpu_memory: str
    gptq_ckpt: Optional[str]
    gptq_wbits: int
    gptq_groupsize: int
    gptq_act_order: bool
    trust_remote_code: bool
    max_num_seqs: int
    enable_tensorizer: Optional[bool]


def get_pydantic_model_from_method(
    meth,
    exclude_fields: Optional[Iterable[str]] = None,
    include_fields: Optional[Dict[str, Any]] = None,
) -> BaseModel:
    # The validate_arguments set Config.extra = "forbid" by default.
    f = validate_arguments(meth, config={"arbitrary_types_allowed": True})
    model = f.model
    model.Config.extra = "ignore"
    model.__fields__.pop("self", None)
    model.__fields__.pop("args", None)
    model.__fields__.pop("kwargs", None)
    pydantic_private_keys = [
        key for key in model.__fields__.keys() if key.startswith("v__")
    ]
    for key in pydantic_private_keys:
        model.__fields__.pop(key)
    if exclude_fields is not None:
        for key in exclude_fields:
            model.__fields__.pop(key, None)
    if include_fields is not None:
        dummy_model = create_model("DummyModel", **include_fields)
        model.__fields__.update(dummy_model.__fields__)
    return model


def fix_forward_ref(model):
    """
    pydantic in Python 3.8 generates ForwardRef field, we replace them
    by the Optional[Any]
    """
    exclude_fields = []
    include_fields = {}
    for key, field in model.__fields__.items():
        if isinstance(field.annotation, ForwardRef):
            exclude_fields.append(key)
            include_fields[key] = (Optional[Any], None)
    if exclude_fields:
        for key in exclude_fields:
            model.__fields__.pop(key, None)
    if include_fields:
        dummy_model = create_model("DummyModel", **include_fields)
        model.__fields__.update(dummy_model.__fields__)
    return model


class ModelAndPrompt(BaseModel):
    model: str
    prompt: str


class CreateCompletionTorch(BaseModel):
    echo: bool = echo_field
    max_tokens: Optional[int] = max_tokens_field
    repetition_penalty: float = repeat_penalty_field
    stop: Optional[Union[str, List[str]]] = stop_field
    stop_token_ids: Optional[Union[int, List[int]]] = none_field
    stream: bool = stream_field
    stream_options: Optional[Union[dict, None]] = stream_option_field
    stream_interval: int = stream_interval_field
    temperature: float = temperature_field
    top_p: float = top_p_field
    top_k: int = top_k_field
    lora_name: Optional[str]
    request_id: Optional[str]


CreateCompletionLlamaCpp: BaseModel
try:
    from llama_cpp import Llama

    CreateCompletionLlamaCpp = get_pydantic_model_from_method(
        Llama.create_completion,
        exclude_fields=["model", "prompt", "grammar", "max_tokens"],
        include_fields={
            "grammar": (Optional[Any], None),
            "max_tokens": (Optional[int], max_tokens_field),
            "lora_name": (Optional[str], None),
            "stream_options": (Optional[Union[dict, None]], None),
        },
    )
except ImportError:
    CreateCompletionLlamaCpp = create_model("CreateCompletionLlamaCpp")


# This type is for openai API compatibility
CreateCompletionOpenAI: BaseModel


from openai.types.completion_create_params import CompletionCreateParamsNonStreaming

CreateCompletionOpenAI = create_model_from_typeddict(
    CompletionCreateParamsNonStreaming,
)
CreateCompletionOpenAI = fix_forward_ref(CreateCompletionOpenAI)


class CreateCompletion(
    ModelAndPrompt,
    CreateCompletionTorch,
    CreateCompletionLlamaCpp,
    CreateCompletionOpenAI,
):
    pass


class CreateChatModel(BaseModel):
    model: str


# Currently, chat calls generates, so the params share the same one.
CreateChatCompletionTorch = CreateCompletionTorch
CreateChatCompletionLlamaCpp: BaseModel = CreateCompletionLlamaCpp


from ._compat import CreateChatCompletionOpenAI


class CreateChatCompletion(  # type: ignore
    CreateChatModel,
    CreateChatCompletionTorch,
    CreateChatCompletionLlamaCpp,
    CreateChatCompletionOpenAI,
):
    pass


class LoRA:
    def __init__(self, lora_name: str, local_path: str):
        self.lora_name = lora_name
        self.local_path = local_path

    def to_dict(self):
        return {
            "lora_name": self.lora_name,
            "local_path": self.local_path,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            lora_name=data["lora_name"],
            local_path=data["local_path"],
        )


class PeftModelConfig:
    def __init__(
        self,
        peft_model: Optional[List[LoRA]] = None,
        image_lora_load_kwargs: Optional[Dict] = None,
        image_lora_fuse_kwargs: Optional[Dict] = None,
    ):
        self.peft_model = peft_model
        self.image_lora_load_kwargs = image_lora_load_kwargs
        self.image_lora_fuse_kwargs = image_lora_fuse_kwargs

    def to_dict(self):
        return {
            "lora_list": [lora.to_dict() for lora in self.peft_model]
            if self.peft_model
            else None,
            "image_lora_load_kwargs": self.image_lora_load_kwargs,
            "image_lora_fuse_kwargs": self.image_lora_fuse_kwargs,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        peft_model_list = data.get("lora_list", None)
        peft_model = (
            [LoRA.from_dict(lora_dict) for lora_dict in peft_model_list]
            if peft_model_list is not None
            else None
        )

        return cls(
            peft_model=peft_model,
            image_lora_load_kwargs=data.get("image_lora_load_kwargs"),
            image_lora_fuse_kwargs=data.get("image_lora_fuse_kwargs"),
        )
