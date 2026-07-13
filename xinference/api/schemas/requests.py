"""Request schemas for Xinference REST API.

This module is intentionally thin and contains only Pydantic models used by the
FastAPI layer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from ..._compat import BaseModel, Field
from ...types import CreateCompletion


class CreateCompletionRequest(CreateCompletion):
    class Config:
        schema_extra = {
            "example": {
                "prompt": "\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n",
                "stop": ["\n", "###"],
            }
        }


class CreateEmbeddingRequest(BaseModel):
    model: str
    input: Union[
        str, List[str], List[int], List[List[int]], Dict[str, str], List[Dict[str, str]]
    ] = Field(description="The input to embed.")
    user: Optional[str] = None
    # Truncate each input to this many tokens before encoding. Mirrors the
    # vLLM LLM semantics: None = no truncation, >0 = cap at N tokens,
    # <0 = cap at the model's max_tokens. Honored by the embedding model base
    # class (see ``EmbeddingModel._truncate_sentences``).
    truncate_prompt_tokens: Optional[int] = None

    class Config:
        schema_extra = {
            "example": {
                "input": "The food was delicious and the waiter...",
            }
        }


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: Optional[bool] = False
    return_len: Optional[bool] = False
    max_chunks_per_doc: Optional[int] = None
    kwargs: Optional[str] = None


class TextToImageRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]] = Field(description="The input to embed.")
    n: Optional[int] = 1
    response_format: Optional[str] = "url"
    size: Optional[str] = "1024*1024"
    kwargs: Optional[str] = None
    user: Optional[str] = None


class SDAPIOptionsRequest(BaseModel):
    sd_model_checkpoint: Optional[str] = None


class SDAPITxt2imgRequst(BaseModel):
    model: Optional[str]
    prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    steps: Optional[int] = None
    seed: Optional[int] = -1
    cfg_scale: Optional[float] = 7.0
    override_settings: Optional[dict] = {}
    width: Optional[int] = 512
    height: Optional[int] = 512
    sampler_name: Optional[str] = None
    denoising_strength: Optional[float] = None
    kwargs: Optional[str] = None
    user: Optional[str] = None


class SDAPIImg2imgRequst(BaseModel):
    model: Optional[str]
    init_images: Optional[list]
    prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    steps: Optional[int] = None
    seed: Optional[int] = -1
    cfg_scale: Optional[float] = 7.0
    override_settings: Optional[dict] = {}
    width: Optional[int] = 512
    height: Optional[int] = 512
    sampler_name: Optional[str] = None
    denoising_strength: Optional[float] = None
    kwargs: Optional[str] = None
    user: Optional[str] = None


class TextToVideoRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]] = Field(description="The input to embed.")
    n: Optional[int] = 1
    kwargs: Optional[str] = None
    user: Optional[str] = None


class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: Optional[str]
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0
    stream: Optional[bool] = False
    kwargs: Optional[str] = None


class RegisterModelRequest(BaseModel):
    model: str
    worker_ip: Optional[str]
    persist: bool


class AutoConfigLLMRequest(BaseModel):
    model_path: str
    model_family: str


class UpdateModelRequest(BaseModel):
    model_type: str
