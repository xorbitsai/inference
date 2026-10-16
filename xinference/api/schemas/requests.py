"""Request schemas for Xinference REST API.

This module is intentionally thin and contains only Pydantic models used by the
FastAPI layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

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
        str,
        List[str],
        List[int],
        List[List[int]],
        Dict[str, Any],
        List[Dict[str, Any]],
        List[Union[str, Dict[str, Any]]],
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
    query: Union[str, Dict[str, Any]]
    documents: List[Union[str, Dict[str, Any]]]
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
    model: Optional[str] = None
    prompt: str = ""
    negative_prompt: str = ""
    steps: Optional[int] = Field(None, ge=1)
    seed: int = -1
    subseed: int = -1
    subseed_strength: float = Field(0.0, ge=0, le=1)
    seed_resize_from_h: int = Field(0, ge=0)
    seed_resize_from_w: int = Field(0, ge=0)
    batch_size: int = Field(1, ge=1)
    n_iter: int = Field(1, ge=1)
    cfg_scale: float = 7.0
    override_settings: dict = Field(default_factory=dict)
    width: int = Field(512, ge=8)
    height: int = Field(512, ge=8)
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    denoising_strength: Optional[float] = Field(None, gt=0, le=1)
    kwargs: Optional[str] = None
    user: Optional[str] = None
    enable_hr: bool = False
    hr_scale: float = Field(2.0, ge=1)
    hr_upscaler: str = "Latent"
    hr_second_pass_steps: int = Field(0, ge=0)
    alwayson_scripts: dict = Field(default_factory=dict)
    request_id: Optional[str] = None


class SDAPIImg2imgRequst(SDAPITxt2imgRequst):
    init_images: List[str] = Field(..., min_items=1)
    mask: Optional[str] = None
    mask_blur: int = Field(0, ge=0)
    inpaint_full_res: bool = False
    inpaint_full_res_padding: int = Field(0, ge=0)
    inpainting_mask_invert: int = Field(0, ge=0, le=1)
    resize_mode: int = Field(0, ge=0, le=2)


class SDAPIControlNetDetect(BaseModel):
    controlnet_masks: List[str] = Field(default_factory=list)
    low_vram: bool = False
    controlnet_module: str = "none"
    controlnet_input_images: List[str] = Field(default_factory=list)
    controlnet_images: List[str] = Field(default_factory=list)
    controlnet_processor_res: int = Field(512, ge=64)
    controlnet_threshold_a: float = 64
    controlnet_threshold_b: float = 64


class SDAPIProgress(BaseModel):
    request_id: str


class SDAPIInterrupt(BaseModel):
    model: str
    request_id: str


class TextToVideoRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]] = Field(description="The input to embed.")
    n: Optional[int] = 1
    kwargs: Optional[str] = None
    user: Optional[str] = None


class WorldGenerationRequest(BaseModel):
    model: str
    prompt: str
    image: Optional[str] = None
    video: Optional[str] = None
    generation_config: Dict[str, Any] = Field(default_factory=dict)
    extra_body: Dict[str, Any] = Field(default_factory=dict)
    user: Optional[str] = None

    class Config:
        extra = "forbid"


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
