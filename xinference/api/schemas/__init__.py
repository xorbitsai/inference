"""Pydantic schemas for Xinference REST API."""

from .requests import (
    AutoConfigLLMRequest,
    CreateCompletionRequest,
    CreateEmbeddingRequest,
    RegisterModelRequest,
    RerankRequest,
    SDAPIImg2imgRequst,
    SDAPIOptionsRequest,
    SDAPITxt2imgRequst,
    SpeechRequest,
    TextToImageRequest,
    TextToVideoRequest,
    UpdateModelRequest,
)

__all__ = [
    "AutoConfigLLMRequest",
    "CreateCompletionRequest",
    "CreateEmbeddingRequest",
    "RegisterModelRequest",
    "RerankRequest",
    "SDAPIImg2imgRequst",
    "SDAPIOptionsRequest",
    "SDAPITxt2imgRequst",
    "SpeechRequest",
    "TextToImageRequest",
    "TextToVideoRequest",
    "UpdateModelRequest",
]
