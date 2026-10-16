"""Pydantic schemas for Xinference REST API."""

from .requests import (
    AutoConfigLLMRequest,
    CreateCompletionRequest,
    CreateEmbeddingRequest,
    RegisterModelRequest,
    RerankRequest,
    SDAPIControlNetDetect,
    SDAPIImg2imgRequst,
    SDAPIInterrupt,
    SDAPIOptionsRequest,
    SDAPIProgress,
    SDAPITxt2imgRequst,
    SpeechRequest,
    TextToImageRequest,
    TextToVideoRequest,
    UpdateModelRequest,
    WorldGenerationRequest,
)

__all__ = [
    "AutoConfigLLMRequest",
    "CreateCompletionRequest",
    "CreateEmbeddingRequest",
    "RegisterModelRequest",
    "RerankRequest",
    "SDAPIControlNetDetect",
    "SDAPIProgress",
    "SDAPIInterrupt",
    "SDAPIImg2imgRequst",
    "SDAPIOptionsRequest",
    "SDAPITxt2imgRequst",
    "SpeechRequest",
    "TextToImageRequest",
    "TextToVideoRequest",
    "UpdateModelRequest",
    "WorldGenerationRequest",
]
