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

from .deepdoc import DeepDocModel
from .deepseek_ocr import DeepSeekOCRModel
from .dots_ocr import DotsOCRModel
from .got_ocr2 import GotOCR2Model
from .hunyuan_ocr import HunyuanOCRModel
from .jina_ocr import JinaOCRModel
from .mlx import MLXDeepSeekOCRModel
from .monkeyocr import MonkeyOCRModel
from .navidc_ocr import NaviDCOCRModel
from .ocr_family import SUPPORTED_ENGINES
from .ovisocr2 import OvisOCR2Model
from .ovisocr3 import OvisOCR3Model
from .paddleocr_vl import PaddleOCRVLModel
from .sglang import SGLangOvisOCR3Model
from .teleocr import LlamaCppTeleOCRModel, TeleOCRModel
from .unlimited_ocr import UnlimitedOCRModel
from .unlimited_ocr2 import UnlimitedOCR2Model
from .vllm import (
    VLLMDeepSeekOCRModel,
    VLLMGotOCR2Model,
    VLLMHunyuanOCRModel,
    VLLMNaviDCOCRModel,
    VLLMOvisOCR2Model,
    VLLMOvisOCR3Model,
    VLLMPaddleOCRVLModel,
    VLLMTeleOCRModel,
)
from .xiaomi_ocr import XiaomiOCRModel

__all__ = [
    "DeepDocModel",
    "DeepSeekOCRModel",
    "DotsOCRModel",
    "GotOCR2Model",
    "HunyuanOCRModel",
    "JinaOCRModel",
    "MonkeyOCRModel",
    "NaviDCOCRModel",
    "OvisOCR2Model",
    "OvisOCR3Model",
    "PaddleOCRVLModel",
    "TeleOCRModel",
    "LlamaCppTeleOCRModel",
    "UnlimitedOCRModel",
    "UnlimitedOCR2Model",
    "XiaomiOCRModel",
]


def register_builtin_ocr_engines() -> None:
    SUPPORTED_ENGINES["transformers"] = [
        DeepSeekOCRModel,
        DotsOCRModel,
        GotOCR2Model,
        HunyuanOCRModel,
        JinaOCRModel,
        MonkeyOCRModel,
        NaviDCOCRModel,
        OvisOCR2Model,
        OvisOCR3Model,
        PaddleOCRVLModel,
        TeleOCRModel,
        UnlimitedOCRModel,
        UnlimitedOCR2Model,
        XiaomiOCRModel,
    ]
    SUPPORTED_ENGINES["vllm"] = [
        VLLMDeepSeekOCRModel,
        VLLMHunyuanOCRModel,
        VLLMNaviDCOCRModel,
        VLLMOvisOCR2Model,
        VLLMOvisOCR3Model,
        VLLMTeleOCRModel,
    ]
    SUPPORTED_ENGINES["llama.cpp"] = [LlamaCppTeleOCRModel]
    SUPPORTED_ENGINES["sglang"] = [SGLangOvisOCR3Model]
    SUPPORTED_ENGINES["mlx"] = [MLXDeepSeekOCRModel]
    # DeepDoc runs on onnxruntime via its own engine name
    SUPPORTED_ENGINES["deepdoc"] = [DeepDocModel]
