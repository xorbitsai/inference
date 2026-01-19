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

from .deepseek_ocr import DeepSeekOCRModel
from .got_ocr2 import GotOCR2Model
from .hunyuan_ocr import HunyuanOCRModel
from .mineru import MinerUModel
from .mlx import MLXDeepSeekOCRModel
from .ocr_family import SUPPORTED_ENGINES
from .paddleocr_vl import PaddleOCRVLModel
from .vllm import (
    VLLMDeepSeekOCRModel,
    VLLMGotOCR2Model,
    VLLMHunyuanOCRModel,
    VLLMMinerUModel,
    VLLMPaddleOCRVLModel,
)

__all__ = [
    "DeepSeekOCRModel",
    "GotOCR2Model",
    "HunyuanOCRModel",
    "MinerUModel",
    "PaddleOCRVLModel",
]


def register_builtin_ocr_engines() -> None:
    SUPPORTED_ENGINES["transformers"] = [
        DeepSeekOCRModel,
        GotOCR2Model,
        HunyuanOCRModel,
        MinerUModel,
        PaddleOCRVLModel,
    ]
    SUPPORTED_ENGINES["vllm"] = [
        VLLMDeepSeekOCRModel,
        VLLMHunyuanOCRModel,
        VLLMMinerUModel,
    ]
    SUPPORTED_ENGINES["mlx"] = [MLXDeepSeekOCRModel]
