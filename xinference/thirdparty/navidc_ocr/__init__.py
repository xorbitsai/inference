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

"""Register the vendored NaviDC-OCR model implementation with vLLM."""

from importlib.metadata import version as package_version

from packaging.version import InvalidVersion, Version

MODEL_ARCHITECTURE = "NaviOCRForConditionalGeneration"
MODEL_CLASS = (
    "xinference.thirdparty.navidc_ocr.qwen2_5_vl:"
    "Qwen2_5_VLForConditionalGeneration"
)
COMPAT_MODEL_CLASS = (
    "xinference.thirdparty.navidc_ocr.vllm_compat:"
    "NaviOCRForConditionalGeneration"
)
UPSTREAM_COMMIT = "2e79d29bf32d4e8997b7cbd2ee619a12bfc8d616"


def _get_model_class() -> str:
    try:
        if Version(package_version("vllm")) >= Version("0.23.0"):
            return COMPAT_MODEL_CLASS
    except (ImportError, InvalidVersion):
        pass
    return MODEL_CLASS


def register() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(MODEL_ARCHITECTURE, _get_model_class())
