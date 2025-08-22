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

import collections.abc
import logging
import platform
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Union

from ...types import PeftModelConfig
from ..core import CacheableModelSpec, VirtualEnvSettings
from ..utils import ModelInstanceInfoMixin
from .ocr.got_ocr2 import GotOCR2Model
from .stable_diffusion.core import DiffusionModel
from .stable_diffusion.mlx import MLXDiffusionModel

logger = logging.getLogger(__name__)

IMAGE_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)
BUILTIN_IMAGE_MODELS: Dict[str, List["ImageModelFamilyV2"]] = {}


def get_image_model_descriptions():
    import copy

    return copy.deepcopy(IMAGE_MODEL_DESCRIPTIONS)


class ImageModelFamilyV2(CacheableModelSpec, ModelInstanceInfoMixin):
    version: Literal[2] = 2
    model_family: str
    model_name: str
    model_id: str
    model_revision: str
    model_hub: str = "huggingface"
    model_ability: Optional[List[str]]
    controlnet: Optional[List["ImageModelFamilyV2"]]
    default_model_config: Optional[dict] = {}
    default_generate_config: Optional[dict] = {}
    gguf_model_id: Optional[str]
    gguf_quantizations: Optional[List[str]]
    gguf_model_file_name_template: Optional[str]
    lightning_model_id: Optional[str]
    lightning_versions: Optional[List[str]]
    lightning_model_file_name_template: Optional[str]

    virtualenv: Optional[VirtualEnvSettings]

    class Config:
        extra = "allow"

    def to_description(self):
        if self.controlnet is not None:
            controlnet = [cn.dict() for cn in self.controlnet]
        else:
            controlnet = self.controlnet
        return {
            "model_type": "image",
            "address": getattr(self, "address", None),
            "accelerators": getattr(self, "accelerators", None),
            "model_name": self.model_name,
            "model_family": self.model_family,
            "model_revision": self.model_revision,
            "model_ability": self.model_ability,
            "controlnet": controlnet,
        }

    def to_version_info(self):
        from .cache_manager import ImageCacheManager
        from .utils import get_model_version

        cache_manager = ImageCacheManager(self)

        if not self.controlnet:
            return [
                {
                    "model_version": get_model_version(self, None),
                    "model_file_location": cache_manager.get_cache_dir(),
                    "cache_status": cache_manager.get_cache_status(),
                    "controlnet": "zoe-depth",
                }
            ]
        else:
            res = []
            for cn in self.controlnet:
                res.append(
                    {
                        "model_version": get_model_version(self, cn),
                        "model_file_location": cache_manager.get_cache_dir(),
                        "cache_status": cache_manager.get_cache_status(),
                        "controlnet": cn.model_name,
                    }
                )
            return res


def generate_image_description(
    image_model: ImageModelFamilyV2,
) -> Dict[str, List[Dict]]:
    res = defaultdict(list)
    res[image_model.model_name].extend(image_model.to_version_info())
    return res


def match_diffusion(
    model_name: str,
    download_hub: Optional[
        Literal["huggingface", "modelscope", "openmind_hub", "csghub"]
    ] = None,
) -> ImageModelFamilyV2:
    from ..utils import download_from_modelscope
    from . import BUILTIN_IMAGE_MODELS
    from .custom import get_user_defined_images

    for model_spec in get_user_defined_images():
        if model_spec.model_name == model_name:
            return model_spec

    if model_name in BUILTIN_IMAGE_MODELS:
        if download_hub == "modelscope" or download_from_modelscope():
            return (
                [
                    x
                    for x in BUILTIN_IMAGE_MODELS[model_name]
                    if x.model_hub == "modelscope"
                ]
                + [
                    x
                    for x in BUILTIN_IMAGE_MODELS[model_name]
                    if x.model_hub == "huggingface"
                ]
            )[0]
        else:
            return [
                x
                for x in BUILTIN_IMAGE_MODELS[model_name]
                if x.model_hub == "huggingface"
            ][0]
    else:
        raise ValueError(
            f"Image model {model_name} not found, available"
            f"model list: {BUILTIN_IMAGE_MODELS.keys()}"
        )


def create_ocr_model_instance(
    model_uid: str,
    model_spec: ImageModelFamilyV2,
    model_path: Optional[str] = None,
    **kwargs,
) -> GotOCR2Model:
    from .cache_manager import ImageCacheManager

    if not model_path:
        cache_manager = ImageCacheManager(model_spec)
        model_path = cache_manager.cache()
    model = GotOCR2Model(
        model_uid,
        model_path,
        model_spec=model_spec,
        **kwargs,
    )
    return model


def create_image_model_instance(
    model_uid: str,
    model_name: str,
    peft_model_config: Optional[PeftModelConfig] = None,
    download_hub: Optional[
        Literal["huggingface", "modelscope", "openmind_hub", "csghub"]
    ] = None,
    model_path: Optional[str] = None,
    gguf_quantization: Optional[str] = None,
    gguf_model_path: Optional[str] = None,
    lightning_version: Optional[str] = None,
    lightning_model_path: Optional[str] = None,
    **kwargs,
) -> Union[DiffusionModel, MLXDiffusionModel, GotOCR2Model]:
    from .cache_manager import ImageCacheManager

    model_spec = match_diffusion(model_name, download_hub)
    if model_spec.model_ability and "ocr" in model_spec.model_ability:
        return create_ocr_model_instance(
            model_uid=model_uid,
            model_name=model_name,
            model_spec=model_spec,
            model_path=model_path,
            **kwargs,
        )

    # use default model config
    model_default_config = (model_spec.default_model_config or {}).copy()
    model_default_config.update(kwargs)
    kwargs = model_default_config

    controlnet = kwargs.get("controlnet")
    # Handle controlnet
    if controlnet is not None:
        if isinstance(controlnet, str):
            controlnet = [controlnet]
        elif not isinstance(controlnet, collections.abc.Sequence):
            raise ValueError("controlnet should be a str or a list of str.")
        elif set(controlnet) != len(controlnet):
            raise ValueError("controlnet should be a list of unique str.")
        elif not model_spec.controlnet:
            raise ValueError(f"Model {model_name} has empty controlnet list.")

        controlnet_model_paths = []
        assert model_spec.controlnet is not None
        for name in controlnet:
            for cn_model_spec in model_spec.controlnet:
                if cn_model_spec.model_name == name:
                    cn_cache_manager = ImageCacheManager(cn_model_spec)
                    controlnet_model_path = cn_cache_manager.cache()
                    controlnet_model_paths.append(controlnet_model_path)
                    break
            else:
                raise ValueError(
                    f"controlnet `{name}` is not supported for model `{model_name}`."
                )
        if len(controlnet_model_paths) == 1:
            kwargs["controlnet"] = (controlnet[0], controlnet_model_paths[0])
        else:
            kwargs["controlnet"] = [
                (n, path) for n, path in zip(controlnet, controlnet_model_paths)
            ]
    cache_manager = ImageCacheManager(model_spec)
    if not model_path:
        model_path = cache_manager.cache()
    if not gguf_model_path and gguf_quantization:
        gguf_model_path = cache_manager.cache_gguf(gguf_quantization)
    if not lightning_model_path and lightning_version:
        lightning_model_path = cache_manager.cache_lightning(lightning_version)
    if peft_model_config is not None:
        lora_model = peft_model_config.peft_model
        lora_load_kwargs = peft_model_config.image_lora_load_kwargs
        lora_fuse_kwargs = peft_model_config.image_lora_fuse_kwargs
    else:
        lora_model = None
        lora_load_kwargs = None
        lora_fuse_kwargs = None

    if (
        platform.system() == "Darwin"
        and "arm" in platform.machine().lower()
        and MLXDiffusionModel.support_model(model_name)
    ):
        # Mac with M series silicon chips
        model_cls = MLXDiffusionModel
    else:
        model_cls = DiffusionModel  # type: ignore

    model = model_cls(
        model_uid,
        model_path,
        lora_model=lora_model,
        lora_load_kwargs=lora_load_kwargs,
        lora_fuse_kwargs=lora_fuse_kwargs,
        model_spec=model_spec,
        gguf_model_path=gguf_model_path,
        lightning_model_path=lightning_model_path,
        **kwargs,
    )
    return model
