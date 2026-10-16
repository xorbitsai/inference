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

import logging
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image

from ..._compat import BaseModel
from ...device_utils import get_available_device
from ...thirdparty.controlnet.supported_preprocessor import Preprocessor
from ...thirdparty.controlnet.utils import (
    encode_tensor_to_base64,
    encode_to_base64,
    to_base64_nparray,
)
from .lvminthin import lvmin_thin, nake_nms
from .utils import decode_base64_to_image, get_unique_axis0

if TYPE_CHECKING:
    from .core import ImageModelFamilyV2

logger = logging.getLogger(__name__)


class ControlNetUnit(BaseModel):
    image: str = ""
    mask: Optional[str] = None
    module: str = "none"
    model: str = "None"
    weight: float = 1.0
    resize_mode: str = "Resize and Fill"
    low_vram: bool = False
    processor_res: int = 512
    threshold_a: float = 64
    threshold_b: float = 64
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    control_mode: str = "Balanced"
    pixel_perfect: bool = False
    guessmode: int = 0  # deprecated: use control_mode
    hr_option: str = "Both"  # Both, Low res only, High res only
    enabled: bool = True


class ResizeMode(Enum):
    """
    Resize modes for ControlNet input images.
    """

    RESIZE = "Just Resize"
    INNER_FIT = "Crop and Resize"
    OUTER_FIT = "Resize and Fill"

    @classmethod
    def _missing_(cls, value):
        return {
            0: cls.RESIZE,
            1: cls.INNER_FIT,
            2: cls.OUTER_FIT,
            "0": cls.RESIZE,
            "1": cls.INNER_FIT,
            "2": cls.OUTER_FIT,
        }.get(value)

    def int_value(self):
        if self == ResizeMode.RESIZE:
            return 0
        elif self == ResizeMode.INNER_FIT:
            return 1
        elif self == ResizeMode.OUTER_FIT:
            return 2
        assert False, "NOTREACHED"


def extract_controlnet_params(alwayson_scripts: dict) -> List[ControlNetUnit]:
    controlnet_units = (
        alwayson_scripts.get("ControlNet") or alwayson_scripts.get("controlnet") or {}
    ).get("args", [])
    return [ControlNetUnit.parse_obj(arg) for arg in controlnet_units]


def generate_controlnet_kwargs_for_text2image(
    model_family: "ImageModelFamilyV2", kwargs: dict, alwayson_scripts: dict
):
    _generate_contronet_kwargs(model_family, kwargs, alwayson_scripts, "text2image")


def generate_controlnet_kwargs_for_image2image(
    model_family: "ImageModelFamilyV2", kwargs: dict, alwayson_scripts: dict
):
    _generate_contronet_kwargs(model_family, kwargs, alwayson_scripts, "image2image")


def _generate_contronet_kwargs(
    model_family: "ImageModelFamilyV2",
    kwargs: dict,
    alwayson_scripts: dict,
    sd_type: str,
):
    from .cache_manager import ImageCacheManager

    is_hires = kwargs.pop("_sdapi_hires", False)
    controlnet_units = [
        unit
        for unit in extract_controlnet_params(alwayson_scripts)
        if unit.hr_option == "Both" or (unit.hr_option == "High res only") == is_hires
    ]

    if not controlnet_units or not any(u.enabled for u in controlnet_units):
        return

    source_image = kwargs.get("image")
    kwargs["control_guidance_start"] = control_guidance_start = []
    kwargs["control_guidance_end"] = control_guidance_end = []
    kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale = []
    kwargs["controlnet"] = []
    if sd_type == "text2image":
        kwargs["image"] = controlnet_images = []  # type: ignore
    elif sd_type == "image2image":
        kwargs["control_image"] = controlnet_images = []
    else:
        raise NotImplementedError

    for unit in controlnet_units:
        if not unit.enabled:
            continue

        if not unit.image and source_image is not None:
            from .utils import encode_pil_to_base64

            unit.image = encode_pil_to_base64(
                source_image[0] if isinstance(source_image, list) else source_image
            )
        if unit.control_mode in ("ControlNet is more important", "2"):
            kwargs["guess_mode"] = True

        if "reference" in unit.module:
            # reference requiries no model
            cn = ("reference", unit.module)
            kwargs["controlnet"].append(cn)
            kwargs["ref_image"] = decode_base64_to_image(unit.image)
            if unit.module == "reference_adain+attn":
                kwargs["reference_adain"] = True
                kwargs["reference_attn"] = True
            elif unit.module == "reference_adain":
                kwargs["reference_adain"] = True
                kwargs["reference_attn"] = False
            else:
                assert unit.module == "reference_only"
                kwargs["reference_adain"] = False
                kwargs["reference_attn"] = True
            continue

        width, height = tuple(int(s) for s in kwargs["size"].split("*", 1))
        resize_mode = ResizeMode(unit.resize_mode)
        if unit.pixel_perfect:
            unit.processor_res = pixel_perfect_resolution(
                to_base64_nparray(unit.image),
                target_H=height,
                target_W=width,
                resize_mode=resize_mode,
            )
            logger.debug("Controlnet preprocessor resolution = %s", unit.processor_res)

        control_guidance_start.append(unit.guidance_start)
        control_guidance_end.append(unit.guidance_end)
        controlnet_conditioning_scale.append(unit.weight)

        detected = detect(  # type: ignore
            unit.module,
            [unit.image],
            unit.processor_res,
            unit.threshold_a,
            unit.threshold_b,
            [unit.mask],  # type: ignore
            unit.low_vram,
            return_raw=True,
        )
        if detect_images := detected.get("images"):
            detect_image = detect_images[0]
            _, detected_map = detectmap_proc(
                detect_image, unit.module, resize_mode, height, width
            )
            controlnet_images.append(Image.fromarray(detected_map))
        else:
            raise NotImplementedError

        model = unit.model.rsplit(None, 1)[0]
        found = False
        for cn in model_family.controlnet or []:  # type: ignore
            if cn.model_name == model or cn.model_name in model:  # type: ignore
                found = True
                path = ImageCacheManager(cn).cache()
                kwargs["controlnet"].append((model, path))
                break
        if not found:
            raise ValueError(f"Cannot support model: {model}")

    if len(controlnet_conditioning_scale) == 1:
        kwargs["control_guidance_start"] = control_guidance_start[0]
        kwargs["control_guidance_end"] = control_guidance_end[0]
        kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale[0]
        if sd_type == "text2image":
            kwargs["image"] = controlnet_images[0]
        elif sd_type == "image2image":
            kwargs["control_image"] = controlnet_images[0]
    if len(kwargs["controlnet"]) == 1:
        kwargs["controlnet"] = kwargs["controlnet"][0]


def detect(
    controlnet_module: str = "none",
    controlnet_input_images: List[str] = [],
    controlnet_processor_res: int = -1,
    controlnet_threshold_a: float = -1,
    controlnet_threshold_b: float = -1,
    controlnet_masks: List[str] = [],
    low_vram: bool = False,
    return_raw: bool = False,
):
    controlnet_input_images = controlnet_input_images or []
    controlnet_masks = controlnet_masks or []

    preprocessor = Preprocessor.get_preprocessor(controlnet_module)

    if preprocessor is None:
        raise ValueError("Module not supported")

    if controlnet_module in (
        "clip_vision",
        "revision_clipvision",
        "revision_ignore_prompt",
        "ip-adapter-auto",
    ):
        raise ValueError("Module not supported")

    if len(controlnet_input_images) == 0:
        raise ValueError("No image selected")

    if preprocessor.requires_mask and len(controlnet_masks) != len(
        controlnet_input_images
    ):
        raise ValueError(
            f"Preprocessor {controlnet_module} requires `controlnet_masks` param."
        )

    logger.info(
        f"Detecting {str(len(controlnet_input_images))} images with the {controlnet_module} module."
    )

    unit = ControlNetUnit(
        enabled=True,
        module=preprocessor.label,
        processor_res=controlnet_processor_res,
        threshold_a=controlnet_threshold_a,
        threshold_b=controlnet_threshold_b,
    )

    tensors = []
    images = []
    poses = []

    try:
        for i, input_image in enumerate(controlnet_input_images):
            img = to_base64_nparray(input_image)
            # Has mask.
            if i < len(controlnet_masks) and controlnet_masks[i]:
                if preprocessor.accepts_mask:
                    mask = to_base64_nparray(controlnet_masks[i])[:, :, :1]
                    img = np.concatenate([img, mask], axis=2)
                else:
                    logger.warning(
                        f"Preprocessor {controlnet_module} does not accept mask. Mask ignored"
                    )

            class JsonAcceptor:
                value: Optional[dict]

                def __init__(self) -> None:
                    self.value = None

                def accept(self, json_dict: dict) -> None:
                    self.value = json_dict

            json_acceptor = JsonAcceptor()
            result = preprocessor.cached_call(
                img,
                resolution=unit.processor_res,
                slider_1=unit.threshold_a,
                slider_2=unit.threshold_b,
                json_pose_callback=json_acceptor.accept,
                low_vram=low_vram,
            )
            if preprocessor.returns_image:
                if return_raw:
                    images.append(result.value)
                else:
                    images.append(encode_to_base64(result.display_images[0]))
            else:
                if return_raw:
                    tensors.append(result.value)
                else:
                    tensors.append(encode_tensor_to_base64(result.value))

            if "openpose" in controlnet_module:
                assert json_acceptor.value is not None
                poses.append(json_acceptor.value)

    finally:
        preprocessor.unload()

    res = {"info": "Success"}
    if poses:
        res["poses"] = poses  # type: ignore
    if images:
        res["images"] = images  # type: ignore
    if tensors:
        res["tensor"] = tensors  # type: ignore

    return res


def _list_models(model_family) -> Mapping[str, Optional[str]]:
    models = {"none": "None"}
    for controlnet in model_family.controlnet or []:
        name = controlnet.model_name + "_fp16"
        name_and_hash = name + " [" + (controlnet.model_revision or "")[:8] + "]"
        models[name] = name_and_hash
    return models


def list_models(model_family) -> Dict[str, list]:
    return {"model_list": list(_list_models(model_family).values())}


def list_modules(alias_names: bool = False) -> Dict[str, list]:
    modules = [
        (p.label if alias_names else p.name)
        for p in Preprocessor.get_sorted_preprocessors()
    ]
    return {"module_list": modules}


def select_control_type(
    control_type: str,
    model_family,
) -> Tuple[List[str], List[str], str, str]:
    pattern = control_type.lower()
    all_models = list(_list_models(model_family))

    if pattern == "all":
        return (
            [p.label for p in Preprocessor.get_sorted_preprocessors()],
            all_models,
            "none",  # default option
            "None",  # default model
        )

    filtered_model_list = [
        model
        for model in all_models
        if model.lower() == "none"
        or (
            pattern in model.lower()
            or any(
                a in model.lower() for a in Preprocessor.tag_to_filters(control_type)
            )
        )
    ]
    assert len(filtered_model_list) > 0, "'None' model should always be available."
    if len(filtered_model_list) == 1:
        default_model = "None"
    else:
        default_model = filtered_model_list[1]
        for x in filtered_model_list:
            if "11" in x.split("[")[0]:
                default_model = x
                break

    return (
        [p.label for p in Preprocessor.get_filtered_preprocessors(control_type)],
        filtered_model_list,
        Preprocessor.get_default_preprocessor(control_type).label,
        default_model,
    )


def control_types(model_family):
    def format_control_type(
        filtered_preprocessor_list,
        filtered_model_list,
        default_option,
        default_model,
    ):
        return {
            "module_list": filtered_preprocessor_list,
            "model_list": filtered_model_list,
            "default_option": default_option,
            "default_model": default_model,
        }

    return {
        "control_types": {
            control_type: format_control_type(
                *select_control_type(control_type, model_family)
            )
            for control_type in Preprocessor.get_all_preprocessor_tags()
        }
    }


def pixel_perfect_resolution(
    image: np.ndarray,
    target_H: int,
    target_W: int,
    resize_mode: ResizeMode,
) -> int:
    """
    Calculate the estimated resolution for resizing an image while preserving aspect ratio.

    The function first calculates scaling factors for height and width of the image based on the target
    height and width. Then, based on the chosen resize mode, it either takes the smaller or the larger
    scaling factor to estimate the new resolution.

    If the resize mode is OUTER_FIT, the function uses the smaller scaling factor, ensuring the whole image
    fits within the target dimensions, potentially leaving some empty space.

    If the resize mode is not OUTER_FIT, the function uses the larger scaling factor, ensuring the target
    dimensions are fully filled, potentially cropping the image.

    After calculating the estimated resolution, the function prints some debugging information.

    Args:
        image (np.ndarray): A 3D numpy array representing an image. The dimensions represent [height, width, channels].
        target_H (int): The target height for the image.
        target_W (int): The target width for the image.
        resize_mode (ResizeMode): The mode for resizing.

    Returns:
        int: The estimated resolution after resizing.
    """
    raw_H, raw_W, _ = image.shape

    k0 = float(target_H) / float(raw_H)
    k1 = float(target_W) / float(raw_W)

    if resize_mode == ResizeMode.OUTER_FIT:
        estimation = min(k0, k1) * float(min(raw_H, raw_W))
    else:
        estimation = max(k0, k1) * float(min(raw_H, raw_W))

    logger.debug("Pixel Perfect Computation:")
    logger.debug(f"resize_mode = {resize_mode}")
    logger.debug(f"raw_H = {raw_H}")
    logger.debug(f"raw_W = {raw_W}")
    logger.debug(f"target_H = {target_H}")
    logger.debug(f"target_W = {target_W}")
    logger.debug(f"estimation = {estimation}")

    return int(np.round(estimation))


def get_pytorch_control(x: np.ndarray) -> torch.Tensor:
    # A very safe method to make sure that Apple/Mac works
    y = x

    # below is very boring but do not change these. If you change these Apple or Mac may fail.
    y = torch.from_numpy(y)
    y = y.float() / 255.0
    y = rearrange(y, "h w c -> 1 c h w")
    y = y.clone()
    y = y.to(get_available_device())
    y = y.clone()
    return y


def detectmap_proc(
    detected_map: np.ndarray, module: str, resize_mode: ResizeMode, h: int, w: int
):
    from ...thirdparty.controlnet.annotator.util import HWC3

    if "inpaint" in module:
        detected_map = detected_map.astype(np.float32)
    else:
        detected_map = HWC3(detected_map)

    def safe_numpy(x):
        # A very safe method to make sure that Apple/Mac works
        y = x

        # below is very boring but do not change these. If you change these Apple or Mac may fail.
        y = y.copy()
        y = np.ascontiguousarray(y)
        y = y.copy()
        return y

    def high_quality_resize(x, size):
        # Written by lvmin
        # Super high-quality control map up-scaling, considering binary, seg, and one-pixel edges

        inpaint_mask = None
        if x.ndim == 3 and x.shape[2] == 4:
            inpaint_mask = x[:, :, 3]
            x = x[:, :, 0:3]

        if x.shape[0] != size[1] or x.shape[1] != size[0]:
            new_size_is_smaller = (size[0] * size[1]) < (x.shape[0] * x.shape[1])
            new_size_is_bigger = (size[0] * size[1]) > (x.shape[0] * x.shape[1])
            unique_color_count = len(get_unique_axis0(x.reshape(-1, x.shape[2])))
            is_one_pixel_edge = False
            is_binary = False
            if unique_color_count == 2:
                is_binary = np.min(x) < 16 and np.max(x) > 240
                if is_binary:
                    xc = x
                    xc = cv2.erode(
                        xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1
                    )
                    xc = cv2.dilate(
                        xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1
                    )
                    one_pixel_edge_count = np.where(xc < x)[0].shape[0]
                    all_edge_count = np.where(x > 127)[0].shape[0]
                    is_one_pixel_edge = one_pixel_edge_count * 2 > all_edge_count

            if 2 < unique_color_count < 200:
                interpolation = cv2.INTER_NEAREST
            elif new_size_is_smaller:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = (
                    cv2.INTER_CUBIC
                )  # Must be CUBIC because we now use nms. NEVER CHANGE THIS

            y = cv2.resize(x, size, interpolation=interpolation)
            if inpaint_mask is not None:
                inpaint_mask = cv2.resize(
                    inpaint_mask, size, interpolation=interpolation
                )

            if is_binary:
                y = np.mean(y.astype(np.float32), axis=2).clip(0, 255).astype(np.uint8)
                if is_one_pixel_edge:
                    y = nake_nms(y)
                    _, y = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    y = lvmin_thin(y, prunings=new_size_is_bigger)
                else:
                    _, y = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                y = np.stack([y] * 3, axis=2)
        else:
            y = x

        if inpaint_mask is not None:
            inpaint_mask = (inpaint_mask > 127).astype(np.float32) * 255.0
            inpaint_mask = inpaint_mask[:, :, None].clip(0, 255).astype(np.uint8)
            y = np.concatenate([y, inpaint_mask], axis=2)

        return y

    if resize_mode == ResizeMode.RESIZE:
        detected_map = high_quality_resize(detected_map, (w, h))
        detected_map = safe_numpy(detected_map)
        return get_pytorch_control(detected_map), detected_map

    old_h, old_w, _ = detected_map.shape
    old_w = float(old_w)
    old_h = float(old_h)
    k0 = float(h) / old_h
    k1 = float(w) / old_w

    safeint = lambda x: int(np.round(x))

    if resize_mode == ResizeMode.OUTER_FIT:
        k = min(k0, k1)
        borders = np.concatenate(
            [
                detected_map[0, :, :],
                detected_map[-1, :, :],
                detected_map[:, 0, :],
                detected_map[:, -1, :],
            ],
            axis=0,
        )
        high_quality_border_color = np.median(borders, axis=0).astype(
            detected_map.dtype
        )
        if len(high_quality_border_color) == 4:
            # Inpaint hijack
            high_quality_border_color[3] = 255
        high_quality_background = np.tile(
            high_quality_border_color[None, None], [h, w, 1]
        )
        detected_map = high_quality_resize(
            detected_map, (safeint(old_w * k), safeint(old_h * k))
        )
        new_h, new_w, _ = detected_map.shape
        pad_h = max(0, (h - new_h) // 2)
        pad_w = max(0, (w - new_w) // 2)
        high_quality_background[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = (
            detected_map
        )
        detected_map = high_quality_background
        detected_map = safe_numpy(detected_map)
        return get_pytorch_control(detected_map), detected_map
    else:
        k = max(k0, k1)
        detected_map = high_quality_resize(
            detected_map, (safeint(old_w * k), safeint(old_h * k))
        )
        new_h, new_w, _ = detected_map.shape
        pad_h = max(0, (new_h - h) // 2)
        pad_w = max(0, (new_w - w) // 2)
        detected_map = detected_map[pad_h : pad_h + h, pad_w : pad_w + w]
        detected_map = safe_numpy(detected_map)
        return get_pytorch_control(detected_map), detected_map
