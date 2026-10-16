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
import math
import os
import re
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

from huggingface_hub import hf_hub_download
from PIL import Image, ImageChops
from xoscar.utils import lazy_import

from ...constants import XINFERENCE_CACHE_DIR
from ...device_utils import get_available_device
from ._compat import LANCZOS
from .utils import create_binary_mask, resize_image

if TYPE_CHECKING:
    from .sdapi import SDAPIDiffusionModelMixin


logger = logging.getLogger(__name__)
adetailer_module = lazy_import("adetailer")
adetailer_to_download_model_to_repo_id = {
    "face_yolov8n.pt": "Bingsu/adetailer",
    "face_yolov8s.pt": "Bingsu/adetailer",
    "hand_yolov8n.pt": "Bingsu/adetailer",
    "person_yolov8n-seg.pt": "Bingsu/adetailer",
    "person_yolov8s-seg.pt": "Bingsu/adetailer",
    "yolov8x-worldv2.pt": "Bingsu/yolo-world-mirror",
}
mediapipe_models = {
    "mediapipe_face_full": "mediapipe_face_full",
    "mediapipe_face_short": "mediapipe_face_short",
    "mediapipe_face_mesh": "mediapipe_face_mesh",
    "mediapipe_face_mesh_eyes_only": "mediapipe_face_mesh_eyes_only",
}
available_adetailer_models = list(adetailer_to_download_model_to_repo_id) + list(
    mediapipe_models
)
adetailer_dir = os.path.join(XINFERENCE_CACHE_DIR, "adetailer")


def extract_adetailer_params(alwayson_scripts: dict):
    adetailers = alwayson_scripts.get("ADetailer", {}).get("args", [])
    if not adetailers or not adetailers[0]:
        return []
    result = []
    for arg in adetailers[1:]:
        if not isinstance(arg, dict):
            continue
        arg = dict(arg)
        if "ad_mask_k_largest" in arg:
            arg.setdefault("ad_mask_k", arg.pop("ad_mask_k_largest"))
        result.append(adetailer_module.ADetailerArgs.parse_obj(arg))
    return result


def _get_model(model_name: str) -> str:
    if model_name in adetailer_to_download_model_to_repo_id:
        model_path = os.path.join(adetailer_dir, model_name)
        if os.path.exists(model_path):
            return model_path
        repo_id = adetailer_to_download_model_to_repo_id[model_name]
        os.makedirs(adetailer_dir, exist_ok=True)
        hf_hub_download(repo_id, model_name, local_dir=adetailer_dir)
        return model_path
    if model_name in mediapipe_models:
        return mediapipe_models[model_name]
    msg = f"ADetailer Model {model_name!r} not found. Available models: {available_adetailer_models}"
    raise ValueError(msg)


def _get_prompts(ad_prompt: str, prompt: str) -> List[str]:
    prompts = re.split(r"\s*\[SEP\]\s*", ad_prompt)
    for i in range(len(prompts)):
        if not prompts[i]:
            prompts[i] = prompt
        elif "[PROMPT]" in prompts[i]:
            prompts[i] = prompts[i].replace("[PROMPT]", prompt)
    return prompts


def process_adetailer(
    model: "SDAPIDiffusionModelMixin",
    sd_type: str,
    image: Image,
    kwargs: dict,
    alwayson_scripts: dict,
    _cancel_event=None,
) -> Image:
    script_args = (alwayson_scripts.get("ADetailer") or {}).get("args") or []
    if not script_args or not script_args[0]:
        return image
    if sd_type == "img2img" and len(script_args) > 1 and script_args[1] is True:
        return image

    from adetailer import (
        ADetailerArgs,
        PredictOutput,
        mediapipe_predict,
        ultralytics_predict,
    )
    from adetailer.args import BBOX_SORTBY, InpaintBBoxMatchMode
    from adetailer.common import ensure_pil_image
    from adetailer.mask import (
        filter_by_ratio,
        filter_k_largest,
        has_intersection,
        mask_preprocess,
        sort_bboxes,
    )
    from adetailer.opts import dynamic_denoise_strength, optimal_crop_size

    kwargs = dict(kwargs, _cancel_event=_cancel_event)
    override_settings = kwargs.get("override_settings", {})
    if "size" in kwargs:
        width, height = tuple(int(i) for i in kwargs["size"].split("*", 1))
    elif "width" in kwargs:
        width, height = kwargs["width"], kwargs["height"]
    else:
        width, height = image.size

    def get_dynamic_denoise_strength(
        denoise_strength: float, bbox: Sequence[Any], image_size: tuple[int, int]
    ):
        denoise_power = override_settings.get("ad_dynamic_denoise_power", 0)
        if denoise_power == 0:
            return denoise_strength

        modified_strength = dynamic_denoise_strength(
            denoise_power=denoise_power,
            denoise_strength=denoise_strength,
            bbox=bbox,
            image_size=image_size,
        )

        logger.debug(
            f"ADetailer: dynamic denoising -- {denoise_strength:.2f} -> {modified_strength:.2f}"
        )

        return modified_strength

    def get_optimal_crop_image_size(
        inpaint_width: int, inpaint_height: int, bbox: Sequence[Any]
    ) -> Tuple[int, int]:
        calculate_optimal_crop = override_settings.get(
            "ad_match_inpaint_bbox_size", InpaintBBoxMatchMode.OFF.value
        )

        optimal_resolution: Optional[Tuple[int, int]] = None

        # Off
        if calculate_optimal_crop == InpaintBBoxMatchMode.OFF.value:
            return (inpaint_width, inpaint_height)

        # Strict (SDXL only)
        if calculate_optimal_crop == InpaintBBoxMatchMode.STRICT.value:
            if not getattr(model._model_spec, "model_base", None) == "SDXL":  # type: ignore
                msg = "[-] ADetailer: strict inpaint bounding box size matching is only available for SDXL. Use Free mode instead."
                logger.warning(msg)
                return (inpaint_width, inpaint_height)

            optimal_resolution = optimal_crop_size.sdxl(
                inpaint_width, inpaint_height, bbox
            )

        # Free
        elif calculate_optimal_crop == InpaintBBoxMatchMode.FREE.value:
            optimal_resolution = optimal_crop_size.free(
                inpaint_width, inpaint_height, bbox
            )

        if optimal_resolution is None:
            msg = "ADetailer: unsupported inpaint bounding box match mode. Original inpainting dimensions will be used."
            logger.debug(msg)
            return (inpaint_width, inpaint_height)

        # Only use optimal dimensions if they're different enough to current inpaint dimensions.
        if (
            abs(optimal_resolution[0] - inpaint_width) > inpaint_width * 0.1
            or abs(optimal_resolution[1] - inpaint_height) > inpaint_height * 0.1
        ):
            logger.debug(
                f"ADetailer: inpaint dimensions optimized -- {inpaint_width}x{inpaint_height} -> {optimal_resolution[0]}x{optimal_resolution[1]}"
            )

        return optimal_resolution

    def pred_preprocessing(pred: PredictOutput, args: ADetailerArgs):
        pred = filter_by_ratio(
            pred, low=args.ad_mask_min_ratio, high=args.ad_mask_max_ratio
        )
        pred = filter_k_largest(
            pred, k=getattr(args, "ad_mask_k", getattr(args, "ad_mask_k_largest", 0))
        )
        sortby = override_settings.get("ad_bbox_sortby", BBOX_SORTBY[0])
        sortby_idx = BBOX_SORTBY.index(sortby)
        pred = sort_bboxes(pred, sortby_idx)
        masks = mask_preprocess(
            pred.masks,
            kernel=args.ad_dilate_erode,
            x_offset=args.ad_x_offset,
            y_offset=args.ad_y_offset,
            merge_invert=args.ad_mask_merge_invert,
        )
        if (
            sd_type == "img2img"
            and kwargs.get("mask_image")
            and kwargs.get("inpaint_full_res_padding") is not None
        ):
            # in the adetailer plugin is
            # is_img2img_inpaint(p) and not is_inpaint_only_masked(p)
            # means is img2img inpaint and not inpaint only masked
            # inpaint only masked need to set inpaint_full_res to True
            # Xinf sdapi will pass inpaint_full_res_padding
            mask = kwargs.get("mask_image")
            mask = ensure_pil_image(mask, "L")
            if kwargs.get("inpainting_mask_invert", False):
                mask = ImageChops.invert(mask)
            mask = create_binary_mask(mask)

            mask = resize_image(kwargs.get("resize_mode"), mask, width, height)  # type: ignore

            if masks and mask.size != masks[0].size:
                mask = mask.resize(masks[0].size, resample=LANCZOS)
            masks = [m for m in masks if has_intersection(mask, m)]

        logger.debug(f"{len(masks)} masks")
        return masks

    prompt = kwargs.get("prompt")
    negative_prompt = kwargs.get("negative_prompt")
    adetailers: List[ADetailerArgs] = extract_adetailer_params(alwayson_scripts)

    if not adetailers or all(adetailer.need_skip() for adetailer in adetailers):
        return image

    progressor = kwargs["progressor"]
    progressor.split_stages(len(adetailers))

    for adetailer in adetailers:
        if _cancel_event is not None and _cancel_event.is_set():
            raise RuntimeError("Image generation cancelled")
        with progressor:
            if adetailer.need_skip():
                continue

            ad_prompts = _get_prompts(adetailer.ad_prompt, prompt)  # type: ignore
            ad_negative_prompts = _get_prompts(adetailer.ad_negative_prompt, negative_prompt)  # type: ignore

            is_mediapipe = adetailer.is_mediapipe()
            if is_mediapipe:
                pred = mediapipe_predict(
                    adetailer.ad_model, image, adetailer.ad_confidence
                )
            else:
                ad_model = _get_model(adetailer.ad_model)
                pred = ultralytics_predict(
                    ad_model,
                    image=image,
                    confidence=adetailer.ad_confidence,
                    device=get_available_device(),
                    classes=adetailer.ad_model_classes,
                )

            if pred.preview is None:
                logger.debug(f"ADetailer: nothing detected on image")
                continue

            masks = pred_preprocessing(pred, adetailer)

            steps = len(masks)

            if is_mediapipe:
                logger.debug(f"mediapipe: {steps} detected.")

            init_image = image
            if not steps:
                continue
            progressor.split_stages(steps)
            for j in range(steps):
                with progressor:
                    j_ad_prompt = ad_prompts[min(j, len(ad_prompts) - 1)]
                    j_ad_negative_prompt = ad_negative_prompts[
                        min(j, len(ad_negative_prompts) - 1)
                    ]
                    if re.match(r"^\s*\[SKIP\]\s*$", j_ad_prompt):
                        continue

                    new_kwargs = kwargs.copy()
                    new_kwargs["n"] = 1
                    new_kwargs["prompt"] = j_ad_prompt
                    new_kwargs["negative_prompt"] = j_ad_negative_prompt
                    new_kwargs["_return_images"] = True
                    new_kwargs["mask_image"] = masks[j]
                    new_kwargs["mask_blur"] = adetailer.ad_mask_blur
                    new_kwargs["image"] = ensure_pil_image(init_image, "RGB")
                    denoising_strength = adetailer.ad_denoising_strength
                    new_kwargs["strength"] = get_dynamic_denoise_strength(denoising_strength, (masks[j].getbbox() or (0, 0, image.width, image.height)), image.size)  # type: ignore
                    if adetailer.ad_inpaint_only_masked:
                        new_kwargs["padding_mask_crop"] = (
                            adetailer.ad_inpaint_only_masked_padding
                        )

                    # Don't override user-defined dimensions.
                    if not adetailer.ad_use_inpaint_width_height:
                        (
                            new_kwargs["width"],
                            new_kwargs["height"],
                        ) = get_optimal_crop_image_size(
                            width,
                            height,
                            (masks[j].getbbox() or (0, 0, image.width, image.height)),
                        )

                    else:
                        new_kwargs["width"] = adetailer.ad_inpaint_width
                        new_kwargs["height"] = adetailer.ad_inpaint_height
                    new_kwargs["size"] = (
                        f"{new_kwargs.pop('width')}*{new_kwargs.pop('height')}"
                    )
                    if new_kwargs["strength"] == 0:
                        continue
                    steps = (
                        adetailer.ad_steps
                        if adetailer.ad_use_steps
                        else new_kwargs.get("num_inference_steps", 20)
                    )
                    new_kwargs["num_inference_steps"] = math.ceil(
                        steps / new_kwargs["strength"]
                    )
                    if adetailer.ad_use_cfg_scale:
                        new_kwargs["guidance_scale"] = adetailer.ad_cfg_scale
                    if adetailer.ad_use_clip_skip:
                        new_kwargs["clip_skip"] = adetailer.ad_clip_skip
                    if adetailer.ad_use_sampler:
                        new_kwargs["sampler_name"] = adetailer.ad_sampler

                    init_image = model.inpainting(**new_kwargs)[0]  # type: ignore

            image = init_image

    return image
