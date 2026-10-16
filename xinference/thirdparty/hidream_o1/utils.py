import os
import math
import torch
from typing import Optional
from PIL import Image, ImageDraw
import json
from typing import Any, Dict, Iterable, List, Sequence, Tuple

MAX_BOX = 5
PREDEFINED_RESOLUTIONS = [
    (2048, 2048),
    (2304, 1728),
    (1728, 2304),
    (2560, 1440),
    (1440, 2560),
    (2496, 1664),
    (1664, 2496),
    (3104, 1312),
    (1312, 3104),
    (2304, 1792),
    (1792, 2304),
]
DEFAULT_COLORS = [
    (255, 0, 0),
    (0, 180, 0),
    (0, 0, 255),
    (204, 180, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
]

def load_layout_bboxes(layout_bboxes: str) -> Any:
    """Load layout boxes from either a JSON string or a JSON file path."""
    if os.path.exists(layout_bboxes):
        with open(layout_bboxes, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(layout_bboxes)

def _unwrap_boxes(data: Any) -> Any:
    if isinstance(data, dict):
        for key in ("layout_bboxes", "bboxes", "boxes", "bbox_list"):
            if key in data:
                return data[key]
    return data

def _as_bbox_and_text(item: Any) -> Tuple[Sequence[float], str]:
    if isinstance(item, dict):
        bbox = item.get("bbox") or item.get("box")
        text = str(item.get("text") or item.get("label") or "")
        if bbox is None:
            raise ValueError(f"Missing bbox in layout item: {item!r}")
        return bbox, text
    if isinstance(item, (list, tuple)) and len(item) == 4:
        return item, ""
    raise ValueError(f"Unsupported layout bbox item: {item!r}")


def _xxyy_relative_to_absolute_bbox(bbox: Sequence[float], width: int, height: int) -> List[int]:
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got: {bbox!r}")
    x1, x2, y1, y2 = [float(v) for v in bbox]

    # Inference layout input is xxyy relative coordinates: [x1, x2, y1, y2].
    # Values in [0, 1] are the intended format. Keep 0-100 support for convenience.
    max_abs = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if max_abs <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif max_abs <= 100.0:
        x1, x2 = x1 / 100.0 * width, x2 / 100.0 * width
        y1, y2 = y1 / 100.0 * height, y2 / 100.0 * height

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width - 1, int(round(x2))))
    y2 = max(0, min(height - 1, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after scaling/clamping: {[x1, y1, x2, y2]!r}")
    return [x1, y1, x2, y2]

def parse_layout_bboxes(layout_bboxes: Any, width: int, height: int) -> List[Dict[str, Any]]:
    """Convert xxyy relative layout boxes into the training-side bbox layout format."""
    raw_boxes = _unwrap_boxes(layout_bboxes)
    if not isinstance(raw_boxes, list):
        raise ValueError("layout_bboxes must be a list, or a dict containing one of: layout_bboxes/bboxes/boxes")

    parsed = []
    for idx, item in enumerate(raw_boxes):
        bbox, text = _as_bbox_and_text(item)
        parsed.append({
            "bbox": _xxyy_relative_to_absolute_bbox(bbox, width, height),
            "color": "",
            "text": text,
            "image": None,
            "_orig_idx": idx,
        })
    return parsed

def _bbox_area(item: Dict[str, Any]) -> int:
    x1, y1, x2, y2 = item["bbox"]
    return max(0, x2 - x1) * max(0, y2 - y1)

def get_render_params(image_width: int, image_height: int) -> Tuple[int, int]:
    edge = math.sqrt(image_width * image_height)
    max_font_size = int(edge * 0.07)
    max_bbox_line_width = int(edge * 0.05)
    return max_font_size, max_bbox_line_width

def draw_bbox_layout(
    bbox_list: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    max_bbox: int = MAX_BOX,
    max_bbox_line_width: int | None = None,
    bbox_line_gap: int | None = None,
    return_color: bool = False,
):
    """Draw a black layout image with colored boxes, matching the training-side layout style."""
    if max_bbox_line_width is None:
        _, max_bbox_line_width = get_render_params(image_width, image_height)
    if bbox_line_gap is None:
        bbox_line_gap = max(1, max_bbox_line_width // max_bbox)

    image = Image.new("RGB", (image_width, image_height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    color_list = [None] * len(bbox_list)
    sorted_bboxes = sorted(bbox_list, key=_bbox_area, reverse=True)[:max_bbox]

    for sorted_idx, item in enumerate(sorted_bboxes):
        color = DEFAULT_COLORS[sorted_idx % len(DEFAULT_COLORS)]
        orig_idx = int(item.get("_orig_idx", sorted_idx))
        if 0 <= orig_idx < len(color_list):
            color_list[orig_idx] = color
        line_width = max(max_bbox_line_width - sorted_idx * bbox_line_gap, 5)
        draw.rectangle([int(v) for v in item["bbox"]], outline=color, width=line_width)

    if return_color:
        return image, color_list
    return image

def add_outer_border_keep_size(pil: Image.Image, color: Iterable[int], width: int) -> Image.Image:
    """Draw a border inside the image without changing its size."""
    img = pil.convert("RGB").copy()
    color_tuple = tuple(int(c) for c in color)
    width = max(0, int(width))
    if width == 0:
        return img

    draw = ImageDraw.Draw(img)
    w, h = img.size
    for t in range(width):
        draw.rectangle([t, t, w - 1 - t, h - 1 - t], outline=color_tuple)
    return img

def create_layout_reference_images(
    ref_pils: Sequence[str],
    layout_bboxes: Any,
    image_width: int,
    image_height: int,
    ref_max_size: int | None = None,
    patch_size: int = 32,
) -> Tuple[List[str], str]:
    """Create bordered ref images plus one layout image; returns paths to pass as ref_images."""
    parsed_boxes = parse_layout_bboxes(layout_bboxes, image_width, image_height)
    layout_image, color_list = draw_bbox_layout(
        parsed_boxes,
        image_width=image_width,
        image_height=image_height,
        return_color=True,
    )

    output_refs: List[str] = []
    for idx, ref in enumerate(ref_pils):
        if ref_max_size is not None:
            ref = resize_pilimage(ref, ref_max_size, patch_size)
        color = color_list[idx] if idx < len(color_list) and color_list[idx] is not None else DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        line_width = int(math.sqrt(ref.width * ref.height) * 0.04)
        bordered = add_outer_border_keep_size(ref, color, line_width)
        output_refs.append(bordered)
    output_refs.append(layout_image)
    return output_refs


def find_closest_resolution(width, height):
    img_ratio = width / height
    best_res = None
    min_diff = float("inf")
    for w, h in PREDEFINED_RESOLUTIONS:
        ratio = w / h
        diff = abs(ratio - img_ratio)
        if diff < min_diff:
            min_diff = diff
            best_res = (w, h)
    return best_res

def resize_pilimage(pil_image, image_size, patch_size=16, resampler=Image.BICUBIC):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    m = patch_size
    width, height = pil_image.width, pil_image.height
    S_max = image_size * image_size
    scale = S_max / (width * height)
    scale = math.sqrt(scale)

    new_sizes = [
        (round(width * scale) // m * m, round(height * scale) // m * m),
        (round(width * scale) // m * m, math.floor(height * scale) // m * m),
        (math.floor(width * scale) // m * m, round(height * scale) // m * m),
        (math.floor(width * scale) // m * m, math.floor(height * scale) // m * m),
    ]
    new_sizes = sorted(new_sizes, key=lambda x: x[0] * x[1], reverse=True)

    for new_size in new_sizes:
        if new_size[0] * new_size[1] <= S_max:
            break

    s1 = width / new_size[0]
    s2 = height / new_size[1]
    if s1 < s2:
        pil_image = pil_image.resize([new_size[0], round(height / s1)], resample=resampler)
        top = (round(height / s1) - new_size[1]) // 2
        pil_image = pil_image.crop((0, top, new_size[0], top + new_size[1]))
    else:
        pil_image = pil_image.resize([round(width / s2), new_size[1]], resample=resampler)
        left = (round(width / s2) - new_size[0]) // 2
        pil_image = pil_image.crop((left, 0, left + new_size[0], new_size[1]))

    return pil_image

def calculate_dimensions(max_size, ratio):
    width = math.sqrt(max_size * max_size * ratio)
    height = width / ratio
    width = int(width / 32) * 32
    height = int(height / 32) * 32
    return width, height

def get_rope_index_fix_point(
        spatial_merge_size,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        skip_vision_start_token=None,
        fix_point=4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                text_len -= skip_vision_start_token[image_index - 1]
                text_len = max(0, text_len)

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

                if skip_vision_start_token[image_index - 1]:
                    if fix_point > 0:
                        fix_point = fix_point - st_idx
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + fix_point + st_idx)
                    fix_point = 0
                else:
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
        return position_ids, mrope_position_deltas
