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

import logging
from typing import Any, Dict, Iterator, List, Tuple, Union, cast

WeMMInput = Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]
logger = logging.getLogger(__name__)


def is_wemm_model(model_name: str) -> bool:
    return model_name.startswith("WeMM-Embedding-")


def ensure_wemm_video_reader() -> None:
    """Install a PyAV fallback when qwen-vl-utils' torchvision reader is absent.

    torchvision 0.24+ removes ``io.read_video``. qwen-vl-utils 0.0.14 still
    falls back to that function when decord/torchcodec is unavailable or
    unusable. PyAV is already a required dependency of qwen-vl-utils.
    """
    from qwen_vl_utils import vision_process

    if getattr(vision_process, "_xinference_wemm_pyav_fallback", False):
        return
    if callable(getattr(vision_process.io, "read_video", None)):
        return

    def _read_video_pyav(
        item: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any], float]:
        import base64
        import io
        from urllib.parse import unquote_to_bytes

        import av
        import numpy as np
        import torch

        source = item["video"]
        if not isinstance(source, str):
            raise TypeError("The PyAV video fallback requires a string source.")
        if source.startswith("file://"):
            source = source[7:]
        elif source.startswith("data:"):
            header, payload = source.split(",", 1)
            raw = (
                base64.b64decode(payload)
                if ";base64" in header
                else unquote_to_bytes(payload)
            )
            source = io.BytesIO(raw)  # type: ignore[assignment]

        start = float(item.get("video_start", 0.0) or 0.0)
        end_value = item.get("video_end")
        end = float(end_value) if end_value is not None else None
        decoded_frames = []
        with av.open(source) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate or getattr(stream, "guessed_rate", None)
            video_fps = float(rate) if rate else 2.0
            for frame_index, frame in enumerate(container.decode(stream)):
                timestamp = (
                    float(frame.time)
                    if frame.time is not None
                    else frame_index / video_fps
                )
                if timestamp < start:
                    continue
                if end is not None and timestamp > end:
                    break
                decoded_frames.append(frame.to_ndarray(format="rgb24"))

        if not decoded_frames:
            raise ValueError("No video frames were decoded by the PyAV fallback.")
        total_frames = len(decoded_frames)
        nframes = vision_process.smart_nframes(
            item,
            total_frames=total_frames,
            video_fps=video_fps,
        )
        indices = torch.linspace(0, total_frames - 1, nframes).round().long()
        selected = np.stack([decoded_frames[index] for index in indices.tolist()])
        video = torch.from_numpy(selected).permute(0, 3, 1, 2)
        sample_fps = nframes / total_frames * video_fps
        metadata = {
            "fps": video_fps,
            "frames_indices": indices,
            "total_num_frames": total_frames,
            "video_backend": "pyav",
        }
        return video, metadata, sample_fps

    # qwen-vl-utils hard-codes "torchvision" as its final exception fallback,
    # so replace that unavailable slot rather than changing backend selection.
    vision_process.VIDEO_READER_BACKENDS["torchvision"] = _read_video_pyav
    vision_process._xinference_wemm_pyav_fallback = True
    logger.info("Using PyAV as the WeMM-Embedding video reader fallback.")


def _normalize_content_item(item: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(item)
    content_type = item.get("type")
    if content_type in {"audio", "audio_url", "input_audio"} or any(
        key in item for key in ("audio", "audio_url", "input_audio")
    ):
        raise ValueError("WeMM-Embedding does not support audio input.")
    if content_type in {"image", "image_url"} or "image" in item or "image_url" in item:
        value = item.pop("image", None)
        if value is None:
            value = item.pop("image_url", None)
        if isinstance(value, dict):
            value = value.get("url")
        if value is None:
            raise ValueError("WeMM-Embedding image input cannot be empty.")
        item["type"] = "image"
        item["image"] = value
        return item
    if content_type in {"video", "video_url"} or "video" in item or "video_url" in item:
        value = item.pop("video", None)
        if value is None:
            value = item.pop("video_url", None)
        if isinstance(value, dict):
            value = value.get("url")
        if value is None:
            raise ValueError("WeMM-Embedding video input cannot be empty.")
        item["type"] = "video"
        item["video"] = value
        return item
    if content_type == "text" or "text" in item:
        text = item.get("text", "")
        if not isinstance(text, str):
            raise ValueError("WeMM-Embedding text input must be a string.")
        item["type"] = "text"
        item["text"] = text
        return item
    raise ValueError(f"Unsupported WeMM-Embedding content item: {item!r}")


def _normalize_message(message: Dict[str, Any]) -> Dict[str, Any]:
    role = message.get("role", "user")
    content = message.get("content", "")
    if isinstance(content, str):
        return {"role": role, "content": content}
    if not isinstance(content, list):
        raise ValueError("WeMM-Embedding message content must be a string or list.")
    if not all(isinstance(item, dict) for item in content):
        raise ValueError("WeMM-Embedding message content items must be dictionaries.")
    return {
        "role": role,
        "content": [
            _normalize_content_item(cast(Dict[str, Any], item)) for item in content
        ],
    }


def normalize_wemm_messages(sample: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert one API embedding input into WeMM chat messages.

    A flat dictionary supports any ordered combination of ``text``, ``image``
    and ``video``. A role/content dictionary expresses an interleaved message;
    ``{"messages": [...]}`` expresses a multi-message conversation.
    """
    if isinstance(sample, str):
        return [{"role": "user", "content": [{"type": "text", "text": sample}]}]
    if not isinstance(sample, dict):
        raise ValueError("Each WeMM-Embedding input must be a string or dictionary.")
    if "messages" in sample:
        messages = sample["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("WeMM-Embedding `messages` must be a non-empty list.")
        if not all(isinstance(message, dict) for message in messages):
            raise ValueError("WeMM-Embedding `messages` items must be dictionaries.")
        return [_normalize_message(cast(Dict[str, Any], msg)) for msg in messages]
    if "role" in sample or "content" in sample:
        return [_normalize_message(sample)]

    content: List[Dict[str, Any]] = []
    for key, value in sample.items():
        if key in {"audio", "audio_url", "input_audio"}:
            raise ValueError("WeMM-Embedding does not support audio input.")
        if key not in {"text", "image", "image_url", "video", "video_url"}:
            continue
        values = value if key != "text" and isinstance(value, list) else [value]
        for one_value in values:
            content.append(_normalize_content_item({"type": key, key: one_value}))
    if not content:
        raise ValueError(
            "WeMM-Embedding input must contain text, image, video, content, or messages."
        )
    return [{"role": "user", "content": content}]


def normalize_wemm_inputs(inputs: WeMMInput) -> List[List[Dict[str, Any]]]:
    if isinstance(inputs, (str, dict)):
        samples: List[Union[str, Dict[str, Any]]] = [inputs]
    elif isinstance(inputs, list):
        samples = cast(List[Union[str, Dict[str, Any]]], inputs)
    else:
        raise ValueError("Unsupported input type for WeMM-Embedding.")
    return [normalize_wemm_messages(sample) for sample in samples]


def iter_wemm_media(
    messages: List[Dict[str, Any]],
) -> Iterator[Tuple[str, Any]]:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "image":
                yield "image", item.get("image")
            elif item.get("type") == "video":
                yield "video", item.get("video")
