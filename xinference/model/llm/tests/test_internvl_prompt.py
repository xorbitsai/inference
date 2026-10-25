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

# CI: linux

import base64
from io import BytesIO
from typing import cast

import pytest
from PIL import Image

from ....types import ChatCompletionMessage
from ..utils import ChatModelMixin


def test_internvl_system_message_once() -> None:
    prompt, images = ChatModelMixin.get_specific_prompt(
        "InternVL3",
        [
            {"role": "system", "content": "Follow these instructions."},
            {"role": "user", "content": "Hello."},
        ],
    )

    assert prompt.count("Follow these instructions.") == 1
    assert prompt == (
        "<s><|im_start|>system\nFollow these instructions.<|im_end|>\n"
        "<|im_start|>user\nHello.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert images == []


def test_internvl_preserves_all_text_parts_in_order() -> None:
    prompt, images = ChatModelMixin.get_specific_prompt(
        "InternVL3",
        [
            cast(
                ChatCompletionMessage,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Context: PART_A"},
                        {"type": "text", "text": "Question: PART_B"},
                    ],
                },
            )
        ],
    )

    assert prompt == (
        "<s><|im_start|>user\nContext: PART_A Question: PART_B<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert images == []


def test_internvl_list_system_content() -> None:
    prompt, images = ChatModelMixin.get_specific_prompt(
        "InternVL3",
        [
            cast(
                ChatCompletionMessage,
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Follow these"},
                        {"type": "text", "text": "instructions."},
                    ],
                },
            ),
            {"role": "user", "content": "Hello."},
        ],
    )

    assert prompt == (
        "<s><|im_start|>system\nFollow these instructions.<|im_end|>\n"
        "<|im_start|>user\nHello.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert images == []


def test_internvl_string_content_control() -> None:
    prompt, images = ChatModelMixin.get_specific_prompt(
        "InternVL3",
        [
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hi."},
            {"role": "user", "content": "How are you?"},
        ],
    )

    assert prompt == (
        "<s><|im_start|>user\nHello.<|im_end|>\n"
        "<|im_start|>assistant\nHi.<|im_end|>\n"
        "<|im_start|>user\nHow are you?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert images == []


@pytest.mark.parametrize("image_count", [1, 2])
def test_internvl_image_and_text_control(image_count: int) -> None:
    image = BytesIO()
    Image.new("RGB", (2, 3), color="red").save(image, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
    prompt, images = ChatModelMixin.get_specific_prompt(
        "InternVL3",
        [
            cast(
                ChatCompletionMessage,
                {
                    "role": "user",
                    "content": [
                        *[
                            {"type": "image_url", "image_url": {"url": image_url}}
                            for _ in range(image_count)
                        ],
                        {"type": "text", "text": "Describe."},
                    ],
                },
            )
        ],
    )

    placeholders = (
        "<image>\n" if image_count == 1 else "Image-1: <image>\n\nImage-2: <image>\n"
    )
    assert prompt == (
        f"<s><|im_start|>user\n{placeholders}\nDescribe.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert len(images) == image_count
    for decoded_image in images:
        assert decoded_image.mode == "RGB"
        assert decoded_image.size == (2, 3)
        assert decoded_image.getpixel((0, 0)) == (255, 0, 0)
