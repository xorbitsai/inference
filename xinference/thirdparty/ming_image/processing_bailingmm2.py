# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Processor class for BailingMM2."""

import sys
from typing import List, Union, Dict, Optional

import torch
import PIL
from PIL import Image

if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from .bailingmm_utils import process_vision_info, VideoInput, process_ratio, process_reference_vision_info, get_default_image_gen_hw
import torchvision
import math

DEFAULT_IMAGE_PATCH_TOKEN = "<imagePatch>"
DEFAULT_IM_START_TOKEN = "<image>"
DEFAULT_IM_END_TOKEN = "</image>"
DEFAULT_VID_START_TOKEN = "<video>"
DEFAULT_VID_END_TOKEN = "</video>"
DEFAULT_GEN_IMAGE_PATCH_TOKEN = "<gen_imagePatch>"
DEFAULT_GEN_IM_START_TOKEN = "<gen_image>"
DEFAULT_GEN_IM_END_TOKEN = "</gen_image>"
PLACEHOLDER_IMAGE_TOKEN_IN_TEXT = "<imageHere>"
DEFAULT_END_OF_CHUNK_TOKEN = "<end_of_chunk>"

DEFAULT_FRAME_PATCH_TOKEN = "<framePatch>"
DEFAULT_TEXT_TOKEN = '<text>'
DEFAULT_ASR_TOKEN = '<asr>'
DEFAULT_TTS_TOKEN = '<tts>'

USER_PREFIX = "<role>HUMAN</role>"
ASSISTANT_PREFIX = "<role>ASSISTANT</role>"

SYSTEM_PROMPT_LINGV2_FLASH_NOTHINK = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off"
SYSTEM_PROMPT_LINGV2_FLASH_THINK = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking on"

def check_single_quotes(s):
    count = s.count("'")
    if count % 2 != 0:
        return False

    positions = [i for i, char in enumerate(s) if char == "'"]
    for i in range(0, len(positions), 2):
        start = positions[i]
        end = positions[i+1]
        substr = s[start+1:end]
        chinese_count = 0
        for char in substr:
            if '\u4e00' <= char <= '\u9fff':
                chinese_count += 1
        other_count = len(substr) - chinese_count
        total = 3 * chinese_count + other_count
        if total >= 20:
            return False
    return True

def get_text_from_prompt(prompt):
    if "'" in prompt and check_single_quotes(prompt):
        prompt = prompt.replace("'", '"')

    patterns = [r'\"(.*?)\"', r'‘(.*?)’', r'“(.*?)”']


    import re
    texts = []
    patterns = [r'\"(.*?)\"', r'‘(.*?)’', r'“(.*?)”']
    for pattern in patterns:
        texts.extend(re.findall(pattern, prompt))

    if len(texts) == 1:
        assert texts[0] in prompt
        is_remove = False
        remove_keywords = ["remove", "delete", "erase"]
        text_start = min([j for j in [prompt.find(i) for i in ['"', '‘', '“']] if j >= 0])
        for kw in remove_keywords:
            if kw in prompt.lower():
                if prompt.lower().find(kw) < text_start:
                    is_remove = True
                    break

        if is_remove:
            texts = []

    text = " ".join(texts[-1:])
    if len(text) > 0:
        text = f'Text "{text}"'
        text += ". "

    return text

def crop_to_aspect_max(img: Image.Image, target_ratio: float) -> Image.Image:
    """
    Center-crop a PIL.Image to the largest area fitting the target aspect ratio
    (width/height) without resizing. Uses torchvision CenterCrop.

    Args:
        img: PIL.Image.Image input image
        target_ratio: float target aspect ratio (width/height), must be positive

    Returns:
        the center-cropped PIL.Image.Image
    """
    if not isinstance(img, Image.Image):
        raise TypeError("img must be a PIL.Image.Image")
    if not math.isfinite(target_ratio) or target_ratio <= 0:
        raise ValueError("target_ratio must be a positive, finite number")

    W, H = img.size
    if W <= 0 or H <= 0:
        raise ValueError("image size is invalid")

    orig_ratio = W / H

    if orig_ratio >= target_ratio:
        # image is wider than the target: use full height, crop left/right
        new_h = H
        new_w = int(math.floor(target_ratio * H))
        new_w = max(1, min(new_w, W))  # guard against extreme ratios producing invalid sizes
    else:
        # image is narrower than the target: use full width, crop top/bottom
        new_w = W
        new_h = int(math.floor(W / target_ratio))
        new_h = max(1, min(new_h, H))

    crop = torchvision.transforms.CenterCrop((new_h, new_w))  # size is (h, w)
    return crop(img)

def transform_reference_images(
    images,
    image_gen_aspect_ratio=None,
    image_gen_resolution=512,
    image_gen_input_channels=None,
):
    if image_gen_input_channels not in (3, 4):
        raise ValueError(
            "image_gen_input_channels must be explicitly set to 3 or 4 "
            "from the checkpoint capability contract"
        )
    image_mode = "RGB" if image_gen_input_channels == 3 else "RGBA"
    images = [image.convert(image_mode) for image in images]

    ref_pil = images[0]
    if image_gen_aspect_ratio is not None:
        ref_pil = crop_to_aspect_max(ref_pil, image_gen_aspect_ratio)

    ori_h = ref_pil.size[1]
    ori_w = ref_pil.size[0]
    closest_size, _ = process_ratio(ori_h=ori_h, ori_w=ori_w, highres=image_gen_resolution)

    ref_pils = [torchvision.transforms.functional.resize(i, closest_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR) for i in images]

    ref_tensor = torch.cat([
        ((torchvision.transforms.functional.to_tensor(i) - 0.5) * 2.0).unsqueeze(0)
        for i in ref_pils
    ], dim=0)

    return ref_tensor, ref_pil.size[1], ref_pil.size[0]

class BailingMM2ProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        "text_kwargs": {"padding": True, "padding_side": "right"},
        "image_kwargs": {},
        "video_kwargs": {},
    }

class BailingMM2Processor(ProcessorMixin):
    r"""
    Constructs a BailingMM2 processor which wraps a bailingmm2 image processor, bailing audio processor and a LLaMa tokenizer into a single processor.
    Args:
        image_processor ([`BailingMM2ImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        video_token (`str`, *optional*, defaults to `"<video>"`):
            Special token used to denote video location.
    """

    attributes = ["image_processor", "tokenizer"]
    optional_attributes = ["chat_template"]

    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    valid_kwargs = [
        "chat_template",
        "num_image_tokens",
        "image_token",
        "video_token",
    ]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token="<image>",
        video_token="<video>",
        **kwargs: Unpack[BailingMM2ProcessorKwargs],
    ):
        self.image_token = image_token
        self.video_token = video_token
        if chat_template is None:
            chat_template = tokenizer.chat_template

        self.gen_terminator = [tokenizer.eos_token_id]
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        videos: VideoInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        image_gen_highres = 512,
        image_gen_aspect_ratio = None,
        image_gen_ref_images: Union["PIL.Image.Image", list["PIL.Image.Image"]] = None,
        image_gen_input_channels = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        LlavaNextImageProcessor's [`~LlavaNextImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or torch Tensor.
                tensor. Both channels-first and channels-last formats are supported.
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or torch Tensor.
            audios (`Tuple[torch.Tensor, int]`, `List[Tuple[torch.Tensor, int]]`):
                The sequence or batch of audios to be prepared. Each audio can be a 1D torch Tensor (with its sampling rate).
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as a list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **image_num_patches** -- Patch number to be fed to a model. Returned when `images` is not `None`.
            - **image_sizes** -- Size of each image that will be used to unpad an image. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of a video input to be fed to a model. Returned when `videos` is not `None`.
            - **pixel_values_audios** -- Pixel values of an audio input to be fed to a model. Returned when `audios` is not `None`.

        """
        output_kwargs = self._merge_kwargs(
            BailingMM2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        image_inputs = {}
        video_inputs = {}
        image_gen_inputs = {}

        text_in_text = [get_text_from_prompt(i) for i in text]


        default_image_gen_height, default_image_gen_width = get_default_image_gen_hw(image_gen_highres, image_gen_aspect_ratio)

        image_gen_inputs.update({
            "image_gen_text": text_in_text,
            "image_gen_highres": image_gen_highres,
            "image_gen_height": torch.LongTensor([default_image_gen_height] * len(text)),
            "image_gen_width": torch.LongTensor([default_image_gen_width] * len(text)),
        })

        if images is not None:
            image_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]

            text = self._expand_image_tokens(text, image_grid_thw)

            # image_gen_pixel_values_reference, image_gen_height, image_gen_width = None, 512, 512
            if image_gen_ref_images is not None:
                if isinstance(image_gen_ref_images, PIL.Image.Image):
                    image_gen_ref_images = [image_gen_ref_images]
                elif not isinstance(image_gen_ref_images, list) and not isinstance(image_gen_ref_images[0], PIL.Image.Image):
                    raise ValueError("Invalid input image_gen_ref_images. Please provide a PIL.Image.Image, or a list of PIL.Image.Image")

                assert len(image_gen_ref_images) == len(text) # same batch_size

                image_gen_pixel_values_reference, image_gen_height_list, image_gen_width_list = transform_reference_images(
                    image_gen_ref_images,
                    image_gen_aspect_ratio,
                    image_gen_highres,
                    image_gen_input_channels,
                )

                image_gen_inputs.update({
                    "image_gen_pixel_values_reference": image_gen_pixel_values_reference,
                    "image_gen_height": torch.LongTensor([image_gen_height_list] * len(text)),
                    "image_gen_width": torch.LongTensor([image_gen_width_list] * len(text)),
                    #"image_gen_height": torch.LongTensor([ori_h]),
                    #"image_gen_width": torch.LongTensor([ori_w]),
                })

        if videos is not None:
            video_inputs = self.image_processor(images=None, videos=videos, do_resize=False, **output_kwargs["videos_kwargs"])
            video_grid_thw = video_inputs["video_grid_thw"]
            text = self._expand_video_tokens(text, video_grid_thw)

        # Padding side can be in TextKwargs but is not accepted by the tokenizer
        _ = output_kwargs["text_kwargs"].pop("padding_side", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs, **video_inputs, **image_gen_inputs})

    def apply_system_template(self, sys_prompt_exp=None, use_cot_system_prompt=False):
        if use_cot_system_prompt:
            sys_prompt = SYSTEM_PROMPT_LINGV2_FLASH_THINK
        else:
            sys_prompt = SYSTEM_PROMPT_LINGV2_FLASH_NOTHINK
        if sys_prompt_exp is not None:
            sys_prompt = sys_prompt.replace("你是一个友好的AI助手。", sys_prompt_exp)

        return sys_prompt

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]]],
        sys_prompt_exp: Optional[str] = None,
        use_cot_system_prompt: Optional[bool] = False,
        **kwargs,
    ) -> str:
        """
        Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
        conversations to turn them into a single tokenizable string.

        Args:
            conversation (`List[Dict, str, str]`):
                The conversation to format.
            sys_prompt_exp (`Optional[str]`, *optional*):
                The system prompt. If not provided, the processor's sysyetm template is used.
            **kwargs:
                Additional keyword arguments
        """
        text = ""
        sys_prompt = self.apply_system_template(sys_prompt_exp, use_cot_system_prompt)
        text = sys_prompt + self.tokenizer.eos_token

        for idx, message in enumerate(conversation):
            assert message["role"] in ["HUMAN", "ASSISTANT"]
            if idx == len(conversation) - 1:
                assert message["role"] == "HUMAN"

            if message["role"] == "HUMAN":
                text += USER_PREFIX
            elif message["role"] == "ASSISTANT":
                text += ASSISTANT_PREFIX

            image_counts = str(message["content"]).count("<image>")
            video_counts = str(message["content"]).count("<video>")

            for content in message["content"]:
                if content["type"] == "image":
                    num_images = 1 if isinstance(content["image"], (str, Image.Image)) else len(content["image"])
                    if image_counts < num_images:
                        image_placeholder = "<IMAGE>\n" * (num_images - image_counts)
                        text += image_placeholder.rstrip("\n")
                # only one video supported now
                elif content["type"] == "video":
                    assert video_counts <= 1, "Video count must be at most 1!"
                    if video_counts == 0:
                        text += "<VIDEO>"
                elif content["type"] == "audio":
                    raise ValueError("audio input is not supported by Ming Image inference")
                elif content["type"] == "text":
                    text += content['text']
            text += self.tokenizer.eos_token
        text += ASSISTANT_PREFIX

        return text

    def process_vision_info(
        self,
        conversations,
    ):
        return process_vision_info(conversations)

    def process_reference_vision_info(
        self,
        conversations,
    ):
        return process_reference_vision_info(conversations)

    def _expand_image_tokens(
        self,
        text: List[TextInput],
        image_grid_thw: Union[List[int], int],
        special_token: str = "<IMAGE>",
    ):
        prompt_strings = []
        image_index = 0
        num_query_token = torch.prod(image_grid_thw, dim=1) // 4
        for sample in text:
            num_images = sample.count(special_token)
            if num_images > 0:
                for i in range(image_index, num_images + image_index):
                    img_text = DEFAULT_IM_START_TOKEN + num_query_token[i] * DEFAULT_IMAGE_PATCH_TOKEN + DEFAULT_IM_END_TOKEN + "\n"
                    sample = sample.replace(special_token, img_text, 1)
            image_index += num_images
            prompt_strings.append(sample)
        text = [sample for sample in prompt_strings]
        return text

    def _expand_video_tokens(
        self,
        text: List[TextInput],
        video_grid_thw: Union[List[int], int],
        special_token: str = "<VIDEO>",
    ):
        prompt_strings = []
        video_index = 0
        num_query_token = torch.prod(video_grid_thw, dim=1) // 4
        for sample in text:
            num_videos = sample.count(special_token)
            if num_videos > 0:
                for i in range(video_index, num_videos + video_index):
                    video_text = num_query_token[i] * DEFAULT_FRAME_PATCH_TOKEN
                    video_text = DEFAULT_VID_START_TOKEN + video_text + DEFAULT_VID_END_TOKEN + "\n"
                    sample = sample.replace(special_token, video_text, 1)
            video_index += num_videos
            prompt_strings.append(sample)
        text = [sample for sample in prompt_strings]
        return text

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names + image_processor_input_names))


def load_bailingmm2_processor(data_directory):
    """Build the processor from data files with this repository's classes.

    Checkpoint component directories (``mllm/``) carry only data files
    (``preprocessor_config.json``, ``tokenizer_config.json``,
    ``special_tokens_map.json``, ``tokenizer.json``). Loading them through
    ``AutoProcessor`` with ``trust_remote_code=True`` fails because
    Transformers requires the Python implementation inside the loaded
    directory, while the implementation intentionally lives only in this
    repository. Construct the components explicitly instead.
    """
    from .image_processing_bailingmm2 import BailingMM2ImageProcessor
    from .tokenization_bailing import BailingTokenizer

    data_directory = str(data_directory)
    tokenizer = BailingTokenizer.from_pretrained(data_directory)
    image_processor = BailingMM2ImageProcessor.from_pretrained(data_directory)
    return BailingMM2Processor(image_processor=image_processor, tokenizer=tokenizer)
