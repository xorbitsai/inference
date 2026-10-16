import base64
import logging
import math
import os
from io import BytesIO
from tqdm.contrib.concurrent import thread_map

import numpy as np

import requests
import torch

from PIL import Image
from typing import Union, Tuple, List

def fetch_video(ele):
    raise ValueError(
        "video input is not supported by Ming Image inference; "
        "the public API accepts image and text input only"
    )

logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28
MAX_RATIO = 200

VideoInput = Union[
    List["Image.Image"],
    "np.ndarray",
    "torch.Tensor",
    List["np.ndarray"],
    List["torch.Tensor"],
    List[List["Image.Image"]],
    List[List["np.ndarray"]],
    List[List["torch.Tensor"]],
]


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def is_image(image_file):
    if isinstance(image_file, str) and (image_file.startswith("base64,") or image_file.lower().endswith(
            ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))):
        return True
    elif isinstance(image_file, Image.Image):
        return True
    else:
        return False

def is_video(video_file):
    if isinstance(video_file, str) and video_file.lower().endswith(
            ('.mp4', '.mkv', '.avi', '.wmv', '.iso', ".webm")):
        return True
    else:
        return False

def is_audio(audio_file):
    if isinstance(audio_file, str) and audio_file.lower().endswith(
            (".wav", ".mp3", ".aac", ".flac", ".alac", ".m4a", ".ogg", ".wma", ".aiff", ".amr", ".au")):
        return True
    else:
        return False

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image

def fetch_image_wo_resize(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")

    #image = image_obj.convert("RGB")

    return image_obj

def fetch_audio(ele: dict[str, str | torch.Tensor], return_tensor="pt") -> Tuple[Union[torch.Tensor, np.ndarray], int]:
    import torchaudio

    if "audio" in ele:
        audio = ele["audio"]
    else:
        audio = ele["audio_url"]

    if isinstance(audio, torch.Tensor):
        waveform = audio
        sample_rate: int = ele.get("sample_rate", 16000)
    elif audio.startswith("http://") or audio.startswith("https://"):
        audio_file = BytesIO(requests.get(audio, stream=True).content)
        waveform, sample_rate = torchaudio.load(audio_file)
    elif audio.startswith("file://"):
        waveform, sample_rate = torchaudio.load(audio[7:])
    else:
        waveform, sample_rate = torchaudio.load(audio)
    if return_tensor == "pt":
        return waveform, sample_rate
    else:
        return waveform.numpy(), sample_rate

def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                            "image" in ele
                            or "image_url" in ele
                            or "video" in ele
                            or "video_url" in ele
                            or "audio" in ele
                            or "audio_url" in ele
                            or ele["type"] in ["image", "image_url", "video", "video_url", "audio", "audio_url"]
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_reference_vision_info(
    conversations: list[dict] | list[list[dict]],
) -> list[Image.Image] | None:
    vision_infos = extract_vision_info(conversations)
    ## Read images
    image_inputs = []

    def inner_process_func(vision_info):
        if "image" in vision_info or "image_url" in vision_info:
            res_list = []
            if "image" in vision_info and isinstance(vision_info["image"], (tuple, list)):
                for i in range(len(vision_info["image"])):
                    res_list.append(fetch_image_wo_resize({"type": "image", "image": vision_info["image"][i]}))
            elif "image_url" in vision_info and vision_info["image_url"].get("url", None) is not None:
                vision_info["image_url"] = vision_info["image_url"].get("url")
                res_list.extend([fetch_image_wo_resize(vision_info)])
            else:
                res_list.extend([fetch_image_wo_resize(vision_info)])
            return {'image_inputs':res_list}
        else:
            return None

    vision_infos_reslist = thread_map(inner_process_func, vision_infos, disable=True)
    for res in vision_infos_reslist:
        if res is None:
            raise ValueError("image, image_url, video, video_url, audio or audio_url should in content.")
        elif 'image_inputs' in res:
            image_inputs.extend(res['image_inputs'])

    if len(image_inputs) > 1: # multi-image input keeps only the first image as the VAE reference
        image_inputs = [image_inputs[0]]

    if len(image_inputs) == 0:
        image_inputs = None

    return image_inputs


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, list[
    torch.Tensor | list[np.ndarray]] | None]:
    vision_infos = extract_vision_info(conversations)
    ## Read images, videos or audios
    image_inputs = []
    video_inputs = []
    audio_inputs = []

    def inner_process_func(vision_info):
        if "image" in vision_info or "image_url" in vision_info:
            res_list = []
            if "image" in vision_info and isinstance(vision_info["image"], (tuple, list)):
                for i in range(len(vision_info["image"])):
                    res_list.append(fetch_image({"type": "image", "image": vision_info["image"][i]}))
            elif "image_url" in vision_info and vision_info["image_url"].get("url", None) is not None:
                vision_info["image_url"] = vision_info["image_url"].get("url")
                res_list.extend([fetch_image(vision_info)])
            else:
                res_list.extend([fetch_image(vision_info)])
            return {'image_inputs':res_list}

        elif "video" in vision_info or "video_url" in vision_info:
            if "video_url" in vision_info and vision_info["video_url"].get("url", None) is not None:
                data_value = vision_info["video_url"].get("url")
            elif "video" in vision_info and not os.path.isdir(vision_info['video']):
                data_value = vision_info['video']
            else:
                data_value = [os.path.join(vision_info['video'], frame) for frame in sorted(os.listdir(vision_info['video']))]
            vision_info['video']=data_value
            return {"video_inputs": [fetch_video(vision_info)]}

        elif "audio" in vision_info or "audio_url" in vision_info:
            if "audio" in vision_info and isinstance(vision_info["audio"], (tuple, list)):
                return {"audio_inputs":[fetch_audio(info) for info in vision_info["audio"]]}
            elif "audio_url" in vision_info and vision_info["audio_url"].get("url", None) is not None:
                vision_info["audio_url"] = vision_info["audio_url"].get("url")
                return {"audio_inputs":[fetch_audio(vision_info)]}
            else:
                return {"audio_inputs":[fetch_audio(vision_info)]}
        else:
            return None

    vision_infos_reslist = thread_map(inner_process_func, vision_infos, disable=True)
    for res in vision_infos_reslist:
        if res is None:
            raise ValueError("image, image_url, video, video_url, audio or audio_url should in content.")
        elif 'image_inputs' in res:
            image_inputs.extend(res['image_inputs'])
        elif 'video_inputs' in res:
            video_inputs.extend(res['video_inputs'])
        elif 'audio_inputs' in res:
            audio_inputs.extend(res['audio_inputs'])

    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if len(audio_inputs) == 0:
        audio_inputs = None
    return image_inputs, video_inputs, audio_inputs


def get_closest_ratio(height: float, width: float, aspect_ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(aspect_ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return aspect_ratios[closest_ratio], float(closest_ratio)

def process_ratio(ori_h, ori_w, highres=512):
    ASPECT_RATIO_512 = {
        "0.25": [256, 1024], "0.26": [256, 992], "0.27": [256, 960], "0.28": [256, 928],
        "0.32": [288, 896], "0.33": [288, 864], "0.35": [288, 832], "0.4": [320, 800],
        "0.42": [320, 768], "0.48": [352, 736], "0.5": [352, 704], "0.52": [352, 672],
        "0.5455": [384, 704], "0.57": [384, 672], "0.6": [384, 640], "0.65": [416, 640],
        "0.68": [416, 608], "0.72": [416, 576], "0.78": [448, 576],
        "0.82": [448, 544], "0.88": [480, 544], "0.94": [480, 512],
        "1.0": [512, 512], "1.07": [512, 480], "1.13": [544, 480], "1.21": [544, 448],
        "1.29": [576, 448], "1.38": [576, 416],
        "1.46": [608, 416], "1.5385": [640, 416], "1.67": [640, 384], "1.75": [672, 384],
        "1.8333": [704, 384], "2.0": [704, 352], "2.09": [736, 352], "2.4": [768, 320],
        "2.5": [800, 320], "2.89": [832, 288], "3.0": [864, 288], "3.11": [896, 288],
        "3.62": [928, 256], "3.75": [960, 256], "3.88": [992, 256], "4.0": [1024, 256],
    }
    ASPECT_RATIO_1024 = {
        "0.25": [512, 2048], "0.26": [512, 1984], "0.27": [512, 1920], "0.28": [512, 1856],
        "0.32": [576, 1792], "0.33": [576, 1728], "0.35": [576, 1664], "0.4": [640, 1600],
        "0.42": [640, 1536], "0.48": [704, 1472], "0.5": [704, 1408], "0.52": [704, 1344],
        "0.5581": [768, 1376], "0.5625": [720, 1280], "0.5647": [768, 1360], "0.57": [768, 1344],
        "0.6": [768, 1280], "0.622": [816, 1312], "0.625": [800, 1280], "0.65": [832, 1280],
        "0.6582": [832, 1264], "0.6667": [832, 1248], "0.6709": [848, 1264], "0.68": [832, 1216],
        "0.7013": [864, 1232], "0.72": [832, 1152], "0.7467": [896, 1200], "0.75": [864, 1152],
        "0.7568": [896, 1184], "0.78": [896, 1152], "0.8": [896, 1120], "0.8056": [928, 1152],
        "0.82": [896, 1088], "0.88": [960, 1088], "0.94": [960, 1024], "0.9846": [1024, 1040],
        "1.0": [1024, 1024], "1.07": [1024, 960], "1.13": [1088, 960], "1.21": [1088, 896],
        "1.2414": [1152, 928], "1.25": [1120, 896], "1.2807": [1168, 912], "1.29": [1152, 896],
        "1.3333": [1152, 864], "1.3393": [1200, 896], "1.38": [1152, 832], "1.46": [1216, 832],
        "1.4906": [1264, 848], "1.5": [1248, 832], "1.6": [1280, 800], "1.67": [1280, 768],
        "1.75": [1344, 768], "1.7708": [1360, 768], "1.7778": [1280, 720], "2.0": [1408, 704],
        "2.09": [1472, 704], "2.4": [1536, 640], "2.5": [1600, 640], "2.89": [1664, 576],
        "3.0": [1728, 576], "3.11": [1792, 576], "3.62": [1856, 512], "3.75": [1920, 512],
        "3.88": [1984, 512], "4.0": [2048, 512],
    }

    ASPECT_RATIO_672 = {
        "0.28": [352, 1280], "0.32": [384, 1184], "0.38": [416, 1088], "0.44": [448, 1024],
        "0.52": [480, 928], "0.5636": [496, 880], "0.57": [512, 896], "0.65": [544, 832],
        "0.6667": [544, 816], "0.75": [576, 768], "0.8": [576, 720], "0.83": [608, 736],
        "0.91": [640, 704], "1.00": [672, 672], "1.10": [704, 640], "1.21": [736, 608],
        "1.25": [720, 576], "1.33": [768, 576], "1.39": [800, 576], "1.5": [816, 544],
        "1.53": [832, 544], "1.69": [864, 512], "1.75": [896, 512], "1.7742": [880, 496],
        "1.93": [928, 480], "2.00": [960, 480],
        "2.21": [992, 448], "2.29": [1024, 448], "2.54": [1056, 416], "2.62": [1088, 416],
        "2.69": [1120, 416], "3.00": [1152, 384], "3.08": [1184, 384], "3.17": [1216, 384],
        "3.55": [1248, 352], "3.64": [1280, 352],
    }

    ASPECT_RATIO_2048 = {
        "0.25": [1024, 4096], "0.26": [1024, 3968], "0.27": [1024, 3840], "0.28": [1024, 3712],
        "0.32": [1152, 3584], "0.33": [1152, 3456], "0.35": [1152, 3328], "0.4": [1280, 3200],
        "0.42": [1280, 3072], "0.48": [1408, 2944], "0.5": [1408, 2816], "0.52": [1408, 2688],
        "0.5625": [1440, 2560], "0.57": [1536, 2688], "0.6": [1536, 2560], "0.6667": [1664, 2496],
        "0.68": [1664, 2432], "0.72": [1664, 2304], "0.75": [1824, 2432], "0.78": [1792, 2304],
        "0.7917": [1824, 2304], "0.8": [1792, 2240], "0.82": [1792, 2176], "0.88": [1920, 2176],
        "0.94": [1920, 2048],
        "1.0": [2048, 2048], "1.07": [2048, 1920], "1.13": [2176, 1920], "1.21": [2176, 1792],
        "1.25": [2240, 1792], "1.2632": [2304, 1824], "1.29": [2304, 1792],
        "1.3333": [2432, 1824], "1.38": [2304, 1664],
        "1.46": [2432, 1664], "1.5": [2496, 1664], "1.67": [2560, 1536], "1.75": [2688, 1536],
        "1.7778": [2560, 1440], "2.0": [2816, 1408], "2.09": [2944, 1408], "2.4": [3072, 1280],
        "2.5": [3200, 1280], "2.89": [3328, 1152], "3.0": [3456, 1152], "3.11": [3584, 1152],
        "3.62": [3712, 1024], "3.75": [3840, 1024], "3.88": [3968, 1024], "4.0": [4096, 1024],
        "0.2941": [1120, 3808], "0.3043": [1120, 3680], "0.3679": [1248, 3392],
        "0.3846": [1280, 3328], "0.433": [1344, 3104], "0.4574": [1376, 3008],
        "0.4681": [1408, 3008], "0.5465": [1504, 2752], "0.6296": [1632, 2592],
        "0.6582": [1664, 2528], "0.7432": [1760, 2368], "0.8551": [1888, 2208],
        "0.9692": [2016, 2080], "1.0317": [2080, 2016], "1.1695": [2208, 1888],
        "1.3455": [2368, 1760], "1.5192": [2528, 1664], "1.5882": [2592, 1632],
        "1.8298": [2752, 1504], "1.913": [2816, 1472], "2.1364": [3008, 1408],
        "2.186": [3008, 1376], "2.3095": [3104, 1344], "2.6": [3328, 1280],
        "2.7179": [3392, 1248], "3.2857": [3680, 1120], "3.4": [3808, 1120],
    }

    aspect_ratio_dict = {
        512 : ASPECT_RATIO_512,
        672 : ASPECT_RATIO_672,
        1024 : ASPECT_RATIO_1024,
        2048 : ASPECT_RATIO_2048,
    }

    if highres is None or highres is False:
        highres = 512
    elif highres is True:
        highres = 1024

    aspect_ratio = aspect_ratio_dict[min([i for i in aspect_ratio_dict], key=lambda x: abs(x - highres))]

    closest_size, _ = get_closest_ratio(ori_h, ori_w, aspect_ratios=aspect_ratio)
    closest_size = list(map(lambda x: int(x), closest_size))
    if closest_size[0] / ori_h > closest_size[1] / ori_w:
        resize_size = closest_size[0], int(ori_w * closest_size[0] / ori_h)
    else:
        resize_size = int(ori_h * closest_size[1] / ori_w), closest_size[1]
    return closest_size, resize_size


def find_first_index_of_consecutive_ones(lst):
    """
    Given a list of 0s and 1s, return the index of the first 1 of each
    consecutive run of 1s.

    Args:
        lst (list): list of 0s and 1s

    Returns:
        list: indices of the first 1 of each consecutive run of 1s
    """
    result = []
    i = 0
    n = len(lst)

    while i < n:
        if lst[i] == 1:
            # find the start of a consecutive run of 1s
            result.append(i)
            # skip the remainder of the consecutive run of 1s
            while i < n and lst[i] == 1:
                i += 1
        else:
            i += 1

    return result

def merge_consecutive_ones(lst, n):
    """
    Given a list of 0s and 1s, merge every n consecutive 1s of each run
    (length >= 1) into a single 1. Each run must have a length divisible by n.
    The relative order of 0s and 1s is preserved.

    Args:
        lst: list of 0s and 1s
        n: positive integer merge unit size

    Returns:
        list: the merged list
    """
    assert isinstance(lst, list), "input must be a list"
    assert isinstance(n, int) and n > 0, "n must be a positive integer"

    # iterate over the list, extract runs of 1s, and verify each run length is divisible by n
    i = 0
    while i < len(lst):
        if lst[i] == 1:
            count = 0
            start = i
            # count the run of consecutive 1s
            while i < len(lst) and lst[i] == 1:
                count += 1
                i += 1
            # every run of 1s must be divisible by n
            assert count % n == 0, f"run of 1s at index {start} has length {count}, not divisible by n={n}"
        else:
            i += 1

    # build the new list by merging groups
    result = []
    i = 0
    while i < len(lst):
        if lst[i] == 0:
            result.append(0)
            i += 1
        else:
            # process a run of 1s
            count = 0
            while i < len(lst) and lst[i] == 1:
                count += 1
                i += 1
            # merge every n 1s into a single 1
            result.extend([1] * (count // n))

    return result

def get_default_image_gen_hw(image_gen_highres, image_gen_aspect_ratio):
    if image_gen_aspect_ratio is None:
        image_gen_aspect_ratio = 1.0

    closest_size, _ = process_ratio(ori_h=512, ori_w=int(512.0 * image_gen_aspect_ratio), highres=image_gen_highres)
    h, w = closest_size
    return h, w
