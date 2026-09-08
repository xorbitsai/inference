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
import base64
import ipaddress
import logging
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps, PngImagePlugin

from ._compat import LANCZOS

logger = logging.getLogger(__name__)
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from numbers import Integral
from typing import TYPE_CHECKING, Any, List, Optional

from ...constants import XINFERENCE_IMAGE_DIR
from ...types import Image, ImageList

if TYPE_CHECKING:
    from .core import ImageModelFamilyV2


MAX_IMAGE_SEED = 2**31 - 1


def resolve_image_seed_list(seed: Any, n: int) -> Optional[List[int]]:
    """Resolve a per-image seed list, padding missing entries with random seeds."""
    if not isinstance(seed, (list, tuple)):
        return None
    if n < 1:
        raise ValueError("n must be greater than 0")
    if len(seed) > n:
        raise ValueError(f"Expected at most {n} image seeds, got {len(seed)}")

    values = list(seed) + [-1] * (n - len(seed))
    random_source = random.SystemRandom()
    resolved: List[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("Image seeds must be integers")
        value = int(value)
        if value == -1:
            value = random_source.randrange(MAX_IMAGE_SEED + 1)
        elif value < 0 or value > MAX_IMAGE_SEED:
            raise ValueError(
                f"Image seeds must be -1 or between 0 and {MAX_IMAGE_SEED}"
            )
        resolved.append(value)
    return resolved


def get_model_version(
    image_model: "ImageModelFamilyV2", controlnet: Optional["ImageModelFamilyV2"]
) -> str:
    return (
        image_model.model_name
        if controlnet is None
        else f"{image_model.model_name}--{controlnet.model_name}"
    )


def _flatten_images(images):
    if images and isinstance(images[0], (list, tuple)):
        flat_images = []
        for group in images:
            if isinstance(group, (list, tuple)):
                flat_images.extend(group)
            else:
                flat_images.append(group)
        return flat_images
    return images


def _needs_png(image) -> bool:
    if image.mode in ("RGBA", "LA"):
        return True
    if image.mode == "P" and "transparency" in image.info:
        return True
    return False


def handle_image_result(response_format: str, images) -> ImageList:
    images = _flatten_images(images)
    if response_format == "url":
        os.makedirs(XINFERENCE_IMAGE_DIR, exist_ok=True)
        image_list = []
        with ThreadPoolExecutor() as executor:
            for img in images:
                use_png = _needs_png(img)
                suffix = ".png" if use_png else ".jpg"
                path = os.path.join(XINFERENCE_IMAGE_DIR, uuid.uuid4().hex + suffix)
                image_list.append(Image(url=path, b64_json=None))
                fmt = "png" if use_png else "jpeg"
                executor.submit(img.save, path, fmt)
        return ImageList(created=int(time.time()), data=image_list)
    elif response_format == "b64_json":

        def _gen_base64_image(_img):
            buffered = BytesIO()
            fmt = "png" if _needs_png(_img) else "jpeg"
            _img.save(buffered, format=fmt)
            return base64.b64encode(buffered.getvalue()).decode()

        with ThreadPoolExecutor() as executor:
            results = list(map(partial(executor.submit, _gen_base64_image), images))  # type: ignore
            image_list = [Image(url=None, b64_json=s.result()) for s in results]  # type: ignore
        return ImageList(created=int(time.time()), data=image_list)
    else:
        raise ValueError(f"Unsupported response format: {response_format}")


def get_fixed_seed(seed) -> int:
    if seed == "" or seed is None:
        seed = -1
    elif isinstance(seed, str):
        try:
            seed = int(seed)
        except Exception:
            seed = -1

    if seed == -1:
        return int(random.randrange(4294967294))

    return seed


def resize_image(
    resize_mode: int,
    image: PILImage.Image,
    width: int,
    height: int,
    upscaler_name: Optional[str] = None,
) -> PILImage.Image:
    """
    Resizes an image with the specified resize_mode, width, and height.

    Args:
        resize_mode: The mode to use when resizing the image.
            0: Resize the image to the specified width and height.
            1: Resize the image to fill the specified width and height,
               maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            2: Resize the image to fit within the specified width and height,
               maintaining the aspect ratio, and then center the image within the dimensions,
               filling empty with data from image.
        image: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
        upscaler_name: The name of the upscaler to use
    """

    def resize(im: PILImage.Image, w: int, h: int) -> PILImage.Image:
        if upscaler_name is None or upscaler_name == "None" or im.mode == "L":
            return im.resize((w, h), resample=LANCZOS)

        scale = max(w / im.width, h / im.height)

        if scale > 1.0:
            from .upscaler import HiResUpscaler, upscale

            im = upscale(HiResUpscaler(upscaler_name), [im], scale, (w, h))[0]

        if im.width != w or im.height != h:
            im = im.resize((w, h), resample=LANCZOS)

        return im

    if resize_mode == 0:
        res = resize(image, width, height)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = image.width / image.height

        src_w = width if ratio > src_ratio else image.width * height // image.height
        src_h = height if ratio <= src_ratio else image.height * width // image.width

        resized = resize(image, src_w, src_h)
        res = PILImage.new(image.mode, (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = image.width / image.height

        src_w = width if ratio < src_ratio else image.width * height // image.height
        src_h = height if ratio >= src_ratio else image.height * width // image.width

        resized = resize(image, src_w, src_h)
        res = PILImage.new(image.mode, (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            if fill_height > 0:
                res.paste(
                    resized.resize((width, fill_height), box=(0, 0, width, 0)),
                    box=(0, 0),
                )
                res.paste(
                    resized.resize(
                        (width, fill_height),
                        box=(0, resized.height, width, resized.height),
                    ),
                    box=(0, fill_height + src_h),
                )
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            if fill_width > 0:
                res.paste(
                    resized.resize((fill_width, height), box=(0, 0, 0, height)),
                    box=(0, 0),
                )
                res.paste(
                    resized.resize(
                        (fill_width, height),
                        box=(resized.width, 0, resized.width, height),
                    ),
                    box=(fill_width + src_w, 0),
                )

    return res


def fix_image(image: PILImage.Image):
    if image is None:
        return None

    try:
        image = ImageOps.exif_transpose(image)
        image = fix_png_transparency(image)
    except Exception:
        pass

    return image


def fix_png_transparency(image: PILImage.Image):
    if image.mode not in ("RGB", "P") or not isinstance(
        image.info.get("transparency"), bytes
    ):
        return image

    image = image.convert("RGBA")
    return image


def fix_read(fp, **kwargs) -> PILImage.Image:
    image = PILImage.open(fp, **kwargs)
    image = fix_image(image)

    return image


def _public_addresses(url):
    import socket

    parsed = urlparse(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Expected a public HTTP(S) URL without credentials")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    addresses = [
        item[4][0]
        for item in socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    ]
    if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
        raise ValueError("URL must resolve only to public addresses")
    return parsed, addresses


def verify_url(url):
    try:
        _public_addresses(url)
        return True
    except (ValueError, OSError):
        return False


@__import__("contextlib").contextmanager
def _open_public_url(url, headers=None):
    """Pin each connection to validated DNS results, including every redirect."""
    from urllib.parse import urljoin

    import urllib3

    initial_host = urlparse(url).hostname
    for _ in range(6):
        parsed, addresses = _public_addresses(url)
        options = dict(
            host=addresses[0],
            port=parsed.port or (443 if parsed.scheme == "https" else 80),
            timeout=urllib3.Timeout(connect=10, read=30),
        )
        if parsed.scheme == "https":
            pool = urllib3.HTTPSConnectionPool(
                **options,
                server_hostname=parsed.hostname,
                assert_hostname=parsed.hostname,
            )
        else:
            pool = urllib3.HTTPConnectionPool(**options)
        request_headers = {"Host": parsed.netloc}
        if parsed.hostname == initial_host:
            request_headers.update(headers or {})
        response = None
        try:
            response = pool.urlopen(
                "GET",
                parsed.path + ("?" + parsed.query if parsed.query else ""),
                headers=request_headers,
                redirect=False,
                retries=False,
                preload_content=False,
            )
            if response.status in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("Missing redirect location")
                target = urljoin(url, location)
                if parsed.scheme == "https" and urlparse(target).scheme != "https":
                    raise ValueError("HTTPS downloads cannot redirect to HTTP")
                url = target
                continue
            if response.status != 200:
                raise ValueError(f"Download failed with HTTP {response.status}")
            yield response, url
            return
        finally:
            if response is not None:
                response.close()
            pool.close()
    raise ValueError("Too many redirects")


def decode_base64_to_image(encoding: str) -> PILImage.Image:
    if encoding.startswith(("http://", "https://")):
        with _open_public_url(encoding) as (response, _):
            data = response.read(64 * 1024 * 1024 + 1)
            if len(data) > 64 * 1024 * 1024:
                raise ValueError("Image exceeds 64 MiB")
    else:
        if encoding.startswith("data:image/"):
            encoding = encoding.split(",", 1)[1]
        try:
            data = base64.b64decode(encoding, validate=True)
        except Exception as exc:
            raise ValueError("Invalid encoded image") from exc
    try:
        image = fix_read(BytesIO(data))
        image.load()
        return image
    except Exception as exc:
        raise ValueError("Invalid image data") from exc


def encode_pil_to_base64(
    image: PILImage.Image, samples_format: str = "png", jpeg_quality: int = 80
):
    # TODO: samples_format and jpeg_quality is originally defined in shared_opts
    # we need to make it configurable for users
    with BytesIO() as output_bytes:
        if isinstance(image, str):
            return image
        if samples_format.lower() == "png":
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            image.save(
                output_bytes,
                format="PNG",
                pnginfo=(metadata if use_metadata else None),
                quality=jpeg_quality,
            )

        elif samples_format.lower() in ("jpg", "jpeg", "webp"):
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            import piexif
            import piexif.helper

            parameters = image.info.get("parameters", None)
            exif_bytes = piexif.dump(
                {
                    "Exif": {
                        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(
                            parameters or "", encoding="unicode"
                        )
                    }
                }
            )
            if samples_format.lower() in ("jpg", "jpeg"):
                image.save(
                    output_bytes, format="JPEG", exif=exif_bytes, quality=jpeg_quality
                )
            else:
                image.save(
                    output_bytes, format="WEBP", exif=exif_bytes, quality=jpeg_quality
                )

        else:
            raise ValueError("Invalid image format")

        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data).decode()


def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    file_name: Optional[str] = None,
    hash_prefix: Optional[str] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file

        download_url_to_file(
            url, cached_file, progress=progress, hash_prefix=hash_prefix
        )
    return cached_file


def create_binary_mask(image: PILImage.Image, round: bool = True) -> PILImage.Image:
    if image.mode == "RGBA" and image.getextrema()[-1] != (255, 255):
        if round:
            image = (
                image.split()[-1].convert("L").point(lambda x: 255 if x > 128 else 0)
            )
        else:
            image = image.split()[-1].convert("L")
    else:
        image = image.convert("L")
    return image


CHUNK_SIZE = 1638400


def download_civitai_model(url: str, output_path: str, token: str = ""):
    """Download a CivitAI artifact atomically without forwarding its token to a CDN."""
    import tempfile
    from email.message import Message

    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "civitai.com":
        raise ValueError("Expected an HTTPS civitai.com URL")
    os.makedirs(output_path, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with _open_public_url(url, headers) as (response, final_url):
        disposition = (
            response.headers.get("Content-Disposition")
            or parse_qs(urlparse(final_url).query).get(
                "response-content-disposition", [""]
            )[0]
        )
        message = Message()
        message["Content-Disposition"] = disposition
        filename = message.get_filename() or os.path.basename(urlparse(final_url).path)
        filename = os.path.basename(unquote(filename).replace("\\", "/"))
        if filename in ("", ".", ".."):
            raise ValueError("Missing artifact filename")
        destination = os.path.join(output_path, filename)
        if os.path.isfile(destination):
            return destination
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output_path, delete=False) as stream:
                temporary = stream.name
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            os.replace(temporary, destination)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        return destination


def get_unique_axis0(data):
    arr = np.asanyarray(data)
    idxs = np.lexsort(arr.T)
    arr = arr[idxs]
    unique_idxs = np.empty(len(arr), dtype=np.bool_)
    unique_idxs[:1] = True
    unique_idxs[1:] = np.any(arr[:-1, :] != arr[1:, :], axis=-1)
    return arr[unique_idxs]


def make_valid_filename(value: str) -> str:
    import re

    value = re.sub(r"[^\w.\-]", "_", value)
    if value in ("", ".", ".."):
        raise ValueError("Invalid artifact name")
    return value
