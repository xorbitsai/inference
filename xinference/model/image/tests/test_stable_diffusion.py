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
import base64
import io
import logging
import os.path
import shutil
import tempfile
from io import BytesIO

import pytest
from PIL import Image

from ..core import ImageModelFamilyV1, cache
from ..stable_diffusion.core import DiffusionModel

TEST_MODEL_SPEC = ImageModelFamilyV1(
    model_family="stable_diffusion",
    model_name="small-stable-diffusion-v0",
    model_id="OFA-Sys/small-stable-diffusion-v0",
    model_revision="38e10e5e71e8fbf717a47a81e7543cd01c1a8140",
)

logger = logging.getLogger(__name__)


def test_model():
    model_path = None
    try:
        model_path = cache(TEST_MODEL_SPEC)
        model = DiffusionModel("mock", model_path)
        # input is a string
        input_text = "an apple"
        model.load()
        r = model.text_to_image(input_text, size="256*256")
        assert len(r["data"]) == 1
        assert os.path.exists(r["data"][0]["url"])
        r = model.text_to_image(input_text, size="256*256", response_format="b64_json")
        assert len(r["data"]) == 1
        b64_json = r["data"][0]["b64_json"]
        image_bytes = base64.b64decode(b64_json)
        img = Image.open(BytesIO(image_bytes))
        assert img.size == (256, 256)
    finally:
        if model_path is not None:
            shutil.rmtree(model_path)


@pytest.mark.skip(reason="Stable diffusion controlnet requires too many GRAM.")
def test_restful_api_for_image_with_canny_controlnet(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="my_controlnet",
        model_name="stable-diffusion-xl-base-1.0",
        model_type="image",
        controlnet="canny",
    )
    model = client.get_model(model_uid)

    import cv2
    import numpy as np
    from diffusers.utils import load_image
    from PIL import Image

    image = load_image(
        "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
    )
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
    negative_prompt = "low quality, bad quality, sketches"
    bio = io.BytesIO()
    image.save(bio, format="png")
    r = model.image_to_image(
        image=bio.getvalue(),
        prompt=prompt,
        negative_prompt=negative_prompt,
        controlnet_conditioning_scale=0.5,
        num_inference_steps=25,
    )
    logger.info("test result %s", r)


@pytest.mark.skip(reason="Stable diffusion controlnet requires too many GRAM.")
def test_restful_api_for_image_with_mlsd_controlnet(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="my_controlnet",
        model_name="stable-diffusion-v1.5",
        model_type="image",
        controlnet="mlsd",
    )
    model = client.get_model(model_uid)

    from controlnet_aux import MLSDdetector
    from diffusers.utils import load_image

    mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")

    # Replace the image path for your test.
    image_path = os.path.expanduser("~/draft.png")
    logger.info("Image path: %s", image_path)
    image = load_image(image_path)
    image = mlsd(image)
    prompt = (
        "a modern house, use glass window, best quality, 8K wallpaper,(realistic:1.3), "
        "photorealistic, photo realistic, hyperrealistic, orante, super detailed, "
        "intricate, dramatic, morning lighting, shadows, high dynamic range,wooden,blue sky"
    )
    negative_prompt = (
        "low quality, bad quality, sketches, signature, soft, blurry, drawing, "
        "sketch, poor quality, ugly, text, type, word, logo, pixelated, "
        "low resolution, saturated, high contrast, oversharpened"
    )
    bio = io.BytesIO()
    image.save(bio, format="png")
    r = model.image_to_image(
        image=bio.getvalue(),
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=20,
    )
    logger.info("test result %s", r)


@pytest.mark.parametrize("model_name", ["sd-turbo", "sdxl-turbo"])
def test_restful_api_for_sd_turbo(setup, model_name):
    if model_name == "sdxl-turbo":
        pytest.skip("sdxl-turbo cost too many resources.")

    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="my_controlnet",
        model_name=model_name,
        model_type="image",
    )
    model = client.get_model(model_uid)

    r = model.text_to_image(
        prompt="A cinematic shot of a baby raccoon wearing an intricate italian priest robe.",
        size="512*512",
        num_inference_steps=10,
    )
    logger.info("test result %s", r)
    from PIL import Image

    with open(r["data"][0]["url"], "rb") as f:
        img = Image.open(f)
        assert img.size == (512, 512)

    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    completion = client.images.generate(
        model=model_uid,
        prompt="A cinematic shot of a baby raccoon wearing an intricate italian priest robe.",
        size="512*512",
        response_format="b64_json",
    )
    img_bytes = base64.b64decode(completion.data[0].b64_json)
    img = Image.open(BytesIO(img_bytes))
    assert img.size == (512, 512)


@pytest.mark.skip(reason="Stable diffusion inpainting requires too many GRAM.")
def test_restful_api_for_sd_inpainting(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="my_inpainting",
        model_name="stable-diffusion-2-inpainting",
        model_type="image",
    )
    model = client.get_model(model_uid)

    from diffusers.utils import load_image

    # Replace the image path for your test.
    image_path = os.path.expanduser("~/raw.jpg")
    logger.info("Image path: %s", image_path)
    image = load_image(image_path)
    bio = io.BytesIO()
    image.save(bio, format="png")
    mask_image_path = os.path.expanduser("~/mask.jpg")
    logger.info("Mask image path: %s", mask_image_path)
    mask_image = load_image(mask_image_path)
    bio2 = io.BytesIO()
    mask_image.save(bio2, format="png")

    r = model.inpainting(
        prompt="desert, clear sky, white clouds",
        image=bio.getvalue(),
        mask_image=bio2.getvalue(),
        num_inference_steps=10,
    )
    logger.info("test result %s", r)
    from PIL import Image

    with open(r["data"][0]["url"], "rb") as f:
        img = Image.open(f)
        assert img.size == image.size


def test_get_cache_status():
    from ..core import get_cache_status

    model_path = None
    try:
        assert get_cache_status(TEST_MODEL_SPEC) is False
        model_path = cache(TEST_MODEL_SPEC)
        assert get_cache_status(TEST_MODEL_SPEC) is True
    finally:
        if model_path is not None:
            shutil.rmtree(model_path)


def test_register_custom_image():
    from ..custom import (
        CustomImageModelFamilyV1,
        get_user_defined_images,
        register_image,
        unregister_image,
    )

    tmp_dir = tempfile.mktemp()

    model_spec = CustomImageModelFamilyV1(
        model_family="stable_diffusion",
        model_name="my-custom-image",
        model_id="my-custom-image",
        model_uri=os.path.abspath(tmp_dir),
    )

    register_image(model_spec, persist=False)
    assert model_spec in get_user_defined_images()

    unregister_image(model_spec.model_name, raise_error=False)
    assert model_spec not in get_user_defined_images()


def test_persist_custom_image():
    from ....constants import XINFERENCE_MODEL_DIR
    from ..custom import (
        CustomImageModelFamilyV1,
        get_user_defined_images,
        register_image,
        unregister_image,
    )

    tmp_dir = tempfile.mktemp()

    model_spec = CustomImageModelFamilyV1(
        model_family="stable_diffusion",
        model_name="my-custom-image",
        model_id="my-custom-image",
        model_uri=os.path.abspath(tmp_dir),
    )

    register_image(model_spec, persist=True)
    assert model_spec in get_user_defined_images()
    assert f"{model_spec.model_name}.json" in os.listdir(
        os.path.join(XINFERENCE_MODEL_DIR, "image")
    )

    unregister_image(model_spec.model_name)
    assert model_spec not in get_user_defined_images()
    assert f"{model_spec.model_name}.json" not in os.listdir(
        os.path.join(XINFERENCE_MODEL_DIR, "image")
    )


def test_launch_custom_image(setup):
    endpoint, _ = setup
    from ....client import Client
    from ....constants import XINFERENCE_CACHE_DIR
    from ....core.utils import json_dumps

    client = Client(endpoint)

    model_path = os.path.join(XINFERENCE_CACHE_DIR, "sd-turbo")

    my_model = {
        "model_family": "stable_diffusion",
        "model_uid": "my_sd",
        "model_name": "my_sd",
        "model_uri": model_path,
    }

    client.register_model(
        model_type="image",
        model=json_dumps(my_model),
        persist=False,
    )

    model_uid = client.launch_model(
        model_uid="my_image",
        model_name="my_sd",
        model_type="image",
    )
    model = client.get_model(model_uid)

    r = model.text_to_image(
        prompt="A cinematic shot of a baby raccoon wearing an intricate italian priest robe.",
        size="512*512",
        num_inference_steps=10,
    )
    logger.info("test result %s", r)

    client.unregister_model(model_type="image", model_name=my_model["model_name"])

    from PIL import Image

    with open(r["data"][0]["url"], "rb") as f:
        img = Image.open(f)
        assert img.size == (512, 512)

    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    completion = client.images.generate(
        model=model_uid,
        prompt="A cinematic shot of a baby raccoon wearing an intricate italian priest robe.",
        size="512*512",
        response_format="b64_json",
    )
    img_bytes = base64.b64decode(completion.data[0].b64_json)
    img = Image.open(BytesIO(img_bytes))
    assert img.size == (512, 512)


@pytest.mark.skip(reason="Stable diffusion controlnet requires too many GRAM.")
def test_launch_custom_image_with_controlnet(setup):
    endpoint, _ = setup
    from ....client import Client
    from ....constants import XINFERENCE_CACHE_DIR
    from ....core.utils import json_dumps

    client = Client(endpoint)

    model_path = os.path.join(XINFERENCE_CACHE_DIR, "stable-diffusion-v1.5")
    controlnet_path = os.path.join(XINFERENCE_CACHE_DIR, "mlsd")

    my_controlnet = {
        "model_family": "controlnet",
        "model_uid": "my_controlnet",
        "model_name": "my_controlnet",
        "model_uri": controlnet_path,
    }

    my_model = {
        "model_family": "stable_diffusion",
        "model_uid": "my_sd",
        "model_name": "my_sd",
        "model_uri": model_path,
        "controlnet": [
            my_controlnet,
        ],
    }

    client.register_model(
        model_type="image",
        model=json_dumps(my_model),
        persist=False,
    )

    model_uid = client.launch_model(
        model_uid="my_image",
        model_name="my_sd",
        model_type="image",
        controlnet="my_controlnet",
    )
    model = client.get_model(model_uid)

    from controlnet_aux import MLSDdetector
    from diffusers.utils import load_image

    mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")

    # Replace the image path for your test.
    image_path = os.path.expanduser("~/draft.png")
    logger.info("Image path: %s", image_path)
    image = load_image(image_path)
    image = mlsd(image)
    prompt = (
        "a modern house, use glass window, best quality, 8K wallpaper,(realistic:1.3), "
        "photorealistic, photo realistic, hyperrealistic, orante, super detailed, "
        "intricate, dramatic, morning lighting, shadows, high dynamic range,wooden,blue sky"
    )
    negative_prompt = (
        "low quality, bad quality, sketches, signature, soft, blurry, drawing, "
        "sketch, poor quality, ugly, text, type, word, logo, pixelated, "
        "low resolution, saturated, high contrast, oversharpened"
    )
    bio = io.BytesIO()
    image.save(bio, format="png")
    r = model.image_to_image(
        image=bio.getvalue(),
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=20,
    )
    logger.info("test result %s", r)

    client.unregister_model(model_type="image", model_name=my_model["model_name"])
    client.unregister_model(model_type="image", model_name=my_controlnet["model_name"])
