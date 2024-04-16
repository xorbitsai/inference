import io
from xinference.client import Client
from xinference.core.utils import json_dumps

client = Client("http://127.0.0.1:9997")

my_model = {
    "model_family": "stable_diffusion",
    "model_uid": "my_image",
    "model_name": "my_sd",
    "model_uri": "/new_data3/xprobe/cache/stable-diffusion-v1.5",  # your model path
    "controlnet": [{
        "model_family": "controlnet",
        "model_uid": "my_controlnet",
        "model_name": "my_controlnet",
        "model_uri": "/new_data3/xprobe/cache/mlsd",  # your controlnet path
    }]
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
image_path = "/Users/xprobe/python/src/Xinference/draft.jpg" # your image path
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
print("test result %s", r)
