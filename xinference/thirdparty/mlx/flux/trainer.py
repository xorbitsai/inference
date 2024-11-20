import mlx.core as mx
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

from .datasets import Dataset
from .flux import FluxPipeline


class Trainer:

    def __init__(self, flux: FluxPipeline, dataset: Dataset, args):
        self.flux = flux
        self.dataset = dataset
        self.args = args
        self.latents = []
        self.t5_features = []
        self.clip_features = []

    def _random_crop_resize(self, img):
        resolution = self.args.resolution
        width, height = img.size

        a, b, c, d = mx.random.uniform(shape=(4,), stream=mx.cpu).tolist()

        # Random crop the input image between 0.8 to 1.0 of its original dimensions
        crop_size = (
            max((0.8 + 0.2 * a) * width, resolution[0]),
            max((0.8 + 0.2 * b) * height, resolution[1]),
        )
        pan = (width - crop_size[0], height - crop_size[1])
        img = img.crop(
            (
                pan[0] * c,
                pan[1] * d,
                crop_size[0] + pan[0] * c,
                crop_size[1] + pan[1] * d,
            )
        )

        # Fit the largest rectangle with the ratio of resolution in the image
        # rectangle.
        width, height = crop_size
        ratio = resolution[0] / resolution[1]
        r1 = (height * ratio, height)
        r2 = (width, width / ratio)
        r = r1 if r1[0] <= width else r2
        img = img.crop(
            (
                (width - r[0]) / 2,
                (height - r[1]) / 2,
                (width + r[0]) / 2,
                (height + r[1]) / 2,
            )
        )

        # Finally resize the image to resolution
        img = img.resize(resolution, Image.LANCZOS)

        return mx.array(np.array(img))

    def _encode_image(self, input_img: ImageFile.ImageFile, num_augmentations: int):
        for i in range(num_augmentations):
            img = self._random_crop_resize(input_img)
            img = (img[:, :, :3].astype(self.flux.dtype) / 255) * 2 - 1
            x_0 = self.flux.ae.encode(img[None])
            x_0 = x_0.astype(self.flux.dtype)
            mx.eval(x_0)
            self.latents.append(x_0)

    def _encode_prompt(self, prompt):
        t5_tok, clip_tok = self.flux.tokenize([prompt])
        t5_feat = self.flux.t5(t5_tok)
        clip_feat = self.flux.clip(clip_tok).pooled_output
        mx.eval(t5_feat, clip_feat)
        self.t5_features.append(t5_feat)
        self.clip_features.append(clip_feat)

    def encode_dataset(self):
        """Encode the images & prompt in the latent space to prepare for training."""
        self.flux.ae.eval()
        for image, prompt in tqdm(self.dataset, desc="encode dataset"):
            self._encode_image(image, self.args.num_augmentations)
            self._encode_prompt(prompt)

    def iterate(self, batch_size):
        xs = mx.concatenate(self.latents)
        t5 = mx.concatenate(self.t5_features)
        clip = mx.concatenate(self.clip_features)
        mx.eval(xs, t5, clip)
        n_aug = self.args.num_augmentations
        while True:
            x_indices = mx.random.permutation(len(self.latents))
            c_indices = x_indices // n_aug
            for i in range(0, len(self.latents), batch_size):
                x_i = x_indices[i : i + batch_size]
                c_i = c_indices[i : i + batch_size]
                yield xs[x_i], t5[c_i], clip[c_i]
