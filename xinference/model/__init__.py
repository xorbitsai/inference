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

import logging
import os
import urllib.request
import warnings
from typing import Callable, List, Optional, Type

import requests
from requests import RequestException
from tqdm import tqdm

from ..constants import XINFERENCE_CACHE_DIR

logger = logging.getLogger(__name__)


class ModelSpec:
    def __init__(
        self,
        model_name: str,
        model_format: str,
        model_size_in_billions: int,
        quantization: str,
        url: str,
    ):
        self.model_name = model_name
        self.model_format = model_format
        self.model_size_in_billions = model_size_in_billions
        self.quantization = quantization
        self.url = url

    def __str__(self):
        return (
            f"{self.model_name}-{self.model_format}-{self.model_size_in_billions}b"
            f"-{self.quantization}"
        )

    def match(
        self,
        model_name: str,
        model_format: Optional[str] = None,
        model_size_in_billions: Optional[int] = None,
        quantization: Optional[str] = None,
    ) -> bool:
        return (
            model_name == self.model_name
            and (
                model_size_in_billions is None
                or model_size_in_billions == self.model_size_in_billions
            )
            and (model_format is None or model_format == self.model_format)
            and (quantization is None or quantization == self.quantization)
        )


class ModelFamily:
    def __init__(
        self,
        model_name: str,
        model_format: str,
        model_sizes_in_billions: List[int],
        quantizations: List[str],
        url_generator: Callable[[int, str], str],
        url_rp_generator: Callable[[int, str], str],
        cls: Type,
    ):
        self.model_name = model_name
        self.model_sizes_in_billions = model_sizes_in_billions
        self.model_format = model_format
        self.quantizations = quantizations
        self.url_generator = url_generator
        self.url_rp_generator = url_rp_generator
        self.cls = cls

    def __str__(self):
        return f"{self.model_name}-{self.model_format}"

    def __iter__(self):
        model_specs = []
        for model_size in self.model_sizes_in_billions:
            for quantization in self.quantizations:
                model_specs.append(
                    ModelSpec(
                        model_name=self.model_name,
                        model_size_in_billions=model_size,
                        model_format=self.model_format,
                        quantization=quantization,
                        url=self.url_generator(model_size, quantization),
                    )
                )
        return iter(model_specs)

    def match(
        self,
        model_name: str,
        model_format: Optional[str] = None,
        model_size_in_billions: Optional[int] = None,
        quantization: Optional[str] = None,
    ) -> Optional[ModelSpec]:
        for model_spec in self:
            if model_spec.match(
                model_name=model_name,
                model_format=model_format,
                model_size_in_billions=model_size_in_billions,
                quantization=quantization,
            ):
                return model_spec
        return None

    def generate_cache_path(
        self,
        model_size_in_billions: Optional[int] = None,
        quantization: Optional[str] = None,
    ):
        full_name = f"{str(self)}-{model_size_in_billions}b-{quantization}"
        save_dir = os.path.join(XINFERENCE_CACHE_DIR, full_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "model.bin")
        meta_path = os.path.join(save_dir, "meta")
        return save_path, meta_path

    def cache(
        self,
        model_size_in_billions: Optional[int] = None,
        quantization: Optional[str] = None,
    ) -> str:
        # by default, choose the smallest size.
        model_size_in_billions = (
            model_size_in_billions or self.model_sizes_in_billions[0]
        )
        # by default, choose the most coarse-grained quantization.
        quantization = quantization or self.quantizations[0]

        url = self.url_generator(model_size_in_billions, quantization)
        rp_url = self.url_rp_generator(model_size_in_billions, quantization)

        try:
            rp_fetch = requests.get(rp_url)
        except RequestException as e:
            raise RequestException(f"Request failed: {str(e)}")

        res_content = rp_fetch.content
        splitted_res_content = str(res_content).split()
        digits = "".join([char for char in splitted_res_content[3] if char.isdigit()])
        expected_size = int(digits)

        if self.model_format == "pytorch":
            return url

        full_name = f"{str(self)}-{model_size_in_billions}b-{quantization}"
        save_path, meta_path = self.generate_cache_path(
            model_size_in_billions, quantization
        )
        if os.path.exists(meta_path) and os.path.exists(save_path):
            # TODO: verify the integrity.
            if os.path.getsize(save_path) == expected_size:
                return save_path
            else:
                warnings.warn(
                    "Model size doesn't match, try to update it...", RuntimeWarning
                )

        try:
            if os.path.exists(save_path):
                os.remove(save_path)
            with tqdm(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc=f"Downloading {full_name}",
            ) as progress:
                urllib.request.urlretrieve(
                    url,
                    save_path,
                    reporthook=lambda blocknum, blocksize, totalsize: progress.update(
                        blocksize
                    ),
                )
            # write a meta file to record if download finished
            with open(meta_path, "w") as f:
                f.write(full_name)
            # TODO: verify the integrity.
            if os.path.getsize(save_path) != expected_size:
                os.remove(save_path)
                raise RuntimeError(f"Failed to download {full_name} from {url}")
        except:
            if os.path.exists(save_path):
                os.remove(save_path)
            raise RuntimeError(f"Failed to download {full_name} from {url}")

        return save_path


MODEL_FAMILIES: List[ModelFamily] = []


def install():
    from .llm import install as llm_install

    llm_install()
