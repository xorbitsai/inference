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
import shutil
from typing import Optional, Type

from ..common import PLEXAR_CACHE_DIR
import urllib.request

import os
from tqdm import tqdm


class ModelSpec:
    name: str
    n_parameters_in_billions: Optional[int] = None
    format: Optional[str] = None
    quantization: Optional[str] = None
    url: Optional[str] = None
    cls: Optional[Type] = None

    def __init__(
        self,
        name: str,
        n_parameters_in_billions: Optional[int],
        fmt: Optional[str] = None,
        quantization: Optional[str] = None,
        url: Optional[str] = None,
        cls: Optional[Type] = None
    ):
        self.name = name
        self.n_parameters_in_billions = n_parameters_in_billions
        self.format = fmt
        self.quantization = quantization
        self.url = url
        self.cls = cls

    def __str__(self):
        return f"{self.name}-{self.n_parameters_in_billions}b-{self.format}-{self.quantization}"

    def cache(self):
        assert self.url is not None

        save_path = os.path.join(
            PLEXAR_CACHE_DIR,
            str(self),
            "model.bin"
        )

        if os.path.exists(save_path):
            os.remove(save_path)

        with tqdm(
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc="Downloading",
            ncols=80
        ) as progress:
            urllib.request.urlretrieve(
                self.url,
                save_path,
                reporthook=lambda blocknum, blocksize, totalsize: progress.update(blocksize)
            )


MODEL_SPECS = []


def install():
    from .llm import install as llm_install

    llm_install()
