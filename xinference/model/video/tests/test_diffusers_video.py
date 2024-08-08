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
import logging
import os.path
import shutil
from io import BytesIO

from PIL import Image

from ..core import cache
from ..diffusers import DiffUsersVideoModel
from .. import BUILTIN_VIDEO_MODELS


logger = logging.getLogger(__name__)


def test_model():
    test_model_spec = next(iter(BUILTIN_VIDEO_MODELS.values()))
    model_path = cache(test_model_spec)
    model = DiffUsersVideoModel("mock", model_path, test_model_spec)
    # input is a string
    input_text = "an apple"
    model.load()
    r = model.text_to_image(input_text)
