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

import os
from pathlib import Path

XINFERENCE_ENV_ENDPOINT = "XINFERENCE_ENDPOINT"
XINFERENCE_ENV_MODEL_SRC = "XINFERENCE_MODEL_SRC"
XINFERENCE_ENV_CSG_TOKEN = "XINFERENCE_CSG_TOKEN"
XINFERENCE_ENV_CSG_ENDPOINT = "XINFERENCE_CSG_ENDPOINT"
XINFERENCE_ENV_HOME_PATH = "XINFERENCE_HOME"
XINFERENCE_ENV_HEALTH_CHECK_FAILURE_THRESHOLD = (
    "XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD"
)
XINFERENCE_ENV_HEALTH_CHECK_INTERVAL = "XINFERENCE_HEALTH_CHECK_INTERVAL"
XINFERENCE_ENV_HEALTH_CHECK_TIMEOUT = "XINFERENCE_HEALTH_CHECK_TIMEOUT"
XINFERENCE_ENV_DISABLE_HEALTH_CHECK = "XINFERENCE_DISABLE_HEALTH_CHECK"
XINFERENCE_ENV_DISABLE_METRICS = "XINFERENCE_DISABLE_METRICS"
XINFERENCE_ENV_TRANSFORMERS_ENABLE_BATCHING = "XINFERENCE_TRANSFORMERS_ENABLE_BATCHING"


def get_xinference_home() -> str:
    home_path = os.environ.get(XINFERENCE_ENV_HOME_PATH)
    if home_path is None:
        home_path = str(Path.home() / ".xinference")
    else:
        # if user has already set `XINFERENCE_HOME` env, change huggingface and modelscope default download path
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(home_path, "huggingface")
        os.environ["MODELSCOPE_CACHE"] = os.path.join(home_path, "modelscope")
    # In multi-tenant mode,
    # gradio's temporary files are stored in their respective home directories,
    # to prevent insufficient permissions
    os.environ["GRADIO_TEMP_DIR"] = os.path.join(home_path, "tmp", "gradio")
    return home_path


XINFERENCE_HOME = get_xinference_home()
XINFERENCE_CACHE_DIR = os.path.join(XINFERENCE_HOME, "cache")
XINFERENCE_TENSORIZER_DIR = os.path.join(XINFERENCE_HOME, "tensorizer")
XINFERENCE_MODEL_DIR = os.path.join(XINFERENCE_HOME, "model")
XINFERENCE_LOG_DIR = os.path.join(XINFERENCE_HOME, "logs")
XINFERENCE_IMAGE_DIR = os.path.join(XINFERENCE_HOME, "image")
XINFERENCE_VIDEO_DIR = os.path.join(XINFERENCE_HOME, "video")
XINFERENCE_AUTH_DIR = os.path.join(XINFERENCE_HOME, "auth")
XINFERENCE_CSG_ENDPOINT = str(
    os.environ.get(XINFERENCE_ENV_CSG_ENDPOINT, "https://hub-stg.opencsg.com/")
)

XINFERENCE_DEFAULT_LOCAL_HOST = "127.0.0.1"
XINFERENCE_DEFAULT_DISTRIBUTED_HOST = "0.0.0.0"
XINFERENCE_DEFAULT_ENDPOINT_PORT = 9997
XINFERENCE_DEFAULT_LOG_FILE_NAME = "xinference.log"
XINFERENCE_LOG_MAX_BYTES = 100 * 1024 * 1024
XINFERENCE_LOG_BACKUP_COUNT = 30
XINFERENCE_LOG_ARG_MAX_LENGTH = 100
XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD = int(
    os.environ.get(XINFERENCE_ENV_HEALTH_CHECK_FAILURE_THRESHOLD, 5)
)
XINFERENCE_HEALTH_CHECK_INTERVAL = int(
    os.environ.get(XINFERENCE_ENV_HEALTH_CHECK_INTERVAL, 5)
)
XINFERENCE_HEALTH_CHECK_TIMEOUT = int(
    os.environ.get(XINFERENCE_ENV_HEALTH_CHECK_TIMEOUT, 10)
)
XINFERENCE_DISABLE_HEALTH_CHECK = bool(
    int(os.environ.get(XINFERENCE_ENV_DISABLE_HEALTH_CHECK, 0))
)
XINFERENCE_DISABLE_METRICS = bool(
    int(os.environ.get(XINFERENCE_ENV_DISABLE_METRICS, 0))
)
XINFERENCE_TRANSFORMERS_ENABLE_BATCHING = bool(
    int(os.environ.get(XINFERENCE_ENV_TRANSFORMERS_ENABLE_BATCHING, 0))
)
