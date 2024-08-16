# Copyright 2022-2024 XProbe Inc.
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
import io
import logging
import os
import tempfile
import zipfile
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ....constants import XINFERENCE_TENSORIZER_DIR
from ....device_utils import get_available_device

logger = logging.getLogger(__name__)

__all__ = [
    "get_tensorizer_dir",
    "check_tensorizer_integrity",
    "load_from_tensorizer",
    "save_to_tensorizer",
    "_load_pretrained_from_tensorizer",
    "_load_model_from_tensorizer",
    "_tensorizer_serialize_model",
    "_tensorizer_serialize_pretrained",
    "_file_is_non_empty",
]


def _filter_kwargs(kwargs):
    kwargs["trust_remote_code"] = kwargs.get("trust_remote_code", True)
    return {
        k: v for k, v in kwargs.items() if k in ["code_revision", "trust_remote_code"]
    }


def _file_is_non_empty(
    path: str,
) -> bool:
    try:
        return os.stat(path).st_size > 0
    except FileNotFoundError:
        return False


def get_tensorizer_dir(model_path: str) -> str:
    model_dir = os.path.basename(model_path.rstrip("/"))
    return f"{XINFERENCE_TENSORIZER_DIR}/{model_dir}"


def check_tensorizer_integrity(
    model_path: str,
    components: Optional[List[str]] = None,
    model_prefix: Optional[str] = "model",
) -> bool:
    tensorizer_dir = get_tensorizer_dir(model_path)
    dir = tensorizer_dir.rstrip("/")
    tensors_uri: str = f"{dir}/{model_prefix}.tensors"
    # iterate over components and get their paths
    paths = [tensors_uri]
    if components is not None:
        for component in components:
            component_uri: str = f"{tensorizer_dir.rstrip('/')}/{component}.zip"
            paths.append(component_uri)
    return all(_file_is_non_empty(path) for path in paths)


def load_from_tensorizer(
    model_path: str,
    components: Optional[List[Tuple[str, Any, Dict[str, Any]]]] = None,
    model_class: Any = None,
    config_class: Any = None,
    model_prefix: Optional[str] = "model",
    **kwargs,
):
    kwargs = _filter_kwargs(kwargs)
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError:
        error_message = "Failed to import module 'transformers'"
        installation_guide = [
            "Please make sure 'transformers' is installed. ",
            "You can install it by `pip install transformers`\n",
        ]
        raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

    model_class = model_class or AutoModel
    config_class = config_class or AutoConfig

    tensorizer_dir = get_tensorizer_dir(model_path)
    logger.debug(f"Loading from tensorizer: {tensorizer_dir}")

    device = get_available_device()
    tensorizer_model = (
        _load_model_from_tensorizer(
            model_path,
            tensorizer_dir,
            model_class,
            config_class,
            model_prefix,
            device,
            **kwargs,
        )
        .to(device)
        .eval()
    )

    tensorizer_components = []

    if components is not None:
        for component, component_class, kwargs in components:
            deserialized_component = _load_pretrained_from_tensorizer(
                component_class, tensorizer_dir, component, **kwargs
            )
            tensorizer_components.append(deserialized_component)

    return tensorizer_model, *tensorizer_components


def _load_pretrained_from_tensorizer(
    component_class: Any,
    tensorizer_dir: str,
    prefix: str,
    **kwargs,
):
    logger.debug(f"Loading components from tensorizer: {component_class} {kwargs}")

    try:
        from tensorizer import stream_io
    except ImportError:
        error_message = "Failed to import module 'tensorizer'"
        installation_guide = [
            "Please make sure 'tensorizer' is installed.\n",
        ]
        raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

    _read_stream = partial(stream_io.open_stream, mode="rb")

    logger.debug(f"Loading pretrained from tensorizer: {tensorizer_dir}")
    load_path: str = f"{tensorizer_dir.rstrip('/')}/{prefix}.zip"
    logger.info(f"Loading {load_path}")
    with io.BytesIO() as downloaded:
        # Download to a BytesIO object first, because ZipFile doesn't play nice
        # with streams that don't fully support random access
        with _read_stream(load_path) as stream:
            downloaded.write(stream.read())
        downloaded.seek(0)
        with zipfile.ZipFile(
            downloaded, mode="r"
        ) as file, tempfile.TemporaryDirectory() as directory:
            file.extractall(path=directory)
            return component_class.from_pretrained(
                directory, cache_dir=None, local_files_only=True, **kwargs
            )


def _load_model_from_tensorizer(
    model_path: str,
    tensorizer_dir: str,
    model_class,
    config_class,
    model_prefix: Optional[str] = "model",
    device=None,
    dtype=None,
    **kwargs,
):
    logger.debug(f"Loading model from tensorizer: {tensorizer_dir} {kwargs}")

    # assert device is not None
    if device is None:
        raise ValueError("device must be specified")

    import time

    try:
        import torch
    except ImportError:
        error_message = "Failed to import module 'torch'"
        installation_guide = [
            "Please make sure 'torch' is installed.\n",
        ]
        raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

    try:
        from transformers import PretrainedConfig
    except ImportError:
        error_message = "Failed to import module 'transformers'"
        installation_guide = [
            "Please make sure 'transformers' is installed. ",
            "You can install it by `pip install transformers`\n",
        ]

        raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

    try:
        from tensorizer import TensorDeserializer, stream_io, utils
    except ImportError:
        error_message = "Failed to import module 'tensorizer'"
        installation_guide = [
            "Please make sure 'tensorizer' is installed.\n",
        ]
        raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

    if model_prefix is None:
        model_prefix = "model"

    dir: str = tensorizer_dir.rstrip("/")
    tensors_uri: str = f"{dir}/{model_prefix}.tensors"

    _read_stream = partial(stream_io.open_stream, mode="rb")

    if config_class is None:
        config_loader = model_class.load_config
    else:
        config_loader = config_class.from_pretrained
    try:
        config, _ = config_loader(model_path, return_unused_kwargs=True, **kwargs)
        if isinstance(config, PretrainedConfig):
            config.gradient_checkpointing = True
    except ValueError:
        config = config_loader(model_path, **kwargs)

    with utils.no_init_or_tensor():
        model_loader = getattr(model_class, "from_config", model_class)
        model = model_loader(config, **kwargs)

    is_cuda: bool = torch.device(device).type == "cuda"
    ram_usage = utils.get_mem_usage()
    logger.info(f"Loading {tensors_uri}, {ram_usage}")
    begin_load = time.perf_counter()

    with _read_stream(tensors_uri) as tensor_stream, TensorDeserializer(
        tensor_stream, device=device, dtype=dtype, plaid_mode=is_cuda
    ) as tensor_deserializer:
        tensor_deserializer.load_into_module(model)
        tensor_load_s = time.perf_counter() - begin_load
        bytes_read: int = tensor_deserializer.total_bytes_read

    rate_str = utils.convert_bytes(bytes_read / tensor_load_s)
    tensors_sz = utils.convert_bytes(bytes_read)
    logger.info(
        f"Model tensors loaded in {tensor_load_s:0.2f}s, read "
        f"{tensors_sz} @ {rate_str}/s, {utils.get_mem_usage()}"
    )

    return model


def save_to_tensorizer(
    model_path: str,
    model,
    components: Optional[List[Tuple[str, Any]]] = None,
    model_prefix: Optional[str] = "model",
    force: Optional[bool] = False,
    **kwargs,
):
    kwargs = _filter_kwargs(kwargs)
    _tensorizer_serialize_model(model_path, model, model_prefix, force, **kwargs)

    if components is not None:
        for component_prefix, component in components:
            _tensorizer_serialize_pretrained(model_path, component, component_prefix)


def _tensorizer_serialize_model(
    model_path: str,
    model,
    model_prefix: Optional[str] = "model",
    force: Optional[bool] = False,
    **kwargs,
):
    try:
        from tensorizer import TensorSerializer, stream_io
    except ImportError:
        error_message = "Failed to import module 'tensorizer'"
        installation_guide = [
            "Please make sure 'tensorizer' is installed.\n",
        ]
        raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

    tensorizer_dir = get_tensorizer_dir(model_path)
    tensor_path: str = f"{tensorizer_dir}/{model_prefix}.tensors"

    _write_stream = partial(stream_io.open_stream, mode="wb+")

    if os.path.exists(tensor_path):
        logger.info(f"Cache {tensor_path} exists, skip tensorizer serialize model")
        return

    logger.info(f"Writing tensors to {tensor_path}")
    with _write_stream(tensor_path) as f:
        serializer = TensorSerializer(f)
        serializer.write_module(model, include_non_persistent_buffers=False)
        serializer.close()

    logger.info(f"Tensorizer serialize model done: {tensor_path}")


def _tensorizer_serialize_pretrained(
    model_path: str, component, prefix: str = "pretrained"
):
    try:
        from tensorizer import stream_io
    except ImportError:
        error_message = "Failed to import module 'tensorizer'"
        installation_guide = [
            "Please make sure 'tensorizer' is installed.\n",
        ]
        raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

    tensorizer_dir = get_tensorizer_dir(model_path)
    save_path: str = f"{tensorizer_dir.rstrip('/')}/{prefix}.zip"

    if os.path.exists(save_path):
        logger.info(f"Cache {save_path} exists, skip tensorizer serialize pretrained")
        return

    logger.info(f"Writing component to {save_path}")
    _write_stream = partial(stream_io.open_stream, mode="wb+")

    with _write_stream(save_path) as stream, zipfile.ZipFile(
        stream, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=5
    ) as file, tempfile.TemporaryDirectory() as directory:
        if hasattr(component, "save_pretrained"):
            component.save_pretrained(directory)
        else:
            logger.warning("The component does not have a 'save_pretrained' method.")
        for path in Path(directory).iterdir():
            file.write(filename=path, arcname=path.name)

    logger.info(f"Tensorizer serialize pretrained done: {save_path}")
