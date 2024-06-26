import io
import json
import logging
import os
import tempfile
import zipfile
from functools import partial
from pathlib import Path
from typing import Any, List, Optional

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


def _file_is_non_empty(
    path: str,
) -> bool:
    try:
        return os.stat(path).st_size > 0
    except FileNotFoundError:
        return False


def get_tensorizer_dir(model_path: str) -> str:
    _, model_dir_name = model_path.rsplit("/", 1)
    return f"{XINFERENCE_TENSORIZER_DIR}/{model_dir_name}"


def check_tensorizer_integrity(
    model_path: str,
    components: Optional[List[str]] = None,
    model_prefix: Optional[str] = "model",
) -> bool:
    tensorizer_dir = get_tensorizer_dir(model_path)
    dir = tensorizer_dir.rstrip("/")
    config_uri: str = f"{dir}/{model_prefix}-config.json"
    tensors_uri: str = f"{dir}/{model_prefix}.tensors"
    # iterate over components and get their paths
    paths = [config_uri, tensors_uri]
    if components is not None:
        for component in components:
            component_uri: str = f"{tensorizer_dir.rstrip('/')}/{component}.zip"
            paths.append(component_uri)
    return all(_file_is_non_empty(path) for path in paths)


def load_from_tensorizer(
    model_path: str,
    components: Optional[List[tuple[str, Any]]] = None,
    model_class: Any = None,
    config_class: Any = None,
    model_prefix: Optional[str] = "model",
):
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
            tensorizer_dir,
            model_class,
            config_class,
            model_prefix,
            device,
        )
        .to(device)
        .eval()
    )

    tensorizer_components = []

    if components is not None:
        for component, component_class in components:
            deserialized_component = _load_pretrained_from_tensorizer(
                component_class, tensorizer_dir, component
            )
            tensorizer_components.append(deserialized_component)

    return tensorizer_model, *tensorizer_components


def _load_pretrained_from_tensorizer(
    component_class: Any,
    tensorizer_dir: str,
    prefix: str,
):
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
                directory, cache_dir=None, local_files_only=True
            )


def _load_model_from_tensorizer(
    tensorizer_dir: str,
    model_class,
    config_class,
    model_prefix: Optional[str] = "model",
    device=None,
    dtype=None,
):
    logger.debug(f"Loading model from tensorizer: {tensorizer_dir}")

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
    config_uri: str = f"{dir}/{model_prefix}-config.json"
    tensors_uri: str = f"{dir}/{model_prefix}.tensors"

    _read_stream = partial(stream_io.open_stream, mode="rb")

    if config_class is None:
        config_loader = model_class.load_config
    else:
        config_loader = config_class.from_pretrained
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config_path = os.path.join(temp_dir, "config.json")
            with open(temp_config_path, "wb") as temp_config:
                logger.info(f"Loading {config_uri}")
                with _read_stream(config_uri) as config_file:
                    temp_config.write(config_file.read())
            config = config_loader(temp_dir)
            if isinstance(config, PretrainedConfig):
                config.gradient_checkpointing = True
    except ValueError:
        config = config_loader(config_uri)
    with utils.no_init_or_tensor():
        # AutoModels instantiate from a config via their from_config()
        # method, while other classes can usually be instantiated directly.
        model_loader = getattr(model_class, "from_config", model_class)
        model = model_loader(config)

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
    components: Optional[List[tuple[str, Any]]] = None,
    model_config: Optional[Any] = None,
    model_prefix: Optional[str] = "model",
    force: Optional[bool] = False,
):
    if model_config is None:
        try:
            from transformers import AutoConfig
        except ImportError:
            error_message = "Failed to import module 'transformers'"
            installation_guide = [
                "Please make sure 'transformers' is installed. ",
                "You can install it by `pip install transformers`\n",
            ]
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        model_config = AutoConfig.from_pretrained(model_path)

    _tensorizer_serialize_model(
        model_path,
        model,
        model_config,
        model_prefix,
        force,
    )

    if components is not None:
        for component_prefix, component in components:
            _tensorizer_serialize_pretrained(model_path, component, component_prefix)


def _tensorizer_serialize_model(
    model_path: str,
    model,
    model_config: Optional[Any] = None,
    model_prefix: Optional[str] = "model",
    force: Optional[bool] = False,
):
    try:
        from tensorizer import TensorSerializer, stream_io
    except ImportError:
        error_message = "Failed to import module 'tensorizer'"
        installation_guide = [
            "Please make sure 'tensorizer' is installed.\n",
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

    if model_config is None:
        model_config = PretrainedConfig.from_pretrained(model_path)

    tensorizer_dir = get_tensorizer_dir(model_path)
    dir_prefix: str = f"{tensorizer_dir}/{model_prefix}"
    config_path: str = f"{dir_prefix}-config.json"
    tensor_path: str = f"{dir_prefix}.tensors"
    logger.info(f"Tensorizer serialize model: {tensor_path}")

    paths = (config_path, tensor_path)

    use_cache = not force and all(_file_is_non_empty(path) for path in paths)
    if use_cache:
        logger.info(f"Cache {tensor_path} exists, skip tensorizer serialize model")
        return tensor_path

    _write_stream = partial(stream_io.open_stream, mode="wb+")

    if not use_cache:
        logger.info(f"Writing config to {config_path}")
        with _write_stream(config_path) as f:
            config_dict = (
                model_config.to_dict()
                if hasattr(model_config, "to_dict")
                else model_config
            )
            f.write(json.dumps(config_dict, indent=2).encode("utf-8"))
        logger.info(f"Writing tensors to {tensor_path}")
        with _write_stream(tensor_path) as f:
            serializer = TensorSerializer(f)
            serializer.write_module(model, include_non_persistent_buffers=False)
            serializer.close()

    logger.info(f"Tensorizer serialize model done: {tensor_path}")
    return tensor_path


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
    logger.info(f"Tensorizer serialize pretrained: {save_path}")

    if os.path.exists(save_path):
        logger.info(f"Cache {save_path} exists, skip tensorizer serialize pretrained")
        return save_path

    _write_stream = partial(stream_io.open_stream, mode="wb+")

    with _write_stream(save_path) as stream, zipfile.ZipFile(
        stream, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=5
    ) as file, tempfile.TemporaryDirectory() as directory:
        if hasattr(component, "save_pretrained"):
            component.save_pretrained(directory)
        else:
            logger.warn("The component does not have a 'save_pretrained' method.")
        for path in Path(directory).iterdir():
            file.write(filename=path, arcname=path.name)

    logger.info(f"Tensorizer serialize pretrained done: {save_path}")
