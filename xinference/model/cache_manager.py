import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import CacheableModelSpec


logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self, model_family: "CacheableModelSpec"):
        from ..constants import XINFERENCE_CACHE_DIR

        self._model_family = model_family
        self._v2_cache_dir_prefix = os.path.join(XINFERENCE_CACHE_DIR, "v2")
        os.makedirs(self._v2_cache_dir_prefix, exist_ok=True)

    def get_cache_dir(self):
        return os.path.join(self._v2_cache_dir_prefix, self._model_family.model_name)

    def get_cache_status(self):
        cache_dir = self.get_cache_dir()
        return os.path.exists(cache_dir)

    def _cache_from_uri(self, model_spec: "CacheableModelSpec") -> str:
        from .utils import parse_uri

        cache_dir = self.get_cache_dir()
        if os.path.exists(cache_dir):
            logger.info("cache %s exists", cache_dir)
            return cache_dir

        assert model_spec.model_uri is not None
        src_scheme, src_root = parse_uri(model_spec.model_uri)
        if src_root.endswith("/"):
            # remove trailing path separator.
            src_root = src_root[:-1]

        if src_scheme == "file":
            if not os.path.isabs(src_root):
                raise ValueError(
                    f"Model URI cannot be a relative path: {model_spec.model_uri}"
                )
            os.symlink(src_root, cache_dir, target_is_directory=True)
            return cache_dir
        else:
            raise ValueError(f"Unsupported URL scheme: {src_scheme}")

    def _cache(self) -> str:
        from .utils import IS_NEW_HUGGINGFACE_HUB, create_symlink, retry_download

        if (
            hasattr(self._model_family, "model_uri")
            and getattr(self._model_family, "model_uri", None) is not None
        ):
            logger.info(f"Model caching from URI: {self._model_family.model_uri}")
            return self._cache_from_uri(model_spec=self._model_family)

        cache_dir = self.get_cache_dir()
        if self.get_cache_status():
            return cache_dir

        from_modelscope: bool = self._model_family.model_hub == "modelscope"
        if from_modelscope:
            from modelscope.hub.snapshot_download import (
                snapshot_download as ms_download,
            )

            download_dir = retry_download(
                ms_download,
                self._model_family.model_name,
                None,
                self._model_family.model_id,
                revision=self._model_family.model_revision,
            )
            create_symlink(download_dir, cache_dir)
        else:
            from huggingface_hub import snapshot_download as hf_download

            use_symlinks = {}
            if not IS_NEW_HUGGINGFACE_HUB:
                use_symlinks = {"local_dir_use_symlinks": True, "local_dir": cache_dir}
            download_dir = retry_download(
                hf_download,
                self._model_family.model_name,
                None,
                self._model_family.model_id,
                revision=self._model_family.model_revision,
                **use_symlinks,
            )
            if IS_NEW_HUGGINGFACE_HUB:
                create_symlink(download_dir, cache_dir)
        return cache_dir

    def cache(self) -> str:
        return self._cache()
