import os
from typing import Optional

from ..cache_manager import CacheManager


class VideoCacheManager(CacheManager):
    def cache_gguf(self, quantization: Optional[str] = None):
        from ..utils import IS_NEW_HUGGINGFACE_HUB, retry_download, symlink_local_file
        from .core import VideoModelFamilyV2

        if not quantization:
            return None

        assert isinstance(self._model_family, VideoModelFamilyV2)
        cache_dir = self.get_cache_dir()

        if not self._model_family.gguf_model_file_name_template:
            raise NotImplementedError(
                f"{self._model_family.model_name} does not support GGUF quantization"
            )
        if quantization not in (self._model_family.gguf_quantizations or []):
            raise ValueError(
                f"Cannot support quantization {quantization}, "
                f"available quantizations: {self._model_family.gguf_quantizations}"
            )

        filename = self._model_family.gguf_model_file_name_template.format(
            quantization=quantization
        )
        full_path = os.path.join(cache_dir, filename)

        if self._model_family.model_hub == "huggingface":
            import huggingface_hub

            use_symlinks = {}
            if not IS_NEW_HUGGINGFACE_HUB:
                use_symlinks = {"local_dir_use_symlinks": True, "local_dir": cache_dir}
            download_file_path = retry_download(
                huggingface_hub.hf_hub_download,
                self._model_family.model_name,
                None,
                self._model_family.gguf_model_id,
                filename=filename,
                **use_symlinks,
            )
            if IS_NEW_HUGGINGFACE_HUB:
                symlink_local_file(download_file_path, cache_dir, filename)
        elif self._model_family.model_hub == "modelscope":
            from modelscope.hub.file_download import model_file_download

            download_file_path = retry_download(
                model_file_download,
                self._model_family.model_name,
                None,
                self._model_family.gguf_model_id,
                filename,
                revision=self._model_family.model_revision,
            )
            symlink_local_file(download_file_path, cache_dir, filename)
        else:
            raise NotImplementedError

        return full_path
