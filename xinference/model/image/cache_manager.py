import os
from typing import Optional

from ..cache_manager import CacheManager


class ImageCacheManager(CacheManager):
    def __init__(self, model_family):
        super().__init__(model_family)
        model_format = getattr(model_family, "model_format", None)
        quantization = getattr(model_family, "quantization", None)
        suffix_parts = []
        if model_format:
            suffix_parts.append(model_format)
        if quantization:
            suffix_parts.append(quantization)
        if suffix_parts:
            suffix = "-".join(suffix_parts)
            self._cache_dir = os.path.join(
                self._v2_cache_dir_prefix,
                f"{self._model_family.model_name.replace('.', '_')}-{suffix}",
            )

    def cache_ocr_gguf(self) -> str:
        """Download one GGUF quantization and its vision projector."""
        from ..utils import IS_NEW_HUGGINGFACE_HUB, retry_download, symlink_local_file

        family = self._model_family
        quantization = getattr(family, "quantization", None)
        template = getattr(family, "model_file_name_template", None)
        projector = getattr(family, "multimodal_projector", None)
        if not quantization or not template or not projector:
            raise ValueError(
                "OCR GGUF requires a quantization and multimodal projector"
            )

        cache_dir = self.get_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        model_filename = template.format(quantization=quantization)
        for filename in (model_filename, projector):
            local_path = os.path.join(cache_dir, filename)
            if os.path.isfile(local_path):
                continue
            if family.model_hub == "huggingface":
                from huggingface_hub import hf_hub_download

                download_kwargs = {"revision": family.model_revision}
                if not IS_NEW_HUGGINGFACE_HUB:
                    download_kwargs.update(
                        {"local_dir": cache_dir, "local_dir_use_symlinks": True}
                    )
                downloaded = retry_download(
                    hf_hub_download,
                    family.model_name,
                    None,
                    family.model_id,
                    filename=filename,
                    **download_kwargs,
                )
                if IS_NEW_HUGGINGFACE_HUB:
                    symlink_local_file(downloaded, cache_dir, filename)
            elif family.model_hub == "modelscope":
                from modelscope.hub.file_download import model_file_download

                downloaded = retry_download(
                    model_file_download,
                    family.model_name,
                    None,
                    family.model_id,
                    filename,
                    revision=family.model_revision,
                )
                symlink_local_file(downloaded, cache_dir, filename)
            else:
                raise ValueError(f"Unsupported model hub: {family.model_hub}")
        return os.path.join(cache_dir, model_filename)

    def cache_gguf(self, quantization: Optional[str] = None):
        from ..utils import IS_NEW_HUGGINGFACE_HUB, retry_download, symlink_local_file
        from .core import ImageModelFamilyV2

        if not quantization:
            return None

        assert isinstance(self._model_family, ImageModelFamilyV2)
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

        filename = self._model_family.gguf_model_file_name_template.format(quantization=quantization)  # type: ignore
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

    def cache_lightning(self, lightning_version: Optional[str] = None):
        from ..utils import IS_NEW_HUGGINGFACE_HUB, retry_download, symlink_local_file
        from .core import ImageModelFamilyV2

        if not lightning_version:
            return None

        assert isinstance(self._model_family, ImageModelFamilyV2)
        cache_dir = self.get_cache_dir()

        if not self._model_family.lightning_model_file_name_template:
            raise NotImplementedError(
                f"{self._model_family.model_name} does not support lightning"
            )
        if lightning_version not in (self._model_family.lightning_versions or []):
            raise ValueError(
                f"Cannot support lightning version {lightning_version}, "
                f"available lightning version: {self._model_family.lightning_versions}"
            )

        filename = self._model_family.lightning_model_file_name_template.format(lightning_version=lightning_version)  # type: ignore
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
                self._model_family.lightning_model_id,
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
                self._model_family.lightning_model_id,
                filename,
                revision=self._model_family.model_revision,
            )
            symlink_local_file(download_file_path, cache_dir, filename)
        else:
            raise NotImplementedError

        return full_path
