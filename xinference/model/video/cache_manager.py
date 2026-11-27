import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

from ..cache_manager import CacheManager
from .ltx25_checkpoint import (
    LTX25_BASE_FILES,
    _download_ltx25_base,
    find_phai_m1_checkpoint,
)

if TYPE_CHECKING:
    from .core import VideoModelFamilyV2


class VideoCacheManager(CacheManager):
    _model_family: "VideoModelFamilyV2"

    def _phai_m1_base_path(self) -> Path:
        hub = self._model_family.model_hub
        return Path(self._v2_cache_dir_prefix) / f"Phai-M1-LTX-2.5-{hub}"

    def _phai_m1_filename(self) -> str:
        return (self._model_family.default_model_config or {})["transformer_filename"]

    def get_cache_status(self) -> bool:
        if self._model_family.model_family != "Phai-M1":
            return super().get_cache_status()
        checkpoint = find_phai_m1_checkpoint(
            self.get_cache_dir(), self._phai_m1_filename()
        )
        base_path = self._phai_m1_base_path()
        return checkpoint is not None and all(
            (base_path / name).is_file() for name in LTX25_BASE_FILES
        )

    def _cache_phai_m1_base(self) -> str:
        from filelock import FileLock

        from ..utils import symlink_local_file

        base_path = self._phai_m1_base_path()
        if all((base_path / name).is_file() for name in LTX25_BASE_FILES):
            return str(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(f"{base_path}.lock", preserve_lock_file=True):
            missing = [
                name for name in LTX25_BASE_FILES if not (base_path / name).is_file()
            ]
            if missing:
                snapshot = Path(_download_ltx25_base(self._model_family.model_hub))
                for name in missing:
                    source = snapshot / name
                    if not source.is_file():
                        raise FileNotFoundError(
                            f"LTX-2.5 base artifact not found: {source}"
                        )
                    symlink_local_file(str(source), str(base_path), name)
        return str(base_path)

    def _download_phai_m1_checkpoint(self, filename: str) -> str:
        from ..utils import retry_download

        spec = self._model_family
        if spec.model_hub == "modelscope":
            from modelscope.hub.file_download import model_file_download

            download = model_file_download
            remote_filename = f"diffusion_models/{filename}"
        else:
            from huggingface_hub import hf_hub_download

            download = hf_hub_download
            remote_filename = filename
        return retry_download(
            download,
            spec.model_name,
            None,
            spec.model_id,
            remote_filename,
            revision=spec.model_revision,
        )

    def cache_phai_m1(
        self,
        model_path: Optional[str] = None,
        base_model_path: Optional[str] = None,
    ) -> Tuple[str, str]:
        from ..utils import symlink_local_file

        cache_dir = self.get_cache_dir()
        filename = self._phai_m1_filename()
        checkpoint = find_phai_m1_checkpoint(model_path or cache_dir, filename)
        if checkpoint is None and model_path:
            raise FileNotFoundError(f"Phai-M1 checkpoint not found in {model_path}")
        if base_model_path is None:
            base_model_path = self._cache_phai_m1_base()
        elif not (Path(base_model_path) / "model_index.json").is_file():
            raise FileNotFoundError(
                f"LTX-2.5 Diffusers model_index.json not found in {base_model_path}"
            )
        if checkpoint is None:
            downloaded = self._download_phai_m1_checkpoint(filename)
            checkpoint = Path(symlink_local_file(downloaded, cache_dir, filename))

        if not model_path:
            # Older caches could mirror the sibling variant from a shared snapshot.
            for directory in (Path(cache_dir), Path(cache_dir) / "diffusion_models"):
                for link in directory.glob(
                    "Phai-M1-22b-*-transformer-bf16.safetensors"
                ):
                    if link.name != filename and link.is_symlink():
                        link.unlink(missing_ok=True)
        return str(checkpoint), base_model_path

    def cache(self) -> str:
        if self._model_family.model_family == "Phai-M1":
            self.cache_phai_m1()
            return self.get_cache_dir()
        return super().cache()

    def check_lightning(self, lightning_version: str) -> str:
        """Return the lightning file name for a supported version, else raise.

        Nothing is downloaded, so callers can reject a bad version before
        caching anything.
        """
        from .core import VideoModelFamilyV2

        assert isinstance(self._model_family, VideoModelFamilyV2)
        filename_template = self._model_family.lightning_model_file_name_template
        if not filename_template or not self._model_family.lightning_model_id:
            raise NotImplementedError(
                f"{self._model_family.model_name} does not support lightning"
            )
        if lightning_version not in (self._model_family.lightning_versions or []):
            raise ValueError(
                f"Cannot support lightning version {lightning_version}, "
                f"available lightning versions: {self._model_family.lightning_versions}"
            )
        return filename_template.format(lightning_version=lightning_version)

    def cache_lightning(self, lightning_version: Optional[str] = None):
        from ..utils import IS_NEW_HUGGINGFACE_HUB, retry_download, symlink_local_file
        from .core import VideoModelFamilyV2

        if not lightning_version:
            return None

        assert isinstance(self._model_family, VideoModelFamilyV2)
        cache_dir = self.get_cache_dir()
        filename = self.check_lightning(lightning_version)
        full_path = os.path.join(cache_dir, filename)
        lightning_revision = self._model_family.lightning_model_revision

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
                revision=lightning_revision,
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
                revision=lightning_revision or "master",
            )
            symlink_local_file(download_file_path, cache_dir, filename)
        else:
            raise NotImplementedError(
                f"Unsupported lightning model hub: {self._model_family.model_hub}"
            )

        return full_path

    def check_gguf(self, quantization: str) -> str:
        """Return the GGUF file name for a supported quantization, else raise."""
        from .core import VideoModelFamilyV2

        assert isinstance(self._model_family, VideoModelFamilyV2)
        filename_template = self._model_family.gguf_model_file_name_template
        if not filename_template:
            raise NotImplementedError(
                f"{self._model_family.model_name} does not support GGUF quantization"
            )
        if quantization not in (self._model_family.gguf_quantizations or []):
            raise ValueError(
                f"Cannot support quantization {quantization}, "
                f"available quantizations: {self._model_family.gguf_quantizations}"
            )
        return filename_template.format(quantization=quantization)

    def cache_gguf(self, quantization: Optional[str] = None):
        from ..utils import IS_NEW_HUGGINGFACE_HUB, retry_download, symlink_local_file
        from .core import VideoModelFamilyV2

        if not quantization:
            return None

        assert isinstance(self._model_family, VideoModelFamilyV2)
        cache_dir = self.get_cache_dir()

        filename = self.check_gguf(quantization)
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
