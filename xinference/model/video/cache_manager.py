import logging
import os
import shutil
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Sequence, Tuple

from ..cache_manager import CacheManager
from .ltx25_checkpoint import (
    LTX25_BASE_FILES,
    LTX25_MODEL_SOURCES,
    _download_ltx25_base,
    find_phai_m1_checkpoint,
)

if TYPE_CHECKING:
    from .core import VideoModelFamilyV2

logger = logging.getLogger(__name__)
PHAI_M1_VARIANTS = ("dev", "distilled")


def _missing_ltx25_base_files(base_path: Path) -> list[str]:
    return [name for name in LTX25_BASE_FILES if not (base_path / name).is_file()]


def phai_m1_base_lock_path(cache_root: Path, hub: str) -> Path:
    # Reuse the lock shared with older workers which used a standalone base view.
    return cache_root / f"Phai-M1-LTX-2.5-{hub}.lock"


@contextmanager
def _phai_m1_download_lock(cache_root: Path, hub: str) -> Iterator[None]:
    from filelock import FileLock, Timeout

    from ..utils import _cancellable_downloaders_var

    lock = FileLock(
        str(phai_m1_base_lock_path(cache_root, hub)), preserve_lock_file=True
    )
    downloaders = _cancellable_downloaders_var.get()
    downloader = downloaders[-1] if downloaders else None
    while True:
        if downloader is not None and downloader.cancelled:
            downloader.raise_error()
        try:
            lock.acquire(timeout=0.25 if downloader is not None else -1)
        except Timeout:
            continue
        break
    try:
        if downloader is not None and downloader.cancelled:
            downloader.raise_error()
        yield
    finally:
        lock.release()


def _preserve_legacy_base_file(source: Path, dest: Path) -> None:
    """Keep a legacy local copy even if a cross-device copy must be retried."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    try:
        os.link(source, dest)
    except OSError:
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            shutil.copy2(source, temp_path)
            temp_path.replace(dest)
        finally:
            temp_path.unlink(missing_ok=True)


class VideoCacheManager(CacheManager):
    _model_family: "VideoModelFamilyV2"

    def _phai_m1_base_path(self) -> Path:
        hub = self._model_family.model_hub
        return Path(self.get_cache_dir()) / f"ltx-2.5-{hub}"

    def _phai_m1_filename(self) -> str:
        return (self._model_family.default_model_config or {})["transformer_filename"]

    def get_cache_status(self) -> bool:
        if self._model_family.model_family != "Phai-M1":
            return super().get_cache_status()
        checkpoint = find_phai_m1_checkpoint(
            self.get_cache_dir(), self._phai_m1_filename()
        )
        if checkpoint is None:
            return False
        return not _missing_ltx25_base_files(self._phai_m1_base_path())

    @staticmethod
    def migrate_legacy_phai_m1_bases() -> None:
        """Attach the previous standalone cache to its existing variants."""
        from filelock import FileLock

        from ...constants import XINFERENCE_CACHE_DIR

        cache_root = Path(XINFERENCE_CACHE_DIR) / "v2"
        for hub in LTX25_MODEL_SOURCES:
            legacy = cache_root / f"Phai-M1-LTX-2.5-{hub}"
            if legacy.is_dir() and not legacy.is_symlink():
                try:
                    with FileLock(
                        str(phai_m1_base_lock_path(cache_root, hub)),
                        preserve_lock_file=True,
                    ):
                        VideoCacheManager._migrate_phai_m1_base(cache_root, hub)
                except OSError:
                    logger.warning(
                        "Cannot migrate Phai-M1 base cache %s", legacy, exc_info=True
                    )

    @staticmethod
    def _migrate_phai_m1_base(cache_root: Path, hub: str) -> None:
        from ..utils import (
            _record_cache_source_path,
            get_cache_source_paths,
            symlink_local_file,
        )

        legacy = cache_root / f"Phai-M1-LTX-2.5-{hub}"
        if not legacy.is_dir() or legacy.is_symlink():
            return
        available = {name for name in LTX25_BASE_FILES if (legacy / name).is_file()}
        if not available:
            return
        owners = []
        for variant in PHAI_M1_VARIANTS:
            entry = cache_root / f"Phai-M1-{variant}"
            filename = f"Phai-M1-22b-{variant}-transformer-bf16.safetensors"
            if not entry.is_symlink() and find_phai_m1_checkpoint(str(entry), filename):
                owners.append(entry)
        if not owners:
            return

        sources = get_cache_source_paths(str(legacy))
        for entry in owners:
            view = entry / f"ltx-2.5-{hub}"
            for name in _missing_ltx25_base_files(view):
                if name not in available:
                    continue
                source = legacy / name
                if source.is_symlink():
                    symlink_local_file(str(source), str(entry), f"{view.name}/{name}")
                else:
                    _preserve_legacy_base_file(source, view / name)
            for source_path in sources:
                _record_cache_source_path(str(entry), source_path)
        # Only the standalone view goes away. Each variant now references
        # the same files through its own cache entry.
        shutil.rmtree(legacy)

    def _cache_phai_m1_base(self) -> str:
        from ..utils import symlink_local_file

        hub = self._model_family.model_hub
        cache_root = Path(self._v2_cache_dir_prefix)
        base_path = self._phai_m1_base_path()
        self._migrate_phai_m1_base(cache_root, hub)
        missing = _missing_ltx25_base_files(base_path)
        if not missing:
            return str(base_path)
        reusable = self._find_phai_m1_base(cache_root, hub, base_path, missing)
        source_dir = reusable or Path(_download_ltx25_base(hub))
        for name in missing:
            source = source_dir / name
            if not source.is_file():
                raise FileNotFoundError(f"LTX-2.5 base artifact not found: {source}")
            symlink_local_file(
                str(source), self.get_cache_dir(), f"{base_path.name}/{name}"
            )
        return str(base_path)

    @staticmethod
    def _find_phai_m1_base(
        cache_root: Path, hub: str, base_path: Path, files: Sequence[str]
    ) -> Optional[Path]:
        for variant in PHAI_M1_VARIANTS:
            other = cache_root / f"Phai-M1-{variant}" / f"ltx-2.5-{hub}"
            if other != base_path and all(
                (other / name).is_symlink() and (other / name).is_file()
                for name in files
            ):
                return other
        return None

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
        if (
            base_model_path is not None
            and not (Path(base_model_path) / "model_index.json").is_file()
        ):
            raise FileNotFoundError(
                f"LTX-2.5 Diffusers model_index.json not found in {base_model_path}"
            )
        lock = (
            nullcontext()
            if model_path and base_model_path is not None
            else _phai_m1_download_lock(
                Path(self._v2_cache_dir_prefix), self._model_family.model_hub
            )
        )
        with lock:
            checkpoint = find_phai_m1_checkpoint(model_path or cache_dir, filename)
            if checkpoint is None:
                if model_path:
                    raise FileNotFoundError(
                        f"Phai-M1 checkpoint not found in {model_path}"
                    )
                downloaded = self._download_phai_m1_checkpoint(filename)
                checkpoint = Path(symlink_local_file(downloaded, cache_dir, filename))
            if base_model_path is None:
                base_model_path = self._cache_phai_m1_base()
            if not model_path:
                # Older caches could mirror the sibling variant from a shared snapshot.
                for directory in (
                    Path(cache_dir),
                    Path(cache_dir) / "diffusion_models",
                ):
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
