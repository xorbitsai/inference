# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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

import logging
import os
from typing import TYPE_CHECKING, Optional

from ..cache_manager import CacheManager

if TYPE_CHECKING:
    from .llm_family import LLMFamilyV2


logger = logging.getLogger(__name__)


class LLMCacheManager(CacheManager):
    def __init__(
        self,
        llm_family: "LLMFamilyV2",
        multimodal_projector: Optional[str] = None,
        use_draft_model: bool = False,
        draft_quantization: Optional[str] = None,
    ):
        super().__init__(llm_family)
        self._llm_family = llm_family
        self._model_name = llm_family.model_name
        self._model_format = llm_family.model_specs[0].model_format
        self._model_size_in_billions = getattr(
            llm_family.model_specs[0], "model_size_in_billions", None
        )
        self._quantization = llm_family.model_specs[0].quantization
        self._model_uri = llm_family.model_specs[0].model_uri
        self._multimodal_projector = multimodal_projector
        self._model_id = llm_family.model_specs[0].model_id
        self._model_hub = llm_family.model_specs[0].model_hub
        self._model_revision = llm_family.model_specs[0].model_revision
        # Set when caching a gguf drafter: a single file rather than a snapshot.
        self._draft_file_name: Optional[str] = None
        self._use_draft_model = use_draft_model
        self._cache_dir = os.path.join(
            self._v2_cache_dir_prefix,
            f"{self._model_name.replace('.', '_')}-{self._model_format}-"
            f"{self._model_size_in_billions}b-{self._quantization}",
        )
        if use_draft_model:
            # Cache the paired drafter checkpoint instead of the target model,
            # reusing the download logic below with a dedicated cache dir.
            spec = llm_family.model_specs[0]
            draft_model_id = getattr(spec, "draft_model_id", None)
            draft_template = getattr(spec, "draft_model_file_name_template", None)
            if not draft_model_id and draft_template:
                # a gguf drafter usually ships inside the target's own repo
                draft_model_id = spec.model_id
            if not draft_model_id:
                raise ValueError(
                    f"Model {self._model_name} ({self._model_format}, "
                    f"{self._model_size_in_billions}b, {self._quantization}) does not "
                    f"declare a drafter model on hub {self._model_hub}. Please pass "
                    f"`draft_model_path` to use a local drafter."
                )
            draft_quantizations = getattr(spec, "draft_quantizations", None) or []
            if draft_quantization is None:
                draft_quantization = (
                    draft_quantizations[0] if draft_quantizations else None
                )
            elif draft_quantizations and draft_quantization not in draft_quantizations:
                raise ValueError(
                    f"Drafter quantization {draft_quantization!r} is not available "
                    f"for {self._model_name} ({self._model_format}, "
                    f"{self._model_size_in_billions}b), "
                    f"available: {draft_quantizations}"
                )
            self._draft_quantization = draft_quantization
            if "{draft_quantization}" in draft_model_id:
                if not draft_quantization:
                    raise ValueError(
                        f"Model {self._model_name} declares the templated drafter "
                        f"{draft_model_id!r} but no `draft_quantizations`."
                    )
                draft_model_id = draft_model_id.format(
                    draft_quantization=draft_quantization
                )
            self._model_id = draft_model_id
            self._model_revision = getattr(spec, "draft_model_revision", None)
            if draft_template:
                if not draft_quantization:
                    raise ValueError(
                        f"Model {self._model_name} declares the drafter file "
                        f"{draft_template!r} but no `draft_quantizations`."
                    )
                self._draft_file_name = draft_template.format(
                    draft_quantization=draft_quantization
                )
            elif self._model_format == "ggufv2":
                # A gguf download is per file, so without a template there is no
                # drafter file to ask for; falling through would request the
                # target's file name from the drafter repository.
                raise ValueError(
                    f"Model {self._model_name} declares the drafter "
                    f"{draft_model_id!r} but no `draft_model_file_name_template`, "
                    f"which a gguf drafter needs. Pass `draft_model_path` to use "
                    f"a local drafter file instead."
                )
            # The drafter is downloaded on its own, never through the target's
            # local URI or its multimodal projector.
            self._model_uri = None
            self._multimodal_projector = None
            # Keep the quantization in the path so drafters converted differently
            # do not collide, and so ``list_deletable_models`` can find them all
            # by the ``-draft`` prefix.
            suffix = f"-draft-{draft_quantization}" if draft_quantization else "-draft"
            self._cache_dir = f"{self._cache_dir}{suffix}"

    def _gguf_file_names(self):
        """File list for a gguf download: the drafter is a single file."""
        from ..utils import generate_model_file_names_with_quantization_parts

        if self._use_draft_model:
            # guarded in __init__; never fall through to the target's file names,
            # they do not exist in the drafter repository
            assert self._draft_file_name, "no drafter file to download"
            return [self._draft_file_name], self._draft_file_name, False
        return generate_model_file_names_with_quantization_parts(
            self._llm_family.model_specs[0], self._multimodal_projector
        )

    def cache_uri(self) -> str:
        from ..utils import parse_uri

        cache_dir = self.get_cache_dir()
        assert self._model_uri is not None
        src_scheme, src_root = parse_uri(self._model_uri)
        if src_root.endswith("/"):
            # remove trailing path separator.
            src_root = src_root[:-1]

        if src_scheme == "file":
            if not os.path.isabs(src_root):
                raise ValueError(
                    f"Model URI cannot be a relative path: {self._model_uri}"
                )
            if not os.path.exists(src_root):
                raise ValueError(
                    f"Model URI path does not exist: {src_root}. "
                    f"Please check the `model_uri` of model {self._model_name!r}."
                )
            if os.path.islink(cache_dir):
                # ``os.path.exists`` follows the link, so a *dangling* symlink
                # (its target was removed) reports ``False`` while the link
                # entry itself still occupies ``cache_dir`` -- the ``os.symlink``
                # call below would then raise ``FileExistsError``. Reuse the
                # link if it already points at ``src_root``; otherwise drop the
                # stale entry so we can recreate it.
                if os.path.realpath(cache_dir) == os.path.realpath(src_root):
                    logger.info(f"Cache {cache_dir} exists")
                    return cache_dir
                logger.info(f"Removing stale cache symlink {cache_dir}")
                os.unlink(cache_dir)
            elif os.path.exists(cache_dir):
                logger.info(f"Cache {cache_dir} exists")
                return cache_dir

            # The cache prefix directory is only created once per process
            # (guarded by ``CacheManager.is_initialized``), so it may be
            # missing here; ensure the parent exists before symlinking to
            # avoid a cryptic ``FileNotFoundError: src -> dst``.
            os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
            os.symlink(src_root, cache_dir, target_is_directory=True)
            return cache_dir
        else:
            raise ValueError(f"Unsupported URL scheme: {src_scheme}")

    def cache_from_huggingface(self) -> str:
        """
        Cache model from Hugging Face. Return the cache directory.
        """
        import huggingface_hub

        from ..utils import (
            IS_NEW_HUGGINGFACE_HUB,
            create_symlink,
            merge_cached_files,
            retry_download,
            symlink_local_file,
        )

        cache_dir = self.get_cache_dir()
        if self.get_cache_status():
            return cache_dir

        cache_config = (
            self._llm_family.cache_config.copy()
            if self._llm_family.cache_config
            else {}
        )
        use_symlinks = {}
        if not IS_NEW_HUGGINGFACE_HUB:
            use_symlinks = {"local_dir_use_symlinks": True, "local_dir": cache_dir}
            cache_config = {**cache_config, **use_symlinks}

        if self._model_format in ["pytorch", "gptq", "awq", "fp4", "fp8", "bnb", "mlx"]:
            download_dir = retry_download(
                huggingface_hub.snapshot_download,
                self._model_name,
                {
                    "model_size": self._model_size_in_billions,
                    "model_format": self._model_format,
                },
                self._model_id,
                revision=self._model_revision,
                **cache_config,
            )
            if IS_NEW_HUGGINGFACE_HUB:
                create_symlink(download_dir, cache_dir)
        elif self._model_format in ["ggufv2"]:
            file_names, final_file_name, need_merge = self._gguf_file_names()

            for file_name in file_names:
                download_file_path = retry_download(
                    huggingface_hub.hf_hub_download,
                    self._model_name,
                    {
                        "model_size": self._model_size_in_billions,
                        "model_format": self._model_format,
                    },
                    self._model_id,
                    revision=self._model_revision,
                    filename=file_name,
                    **use_symlinks,
                )
                if IS_NEW_HUGGINGFACE_HUB:
                    symlink_local_file(download_file_path, cache_dir, file_name)

            if need_merge:
                merge_cached_files(cache_dir, file_names, final_file_name)
        else:
            raise ValueError(f"Unsupported model format: {self._model_format}")

        return cache_dir

    def cache_from_modelscope(self) -> str:
        """
        Cache model from Modelscope. Return the cache directory.
        """
        from modelscope.hub.file_download import model_file_download
        from modelscope.hub.snapshot_download import snapshot_download

        from ..utils import (
            create_symlink,
            merge_cached_files,
            retry_download,
            symlink_local_file,
        )

        cache_dir = self.get_cache_dir()
        if self.get_cache_status():
            return cache_dir

        cache_config = (
            self._llm_family.cache_config.copy()
            if self._llm_family.cache_config
            else {}
        )
        if self._model_format in [
            "pytorch",
            "gptq",
            "awq",
            "fp4",
            "bnb",
            "fp8",
            "bnb",
            "mlx",
        ]:
            download_dir = retry_download(
                snapshot_download,
                self._model_name,
                {
                    "model_size": self._model_size_in_billions,
                    "model_format": self._model_format,
                },
                self._model_id,
                revision=self._model_revision,
                **cache_config,
            )
            create_symlink(download_dir, cache_dir)

        elif self._model_format in ["ggufv2"]:
            file_names, final_file_name, need_merge = self._gguf_file_names()

            for filename in file_names:
                download_path = retry_download(
                    model_file_download,
                    self._model_name,
                    {
                        "model_size": self._model_size_in_billions,
                        "model_format": self._model_format,
                    },
                    self._model_id,
                    filename,
                    revision=self._model_revision,
                )
                symlink_local_file(download_path, cache_dir, filename)

            if need_merge:
                merge_cached_files(cache_dir, file_names, final_file_name)
        else:
            raise ValueError(f"Unsupported format: {self._model_format}")

        return cache_dir

    def cache_from_openmind_hub(self) -> str:
        """
        Cache model from openmind_hub. Return the cache directory.
        """
        from openmind_hub import snapshot_download

        from ..utils import create_symlink, retry_download

        cache_dir = self.get_cache_dir()
        if self.get_cache_status():
            return cache_dir

        if self._model_format in ["pytorch", "mindspore"]:
            download_dir = retry_download(
                snapshot_download,
                self._model_name,
                {
                    "model_size": self._model_size_in_billions,
                    "model_format": self._model_format,
                },
                self._model_id,
                revision=self._model_revision,
            )
            create_symlink(download_dir, cache_dir)

        else:
            raise ValueError(f"Unsupported format: {self._model_format}")
        return cache_dir

    def cache_from_csghub(self) -> str:
        """
        Cache model from CSGHub. Return the cache directory.
        """
        from pycsghub.file_download import file_download
        from pycsghub.snapshot_download import snapshot_download

        from ...constants import XINFERENCE_CSG_ENDPOINT, XINFERENCE_ENV_CSG_TOKEN
        from ..utils import (
            create_symlink,
            merge_cached_files,
            retry_download,
            symlink_local_file,
        )

        cache_dir = self.get_cache_dir()
        if self.get_cache_status():
            return cache_dir

        if self._model_format in ["pytorch", "gptq", "awq", "fp4", "fp8", "bnb", "mlx"]:
            download_dir = retry_download(
                snapshot_download,
                self._model_name,
                {
                    "model_size": self._model_size_in_billions,
                    "model_format": self._model_format,
                },
                self._model_id,
                endpoint=XINFERENCE_CSG_ENDPOINT,
                token=os.environ.get(XINFERENCE_ENV_CSG_TOKEN),
            )
            create_symlink(download_dir, cache_dir)
        elif self._model_format in ["ggufv2"]:
            file_names, final_file_name, need_merge = self._gguf_file_names()

            for filename in file_names:
                download_path = retry_download(
                    file_download,
                    self._model_name,
                    {
                        "model_size": self._model_size_in_billions,
                        "model_format": self._model_format,
                    },
                    self._model_id,
                    file_name=filename,
                    endpoint=XINFERENCE_CSG_ENDPOINT,
                    token=os.environ.get(XINFERENCE_ENV_CSG_TOKEN),
                )
                symlink_local_file(download_path, cache_dir, filename)

            if need_merge:
                merge_cached_files(cache_dir, file_names, final_file_name)
        else:
            raise ValueError(f"Unsupported format: {self._model_format}")

        return cache_dir

    def cache(self) -> str:
        if self._model_uri is not None:
            return self.cache_uri()
        else:
            if self._model_hub == "huggingface":
                return self.cache_from_huggingface()
            elif self._model_hub == "modelscope":
                return self.cache_from_modelscope()
            elif self._model_hub == "openmind_hub":
                return self.cache_from_openmind_hub()
            elif self._model_hub == "csghub":
                return self.cache_from_csghub()
            else:
                raise ValueError(f"Unknown model hub: {self._model_hub}")
