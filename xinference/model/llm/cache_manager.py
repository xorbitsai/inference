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

import contextlib
import hashlib
import json
import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..cache_manager import CacheManager

if TYPE_CHECKING:
    from .llm_family import LLMFamilyV2


logger = logging.getLogger(__name__)

# Where the files of a user-registered model's cache directory were downloaded from
# are recorded next to the model directories, never inside one, so that no file of a
# repository can pass for the record or be displaced by it. The record maps what a
# launch reuses to its source: GGUF files, including shards, aliases and projectors,
# otherwise the whole directory.
SOURCE_RECORD_DIR = ".model-sources"
WHOLE_DIRECTORY = "."


def _source_of(spec, quantization: Optional[str] = None) -> Dict[str, Any]:
    model_id = spec.model_id
    if model_id and "{" in model_id:
        # Match the selected spec's expansion, including PyTorch's default.
        # An invalid template is match_llm's error if that spec is selected.
        with contextlib.suppress(
            AttributeError, IndexError, KeyError, TypeError, ValueError
        ):
            model_id = model_id.format(
                quantization=spec.quantization if quantization is None else quantization
            )
    return {
        "model_hub": spec.model_hub,
        "model_id": model_id,
        "model_revision": spec.model_revision,
    }


def _describe_source(source: Dict[str, Any]) -> str:
    revision = source.get("model_revision")
    description = f"{source.get('model_hub')}:{source.get('model_id')}"
    return f"{description}@{revision}" if revision else description


class LLMCacheManager(CacheManager):
    @staticmethod
    def get_cache_dir_for_spec(model_name, spec):
        """Compute a cache path without initializing or creating directories."""
        from ...constants import XINFERENCE_CACHE_DIR

        return os.path.join(
            XINFERENCE_CACHE_DIR,
            "v2",
            f"{model_name.replace('.', '_')}-{spec.model_format}-"
            f"{getattr(spec, 'model_size_in_billions', None)}b-{spec.quantization}",
        )

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
        self._reuse_existing_gguf_files = False
        self._model_id = llm_family.model_specs[0].model_id
        self._model_hub = llm_family.model_specs[0].model_hub
        self._model_revision = llm_family.model_specs[0].model_revision
        # Set when caching a gguf drafter: a single file rather than a snapshot.
        self._draft_file_name: Optional[str] = None
        self._use_draft_model = use_draft_model
        self._cache_dir = self.get_cache_dir_for_spec(
            self._model_name, llm_family.model_specs[0]
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
            if "{draft_quantization}" in draft_model_id and not draft_quantization:
                raise ValueError(
                    f"Model {self._model_name} declares the templated drafter "
                    f"{draft_model_id!r} but no `draft_quantizations`."
                )
            # unconditionally, like the file template below: a field carrying
            # some other placeholder has to fail here rather than reach the hub
            # verbatim and 404 at download time
            draft_model_id = self._format_draft_field(
                draft_model_id, draft_quantization, "draft_model_id"
            )
            self._model_id = draft_model_id
            self._model_revision = getattr(spec, "draft_model_revision", None)
            if draft_template:
                if "{draft_quantization}" in draft_template and not draft_quantization:
                    raise ValueError(
                        f"Model {self._model_name} declares the drafter file "
                        f"{draft_template!r} but no `draft_quantizations`."
                    )
                self._draft_file_name = self._format_draft_field(
                    draft_template,
                    draft_quantization,
                    "draft_model_file_name_template",
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

    @staticmethod
    def _format_draft_field(
        template: str, draft_quantization: Optional[str], field: str
    ) -> str:
        """Fill ``{draft_quantization}`` in a drafter field.

        Any other placeholder makes ``str.format`` raise KeyError or IndexError.
        Nothing validates a user-registered spec, so those are normalized into
        the ValueError the callers already expect for a drafter shape that does
        not add up — a listing call must not fail over one bad registration.
        """
        try:
            return template.format(draft_quantization=draft_quantization)
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"{field} {template!r} carries a placeholder other than "
                f"`{{draft_quantization}}`: {exc!r}"
            )

    def get_cache_status(self):
        cache_dir = self.get_cache_dir()
        if not os.path.exists(cache_dir):
            return False
        if (
            self._model_format != "ggufv2"
            or self._model_uri is not None
            or self._use_draft_model
        ):
            return True
        # The directory name does not depend on the file a spec names, so a
        # directory cached under an older spec can lack the file the loader opens.
        from .gguf_model_metadata import resolve_gguf_model_path

        return os.path.exists(
            resolve_gguf_model_path(cache_dir, self._llm_family.model_specs[0])
        )

    def _declared_sources(self) -> Optional[List[Dict[str, Any]]]:
        """The downloads a user registration declares for this cache directory.

        None for a model that is not user-registered: built-in specs name a
        different repository per hub for one directory, so only registrations
        are tied to the source they were downloaded from.
        """
        from .custom import get_user_defined_llm_families

        for family in get_user_defined_llm_families():
            if family.model_name == self._model_name:
                return [
                    _source_of(spec, quantization)
                    for spec in family.model_specs
                    if spec.model_uri is None
                    and self.get_cache_dir_for_spec(self._model_name, spec)
                    == self._cache_dir
                    # match_llm also uses "none" for an omitted PyTorch quantization.
                    for quantization in (
                        (spec.quantization, "none")
                        if spec.model_format == "pytorch"
                        else (spec.quantization,)
                    )
                ]
        return None

    def _source_record_path(self) -> str:
        return os.path.join(
            self._v2_cache_dir_prefix,
            SOURCE_RECORD_DIR,
            os.path.basename(self._cache_dir) + ".json",
        )

    def _source_files(self, downloading: bool) -> List[str]:
        """Files read on a cache hit, or written by a GGUF download."""
        if self._model_format != "ggufv2" or self._model_uri is not None:
            return [WHOLE_DIRECTORY]
        from .gguf_model_metadata import resolve_gguf_model_path

        file_names, final_file_name, need_merge = self._gguf_file_names()
        if downloading:
            if need_merge:
                file_names.append(final_file_name)
        elif not need_merge:
            # Keep the runtime's legacy model.bin fallback.
            file_names = [
                os.path.relpath(
                    resolve_gguf_model_path(
                        self._cache_dir, self._llm_family.model_specs[0]
                    ),
                    self._cache_dir,
                )
            ]
            if self._multimodal_projector:
                file_names.append(self._multimodal_projector)
        return list(
            dict.fromkeys(
                os.path.relpath(
                    os.path.join(self._cache_dir, name), self._cache_dir
                ).replace(os.sep, "/")
                for name in file_names
            )
        )

    def _read_records(self) -> Dict[str, Dict[str, Any]]:
        """The recorded source of each file the directory holds, as far as known.

        Empty for a directory cached before sources were recorded, and for one
        that no longer exists: the record it left behind is not carried over.
        """
        if not os.path.exists(self._cache_dir):
            return {}
        record_path = self._source_record_path()
        try:
            with open(record_path, encoding="utf-8") as fd:
                records = json.load(fd)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            records = None
        if not (
            isinstance(records, dict)
            and records
            and all(isinstance(r, dict) and "model_hub" in r for r in records.values())
        ):
            raise ValueError(
                f"The record of where the cache directory {self._cache_dir} of model "
                f"{self._model_name} came from, {record_path}, cannot be read or is "
                f"invalid, so its files cannot be checked against the registration. "
                f"Remove the directory if the old files are no longer needed, or "
                f"remove the record to use the directory as it is."
            )
        return records

    def _check_source(self, may_record: bool) -> None:
        """Tie the files a user-registered model reuses to the source they came from.

        Unregistering a custom model leaves its downloaded directory behind. When
        the same name is registered again for another repository or revision,
        reusing the directory would serve the old files, so a file is reused only
        if it came from a source the registration still declares. Nothing is
        deleted or moved: only its owner can tell whether the old files are still
        wanted.

        Record new hub downloads before they start, under the files they add: a
        GGUF directory can exist, with or without a record, while the download of
        the selected file is still needed. Missing records and files without an
        entry are accepted for compatibility with older caches, including when a
        record was removed manually, except when completing a split GGUF download.
        Unreadable or invalid records do not authorize reuse.
        """
        self._reuse_existing_gguf_files = False
        declared = None if self._use_draft_model else self._declared_sources()
        if declared is None:
            return  # a drafter, or a model that is not user-registered
        if os.path.islink(self._cache_dir):
            return  # a link to the user's own model_uri directory
        records = self._read_records()
        # A failed download may have left entries for files that never arrived.
        records = {
            key: source
            for key, source in records.items()
            if key == WHOLE_DIRECTORY
            or os.path.exists(os.path.join(self._cache_dir, key))
        }
        downloading = not self.get_cache_status()
        files = self._source_files(downloading)
        source = _source_of(self._llm_family.model_specs[0])

        def refers_to_file(key: str, name: str) -> bool:
            if name == WHOLE_DIRECTORY or key == WHOLE_DIRECTORY or key == name:
                return True
            # Account for aliases and case-insensitive filesystems without
            # conflating distinct files on a case-sensitive volume.
            with contextlib.suppress(OSError):
                return os.path.samefile(
                    os.path.join(self._cache_dir, key),
                    os.path.join(self._cache_dir, name),
                )
            return False

        # A missing main file does not authorize replacing another recorded
        # source's shards or projector, even when both hubs are declared.
        allowed = [source] if downloading and may_record else declared
        used_sources = [
            value
            for key, value in records.items()
            if any(refers_to_file(key, name) for name in files)
        ]
        recorded = next(
            (value for value in used_sources if value not in allowed),
            None,
        )
        if recorded is None and any(value != used_sources[0] for value in used_sources):
            raise ValueError(
                f"Cannot reuse cache directory {self._cache_dir} for model "
                f"{self._model_name}: files used together were downloaded from "
                f"different sources. Remove that directory if the old files are "
                f"no longer needed, or register the model under another name."
            )
        if recorded is None:
            if downloading and may_record:
                if self._model_format == "ggufv2" and self._gguf_file_names()[2]:
                    for name in files:
                        if os.path.exists(
                            os.path.join(self._cache_dir, name)
                        ) and not any(refers_to_file(key, name) for key in records):
                            raise ValueError(
                                f"Cannot complete split GGUF model {self._model_name}: "
                                f"cached file {name} has no source record. Remove the "
                                f"cache directory {self._cache_dir} if the old files "
                                f"are no longer needed, or register the model under "
                                f"another name."
                            )
                additions = {
                    name: source
                    for name in files
                    if name == WHOLE_DIRECTORY
                    or not os.path.exists(os.path.join(self._cache_dir, name))
                }
                # Existing unrecorded files remain legacy; do not claim them.
                self._record_source({**records, **additions})
                self._reuse_existing_gguf_files = self._model_format == "ggufv2"
            return
        current = (
            "a local model_uri"
            if self._model_uri is not None
            else _describe_source(_source_of(self._llm_family.model_specs[0]))
        )
        raise ValueError(
            f"Model {self._model_name} is registered with {current}, but its cache "
            f"directory {self._cache_dir} holds files downloaded from "
            f"{_describe_source(recorded)}. Remove that directory if the old files "
            f"are no longer needed, or register the model under another name."
        )

    def _record_source(self, records: Dict[str, Dict[str, Any]]) -> None:
        record_path = self._source_record_path()
        os.makedirs(os.path.dirname(record_path), exist_ok=True)
        temp_path = f"{record_path}.{os.getpid()}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as fd:
                json.dump(records, fd)
            os.replace(temp_path, record_path)
        finally:
            with contextlib.suppress(OSError):
                os.remove(temp_path)

    def _reuse_gguf_file(self, file_name: str) -> bool:
        # Source checks already covered existing recorded files. Preserve
        # unrecorded legacy files rather than overwrite and misattribute them.
        return self._reuse_existing_gguf_files and os.path.exists(
            os.path.join(self._cache_dir, file_name)
        )

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
        from ..utils import is_remote_uri, link_model_uri, resolve_model_uri

        cache_dir = self.get_cache_dir()
        assert self._model_uri is not None

        if is_remote_uri(self._model_uri) and os.path.exists(cache_dir):
            # Resolving a remote URI costs a network round trip; skip it.
            logger.info(f"Cache {cache_dir} exists")
            return cache_dir

        src_root = resolve_model_uri(self._model_uri, self._model_name)

        if os.path.islink(cache_dir):
            # ``os.path.exists`` follows the link, so a *dangling* symlink
            # (its target was removed) reports ``False`` while the link entry
            # itself still occupies ``cache_dir`` -- the ``os.symlink`` call
            # below would then raise ``FileExistsError``. Reuse the link if it
            # already points at ``src_root``; otherwise drop the stale entry so
            # we can recreate it.
            if os.path.realpath(cache_dir) == os.path.realpath(src_root):
                logger.info(f"Cache {cache_dir} exists")
                return cache_dir
            logger.info(f"Removing stale cache symlink {cache_dir}")
            os.unlink(cache_dir)
        elif os.path.exists(cache_dir):
            logger.info(f"Cache {cache_dir} exists")
            return cache_dir

        # The cache prefix directory is only created once per process (guarded
        # by ``CacheManager.is_initialized``), so it may be missing here; ensure
        # the parent exists before symlinking to avoid a cryptic
        # ``FileNotFoundError: src -> dst``.
        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
        return link_model_uri(src_root, cache_dir)

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
            retry_snapshot_download,
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
            download_dir = retry_snapshot_download(
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
                if self._reuse_gguf_file(file_name):
                    continue
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

            if need_merge and not self._reuse_gguf_file(final_file_name):
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
            retry_snapshot_download,
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
            download_dir = retry_snapshot_download(
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
                if self._reuse_gguf_file(filename):
                    continue
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

            if need_merge and not self._reuse_gguf_file(final_file_name):
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
                if self._reuse_gguf_file(filename):
                    continue
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

            if need_merge and not self._reuse_gguf_file(final_file_name):
                merge_cached_files(cache_dir, file_names, final_file_name)
        else:
            raise ValueError(f"Unsupported format: {self._model_format}")

        return cache_dir

    def _cache_lock_key(self, lock_dir: str) -> str:
        # Parent identity plus filename stays stable before the destination exists.
        parent, name = os.path.split(os.path.realpath(self._cache_dir))
        insensitive = True  # Conservatively serialize aliases if probing fails.
        try:
            with contextlib.ExitStack() as stack:
                # The permanent lock directory is an entry inside the parent whose
                # case sensitivity governs the destination's name, even on macOS.
                probe = lock_dir
                if not os.path.samefile(parent, os.path.dirname(lock_dir)):
                    probe = stack.enter_context(
                        tempfile.TemporaryDirectory(prefix=".case-", dir=parent)
                    )
                alias = os.path.join(parent, os.path.basename(probe).swapcase())
                insensitive = os.path.exists(alias) and os.path.samefile(probe, alias)
        except OSError:
            pass
        if insensitive:
            name = name.casefold()
        identity = os.stat(parent)
        return hashlib.sha256(
            f"{identity.st_dev}:{identity.st_ino}:{name}".encode()
        ).hexdigest()

    def cache(self) -> str:
        if self._model_uri is not None:
            self._check_source(may_record=False)
            return self.cache_uri()
        # SDK downloads share a repository-level temporary directory even
        # across quantizations/revisions. Serialize cache checks and downloads
        # for that repository across replica threads and worker processes.
        from filelock import FileLock

        key = hashlib.sha256(f"{self._model_hub}:{self._model_id}".encode()).hexdigest()
        lock_dir = os.path.join(self._v2_cache_dir_prefix, ".download-locks")
        os.makedirs(lock_dir, exist_ok=True)
        cache_key = self._cache_lock_key(lock_dir)
        # Different repositories can share a destination. Keep its record and
        # cache-status check serialized through the entire download as well.
        with (
            FileLock(os.path.join(lock_dir, key + ".lock")),
            FileLock(os.path.join(lock_dir, f"cache-{cache_key}.lock")),
        ):
            self._check_source(may_record=True)
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
