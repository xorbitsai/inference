from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .codec import patchify_latent
from .duration import build_duration_features
from .tokenizer import PretrainedTextTokenizer

_MANIFEST_INDEX_CACHE_VERSION = 4


def _caption_candidates(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, list):
        candidates: list[str] = []
        for item in raw:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                candidates.append(text)
        return candidates
    text = str(raw).strip()
    return [text] if text else []


def _has_caption(raw: Any) -> bool:
    return bool(_caption_candidates(raw))


def _select_caption(raw: Any) -> str:
    candidates = _caption_candidates(raw)
    if not candidates:
        return ""
    return random.choice(candidates)


def _coerce_latent_shape(latent: torch.Tensor, latent_dim: int) -> torch.Tensor:
    """
    Normalize latent tensor to (T, D).
    Accepts common layouts: (T, D), (D, T), (1, T, D), (1, D, T).
    """
    if latent.ndim == 3 and latent.shape[0] == 1:
        latent = latent[0]
    if latent.ndim != 2:
        raise ValueError(f"Unsupported latent shape: {tuple(latent.shape)}")

    if latent.shape[1] == latent_dim:
        return latent
    if latent.shape[0] == latent_dim:
        return latent.transpose(0, 1).contiguous()
    raise ValueError(
        f"Could not infer latent layout for shape={tuple(latent.shape)} and latent_dim={latent_dim}"
    )


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return int(value)
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


class LatentTextDataset(Dataset):
    """
    Manifest format (JSONL), one sample per line:
      {"text": "...", "latent_path": "path/to/latent.pt", "speaker_id": "..."}
    """

    def __init__(
        self,
        manifest_path: str | Path,
        latent_dim: int,
        max_latent_steps: int | None = None,
        subset_indices: list[int] | None = None,
        enable_caption_condition: bool = False,
        enable_speaker_condition: bool = True,
        caption_key: str = "caption",
        manifest_index: _ManifestIndex | None = None,
        show_manifest_progress: bool = False,
        manifest_progress_desc: str | None = None,
        ref_min_frames: int | None = None,
        ref_max_frames: int | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.manifest_dir = self.manifest_path.parent
        self.latent_dim = latent_dim
        self.max_latent_steps = max_latent_steps
        self.enable_caption_condition = bool(enable_caption_condition)
        self.enable_speaker_condition = bool(enable_speaker_condition)
        self.caption_key = str(caption_key)
        if ref_min_frames is not None and ref_max_frames is not None:
            if int(ref_min_frames) <= 0 or int(ref_max_frames) <= 0:
                raise ValueError(
                    "ref_min_frames and ref_max_frames must be positive when set: "
                    f"got {ref_min_frames=} {ref_max_frames=}"
                )
            if int(ref_min_frames) > int(ref_max_frames):
                raise ValueError(
                    f"ref_min_frames ({ref_min_frames}) must be <= ref_max_frames ({ref_max_frames})"
                )
            self.ref_min_frames: int | None = int(ref_min_frames)
            self.ref_max_frames: int | None = int(ref_max_frames)
        else:
            self.ref_min_frames = None
            self.ref_max_frames = None
        self._manifest_fp = None
        subset_index_tensor: torch.Tensor | None = None
        if subset_indices is not None:
            if isinstance(subset_indices, torch.Tensor):
                subset_index_tensor = subset_indices.detach().to(device="cpu", dtype=torch.int64)
            else:
                subset_index_tensor = torch.tensor(
                    [int(x) for x in subset_indices],
                    dtype=torch.int64,
                )
            subset_index_tensor = subset_index_tensor.contiguous()
            if subset_index_tensor.numel() == 0:
                raise ValueError("subset_indices must contain at least one index.")

        if manifest_index is None:
            manifest_index = _ManifestIndex.build(
                manifest_path=self.manifest_path,
                caption_key=self.caption_key,
                show_progress=show_manifest_progress,
                progress_desc=manifest_progress_desc,
            )
        elif manifest_index.caption_key != self.caption_key:
            raise ValueError(
                "manifest_index caption_key mismatch: "
                f"expected {self.caption_key!r}, got {manifest_index.caption_key!r}"
            )
        self.manifest_index = manifest_index

        if subset_index_tensor is None:
            self.sample_indices: torch.Tensor | None = None
            self.num_samples = len(self.manifest_index.offsets)
        else:
            max_index = len(self.manifest_index.offsets) - 1
            invalid_mask = (subset_index_tensor < 0) | (subset_index_tensor > max_index)
            if bool(invalid_mask.any()):
                invalid_indices = subset_index_tensor[invalid_mask][:8].tolist()
                raise ValueError(
                    f"subset_indices contain out-of-range values for manifest: {invalid_indices}"
                )
            self.sample_indices = subset_index_tensor
            self.num_samples = int(subset_index_tensor.numel())

        if self.sample_indices is None:
            self.sample_speaker_codes = self.manifest_index.speaker_codes
            sample_has_caption = self.manifest_index.has_caption
            self.sample_num_frames = self.manifest_index.num_frames
        else:
            self.sample_speaker_codes = self.manifest_index.speaker_codes[self.sample_indices]
            sample_has_caption = self.manifest_index.has_caption[self.sample_indices]
            self.sample_num_frames = self.manifest_index.num_frames[self.sample_indices]

        valid_speaker_mask = self.sample_speaker_codes >= 0
        self.speaker_labeled_count = int(valid_speaker_mask.sum())
        self.caption_labeled_count = int(sample_has_caption.sum())
        self.speaker_group_offsets: torch.Tensor | None = None
        self.speaker_group_indices: torch.Tensor | None = None
        if self.enable_speaker_condition and self.speaker_labeled_count > 0:
            valid_local_indices = torch.nonzero(valid_speaker_mask, as_tuple=False).flatten()
            valid_codes = self.sample_speaker_codes[valid_local_indices]
            if self.sample_indices is None:
                valid_sample_indices = valid_local_indices
            else:
                valid_sample_indices = self.sample_indices[valid_local_indices]
            num_speakers = len(self.manifest_index.speakers)
            counts = torch.bincount(valid_codes, minlength=num_speakers).to(dtype=torch.int64)
            offsets = torch.empty(num_speakers + 1, dtype=torch.int64)
            offsets[0] = 0
            offsets[1:] = torch.cumsum(counts, dim=0)
            order = torch.argsort(valid_codes, stable=True)
            self.speaker_group_offsets = offsets.contiguous()
            self.speaker_group_indices = valid_sample_indices[order].contiguous()

        if self.num_samples <= 0:
            raise ValueError(f"No valid samples in manifest: {self.manifest_path}")

    def _sample_index(self, index: int) -> int:
        if self.sample_indices is None:
            return int(index)
        return int(self.sample_indices[index])

    def _resolve_latent_path(self, latent_path_raw: str) -> Path:
        latent_path = Path(latent_path_raw).expanduser()
        if not latent_path.is_absolute():
            latent_path = (self.manifest_dir / latent_path).resolve()
        return latent_path

    def _load_latent(self, latent_path_raw: str) -> torch.Tensor:
        latent_path = self._resolve_latent_path(latent_path_raw)
        latent = torch.load(latent_path, map_location="cpu", weights_only=True)
        latent = _coerce_latent_shape(latent, self.latent_dim).float()
        if self.max_latent_steps is not None:
            latent = latent[: self.max_latent_steps]
        return latent

    def _load_ref_latent(self, latent_path_raw: str) -> torch.Tensor:
        """
        Load a latent for reference use. Does not apply ``max_latent_steps`` clamp
        (that limit is a target-side constraint; reference latents may be longer,
        e.g., for 120s concat).
        """
        latent_path = self._resolve_latent_path(latent_path_raw)
        latent = torch.load(latent_path, map_location="cpu", weights_only=True)
        return _coerce_latent_shape(latent, self.latent_dim).float()

    def _sample_ref_concat_indices(
        self,
        target_sample_index: int,
        group_start: int,
        group_end: int,
        ref_min_frames: int,
        ref_max_frames: int,
    ) -> list[int]:
        """
        Pick same-speaker sibling sample indices to concatenate for a long reference.

        Behavior:
        - Exclude the target itself.
        - Skip candidates whose manifest ``num_frames`` is <= 0.
        - Sample a target length uniformly in ``[ref_min_frames, ref_max_frames]``
          and greedily add shuffled candidates until the cumulative length reaches
          or exceeds it. The candidate that causes the crossover is included whole
          (not truncated) to avoid unnatural cut-offs during training.
        - If exhausted before reaching the target, return whatever we have.
        - Return an empty list if no usable candidate exists (caller falls back).
        """
        if self.speaker_group_indices is None:
            return []
        candidates = self.speaker_group_indices[group_start:group_end].tolist()
        candidates = [int(c) for c in candidates if int(c) != int(target_sample_index)]
        if not candidates:
            return []
        random.shuffle(candidates)
        target_frames = random.uniform(float(ref_min_frames), float(ref_max_frames))
        chosen: list[int] = []
        total = 0
        for sample_index in candidates:
            n = int(self.manifest_index.num_frames[sample_index])
            if n <= 0:
                continue
            chosen.append(sample_index)
            total += n
            if total >= target_frames:
                break
        return chosen

    def _build_ref_latent(self, ref_sample_indices: list[int]) -> torch.Tensor:
        """
        Load and concatenate reference latents. Only clamps at the hard
        ``ref_max_frames`` upper bound (model receptive limit); does not truncate
        at the randomly sampled target length.
        """
        pieces: list[torch.Tensor] = []
        for sample_index in ref_sample_indices:
            item = self._read_item_by_sample_index(sample_index)
            pieces.append(self._load_ref_latent(item["latent_path"]))
        ref_latent = torch.cat(pieces, dim=0)
        if self.ref_max_frames is not None and ref_latent.shape[0] > self.ref_max_frames:
            ref_latent = ref_latent[: self.ref_max_frames]
        return ref_latent

    def _manifest_file(self):
        if self._manifest_fp is None or self._manifest_fp.closed:
            self._manifest_fp = self.manifest_path.open("r", encoding="utf-8")
        return self._manifest_fp

    def _read_item(self, index: int) -> dict[str, Any]:
        return self._read_item_by_sample_index(self._sample_index(index))

    def _read_item_by_sample_index(self, sample_index: int) -> dict[str, Any]:
        fp = self._manifest_file()
        fp.seek(int(self.manifest_index.offsets[sample_index]))
        line = fp.readline()
        if line == "":
            raise ValueError(
                f"Unexpected EOF while reading manifest sample_index={sample_index}: {self.manifest_path}"
            )
        item = json.loads(line)
        if "text" not in item or "latent_path" not in item:
            raise ValueError(f"Invalid manifest line (needs text and latent_path): {line.rstrip()}")
        return item

    def __len__(self) -> int:
        return self.num_samples

    def length_bucket_values(self) -> torch.Tensor:
        lengths = self.sample_num_frames.detach().to(device="cpu", dtype=torch.int64).clone()
        if self.max_latent_steps is not None:
            lengths.clamp_(max=int(self.max_latent_steps))
        return lengths

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self._read_item(index)
        latent = self._load_latent(item["latent_path"])

        ref_latent: torch.Tensor | None = None
        has_speaker = False
        if self.enable_speaker_condition and self.speaker_group_offsets is not None:
            speaker_code = int(self.sample_speaker_codes[index])
            if speaker_code >= 0:
                group_start = int(self.speaker_group_offsets[speaker_code])
                group_end = int(self.speaker_group_offsets[speaker_code + 1])
                group_size = group_end - group_start
            else:
                group_start = 0
                group_end = 0
                group_size = 0
            if group_size > 1 and self.speaker_group_indices is not None:
                target_sample_index = self._sample_index(index)
                if self.ref_min_frames is not None and self.ref_max_frames is not None:
                    ref_indices = self._sample_ref_concat_indices(
                        target_sample_index=target_sample_index,
                        group_start=group_start,
                        group_end=group_end,
                        ref_min_frames=self.ref_min_frames,
                        ref_max_frames=self.ref_max_frames,
                    )
                    if ref_indices:
                        ref_latent = self._build_ref_latent(ref_indices)
                        has_speaker = True
                if ref_latent is None:
                    candidate_pos = group_start + random.randrange(group_size - 1)
                    ref_sample_index = int(self.speaker_group_indices[candidate_pos])
                    if ref_sample_index == self._sample_index(index):
                        ref_sample_index = int(self.speaker_group_indices[group_end - 1])
                    ref_item = self._read_item_by_sample_index(ref_sample_index)
                    ref_latent = self._load_latent(ref_item["latent_path"])
                    has_speaker = True

        if ref_latent is None:
            ref_latent = latent
        manifest_num_frames = int(item.get("num_frames", latent.shape[0]))
        num_frames = min(manifest_num_frames, int(latent.shape[0]))
        caption = (
            _select_caption(item.get(self.caption_key)) if self.enable_caption_condition else ""
        )

        return {
            "text": item["text"],
            "caption": caption,
            "has_caption": bool(caption) if self.enable_caption_condition else False,
            "latent": latent,
            "num_frames": num_frames,
            "ref_latent": ref_latent,
            "has_speaker": has_speaker,
        }


@dataclass(frozen=True)
class _ManifestIndex:
    offsets: torch.Tensor
    speaker_codes: torch.Tensor
    speakers: list[str]
    has_caption: torch.Tensor
    num_frames: torch.Tensor
    caption_key: str
    cache_version: int = _MANIFEST_INDEX_CACHE_VERSION

    @staticmethod
    def _cache_path(manifest_path: Path, caption_key: str) -> Path:
        suffix = ".irodori_index.pt"
        if caption_key != "caption":
            safe_caption_key = "".join(
                ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in caption_key
            )
            suffix = f".{safe_caption_key}.irodori_index.pt"
        return manifest_path.with_name(manifest_path.name + suffix)

    @staticmethod
    def _load_cache(manifest_path: Path, caption_key: str) -> _ManifestIndex | None:
        cache_path = _ManifestIndex._cache_path(manifest_path, caption_key)
        if not cache_path.exists():
            return None
        stat = manifest_path.stat()
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        version = payload.get("version")
        if version != _MANIFEST_INDEX_CACHE_VERSION:
            return None
        if payload.get("manifest_size") != stat.st_size:
            return None
        if payload.get("manifest_mtime_ns") != stat.st_mtime_ns:
            return None
        if payload.get("caption_key") != caption_key:
            return None

        offsets = payload.get("offsets")
        has_caption = payload.get("has_caption")
        if not isinstance(offsets, torch.Tensor) or offsets.ndim != 1:
            return None
        if not isinstance(has_caption, torch.Tensor) or has_caption.ndim != 1:
            return None
        if offsets.numel() != has_caption.numel():
            return None
        speaker_codes = payload.get("speaker_codes")
        speakers = payload.get("speakers")
        num_frames = payload.get("num_frames")
        if not isinstance(speaker_codes, torch.Tensor) or speaker_codes.ndim != 1:
            return None
        if not isinstance(speakers, list):
            return None
        if not isinstance(num_frames, torch.Tensor) or num_frames.ndim != 1:
            return None
        if offsets.numel() != speaker_codes.numel() or offsets.numel() != num_frames.numel():
            return None
        speakers = [str(x) for x in speakers]
        return _ManifestIndex(
            offsets=offsets.detach().to(device="cpu", dtype=torch.int64).contiguous(),
            speaker_codes=speaker_codes.detach().to(device="cpu", dtype=torch.int64).contiguous(),
            speakers=speakers,
            has_caption=has_caption.detach().to(device="cpu", dtype=torch.bool).contiguous(),
            num_frames=num_frames.detach().to(device="cpu", dtype=torch.int64).contiguous(),
            caption_key=str(caption_key),
            cache_version=_MANIFEST_INDEX_CACHE_VERSION,
        )

    def _save_cache(self, manifest_path: Path) -> None:
        cache_path = self._cache_path(manifest_path, self.caption_key)
        stat = manifest_path.stat()
        payload = {
            "version": _MANIFEST_INDEX_CACHE_VERSION,
            "manifest_size": stat.st_size,
            "manifest_mtime_ns": stat.st_mtime_ns,
            "caption_key": self.caption_key,
            "offsets": self.offsets.detach().to(device="cpu", dtype=torch.int64).contiguous(),
            "speaker_codes": self.speaker_codes.detach()
            .to(device="cpu", dtype=torch.int64)
            .contiguous(),
            "speakers": self.speakers,
            "has_caption": self.has_caption.detach().to(device="cpu", dtype=torch.bool).contiguous(),
            "num_frames": self.num_frames.detach().to(device="cpu", dtype=torch.int64).contiguous(),
        }
        tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        try:
            torch.save(payload, tmp_path)
            tmp_path.replace(cache_path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    @classmethod
    def build(
        cls,
        manifest_path: Path,
        *,
        caption_key: str = "caption",
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> _ManifestIndex:
        caption_key = str(caption_key)
        cached = cls._load_cache(manifest_path, caption_key)
        if cached is not None:
            if show_progress:
                print(f"Loaded manifest index cache: {cls._cache_path(manifest_path, caption_key)}")
            if cached.cache_version != _MANIFEST_INDEX_CACHE_VERSION:
                cached._save_cache(manifest_path)
            return cached

        offsets: list[int] = []
        speaker_codes: list[int] = []
        speaker_id_to_code: dict[str, int] = {}
        speakers: list[str] = []
        has_caption: list[bool] = []
        num_frames: list[int] = []
        total_bytes = manifest_path.stat().st_size
        if progress_desc is None:
            progress_desc = f"Index Manifest ({manifest_path.name})"
        pbar = tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
            desc=progress_desc,
            disable=not show_progress,
            leave=False,
        )
        with manifest_path.open("r", encoding="utf-8") as f:
            try:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if line == "":
                        break
                    pbar.update(len(line.encode("utf-8")))
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if "text" not in item or "latent_path" not in item:
                        raise ValueError(
                            f"Invalid manifest line (needs text and latent_path): {line.rstrip()}"
                        )
                    offsets.append(offset)
                    speaker_id = item.get("speaker_id")
                    if speaker_id is None:
                        speaker_codes.append(-1)
                    else:
                        speaker_id = str(speaker_id)
                        speaker_code = speaker_id_to_code.get(speaker_id)
                        if speaker_code is None:
                            speaker_code = len(speakers)
                            speaker_id_to_code[speaker_id] = speaker_code
                            speakers.append(speaker_id)
                        speaker_codes.append(int(speaker_code))
                    has_caption.append(_has_caption(item.get(caption_key)))
                    num_frames.append(max(0, int(item.get("num_frames", 0) or 0)))
            finally:
                pbar.close()
        if not offsets:
            raise ValueError(f"No valid samples in manifest: {manifest_path}")
        index = cls(
            offsets=torch.tensor(offsets, dtype=torch.int64),
            speaker_codes=torch.tensor(speaker_codes, dtype=torch.int64),
            speakers=speakers,
            has_caption=torch.tensor(has_caption, dtype=torch.bool),
            num_frames=torch.tensor(num_frames, dtype=torch.int64),
            caption_key=caption_key,
        )
        index._save_cache(manifest_path)
        return index


@dataclass
class TTSCollator:
    tokenizer: PretrainedTextTokenizer
    caption_tokenizer: PretrainedTextTokenizer | None
    latent_dim: int
    latent_patch_size: int
    fixed_target_latent_steps: int | None = None
    fixed_target_full_mask: bool = False
    latent_length_bucket_size: int = 0
    max_text_len: int = 256
    max_caption_len: int | None = None

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [x["text"] for x in batch]
        captions = [x["caption"] for x in batch]
        latents = [x["latent"] for x in batch]  # each: (T, D)
        ref_latents = [x["ref_latent"] for x in batch]  # each: (T_ref, D)
        has_speaker = torch.tensor([bool(x["has_speaker"]) for x in batch], dtype=torch.bool)
        has_caption = torch.tensor([bool(x["has_caption"]) for x in batch], dtype=torch.bool)
        bsz = len(latents)

        text_ids, text_mask = self.tokenizer.batch_encode(texts, max_length=self.max_text_len)
        token_counts = text_mask.sum(dim=1)
        caption_ids = None
        caption_mask = None
        if self.caption_tokenizer is not None:
            max_caption_len = self.max_caption_len
            if max_caption_len is None:
                max_caption_len = self.max_text_len
            caption_ids, caption_mask = self.caption_tokenizer.batch_encode(
                captions,
                max_length=max_caption_len,
            )
            caption_mask = caption_mask & has_caption[:, None]

        if self.fixed_target_latent_steps is not None:
            max_t = int(self.fixed_target_latent_steps)
            if max_t <= 0:
                raise ValueError(
                    f"fixed_target_latent_steps must be > 0, got {self.fixed_target_latent_steps}"
                )
        else:
            max_t = max(x.shape[0] for x in latents)
            bucket_size = int(self.latent_length_bucket_size)
            if bucket_size > 0:
                bucket_size = _round_up_to_multiple(bucket_size, self.latent_patch_size)
                max_t = _round_up_to_multiple(max_t, bucket_size)
        latent_batch = torch.zeros((bsz, max_t, self.latent_dim), dtype=torch.float32)
        latent_mask_valid = torch.zeros((bsz, max_t), dtype=torch.bool)
        for i, latent in enumerate(latents):
            n = min(latent.shape[0], max_t)
            latent_batch[i, :n] = latent[:n]
            latent_mask_valid[i, :n] = True
        latent_mask = latent_mask_valid.clone()
        if self.fixed_target_full_mask:
            latent_mask.fill_(True)

        max_ref_t = max(x.shape[0] for x in ref_latents)
        if self.fixed_target_latent_steps is None:
            bucket_size = int(self.latent_length_bucket_size)
            if bucket_size > 0:
                bucket_size = _round_up_to_multiple(bucket_size, self.latent_patch_size)
                max_ref_t = _round_up_to_multiple(max_ref_t, bucket_size)
        ref_batch = torch.zeros((bsz, max_ref_t, self.latent_dim), dtype=torch.float32)
        ref_mask = torch.zeros((bsz, max_ref_t), dtype=torch.bool)
        for i, ref_latent in enumerate(ref_latents):
            n = ref_latent.shape[0]
            ref_batch[i, :n] = ref_latent
            ref_mask[i, :n] = True

        latent_patched = patchify_latent(latent_batch, self.latent_patch_size)
        # Keep reference in latent-patched space. The model applies an extra
        # speaker_patch_size patching internally for speaker conditioning.
        ref_patched = patchify_latent(ref_batch, self.latent_patch_size)

        def _patch_mask(mask: torch.Tensor) -> torch.Tensor:
            if self.latent_patch_size <= 1:
                return mask
            usable = (mask.shape[1] // self.latent_patch_size) * self.latent_patch_size
            return (
                mask[:, :usable]
                .reshape(bsz, usable // self.latent_patch_size, self.latent_patch_size)
                .all(dim=-1)
            )

        latent_mask_patched = _patch_mask(latent_mask)
        latent_mask_valid_patched = _patch_mask(latent_mask_valid)
        ref_mask_patched = _patch_mask(ref_mask)

        out = {
            "text_ids": text_ids,
            "text_mask": text_mask,
            "num_frames": torch.tensor([int(x["num_frames"]) for x in batch], dtype=torch.long),
            "duration_features": build_duration_features(
                texts,
                token_counts=token_counts,
                max_text_len=self.max_text_len,
                has_speaker=has_speaker,
            ),
            "latent": latent_batch,
            "latent_mask": latent_mask,
            "latent_mask_valid": latent_mask_valid,
            "latent_patched": latent_patched,
            "latent_mask_patched": latent_mask_patched,
            "latent_mask_valid_patched": latent_mask_valid_patched,
            "latent_padding_mask_patched": ~latent_mask_valid_patched,
            "ref_latent": ref_batch,
            "ref_latent_mask": ref_mask,
            "ref_latent_patched": ref_patched,
            "ref_latent_mask_patched": ref_mask_patched,
            "has_speaker": has_speaker,
        }
        if caption_ids is not None and caption_mask is not None:
            out["caption_ids"] = caption_ids
            out["caption_mask"] = caption_mask
            out["has_caption"] = has_caption
        return out
