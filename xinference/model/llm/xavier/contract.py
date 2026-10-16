# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Versioned identities for full-attention KV shared by engine adapters.

Adapters must fingerprint the actual weights, tokenizer assets, effective
attention configuration and effective position/RoPE configuration. A model
name, mutable revision or local path alone does not establish compatibility.
Engine names, physical cache slots and local layer names are not identities.

Version 1 supports unquantized FP16 text models with TP=PP=1 and contiguous
positions starting at zero. Each logical layer uses global indices 0..L-1.
Its canonical payload is [blocks, K/V, tokens, KV heads, head dimension], with
K before V, serialized as little-endian uint8 bytes. Physical layout conversion
belongs to the device backend. This protocol does not change vLLM's V1 keys or
enable cross-engine serving on its own.
"""

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar

_T = TypeVar("_T")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def fingerprint_metadata(metadata: Mapping[str, Any]) -> str:
    """Hash effective semantic configuration, independently of dictionary order."""
    return hashlib.sha256(
        b"xavier-metadata-v1\0" + _canonical_json(metadata)
    ).hexdigest()


def _asset_digest(path: Path, cache_dir: Optional[Path]) -> str:
    if cache_dir is None:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    from filelock import FileLock

    resolved = path.resolve()
    stat = resolved.stat()
    identity = [
        str(resolved),
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        stat.st_dev,
        stat.st_ino,
    ]
    key = hashlib.sha256(str(resolved).encode()).hexdigest()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(cache_dir / (key + ".lock"))):
            entry = cache_dir / (key + ".json")
            try:
                cached = json.loads(entry.read_text())
                value = cached["digest"]
                _require_digest("cached asset", value)
                if cached["identity"] == identity:
                    return value
            except (OSError, ValueError, KeyError, TypeError):
                pass
            value = _asset_digest(resolved, None)
            after = resolved.stat()
            if [
                str(resolved),
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_dev,
                after.st_ino,
            ] != identity:
                raise ValueError("Model asset changed while fingerprinting")
            temporary = entry.with_suffix(f".{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps(dict(identity=identity, digest=value)))
                os.replace(temporary, entry)
            finally:
                temporary.unlink(missing_ok=True)
            return value
    except OSError:
        # Read-only cache locations must not prevent loading verified assets.
        return _asset_digest(resolved, None)


def fingerprint_files(
    files: Mapping[str, Path], *, cache_dir: Optional[Path] = None
) -> str:
    """Hash a complete, adapter-selected asset manifest without local root paths.

    Names identify files within the asset set; paths identify where to read them.
    The adapter must include every relevant weight/tokenizer asset and any custom
    implementation affecting the model. Byte-identical manifests compare equal
    after relocation. Different sharding/serialization conservatively differs.
    """
    if not files or any(not isinstance(name, str) or not name for name in files):
        raise ValueError("A nonempty asset manifest with named files is required")
    manifest = {}
    for name, path in sorted(files.items()):
        manifest[name] = _asset_digest(path, cache_dir)
    return fingerprint_metadata(manifest)


def _require_digest(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_integer(name: str, value: int, minimum: int = 1) -> None:
    if type(value) is not int or not minimum <= value < 2**64:
        raise ValueError(f"Invalid {name}")


def _from_dict(cls: Type[_T], metadata: Mapping[str, Any]) -> _T:
    if not isinstance(metadata, Mapping):
        raise ValueError("KV metadata must be a mapping")
    names = {item.name for item in fields(cls)}  # type: ignore[arg-type]
    if set(metadata) != names:
        raise ValueError("Missing or unknown KV metadata fields")
    return cls(**dict(metadata))


@dataclass(frozen=True)
class KVCacheContract:
    weights_fingerprint: str
    tokenizer_fingerprint: str
    attention_fingerprint: str
    position_fingerprint: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    logical_dtype: str
    protocol_version: int = 1
    layout_version: int = 1
    layout: str = "N2THD"
    transport_dtype: str = "uint8"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    attention_kind: str = "full"
    weight_quantization: str = "none"
    is_multimodal: bool = False
    has_lora: bool = False

    def __post_init__(self) -> None:
        for name in (
            "weights_fingerprint",
            "tokenizer_fingerprint",
            "attention_fingerprint",
            "position_fingerprint",
        ):
            _require_digest(name, getattr(self, name))
        for name in ("num_layers", "num_kv_heads", "head_dim", "block_size"):
            _require_integer(name, getattr(self, name))
        for name in (
            "protocol_version",
            "layout_version",
            "tensor_parallel_size",
            "pipeline_parallel_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) != 1:
                raise ValueError(f"Unsupported {name}")
        for name, expected in (
            ("logical_dtype", "float16"),
            ("layout", "N2THD"),
            ("transport_dtype", "uint8"),
            ("attention_kind", "full"),
            ("weight_quantization", "none"),
        ):
            if getattr(self, name) != expected:
                raise ValueError(f"Unsupported {name}")
        if self.is_multimodal is not False or self.has_lora is not False:
            raise ValueError("KV contract requires text-only models without LoRA")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, metadata: Mapping[str, Any]) -> "KVCacheContract":
        return _from_dict(cls, metadata)

    @cached_property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            b"xavier-kv-contract-v1\0" + _canonical_json(self.to_dict())
        ).hexdigest()

    def require_match(self, other: "KVCacheContract") -> None:
        if not isinstance(other, KVCacheContract):
            raise ValueError("Missing peer KV contract")
        mismatches = [
            item.name
            for item in fields(self)
            if getattr(self, item.name) != getattr(other, item.name)
        ]
        if mismatches:
            raise ValueError("Incompatible KV contract: " + ", ".join(mismatches))

    @property
    def layer_nbytes(self) -> int:
        """Bytes per complete block for one logical layer (two FP16 tensors)."""
        return 2 * self.block_size * self.num_kv_heads * self.head_dim * 2


@dataclass(frozen=True)
class KVBlockKey:
    contract_fingerprint: str
    prefix_tokens: int
    digest: str

    def __post_init__(self) -> None:
        _require_digest("contract_fingerprint", self.contract_fingerprint)
        _require_digest("digest", self.digest)
        _require_integer("prefix_tokens", self.prefix_tokens)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, metadata: Mapping[str, Any]) -> "KVBlockKey":
        return _from_dict(cls, metadata)

    @property
    def storage_key(self) -> int:
        """Retain the entire digest when using Xavier's integer-keyed snapshots."""
        return int(self.digest, 16)


@dataclass(frozen=True)
class KVLayerMetadata:
    """Bind an ordered block payload to its logical layer and cache contract."""

    contract: KVCacheContract
    layer_index: int
    keys: Tuple[KVBlockKey, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.contract, KVCacheContract):
            raise ValueError("Missing KV layer contract")
        _require_integer("layer_index", self.layer_index, minimum=0)
        if self.layer_index >= self.contract.num_layers:
            raise ValueError("KV layer index is outside the contract")
        if not isinstance(self.keys, tuple) or not self.keys:
            raise ValueError("KV layer requires an immutable sequence of block keys")
        for key in self.keys:
            if (
                not isinstance(key, KVBlockKey)
                or key.contract_fingerprint != self.contract.fingerprint
                or key.prefix_tokens % self.contract.block_size
            ):
                raise ValueError("KV layer key belongs to an incompatible contract")
        if len({key.digest for key in self.keys}) != len(self.keys):
            raise ValueError("Duplicate KV layer keys")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract": self.contract.to_dict(),
            "layer_index": self.layer_index,
            "keys": [key.to_dict() for key in self.keys],
        }

    @classmethod
    def from_dict(cls, metadata: Mapping[str, Any]) -> "KVLayerMetadata":
        if not isinstance(metadata, Mapping) or set(metadata) != {
            "contract",
            "layer_index",
            "keys",
        }:
            raise ValueError("Missing or unknown KV layer metadata fields")
        if not isinstance(metadata["keys"], list):
            raise ValueError("KV layer keys must be a list")
        return cls(
            KVCacheContract.from_dict(metadata["contract"]),
            metadata["layer_index"],
            tuple(KVBlockKey.from_dict(key) for key in metadata["keys"]),
        )

    def require_match(self, expected: "KVLayerMetadata") -> None:
        self.contract.require_match(expected.contract)
        if self.layer_index != expected.layer_index:
            raise ValueError("KV logical layer differs from the request")
        if self.keys != expected.keys:
            raise ValueError("KV block keys or order differ from the request")


def build_block_key(
    contract: KVCacheContract,
    token_ids: Sequence[int],
    position_ids: Sequence[int],
    previous: Optional[KVBlockKey] = None,
    cache_salt: str = "",
) -> KVBlockKey:
    """Identify one complete block and its causal prefix, independent of engine."""
    if len(token_ids) != contract.block_size or len(position_ids) != len(token_ids):
        raise ValueError("KV keys require one complete block of tokens and positions")
    if not isinstance(cache_salt, str):
        raise ValueError("cache_salt must be a string")
    start = 0
    if previous is not None:
        if (
            previous.contract_fingerprint != contract.fingerprint
            or previous.prefix_tokens % contract.block_size
        ):
            raise ValueError("Previous KV key belongs to an incompatible prefix")
        start = previous.prefix_tokens
    for offset, (token, position) in enumerate(zip(token_ids, position_ids)):
        if type(token) is not int or not 0 <= token < 2**32:
            raise ValueError("Invalid token ID")
        if type(position) is not int or position != start + offset:
            raise ValueError(
                "KV contract requires contiguous positions starting at zero"
            )
    end = start + contract.block_size
    _require_integer("prefix_tokens", end)
    digest = hashlib.sha256(
        b"xavier-kv-block-v1\0"
        + _canonical_json(
            {
                "contract": contract.fingerprint,
                "previous": previous.digest if previous else None,
                "tokens": list(token_ids),
                "positions": list(position_ids),
                "cache_salt": cache_salt,
            }
        )
    ).hexdigest()
    return KVBlockKey(contract.fingerprint, end, digest)


def build_prefix_keys(
    contract: KVCacheContract,
    token_ids: Sequence[int],
    position_ids: Optional[Sequence[int]] = None,
    cache_salt: str = "",
) -> List[KVBlockKey]:
    """Return complete-block keys; leave the partial tail to local computation."""
    if position_ids is None:
        position_ids = range(len(token_ids))
    if len(position_ids) != len(token_ids):
        raise ValueError("Token and position counts differ")
    if not isinstance(cache_salt, str):
        raise ValueError("cache_salt must be a string")
    for offset, (token, position) in enumerate(zip(token_ids, position_ids)):
        if type(token) is not int or not 0 <= token < 2**32:
            raise ValueError("Invalid token ID")
        if type(position) is not int or position != offset:
            raise ValueError(
                "KV contract requires contiguous positions starting at zero"
            )
    keys: List[KVBlockKey] = []
    for start in range(
        0, len(token_ids) - contract.block_size + 1, contract.block_size
    ):
        keys.append(
            build_block_key(
                contract,
                token_ids[start : start + contract.block_size],
                position_ids[start : start + contract.block_size],
                keys[-1] if keys else None,
                cache_salt,
            )
        )
    return keys
