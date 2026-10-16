"""Checkpoint capability contract for public Ming image inference.

New packages declare their capability in ``transformer/config.json`` via the
``alignment_padding_mode`` / ``multi_frame_output`` pair; the VAE contract is
derived from ``vae/config.json``. Legacy packages that predate the component
metadata are loaded strictly from the root ``inference_profile.json`` during
the compatibility window. The contract is intentionally strict: selecting
behavior from directory names, missing state-dict keys, or task names can
silently load the wrong padding semantics and produce degraded images.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Union


PROFILE_FILENAME = "inference_profile.json"
PROFILE_SCHEMA_VERSION = 1

GENERATION_PROFILE = "generation_edit"
LAYER_PROFILE = "layer_decompose"
VALID_PROFILES = {GENERATION_PROFILE, LAYER_PROFILE}

LEARNED_PADDING = "learned"
ZERO_MASKED_PADDING = "zero_masked"
VALID_PADDING_MODES = {LEARNED_PADDING, ZERO_MASKED_PADDING}

VALID_VAE_SAMPLE_MODES = {"sample", "argmax"}
VALID_TASKS = {"text-to-image", "image-edit", "layer-decompose"}

REQUIRED_PROFILE_FIELDS = {
    "schema_version",
    "inference_profile",
    "alignment_padding_mode",
    "multi_frame_output",
    "vae_input_channels",
    "vae_sample_mode",
}


class InferenceProfileError(ValueError):
    """Raised when a checkpoint does not satisfy the inference contract."""


@dataclass(frozen=True)
class SamplingParameters:
    steps: int
    cfg: float


DEFAULT_SAMPLING_PARAMETERS = {
    GENERATION_PROFILE: SamplingParameters(steps=12, cfg=1.0),
    LAYER_PROFILE: SamplingParameters(steps=12, cfg=2.0),
}


@dataclass(frozen=True)
class InferenceProfile:
    schema_version: int
    inference_profile: str
    alignment_padding_mode: str
    multi_frame_output: bool
    vae_input_channels: int
    vae_sample_mode: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "InferenceProfile":
        missing = sorted(REQUIRED_PROFILE_FIELDS.difference(raw))
        if missing:
            raise InferenceProfileError(
                "checkpoint inference profile is missing required fields: "
                + ", ".join(missing)
            )

        unknown = sorted(set(raw).difference(REQUIRED_PROFILE_FIELDS))
        if unknown:
            raise InferenceProfileError(
                "checkpoint inference profile contains unsupported fields: "
                + ", ".join(unknown)
            )

        if type(raw["schema_version"]) is not int:
            raise InferenceProfileError("schema_version must be an integer")
        if type(raw["multi_frame_output"]) is not bool:
            raise InferenceProfileError("multi_frame_output must be a boolean")
        if type(raw["vae_input_channels"]) is not int:
            raise InferenceProfileError("vae_input_channels must be an integer")

        profile = cls(**{key: raw[key] for key in REQUIRED_PROFILE_FIELDS})
        profile.validate()
        return profile

    def validate(self) -> None:
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise InferenceProfileError(
                f"unsupported profile schema_version={self.schema_version}; "
                f"expected {PROFILE_SCHEMA_VERSION}"
            )
        if self.inference_profile not in VALID_PROFILES:
            raise InferenceProfileError(
                f"inference_profile must be one of {sorted(VALID_PROFILES)}, "
                f"got {self.inference_profile!r}"
            )
        if self.alignment_padding_mode not in VALID_PADDING_MODES:
            raise InferenceProfileError(
                "alignment_padding_mode must be one of "
                f"{sorted(VALID_PADDING_MODES)}, got {self.alignment_padding_mode!r}"
            )
        if self.vae_input_channels not in (3, 4):
            raise InferenceProfileError(
                f"vae_input_channels must be 3 or 4, got {self.vae_input_channels}"
            )
        if self.vae_sample_mode not in VALID_VAE_SAMPLE_MODES:
            raise InferenceProfileError(
                f"vae_sample_mode must be one of {sorted(VALID_VAE_SAMPLE_MODES)}, "
                f"got {self.vae_sample_mode!r}"
            )

        if self.inference_profile == GENERATION_PROFILE:
            if self.alignment_padding_mode != ZERO_MASKED_PADDING:
                raise InferenceProfileError(
                    "generation_edit checkpoints must use zero_masked alignment padding"
                )
            if self.multi_frame_output:
                raise InferenceProfileError(
                    "generation_edit checkpoints must set multi_frame_output=false"
                )
            if self.vae_input_channels != 4:
                raise InferenceProfileError(
                    "generation_edit checkpoints must declare vae_input_channels=4"
                )
            if self.vae_sample_mode != "argmax":
                raise InferenceProfileError(
                    "generation_edit checkpoints must declare vae_sample_mode='argmax'"
                )
        else:
            if self.alignment_padding_mode != LEARNED_PADDING:
                raise InferenceProfileError(
                    "layer_decompose checkpoints must use learned alignment padding"
                )
            if not self.multi_frame_output:
                raise InferenceProfileError(
                    "layer_decompose checkpoints must set multi_frame_output=true"
                )
            if self.vae_input_channels != 4:
                raise InferenceProfileError(
                    "layer_decompose checkpoints must declare vae_input_channels=4"
                )
            if self.vae_sample_mode != "argmax":
                raise InferenceProfileError(
                    "layer_decompose checkpoints must declare vae_sample_mode='argmax'"
                )

    def validate_task(
        self,
        task: str,
        *,
        has_reference_image: bool,
        num_layers: int = 1,
    ) -> None:
        if task not in VALID_TASKS:
            raise InferenceProfileError(
                f"task must be one of {sorted(VALID_TASKS)}, got {task!r}"
            )
        if num_layers < 1:
            raise InferenceProfileError("num_layers must be at least 1")

        if task == "layer-decompose":
            if self.inference_profile != LAYER_PROFILE:
                raise InferenceProfileError(
                    "layer-decompose requires a layer_decompose checkpoint"
                )
            if not has_reference_image:
                raise InferenceProfileError(
                    "layer-decompose requires an input reference image"
                )
            return

        if self.inference_profile != GENERATION_PROFILE:
            raise InferenceProfileError(
                f"{task} requires a generation_edit checkpoint"
            )
        if num_layers != 1:
            raise InferenceProfileError(
                f"{task} does not support num_layers={num_layers}; expected 1"
            )
        if task == "text-to-image" and has_reference_image:
            raise InferenceProfileError("text-to-image does not accept a reference image")
        if task == "image-edit" and not has_reference_image:
            raise InferenceProfileError("image-edit requires a reference image")

    def resolve_sampling_parameters(
        self,
        *,
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
    ) -> SamplingParameters:
        defaults = DEFAULT_SAMPLING_PARAMETERS[self.inference_profile]
        resolved_steps = defaults.steps if steps is None else steps
        resolved_cfg = defaults.cfg if cfg is None else cfg

        if type(resolved_steps) is not int or resolved_steps < 1:
            raise InferenceProfileError("sampling steps must be an integer >= 1")
        if (
            isinstance(resolved_cfg, bool)
            or not isinstance(resolved_cfg, (int, float))
            or not math.isfinite(float(resolved_cfg))
            or resolved_cfg < 0
        ):
            raise InferenceProfileError("CFG must be a finite number >= 0")

        return SamplingParameters(
            steps=resolved_steps,
            cfg=float(resolved_cfg),
        )


def load_inference_profile(model_directory: Union[str, Path]) -> InferenceProfile:
    """Legacy parser: strict root ``inference_profile.json`` (see module doc)."""

    model_directory = Path(model_directory)
    profile_path = model_directory / PROFILE_FILENAME
    if not profile_path.is_file():
        raise InferenceProfileError(
            f"checkpoint must contain {PROFILE_FILENAME}: {profile_path}"
        )
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InferenceProfileError(
            f"failed to read checkpoint inference profile {profile_path}: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise InferenceProfileError("checkpoint inference profile must be a JSON object")
    return InferenceProfile.from_dict(raw)


TRANSFORMER_CONFIG_FILENAME = "transformer/config.json"
VAE_CONFIG_FILENAME = "vae/config.json"
CAPABILITY_FIELDS = ("alignment_padding_mode", "multi_frame_output")
QWEN_VAE_CLASS_NAME = "AutoencoderKLQwenImage"

# The only valid capability pairs and the runtime family they select.
CAPABILITY_PROFILES = {
    (ZERO_MASKED_PADDING, False): GENERATION_PROFILE,
    (LEARNED_PADDING, True): LAYER_PROFILE,
}


def _read_component_config(model_directory: Path, relative: str) -> Mapping[str, Any]:
    config_path = model_directory / relative
    if not config_path.is_file():
        raise InferenceProfileError(
            f"checkpoint is missing component config {relative}: {config_path}"
        )
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InferenceProfileError(
            f"failed to read component config {config_path}: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise InferenceProfileError(f"component config {relative} must be a JSON object")
    return raw


def _derive_vae_contract(model_directory: Path) -> tuple[int, str]:
    """(vae_input_channels, vae_sample_mode) from the VAE component config."""

    config = _read_component_config(model_directory, VAE_CONFIG_FILENAME)
    class_name = config.get("_class_name")
    if class_name != QWEN_VAE_CLASS_NAME:
        raise InferenceProfileError(
            f"unsupported VAE contract: {VAE_CONFIG_FILENAME} _class_name must be "
            f"{QWEN_VAE_CLASS_NAME!r} for argmax reference encoding, got "
            f"{class_name!r}"
        )
    declared = [
        config[key]
        for key in ("input_channels", "in_channels")
        if key in config
    ]
    if not declared:
        raise InferenceProfileError(
            f"{VAE_CONFIG_FILENAME} must declare input_channels or in_channels"
        )
    if len(set(declared)) != 1:
        raise InferenceProfileError(
            f"{VAE_CONFIG_FILENAME} input_channels and in_channels disagree: "
            f"{declared}"
        )
    channels = declared[0]
    if type(channels) is not int or channels != 4:
        raise InferenceProfileError(
            "the supported public families require a 4-channel VAE, got "
            f"{channels!r} in {VAE_CONFIG_FILENAME}"
        )
    return channels, "argmax"


def _profile_from_components(
    model_directory: Path,
    alignment_padding_mode: Any,
    multi_frame_output: Any,
) -> InferenceProfile:
    if type(alignment_padding_mode) is not str:
        raise InferenceProfileError(
            f"{TRANSFORMER_CONFIG_FILENAME} alignment_padding_mode must be a "
            f"string, got {alignment_padding_mode!r}"
        )
    if type(multi_frame_output) is not bool:
        raise InferenceProfileError(
            f"{TRANSFORMER_CONFIG_FILENAME} multi_frame_output must be a "
            f"boolean, got {multi_frame_output!r}"
        )
    pair = (alignment_padding_mode, multi_frame_output)
    if pair not in CAPABILITY_PROFILES:
        raise InferenceProfileError(
            f"unsupported capability pair {pair!r} in "
            f"{TRANSFORMER_CONFIG_FILENAME}; expected one of "
            f"{sorted(CAPABILITY_PROFILES)}"
        )
    channels, sample_mode = _derive_vae_contract(model_directory)
    profile = InferenceProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        inference_profile=CAPABILITY_PROFILES[pair],
        alignment_padding_mode=alignment_padding_mode,
        multi_frame_output=multi_frame_output,
        vae_input_channels=channels,
        vae_sample_mode=sample_mode,
    )
    profile.validate()
    return profile


def load_checkpoint_capabilities(
    model_directory: Union[str, Path],
) -> InferenceProfile:
    """Derive the runtime capability from component configs.

    New packages declare ``alignment_padding_mode`` and
    ``multi_frame_output`` in ``transformer/config.json``; legacy packages
    without them fall back to the strict root ``inference_profile.json``.
    A missing ``transformer/config.json`` means both fields are absent, so
    the legacy path applies. A partial pair is a hard error. When both the
    component fields and the legacy file exist, the component metadata is
    authoritative and the legacy file may only agree with it.
    """

    model_directory = Path(model_directory)
    transformer_path = model_directory / TRANSFORMER_CONFIG_FILENAME
    transformer_config = (
        _read_component_config(model_directory, TRANSFORMER_CONFIG_FILENAME)
        if transformer_path.is_file()
        else {}
    )
    present = [field for field in CAPABILITY_FIELDS if field in transformer_config]
    if len(present) == 1:
        raise InferenceProfileError(
            f"{TRANSFORMER_CONFIG_FILENAME} carries only {present[0]!r}; "
            "alignment_padding_mode and multi_frame_output must be declared "
            "together"
        )
    if not present:
        return load_inference_profile(model_directory)

    profile = _profile_from_components(
        model_directory,
        transformer_config["alignment_padding_mode"],
        transformer_config["multi_frame_output"],
    )
    legacy_path = model_directory / PROFILE_FILENAME
    if legacy_path.is_file():
        legacy = load_inference_profile(model_directory)
        if legacy != profile:
            raise InferenceProfileError(
                f"component configs disagree with legacy {PROFILE_FILENAME}: "
                f"transformer/vae derive {profile!r} but the profile declares "
                f"{legacy!r}"
            )
    return profile


def resolve_model_directory(
    model_name_or_path: Union[str, Path],
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
) -> Path:
    """Resolve a local directory or materialize a Hugging Face Hub snapshot."""

    candidate = Path(model_name_or_path).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    if candidate.exists():
        raise ValueError(f"model path must be a directory: {candidate}")
    if candidate.is_absolute():
        raise FileNotFoundError(f"local model directory does not exist: {candidate}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required when --model is a Hub repository ID"
        ) from error

    snapshot_path = snapshot_download(
        repo_id=str(model_name_or_path),
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_files_only=local_files_only,
        token=token,
    )
    return Path(snapshot_path).resolve()
