from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoConfig, AutoModel

from sensenova_u1 import check_checkpoint_compatibility
from sensenova_u1.models.neo_unify.transformers_compat import pretrained_dtype_kwargs


@dataclass(frozen=True)
class GroupRule:
    name: str
    prefixes: tuple[str, ...] = ()
    contains: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParamEntry:
    name: str
    numel: int
    dtype: str
    bytes: int


@dataclass(frozen=True)
class ParamGroupStat:
    name: str
    params: int
    trainable_params: int
    bytes: int
    entries: tuple[ParamEntry, ...]


@dataclass(frozen=True)
class ParamCountResult:
    model_path: str
    total_params: int
    trainable_params: int
    total_bytes: int
    groups: tuple[ParamGroupStat, ...]


# NOTE on architecture (SenseNova-U1, MoT):
#   * vision_model.*                       -> visual und.
#   * fm_modules.*                         -> generation-only (visual gen., fm_head, timestep/noise embedders)
#   * language_model.* w/ "_mot_gen"       -> generation expert inside the LLM backbone
#   * language_model.* w/o "_mot_gen"      -> understanding expert inside the LLM backbone
#   * language_model.model.embed_tokens.*  -> token input embedding, used by every text token in both
#                                             pathways (image-gen still embeds the text prompt)
#   * language_model.lm_head.*             -> text-token output projection. Also exercised by the
#                                             generation pathway because t2i-reasoning runs a
#                                             thinking phase that emits text tokens before image
#                                             tokens. Hence both belong to the "shared" group.
DEFAULT_GROUPS: tuple[GroupRule, ...] = (
    GroupRule("generation_transformer", prefixes=("fm_modules",)),
    GroupRule(
        "generation_transformer",
        prefixes=("language_model",),
        contains=("_mot_gen",),
    ),
    GroupRule(
        "shared",
        prefixes=(
            "language_model.model.embed_tokens",
            "language_model.lm_head",
        ),
    ),
    GroupRule("understanding_transformer", prefixes=("vision_model",)),
    GroupRule("understanding_transformer", prefixes=("language_model",)),
)


def format_param_count(n: int) -> str:
    """Format a parameter count using SI suffixes (B = Billion = 1e9)."""
    units = (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000))
    for suffix, base in units:
        if abs(n) >= base:
            return f"{n / base:.3f}{suffix}"
    return str(n)


def format_bytes(n: int) -> str:
    """Format a byte count in decimal units (GB = 1e9 bytes), matching SI."""
    units = (("GB", 1_000_000_000), ("MB", 1_000_000), ("KB", 1_000))
    for suffix, base in units:
        if abs(n) >= base:
            return f"{n / base:.3f}{suffix}"
    return f"{n}B"


def build_rules(custom_groups_json: str | None = None) -> tuple[GroupRule, ...]:
    if not custom_groups_json:
        return DEFAULT_GROUPS

    with open(custom_groups_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("custom group config must be a JSON object")

    rules: list[GroupRule] = []
    for group_name, prefixes in raw.items():
        if not isinstance(group_name, str):
            raise ValueError("group name must be string")
        if not isinstance(prefixes, list) or not all(isinstance(x, str) for x in prefixes):
            raise ValueError(f"group '{group_name}' prefixes must be list[str]")
        rules.append(GroupRule(group_name, tuple(prefixes)))
    return tuple(rules)


def _rule_matches(rule: GroupRule, param_name: str) -> bool:
    if rule.prefixes and not any(param_name.startswith(p) for p in rule.prefixes):
        return False
    if rule.contains and not any(c in param_name for c in rule.contains):
        return False
    if rule.excludes and any(e in param_name for e in rule.excludes):
        return False
    return bool(rule.prefixes or rule.contains)


def infer_group(param_name: str, rules: Iterable[GroupRule]) -> str:
    for rule in rules:
        if _rule_matches(rule, param_name):
            return rule.name

    lowered = param_name.lower()
    if "embed" in lowered or "embedding" in lowered:
        return "embedding_misc"
    return "other"


class ModelParamInspector:
    def __init__(
        self,
        model_path: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_path = model_path
        config = AutoConfig.from_pretrained(model_path)
        check_checkpoint_compatibility(config)
        self.model = AutoModel.from_pretrained(model_path, config=config, **pretrained_dtype_kwargs(dtype))

    def count(self, rules: Iterable[GroupRule]) -> ParamCountResult:
        total_params = 0
        trainable_params = 0
        total_bytes = 0
        group_to_params: dict[str, int] = {}
        group_to_trainable: dict[str, int] = {}
        group_to_bytes: dict[str, int] = {}
        group_to_entries: dict[str, list[ParamEntry]] = {}
        seen_param_ids: set[int] = set()

        for name, param in self.model.named_parameters():
            param_id = id(param)
            if param_id in seen_param_ids:
                continue
            seen_param_ids.add(param_id)

            numel = int(param.numel())
            # element_size() reflects the actual per-element byte width of this
            # parameter, which is robust to mixed-dtype checkpoints (e.g. norms
            # forced to fp32 even when the rest is loaded as bf16).
            nbytes = numel * param.element_size()
            total_params += numel
            total_bytes += nbytes
            if param.requires_grad:
                trainable_params += numel

            group = infer_group(name, rules)
            group_to_params[group] = group_to_params.get(group, 0) + numel
            group_to_bytes[group] = group_to_bytes.get(group, 0) + nbytes
            group_to_entries.setdefault(group, []).append(
                ParamEntry(
                    name=name,
                    numel=numel,
                    dtype=str(param.dtype).replace("torch.", ""),
                    bytes=nbytes,
                )
            )
            if param.requires_grad:
                group_to_trainable[group] = group_to_trainable.get(group, 0) + numel

        groups = tuple(
            ParamGroupStat(
                name=k,
                params=v,
                trainable_params=group_to_trainable.get(k, 0),
                bytes=group_to_bytes.get(k, 0),
                entries=tuple(sorted(group_to_entries.get(k, []), key=lambda e: e.numel, reverse=True)),
            )
            for k, v in sorted(group_to_params.items(), key=lambda x: x[1], reverse=True)
        )
        return ParamCountResult(
            model_path=self.model_path,
            total_params=total_params,
            trainable_params=trainable_params,
            total_bytes=total_bytes,
            groups=groups,
        )
