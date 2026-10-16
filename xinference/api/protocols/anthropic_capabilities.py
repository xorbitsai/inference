# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Capability planning helpers for the Anthropic compatibility layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Optional, Tuple


class SupportLevel(str, Enum):
    NATIVE = "native"
    TRANSLATED = "translated"
    EMULATED = "emulated"
    DEGRADED = "degraded"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class AnthropicBackendCapabilities:
    revision: int = 1
    engine: str = "unknown"
    text: SupportLevel = SupportLevel.NATIVE
    vision: SupportLevel = SupportLevel.UNSUPPORTED
    documents: SupportLevel = SupportLevel.EMULATED
    tools: SupportLevel = SupportLevel.UNSUPPORTED
    parallel_tools: SupportLevel = SupportLevel.UNSUPPORTED
    reasoning: SupportLevel = SupportLevel.UNSUPPORTED
    adaptive_thinking: SupportLevel = SupportLevel.UNSUPPORTED
    reasoning_effort: SupportLevel = SupportLevel.DEGRADED
    structured_output: SupportLevel = SupportLevel.UNSUPPORTED
    prompt_cache: SupportLevel = SupportLevel.DEGRADED
    context_management: SupportLevel = SupportLevel.EMULATED
    max_images_per_prompt: Optional[int] = None
    supported_context_edits: FrozenSet[str] = frozenset({"clear_thinking_20251015"})


@dataclass(frozen=True)
class CompatibilityDecision:
    feature: str
    action: str
    source: str
    reason: str


@dataclass(frozen=True)
class CompatibilityPlan:
    allowed: bool
    decisions: Tuple[CompatibilityDecision, ...]
    effective_max_images: Optional[int]


def plan_anthropic_compatibility(
    requested_features: FrozenSet[str],
    content_metrics: Dict[str, int],
    capabilities: AnthropicBackendCapabilities,
) -> CompatibilityPlan:
    decisions = []
    total_images = content_metrics.get("total_images", 0)
    if total_images and capabilities.vision == SupportLevel.UNSUPPORTED:
        decisions.append(
            CompatibilityDecision(
                feature="vision",
                action="rejected",
                source=capabilities.engine,
                reason="backend_does_not_support_vision",
            )
        )
    limit = capabilities.max_images_per_prompt
    if limit is not None and total_images > limit:
        decisions.append(
            CompatibilityDecision(
                feature="vision.max_images_per_prompt",
                action="rejected",
                source=capabilities.engine,
                reason=f"requested_{total_images}_limit_{limit}",
            )
        )

    feature_support = {
        "documents": capabilities.documents,
        "tools": capabilities.tools,
        "parallel_tools": capabilities.parallel_tools,
        "reasoning": capabilities.reasoning,
        "adaptive_thinking": capabilities.adaptive_thinking,
        "reasoning_effort": capabilities.reasoning_effort,
        "structured_output": capabilities.structured_output,
        "prompt_cache": capabilities.prompt_cache,
        "context_management": capabilities.context_management,
    }
    for feature, support in feature_support.items():
        if feature not in requested_features or support == SupportLevel.NATIVE:
            continue
        action = "rejected" if support == SupportLevel.UNSUPPORTED else support.value
        decisions.append(
            CompatibilityDecision(
                feature=feature,
                action=action,
                source=capabilities.engine,
                reason=f"backend_support_is_{support.value}",
            )
        )
    return CompatibilityPlan(
        allowed=not any(decision.action == "rejected" for decision in decisions),
        decisions=tuple(decisions),
        effective_max_images=limit,
    )
