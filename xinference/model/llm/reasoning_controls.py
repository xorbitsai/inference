# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Model-declared reasoning effort and request-local thinking budgets."""

import copy
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from ..._compat import BaseModel, Field, validator


class ReasoningEffortConfig(BaseModel):
    values: List[str]
    integer_range: Optional[Tuple[int, int]] = None
    aliases: Dict[str, int] = Field(default_factory=dict)

    @validator("values")
    def validate_values(cls, value):
        if (
            not value
            or any(not item for item in value)
            or len(set(value)) != len(value)
        ):
            raise ValueError("values must contain distinct, nonempty effort names")
        return value

    @validator("integer_range")
    def validate_range(cls, value):
        if value is not None and not 1 <= value[0] <= value[1]:
            raise ValueError("integer_range must be an increasing positive range")
        return value

    @validator("aliases")
    def validate_aliases(cls, value, values):
        bounds = values.get("integer_range")
        if value and (
            bounds is None
            or any(
                name not in values.get("values", [])
                or not bounds[0] <= number <= bounds[1]
                for name, number in value.items()
            )
        ):
            raise ValueError("aliases must map declared names into integer_range")
        return value

    def normalize(self, value: Any, parameter: str) -> Union[str, int]:
        if isinstance(value, str) and value in self.values:
            return self.aliases.get(value, value)
        if (
            type(value) is int
            and self.integer_range is not None
            and self.integer_range[0] <= value <= self.integer_range[1]
        ):
            return value
        allowed = ", ".join(self.values)
        if self.integer_range is not None:
            allowed += f", or an integer from {self.integer_range[0]} to {self.integer_range[1]}"
        raise ValueError(f"Invalid {parameter}. Supported values are {allowed}.")


def normalize_reasoning_controls(
    config: Dict[str, Any], capability: Optional[ReasoningEffortConfig]
) -> Dict[str, Any]:
    """Validate explicit controls without deriving effort from token limits."""
    config = copy.deepcopy(config)
    context = config.get("chat_template_kwargs") or {}
    if isinstance(context, str):
        context = json.loads(context)
    if not isinstance(context, dict):
        raise ValueError("chat_template_kwargs must be an object")
    has_effort = "reasoning_effort" in config
    has_template_effort = "reasoning_effort" in context
    if has_effort or has_template_effort:
        if capability is None:
            raise NotImplementedError(
                "This model has no declared native reasoning_effort support"
            )
        effort = capability.normalize(
            config["reasoning_effort"] if has_effort else context["reasoning_effort"],
            (
                "reasoning_effort"
                if has_effort
                else "chat_template_kwargs.reasoning_effort"
            ),
        )
        if (
            has_effort
            and has_template_effort
            and effort
            != capability.normalize(
                context["reasoning_effort"], "chat_template_kwargs.reasoning_effort"
            )
        ):
            raise ValueError("Conflicting reasoning_effort values")
        context["reasoning_effort"] = effort
        config.pop("reasoning_effort", None)
    if "thinking_budget" in config:
        budget = config.pop("thinking_budget")
        if "thinking_budget" in context and context["thinking_budget"] != budget:
            raise ValueError("Conflicting thinking_budget values")
        context["thinking_budget"] = budget
    if "thinking_budget" in context:
        budget = context["thinking_budget"]
        if type(budget) is not int or budget < 1:
            raise ValueError("thinking_budget must be a positive integer")
        maximum = config.get("max_tokens")
        if maximum is not None and budget >= maximum:
            raise ValueError("thinking_budget must be less than max_tokens")
        if context.get("enable_thinking") is False or context.get("thinking") is False:
            raise ValueError("thinking_budget requires thinking to be enabled")
        context["enable_thinking"] = True
    if context or "chat_template_kwargs" in config:
        config["chat_template_kwargs"] = context
    return config


def has_reasoning_controls(config: Dict[str, Any]) -> bool:
    if "reasoning_effort" in config or "thinking_budget" in config:
        return True
    context = config.get("chat_template_kwargs") or {}
    if isinstance(context, str):
        context = json.loads(context)
    return isinstance(context, dict) and bool(
        {"reasoning_effort", "thinking_budget"} & context.keys()
    )
