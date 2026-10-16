from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .config import BackendConfig, RouteAction, RoutingRule, RuleMatch
from .tokenizer import TokenBudget


class RouteRejected(ValueError):
    def __init__(self, reason: str, message: str | None = None) -> None:
        super().__init__(message or reason.replace("_", " "))
        self.reason = reason


class ThinkingRejected(RouteRejected):
    """Raised when a Router policy rejects thinking-mode requests."""

    def __init__(
        self, message: str = "Thinking-mode requests are disabled by Router policy"
    ) -> None:
        super().__init__("thinking_not_allowed", message)


class ContextLimitExceeded(RouteRejected):
    def __init__(
        self, total_tokens: int, max_model_len: int, backend_id: str = ""
    ) -> None:
        target = f" for backend {backend_id}" if backend_id else ""
        super().__init__(
            "context_length_exceeded",
            f"Request token budget {total_tokens} exceeds context limit "
            f"{max_model_len}{target}",
        )
        self.total_tokens = total_tokens
        self.max_model_len = max_model_len
        self.backend_id = backend_id


@dataclass(frozen=True)
class RequestProfile:
    budget: TokenBudget
    thinking: bool
    tools_present: bool
    stream: bool

    @property
    def prompt_tokens(self) -> int:
        return self.budget.prompt_tokens

    @property
    def output_tokens(self) -> int:
        return self.budget.output_tokens

    @property
    def reserve_tokens(self) -> int:
        return self.budget.reserve_tokens

    @property
    def total_tokens(self) -> int:
        return self.budget.total_tokens

    @classmethod
    def from_budget(
        cls,
        budget: TokenBudget,
        *,
        tools_present: bool = False,
        stream: bool = False,
    ) -> "RequestProfile":
        return cls(
            budget=budget,
            thinking=budget.enable_thinking,
            tools_present=tools_present,
            stream=stream,
        )


@dataclass(frozen=True)
class RouteDecision:
    backend_id: str
    rule_id: str
    reason: str
    profile: RequestProfile

    @property
    def pool(self) -> str:  # Compatibility header/metrics alias.
        return self.backend_id

    @property
    def budget(self) -> TokenBudget:
        return self.profile.budget


def _matches(match: RuleMatch, profile: RequestProfile) -> bool:
    if (
        match.total_tokens_gte is not None
        and profile.total_tokens < match.total_tokens_gte
    ):
        return False
    if (
        match.total_tokens_lte is not None
        and profile.total_tokens > match.total_tokens_lte
    ):
        return False
    if match.thinking is not None and profile.thinking is not match.thinking:
        return False
    if (
        match.tools_present is not None
        and profile.tools_present is not match.tools_present
    ):
        return False
    if match.stream is not None and profile.stream is not match.stream:
        return False
    return True


class RoutingPolicy:
    """Deterministic first-match typed routing policy.

    The legacy constructor arguments are retained for direct callers and tests; they
    are normalized into the same ordered-rule representation used by V2 configs.
    """

    def __init__(
        self,
        *,
        backends: Iterable[BackendConfig] | None = None,
        rules: Iterable[RoutingRule] | None = None,
        default_action: RouteAction | None = None,
        short_threshold_tokens: int | None = None,
        short_max_model_len: int | None = None,
        long_max_model_len: int | None = None,
        thinking_pool: str | None = None,
    ) -> None:
        backend_values: tuple[BackendConfig, ...]
        rule_values: tuple[RoutingRule, ...]
        if backends is None:
            if None in (
                short_threshold_tokens,
                short_max_model_len,
                long_max_model_len,
                thinking_pool,
            ):
                raise TypeError(
                    "backends/rules or all legacy routing arguments are required"
                )
            assert short_threshold_tokens is not None
            assert short_max_model_len is not None
            assert long_max_model_len is not None
            assert thinking_pool is not None
            backend_values = (
                BackendConfig("short", "short", short_max_model_len, 1, 0, 0, 1),
                BackendConfig("long", "long", long_max_model_len, 1, 0, 0, 1),
            )
            thinking_action = (
                RouteAction(type="reject", reason="thinking_not_allowed")
                if thinking_pool == "reject"
                else RouteAction(type="route", backend_id=str(thinking_pool))
            )
            rule_values = (
                RoutingRule(
                    "thinking-policy", 100, RuleMatch(thinking=True), thinking_action
                ),
                RoutingRule(
                    "short-threshold",
                    50,
                    RuleMatch(total_tokens_lte=short_threshold_tokens),
                    RouteAction(type="route", backend_id="short"),
                ),
                RoutingRule(
                    "long-threshold",
                    40,
                    RuleMatch(total_tokens_gte=short_threshold_tokens + 1),
                    RouteAction(type="route", backend_id="long"),
                ),
            )
            default = RouteAction(type="reject", reason="context_length_exceeded")
        else:
            backend_values = tuple(backends)
            rule_values = tuple(rules or ())
            default = default_action or RouteAction(
                type="reject", reason="no_matching_route"
            )
        self.backends: Mapping[str, BackendConfig] = {
            backend.id: backend for backend in backend_values
        }
        self.rules = tuple(
            sorted(rule_values, key=lambda item: item.priority, reverse=True)
        )
        self.default_action = default

    def _execute(
        self, action: RouteAction, profile: RequestProfile, rule_id: str, reason: str
    ) -> RouteDecision:
        if action.type == "reject":
            if action.reason == "thinking_not_allowed":
                raise ThinkingRejected()
            if action.reason == "context_length_exceeded":
                max_context = max(
                    (backend.max_context_tokens for backend in self.backends.values()),
                    default=0,
                )
                raise ContextLimitExceeded(profile.total_tokens, max_context)
            raise RouteRejected(action.reason or "route_rejected")
        backend = self.backends[action.backend_id]
        if profile.total_tokens > backend.max_context_tokens:
            raise ContextLimitExceeded(
                profile.total_tokens, backend.max_context_tokens, backend.id
            )
        return RouteDecision(
            backend_id=backend.id,
            rule_id=rule_id,
            reason=reason,
            profile=profile,
        )

    def classify(
        self,
        value: TokenBudget | RequestProfile,
        *,
        tools_present: bool = False,
        stream: bool = False,
    ) -> RouteDecision:
        profile = (
            value
            if isinstance(value, RequestProfile)
            else RequestProfile.from_budget(
                value, tools_present=tools_present, stream=stream
            )
        )
        for rule in self.rules:
            if _matches(rule.match, profile):
                reason = (
                    "thinking_policy"
                    if rule.id == "thinking-policy"
                    else (
                        "within_short_threshold"
                        if rule.id == "short-threshold"
                        else (
                            "exceeds_short_threshold"
                            if rule.id == "long-threshold"
                            else "rule_matched"
                        )
                    )
                )
                return self._execute(rule.action, profile, rule.id, reason)
        return self._execute(self.default_action, profile, "default", "default_action")
