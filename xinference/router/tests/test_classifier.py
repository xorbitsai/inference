import pytest

from xinference.router.classifier import (
    ContextLimitExceeded,
    RoutingPolicy,
    ThinkingRejected,
)
from xinference.router.tokenizer import TokenBudget


def budget(total: int, *, thinking: bool = False) -> TokenBudget:
    return TokenBudget(
        prompt_tokens=total - 65,
        output_tokens=1,
        reserve_tokens=64,
        total_tokens=total,
        enable_thinking=thinking,
    )


def policy() -> RoutingPolicy:
    return RoutingPolicy(
        short_threshold_tokens=32768,
        short_max_model_len=131072,
        long_max_model_len=1048576,
        thinking_pool="long",
    )


def test_routes_short_boundary_to_short() -> None:
    decision = policy().classify(budget(32768))
    assert decision.pool == "short"
    assert decision.reason == "within_short_threshold"


def test_routes_above_short_boundary_to_long() -> None:
    decision = policy().classify(budget(32769))
    assert decision.pool == "long"
    assert decision.reason == "exceeds_short_threshold"


def test_routes_thinking_to_long() -> None:
    decision = policy().classify(budget(1024, thinking=True))
    assert decision.pool == "long"
    assert decision.reason == "thinking_policy"


def test_rejects_above_long_context_limit() -> None:
    with pytest.raises(ContextLimitExceeded):
        policy().classify(budget(1048577))


def test_rejects_thinking_when_policy_is_reject() -> None:
    policy = RoutingPolicy(
        short_threshold_tokens=100,
        short_max_model_len=200,
        long_max_model_len=1000,
        thinking_pool="reject",
    )

    with pytest.raises(ThinkingRejected, match="disabled"):
        policy.classify(budget(50, thinking=True))


def typed_policy() -> RoutingPolicy:
    from xinference.router.config import (
        BackendConfig,
        RouteAction,
        RoutingRule,
        RuleMatch,
    )

    backends = (
        BackendConfig("fast", "fast-model", 100, 1, 0, 0, 1),
        BackendConfig("tools", "tools-model", 200, 1, 0, 0, 1),
        BackendConfig("reasoning", "reasoning-model", 300, 1, 0, 0, 1),
        BackendConfig("stream", "stream-model", 400, 1, 0, 0, 1),
    )
    rules = (
        RoutingRule(
            "tools-thinking",
            500,
            RuleMatch(tools_present=True, thinking=True),
            RouteAction(type="route", backend_id="reasoning"),
        ),
        RoutingRule(
            "tools",
            400,
            RuleMatch(tools_present=True),
            RouteAction(type="route", backend_id="tools"),
        ),
        RoutingRule(
            "thinking",
            300,
            RuleMatch(thinking=True),
            RouteAction(type="route", backend_id="reasoning"),
        ),
        RoutingRule(
            "stream",
            200,
            RuleMatch(stream=True),
            RouteAction(type="route", backend_id="stream"),
        ),
        RoutingRule(
            "fast",
            100,
            RuleMatch(total_tokens_lte=100),
            RouteAction(type="route", backend_id="fast"),
        ),
    )
    return RoutingPolicy(
        backends=backends,
        rules=rules,
        default_action=RouteAction(type="reject", reason="no_matching_route"),
    )


def test_typed_rules_use_first_match_across_four_backends() -> None:
    decision = typed_policy().classify(
        budget(50, thinking=True), tools_present=True, stream=True
    )

    assert decision.backend_id == "reasoning"
    assert decision.rule_id == "tools-thinking"
    assert decision.reason == "rule_matched"


def test_typed_rule_conditions_are_combined_with_and() -> None:
    decision = typed_policy().classify(budget(50), tools_present=True)

    assert decision.backend_id == "tools"
    assert decision.rule_id == "tools"


def test_typed_rules_match_stream_and_default_reject() -> None:
    from xinference.router.classifier import RouteRejected

    stream_decision = typed_policy().classify(budget(150), stream=True)
    assert stream_decision.backend_id == "stream"
    assert stream_decision.rule_id == "stream"

    with pytest.raises(RouteRejected, match="no matching route") as exc_info:
        typed_policy().classify(budget(150))
    assert exc_info.value.reason == "no_matching_route"


def test_typed_route_enforces_target_backend_context() -> None:
    with pytest.raises(ContextLimitExceeded, match="backend tools"):
        typed_policy().classify(budget(201), tools_present=True)
