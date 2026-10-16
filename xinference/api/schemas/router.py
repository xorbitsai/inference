# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Schemas for Token-aware Router management and runtime APIs."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlsplit

from ..._compat import BaseModel, Field, validator

_BACKEND_ID_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$"
_RULE_ID_PATTERN = _BACKEND_ID_PATTERN


def normalize_token_router_backend_url(value: str) -> str:
    """Normalize and validate a Token Router Supervisor REST endpoint."""

    normalized = value.strip().rstrip("/")
    try:
        parsed = urlsplit(normalized)
        # Accessing port also validates malformed or out-of-range ports.
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("backend_url must be a valid HTTP(S) URL") from exc

    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("backend_url must start with http:// or https://")
    if parsed.username or parsed.password:
        raise ValueError("backend_url must not contain credentials")
    if parsed.path or parsed.query or parsed.fragment:
        raise ValueError(
            "backend_url must be a base endpoint without path, query, or fragment"
        )
    return normalized


class AdmissionConfig(BaseModel):
    max_active: int = Field(default=1, ge=1)
    max_queue: int = Field(default=0, ge=0)
    queue_timeout_seconds: float = Field(default=5.0, ge=0)
    retry_after_seconds: int = Field(default=1, ge=0)


class RouterBackendConfig(BaseModel):
    """Legacy V1 Short/Long backend."""

    model_uid: str = Field(min_length=1)
    max_context_tokens: int = Field(ge=1)
    admission: AdmissionConfig = Field(default_factory=AdmissionConfig)


class RouterBackendsConfig(BaseModel):
    """Legacy V1 fixed backend map."""

    short: RouterBackendConfig
    long: RouterBackendConfig


class DynamicRouterBackendConfig(RouterBackendConfig):
    """V2 backend with a stable Router-local identifier."""

    id: str = Field(min_length=1, max_length=64, regex=_BACKEND_ID_PATTERN)


class RoutingConfig(BaseModel):
    """Legacy V1 token threshold routing."""

    short_threshold_tokens: int = Field(default=131072, ge=1)
    context_reserve_tokens: int = Field(default=64, ge=0)
    default_output_tokens: int = Field(default=512, ge=1)
    thinking_policy: Literal["short", "long", "reject"] = "long"
    overflow_policy: Literal["reject"] = "reject"


class RoutingRuleMatch(BaseModel):
    total_tokens_gte: Optional[int] = Field(default=None, ge=0)
    total_tokens_lte: Optional[int] = Field(default=None, ge=0)
    thinking: Optional[bool] = None
    tools_present: Optional[bool] = None
    stream: Optional[bool] = None

    @validator("stream", always=True)
    def validate_match(cls, value: Optional[bool], values: Dict[str, Any]):
        fields = (
            values.get("total_tokens_gte"),
            values.get("total_tokens_lte"),
            values.get("thinking"),
            values.get("tools_present"),
            value,
        )
        if all(item is None for item in fields):
            raise ValueError("routing rule match cannot be empty")
        lower = values.get("total_tokens_gte")
        upper = values.get("total_tokens_lte")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("total_tokens_gte cannot exceed total_tokens_lte")
        return value


class RouteAction(BaseModel):
    type: Literal["route"]
    backend_id: str = Field(min_length=1, max_length=64, regex=_BACKEND_ID_PATTERN)


class RejectAction(BaseModel):
    type: Literal["reject"]
    reason: str = Field(min_length=1, max_length=128)


RoutingAction = Union[RouteAction, RejectAction]


class RoutingRule(BaseModel):
    id: str = Field(min_length=1, max_length=64, regex=_RULE_ID_PATTERN)
    priority: int = Field(ge=1, le=10000)
    match: RoutingRuleMatch
    action: RoutingAction


class TypedRoutingConfig(BaseModel):
    evaluation_mode: Literal["first_match"] = "first_match"
    context_reserve_tokens: int = Field(default=64, ge=0)
    default_output_tokens: int = Field(default=512, ge=1)
    rules: list[RoutingRule] = Field(min_items=1, max_items=64)
    default_action: RoutingAction

    @validator("rules")
    def validate_rules(cls, value: list[RoutingRule]) -> list[RoutingRule]:
        ids = [rule.id for rule in value]
        if len(ids) != len(set(ids)):
            raise ValueError("routing rule ids must be unique")
        priorities = [rule.priority for rule in value]
        if len(priorities) != len(set(priorities)):
            raise ValueError("routing rule priorities must be unique")
        return value


class TokenizationConfig(BaseModel):
    executor: Literal["process"] = "process"
    multiprocessing_start_method: Literal["spawn"] = "spawn"
    max_workers: int = Field(default=2, ge=1)
    max_active: int = Field(default=2, ge=1)
    max_queue: int = Field(default=8, ge=0)
    queue_timeout_seconds: float = Field(default=5.0, ge=0)
    retry_after_seconds: int = Field(default=1, ge=0)


class TokenRouterConfigBase(BaseModel):
    config_version: int = Field(default=1)
    virtual_model_uid: str = Field(min_length=1)
    model_type: Literal["LLM"] = "LLM"
    route_profile: Literal["llm_chat"] = "llm_chat"
    strategy: Literal["token_budget", "typed_rules"] = "token_budget"
    tokenizer_asset_id: Optional[str] = Field(default=None, min_length=1)
    tokenizer_path: Optional[str] = Field(default=None, min_length=1)
    backend_url: str = Field(min_length=1)
    model_aliases: list[str] = Field(default_factory=list)
    request_timeout_seconds: float = Field(default=10800.0, gt=0)
    connect_timeout_seconds: float = Field(default=10.0, gt=0)
    backends: Union[RouterBackendsConfig, list[DynamicRouterBackendConfig]]
    routing: Union[TypedRoutingConfig, RoutingConfig]
    tokenization: TokenizationConfig = Field(default_factory=TokenizationConfig)

    @validator("config_version")
    def validate_config_version(cls, value: int) -> int:
        if value not in {1, 2}:
            raise ValueError("config_version must be 1 or 2")
        return value

    @validator("tokenizer_asset_id", "tokenizer_path", pre=True)
    def normalize_tokenizer_source(cls, value: Any):
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @validator("tokenizer_path", always=True)
    def validate_tokenizer_source(
        cls, value: Optional[str], values: Dict[str, Any]
    ) -> Optional[str]:
        if not values.get("tokenizer_asset_id") and not value:
            raise ValueError("tokenizer_asset_id or tokenizer_path must be provided")
        return value

    @validator("backend_url")
    def normalize_backend_url(cls, value: str) -> str:
        return normalize_token_router_backend_url(value)

    @validator("backends")
    def validate_backends(
        cls,
        value: Union[RouterBackendsConfig, list[DynamicRouterBackendConfig]],
        values: Dict[str, Any],
    ):
        version = values.get("config_version", 1)
        if version == 1:
            if not isinstance(value, RouterBackendsConfig):
                raise ValueError("config_version 1 requires short/long backends")
            if value.short.model_uid == value.long.model_uid:
                raise ValueError("short and long backend model_uid must be different")
            return value
        if not isinstance(value, list):
            raise ValueError("config_version 2 requires a dynamic backend list")
        if not 1 <= len(value) <= 16:
            raise ValueError("config_version 2 requires 1 to 16 backends")
        ids = [backend.id for backend in value]
        if len(ids) != len(set(ids)):
            raise ValueError("backend ids must be unique")
        return value

    @validator("routing")
    def validate_routing(
        cls,
        value: Union[TypedRoutingConfig, RoutingConfig],
        values: Dict[str, Any],
    ):
        version = values.get("config_version", 1)
        backends = values.get("backends")
        strategy = values.get("strategy")
        if version == 1:
            if not isinstance(value, RoutingConfig) or not isinstance(
                backends, RouterBackendsConfig
            ):
                raise ValueError("config_version 1 requires legacy routing")
            if strategy != "token_budget":
                raise ValueError("config_version 1 requires token_budget strategy")
            if value.short_threshold_tokens > backends.short.max_context_tokens:
                raise ValueError(
                    "short_threshold_tokens cannot exceed short max_context_tokens"
                )
            if backends.short.max_context_tokens > backends.long.max_context_tokens:
                raise ValueError(
                    "short max_context_tokens cannot exceed long max_context_tokens"
                )
            return value
        if not isinstance(value, TypedRoutingConfig) or not isinstance(backends, list):
            raise ValueError("config_version 2 requires typed routing rules")
        if strategy != "typed_rules":
            raise ValueError("config_version 2 requires typed_rules strategy")
        backend_ids = {backend.id for backend in backends}
        actions = [rule.action for rule in value.rules] + [value.default_action]
        unknown = sorted(
            action.backend_id
            for action in actions
            if isinstance(action, RouteAction) and action.backend_id not in backend_ids
        )
        if unknown:
            raise ValueError(
                "routing actions reference unknown backends: " + ", ".join(unknown)
            )
        return value


class TokenRouterCreate(TokenRouterConfigBase):
    router_uid: str = Field(min_length=1)


class TokenRouterUpdate(TokenRouterConfigBase):
    revision: int = Field(ge=1)


class TokenRouterDeploymentUpdate(BaseModel):
    management_mode: Optional[Literal["external", "managed"]] = None
    desired_replicas: Optional[int] = Field(default=None, ge=0)
    desired_state: Optional[Literal["running", "stopped"]] = None
    placement: Optional[Dict[str, Any]] = None
    rollout: Optional[Dict[str, Any]] = None
    deployment_generation: Optional[int] = Field(default=None, ge=1)


class TokenizerAssetCreate(BaseModel):
    asset_id: str = Field(min_length=1, max_length=128)
    origin: Literal["builtin", "artifact", "shared_fs", "local", "external"]
    revision: str = Field(min_length=1)
    fingerprint: str = Field(regex=r"^sha256:[0-9a-fA-F]{64}$")
    source: Dict[str, Any] = Field(default_factory=dict)
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    display_name: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class TokenizerAssetUpdate(BaseModel):
    origin: Optional[
        Literal["builtin", "artifact", "shared_fs", "local", "external"]
    ] = None
    revision: Optional[str] = Field(default=None, min_length=1)
    fingerprint: Optional[str] = Field(default=None, regex=r"^sha256:[0-9a-fA-F]{64}$")
    source: Optional[Dict[str, Any]] = None
    capabilities: Optional[Dict[str, Any]] = None
    display_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None


class TokenizerAssetBindingCreate(BaseModel):
    node_ids: List[str] = Field(default_factory=list)
    selector: Dict[str, Any] = Field(default_factory=dict)
    desired_state: Literal["present", "absent"] = "present"
    binding_mode: Literal["manual", "on_demand"] = "manual"
    owner_type: str = ""
    owner_id: str = ""


class TokenizerAssetBindingUpdate(BaseModel):
    desired_state: Optional[Literal["present", "absent"]] = None
    binding_mode: Optional[Literal["manual", "on_demand"]] = None
    owner_type: Optional[str] = None
    owner_id: Optional[str] = None


class TokenizerAssetBindingStatus(BaseModel):
    asset_id: str = Field(min_length=1)
    generation: int = Field(ge=1)
    observed_state: Literal[
        "pending",
        "preparing",
        "validating",
        "ready",
        "unavailable",
        "error",
        "removing",
        "absent",
        "stale",
    ]
    observed_revision: str = ""
    observed_fingerprint: str = ""
    local_path: str = ""
    last_error_code: str = ""
    last_error: str = ""


class RouterNodeLabelsUpdate(BaseModel):
    labels: Dict[str, Any] = Field(default_factory=dict)


class RouterNodeRegister(BaseModel):
    node_id: str = Field(min_length=1)
    advertise_host: str = Field(min_length=1)
    port_range_start: int = Field(ge=1024, le=65535)
    port_range_end: int = Field(ge=1024, le=65535)
    max_instances: int = Field(ge=1)
    reported_labels: Dict[str, Any] = Field(default_factory=dict)
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    software_version: str = ""
    software_revision: Optional[str] = None

    @validator("port_range_end")
    def validate_port_range(cls, value: int, values: Dict[str, Any]) -> int:
        start = values.get("port_range_start")
        if start is not None and value < start:
            raise ValueError("port_range_end cannot be smaller than port_range_start")
        return value


class RouterNodeHeartbeat(BaseModel):
    status: str = "ready"
    running_instances: int = Field(default=0, ge=0)
    available_slots: int = Field(default=0, ge=0)
    assignments: list[Dict[str, Any]] = Field(default_factory=list)
    resources: Dict[str, Any] = Field(default_factory=dict)


class RouterNodeStateUpdate(BaseModel):
    desired_state: Literal["active", "cordoned", "draining", "disabled"]


class RouterAssignmentStatus(BaseModel):
    node_id: str = Field(min_length=1)
    assignment_generation: int = Field(ge=1)
    observed_state: Literal[
        "pending",
        "assigned",
        "starting",
        "ready",
        "draining",
        "stopped",
        "failed",
        "crash_loop",
        "port_conflict",
        "stale",
    ]
    pid: Optional[int] = Field(default=None, ge=1)
    instance_id: Optional[str] = None
    listen_port: Optional[int] = Field(default=None, ge=1, le=65535)
    last_error: str = ""
    observed: Dict[str, Any] = Field(default_factory=dict)


class RouterRuntimeRegister(BaseModel):
    router_uid: str = Field(min_length=1)
    instance_id: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    assignment_id: Optional[str] = None
    assignment_generation: Optional[int] = Field(default=None, ge=1)
    node_id: Optional[str] = None
    # Legacy control-plane protocol field kept for backward compatibility.
    version: str = ""
    protocol_version: str = ""
    software_version: str = ""
    software_revision: Optional[str] = None
    acked_revision: int = Field(default=0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RouterRuntimeHeartbeat(BaseModel):
    status: str = "ready"
    metrics: Dict[str, Any] = Field(default_factory=dict)
    backend_health: Dict[str, Any] = Field(default_factory=dict)
    process: Dict[str, Any] = Field(default_factory=dict)


class RouterConfigAck(BaseModel):
    router_uid: str = Field(min_length=1)
    revision: int = Field(ge=1)
    error: str = ""
