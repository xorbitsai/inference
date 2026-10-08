from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]

from .tokenizer_asset import DEFAULT_TOKENIZER_ASSET_FILES
from .tokenizer_assets import (
    BuiltinTokenizerAssetError,
    resolve_builtin_tokenizer_asset,
)

_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


class ConfigError(ValueError):
    """Raised when Router configuration is invalid."""


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean, got {value!r}")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


@dataclass(frozen=True)
class BackendConfig:
    id: str
    model_uid: str
    max_context_tokens: int
    max_active: int
    max_queue: int
    queue_timeout_seconds: float
    retry_after_seconds: int

    @property
    def name(self) -> str:  # V1 compatibility for metrics/tests.
        return self.id


PoolConfig = BackendConfig


@dataclass(frozen=True)
class RuleMatch:
    total_tokens_gte: int | None = None
    total_tokens_lte: int | None = None
    thinking: bool | None = None
    tools_present: bool | None = None
    stream: bool | None = None


@dataclass(frozen=True)
class RouteAction:
    type: str
    backend_id: str = ""
    reason: str = ""


@dataclass(frozen=True)
class RoutingRule:
    id: str
    priority: int
    match: RuleMatch
    action: RouteAction


@dataclass(frozen=True)
class TokenizationConfig:
    max_workers: int
    max_active: int
    max_queue: int
    queue_timeout_seconds: float
    retry_after_seconds: int


@dataclass(frozen=True)
class RouterConfig:
    listen_host: str
    listen_port: int
    backend_url: str
    backend_api_key: str
    require_auth: bool
    logical_model: str
    model_aliases: tuple[str, ...]
    tokenizer_path: Path
    context_reserve_tokens: int
    default_output_tokens: int
    request_timeout_seconds: float
    connect_timeout_seconds: float
    tokenization: TokenizationConfig
    backends: tuple[BackendConfig, ...]
    rules: tuple[RoutingRule, ...]
    default_action: RouteAction
    log_level: str
    config_version: int = 2
    route_profile: str = "llm_chat"
    strategy: str = "typed_rules"
    enabled: bool = True
    revision: int = 0
    router_uid: str = ""
    tokenizer_asset_id: str = ""
    tokenizer_asset_origin: str = ""
    tokenizer_asset_revision: str = ""
    tokenizer_asset_fingerprint: str = ""
    tokenizer_asset_files: tuple[str, ...] = DEFAULT_TOKENIZER_ASSET_FILES
    tokenizer_asset_capabilities: tuple[str, ...] = ("chat", "tools", "thinking")
    legacy_short_threshold_tokens: int | None = None
    legacy_thinking_pool: str = ""

    def backend(self, backend_id: str) -> BackendConfig:
        for backend in self.backends:
            if backend.id == backend_id:
                return backend
        raise ConfigError(f"Unknown backend: {backend_id}")

    def pool(self, name: str) -> BackendConfig:
        return self.backend(name)

    @property
    def short_pool(self) -> BackendConfig:
        return self.backend("short")

    @property
    def long_pool(self) -> BackendConfig:
        return self.backend("long")

    @property
    def short_threshold_tokens(self) -> int:
        if self.legacy_short_threshold_tokens is None:
            raise ConfigError("short threshold is unavailable for V2 typed rules")
        return self.legacy_short_threshold_tokens

    @property
    def short_max_model_len(self) -> int:
        return self.short_pool.max_context_tokens

    @property
    def long_max_model_len(self) -> int:
        return self.long_pool.max_context_tokens

    @property
    def thinking_pool(self) -> str:
        return self.legacy_thinking_pool


def _validate_action(action: RouteAction, backend_ids: set[str], where: str) -> None:
    if action.type == "route":
        if action.backend_id not in backend_ids:
            raise ConfigError(
                f"{where} references unknown backend {action.backend_id!r}"
            )
    elif action.type == "reject":
        if not action.reason:
            raise ConfigError(f"{where} reject action requires a reason")
    else:
        raise ConfigError(f"{where} action type must be route or reject")


def _validate_config(cfg: RouterConfig) -> RouterConfig:
    if not cfg.backend_url or not cfg.backend_url.startswith(("http://", "https://")):
        raise ConfigError("backend URL must start with http:// or https://")
    if not cfg.logical_model:
        raise ConfigError("logical model UID is required")
    if cfg.route_profile != "llm_chat":
        raise ConfigError("route_profile must be llm_chat")
    if not cfg.tokenizer_path.is_dir():
        raise ConfigError(f"Tokenizer path does not exist: {cfg.tokenizer_path}")
    if not 1 <= len(cfg.backends) <= 16:
        raise ConfigError("Router requires 1 to 16 backends")
    backend_ids = [backend.id for backend in cfg.backends]
    if len(backend_ids) != len(set(backend_ids)):
        raise ConfigError("backend ids must be unique")
    for backend in cfg.backends:
        if not _ID_RE.fullmatch(backend.id):
            raise ConfigError(f"Invalid backend id: {backend.id!r}")
        if not backend.model_uid:
            raise ConfigError(f"backends.{backend.id}.model_uid is required")
        if backend.max_context_tokens < 1:
            raise ConfigError(f"Invalid context limit for backend {backend.id}")
        if backend.max_active < 1 or backend.max_queue < 0:
            raise ConfigError(f"Invalid capacity for backend {backend.id}")
        if backend.queue_timeout_seconds < 0 or backend.retry_after_seconds < 0:
            raise ConfigError(f"Invalid timeout for backend {backend.id}")
    if not 1 <= len(cfg.rules) <= 64:
        raise ConfigError("Router requires 1 to 64 routing rules")
    rule_ids = [rule.id for rule in cfg.rules]
    priorities = [rule.priority for rule in cfg.rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise ConfigError("routing rule ids must be unique")
    if len(priorities) != len(set(priorities)):
        raise ConfigError("routing rule priorities must be unique")
    backend_id_set = set(backend_ids)
    for rule in cfg.rules:
        if not _ID_RE.fullmatch(rule.id) or not 1 <= rule.priority <= 10000:
            raise ConfigError(f"Invalid routing rule {rule.id!r}")
        match = rule.match
        if all(
            value is None
            for value in (
                match.total_tokens_gte,
                match.total_tokens_lte,
                match.thinking,
                match.tools_present,
                match.stream,
            )
        ):
            raise ConfigError(f"routing rule {rule.id!r} match cannot be empty")
        if (
            match.total_tokens_gte is not None
            and match.total_tokens_lte is not None
            and match.total_tokens_gte > match.total_tokens_lte
        ):
            raise ConfigError(f"Invalid token range in routing rule {rule.id!r}")
        _validate_action(rule.action, backend_id_set, f"routing rule {rule.id!r}")
        if (
            match.tools_present is True
            and "tools" not in cfg.tokenizer_asset_capabilities
        ):
            raise ConfigError(
                f"routing rule {rule.id!r} requires tools but the Tokenizer "
                "asset does not support tools"
            )
        if (
            match.thinking is True
            and "thinking" not in cfg.tokenizer_asset_capabilities
        ):
            raise ConfigError(
                f"routing rule {rule.id!r} requires thinking but the Tokenizer "
                "asset does not support thinking"
            )
    _validate_action(cfg.default_action, backend_id_set, "default action")
    if cfg.context_reserve_tokens < 0 or cfg.default_output_tokens < 1:
        raise ConfigError("Invalid token budget defaults")
    if cfg.request_timeout_seconds <= 0 or cfg.connect_timeout_seconds <= 0:
        raise ConfigError("backend timeouts must be greater than zero")
    tokenization = cfg.tokenization
    if (
        tokenization.max_workers < 1
        or tokenization.max_active < tokenization.max_workers
    ):
        raise ConfigError(
            "tokenization.max_active must be greater than or equal to max_workers"
        )
    if tokenization.max_queue < 0 or tokenization.queue_timeout_seconds < 0:
        raise ConfigError("Invalid tokenization capacity or timeout")
    return cfg


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Router config does not exist: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError("Router config must be a YAML object")
    return data


def _legacy_pool(data: dict[str, Any], name: str, max_context: int) -> BackendConfig:
    pool = data.get("pools", {}).get(name, {})
    if not isinstance(pool, dict):
        raise ConfigError(f"pools.{name} must be an object")
    prefix = f"ROUTER_{name.upper()}"
    model_uid = os.getenv(f"{prefix}_MODEL_UID", str(pool.get("model_uid", "")))
    return BackendConfig(
        id=name,
        model_uid=model_uid,
        max_context_tokens=max_context,
        max_active=_env_int(f"{prefix}_MAX_ACTIVE", int(pool.get("max_active", 1))),
        max_queue=_env_int(f"{prefix}_MAX_QUEUE", int(pool.get("max_queue", 0))),
        queue_timeout_seconds=_env_float(
            f"{prefix}_QUEUE_TIMEOUT_SECONDS",
            float(pool.get("queue_timeout_seconds", 0)),
        ),
        retry_after_seconds=_env_int(
            f"{prefix}_RETRY_AFTER_SECONDS", int(pool.get("retry_after_seconds", 1))
        ),
    )


def _legacy_rules(threshold: int, thinking_pool: str) -> tuple[RoutingRule, ...]:
    rules: list[RoutingRule] = []
    if thinking_pool == "reject":
        thinking_action = RouteAction(type="reject", reason="thinking_not_allowed")
    else:
        thinking_action = RouteAction(type="route", backend_id=thinking_pool)
    rules.append(
        RoutingRule("thinking-policy", 100, RuleMatch(thinking=True), thinking_action)
    )
    rules.append(
        RoutingRule(
            "short-threshold",
            50,
            RuleMatch(total_tokens_lte=threshold),
            RouteAction(type="route", backend_id="short"),
        )
    )
    rules.append(
        RoutingRule(
            "long-threshold",
            40,
            RuleMatch(total_tokens_gte=threshold + 1),
            RouteAction(type="route", backend_id="long"),
        )
    )
    return tuple(rules)


def _tokenization(value: Mapping[str, Any]) -> TokenizationConfig:
    return TokenizationConfig(
        max_workers=int(value.get("max_workers", 2)),
        max_active=int(value.get("max_active", 2)),
        max_queue=int(value.get("max_queue", 8)),
        queue_timeout_seconds=float(value.get("queue_timeout_seconds", 5)),
        retry_after_seconds=int(value.get("retry_after_seconds", 1)),
    )


def load_config(path: str | Path | None = None) -> RouterConfig:
    """Load the legacy standalone YAML format and normalize it to V2 internally."""
    config_path: str | Path = (
        path if path is not None else os.getenv("ROUTER_CONFIG", "config.yaml")
    )
    data = _load_yaml(Path(config_path).expanduser())
    listen, limits, auth = (
        data.get("listen", {}),
        data.get("limits", {}),
        data.get("auth", {}),
    )
    backend, model = data.get("backend", {}), data.get("model", {})
    tokenization_data = data.get("tokenization", {})
    backend_api_key = os.getenv("XINFERENCE_API_KEY", "")
    require_auth = _env_bool(
        "ROUTER_REQUIRE_AUTH", bool(auth.get("require_auth", True))
    )
    if require_auth and not backend_api_key:
        raise ConfigError("XINFERENCE_API_KEY is required when Router auth is enabled")
    aliases = model.get("aliases", [])
    if not isinstance(aliases, list):
        raise ConfigError("model.aliases must be a list")
    short_max = _env_int(
        "ROUTER_SHORT_MAX_MODEL_LEN", int(limits.get("short_max_model_len", 131072))
    )
    long_max = _env_int(
        "ROUTER_LONG_MAX_MODEL_LEN", int(limits.get("long_max_model_len", 1048576))
    )
    threshold = _env_int(
        "ROUTER_SHORT_THRESHOLD_TOKENS",
        int(limits.get("short_threshold_tokens", 32768)),
    )
    thinking_pool = os.getenv(
        "ROUTER_THINKING_POOL", str(model.get("thinking_pool", "long"))
    )
    cfg = RouterConfig(
        listen_host=os.getenv(
            "ROUTER_LISTEN_HOST", str(listen.get("host", "127.0.0.1"))
        ),
        listen_port=_env_int("ROUTER_LISTEN_PORT", int(listen.get("port", 10080))),
        backend_url=os.getenv("ROUTER_BACKEND_URL", str(backend.get("url", ""))).rstrip(
            "/"
        ),
        backend_api_key=backend_api_key,
        require_auth=require_auth,
        logical_model=os.getenv(
            "ROUTER_LOGICAL_MODEL", str(model.get("logical_model", ""))
        ),
        model_aliases=tuple(str(alias) for alias in aliases),
        tokenizer_path=Path(
            os.getenv("ROUTER_TOKENIZER_PATH", str(model.get("tokenizer_path", "")))
        ).expanduser(),
        context_reserve_tokens=_env_int(
            "ROUTER_CONTEXT_RESERVE_TOKENS",
            int(limits.get("context_reserve_tokens", 64)),
        ),
        default_output_tokens=_env_int(
            "ROUTER_DEFAULT_OUTPUT_TOKENS",
            int(limits.get("default_output_tokens", 512)),
        ),
        request_timeout_seconds=_env_float(
            "ROUTER_REQUEST_TIMEOUT_SECONDS",
            float(backend.get("request_timeout_seconds", 7200)),
        ),
        connect_timeout_seconds=_env_float(
            "ROUTER_CONNECT_TIMEOUT_SECONDS",
            float(backend.get("connect_timeout_seconds", 10)),
        ),
        tokenization=TokenizationConfig(
            max_workers=_env_int(
                "ROUTER_TOKENIZATION_MAX_WORKERS",
                int(tokenization_data.get("max_workers", 2)),
            ),
            max_active=_env_int(
                "ROUTER_TOKENIZATION_MAX_ACTIVE",
                int(tokenization_data.get("max_active", 2)),
            ),
            max_queue=_env_int(
                "ROUTER_TOKENIZATION_MAX_QUEUE",
                int(tokenization_data.get("max_queue", 8)),
            ),
            queue_timeout_seconds=_env_float(
                "ROUTER_TOKENIZATION_QUEUE_TIMEOUT_SECONDS",
                float(tokenization_data.get("queue_timeout_seconds", 5)),
            ),
            retry_after_seconds=_env_int(
                "ROUTER_TOKENIZATION_RETRY_AFTER_SECONDS",
                int(tokenization_data.get("retry_after_seconds", 1)),
            ),
        ),
        backends=(
            _legacy_pool(data, "short", short_max),
            _legacy_pool(data, "long", long_max),
        ),
        rules=_legacy_rules(threshold, thinking_pool),
        default_action=RouteAction(type="reject", reason="context_length_exceeded"),
        log_level=os.getenv("ROUTER_LOG_LEVEL", str(data.get("log_level", "INFO"))),
        config_version=1,
        strategy="token_budget",
        legacy_short_threshold_tokens=threshold,
        legacy_thinking_pool=thinking_pool,
        tokenizer_asset_id=str(model.get("tokenizer_asset_id", "")),
        tokenizer_asset_origin=str(model.get("tokenizer_asset_origin", "")),
        tokenizer_asset_revision=str(model.get("tokenizer_asset_revision", "")),
        tokenizer_asset_fingerprint=str(model.get("tokenizer_asset_fingerprint", "")),
    )
    return _validate_config(cfg)


def _backend(value: Mapping[str, Any], backend_id: str | None = None) -> BackendConfig:
    admission = value.get("admission", {})
    return BackendConfig(
        id=str(backend_id if backend_id is not None else value["id"]),
        model_uid=str(value["model_uid"]),
        max_context_tokens=int(value["max_context_tokens"]),
        max_active=int(admission.get("max_active", 1)),
        max_queue=int(admission.get("max_queue", 0)),
        queue_timeout_seconds=float(admission.get("queue_timeout_seconds", 5)),
        retry_after_seconds=int(admission.get("retry_after_seconds", 1)),
    )


def _action(value: Mapping[str, Any]) -> RouteAction:
    return RouteAction(
        type=str(value.get("type", "")),
        backend_id=str(value.get("backend_id", "")),
        reason=str(value.get("reason", "")),
    )


def _typed_rule(value: Mapping[str, Any]) -> RoutingRule:
    match = value.get("match", {})
    return RoutingRule(
        id=str(value["id"]),
        priority=int(value["priority"]),
        match=RuleMatch(
            total_tokens_gte=match.get("total_tokens_gte"),
            total_tokens_lte=match.get("total_tokens_lte"),
            thinking=match.get("thinking"),
            tools_present=match.get("tools_present"),
            stream=match.get("stream"),
        ),
        action=_action(value["action"]),
    )


def _resolve_tokenizer(data: Mapping[str, Any]) -> tuple[str, str, Path]:
    tokenizer_path = str(data.get("tokenizer_path") or "").strip()
    asset_id = str(data.get("tokenizer_asset_id") or "").strip()
    asset_origin = str(data.get("tokenizer_asset_origin") or "").strip()
    if not tokenizer_path:
        raise ConfigError("tokenizer_path is required")
    if asset_origin == "builtin":
        try:
            local_asset = resolve_builtin_tokenizer_asset(asset_id)
        except (KeyError, BuiltinTokenizerAssetError) as exc:
            raise ConfigError(
                f"Cannot resolve built-in Tokenizer asset {asset_id}: {exc}"
            ) from exc
        expected_revision = str(data.get("tokenizer_asset_revision") or "")
        expected_fingerprint = str(data.get("tokenizer_asset_fingerprint") or "")
        if (
            expected_revision
            and expected_revision != local_asset["tokenizer_asset_revision"]
        ):
            raise ConfigError(
                "Built-in Tokenizer asset revision differs from the control plane"
            )
        if (
            expected_fingerprint
            and expected_fingerprint != local_asset["tokenizer_asset_fingerprint"]
        ):
            raise ConfigError(
                "Built-in Tokenizer asset fingerprint differs from the control plane"
            )
        tokenizer_path = local_asset["tokenizer_path"]
    return asset_id, asset_origin, Path(tokenizer_path).expanduser()


def _control_plane_capabilities(data: Mapping[str, Any]) -> tuple[str, ...]:
    raw = data.get("tokenizer_asset_capabilities")
    if raw is None:
        return ("chat", "tools", "thinking")
    if isinstance(raw, Mapping):
        return tuple(str(name) for name, enabled in raw.items() if enabled)
    return tuple(str(item) for item in raw)


def config_from_control_plane(
    data: dict[str, Any],
    *,
    listen_host: str = "127.0.0.1",
    listen_port: int = 10080,
    log_level: str = "INFO",
) -> RouterConfig:
    """Convert V1 or V2 Supervisor data to one immutable V2 runtime config."""
    try:
        version = int(data.get("config_version", 1))
        routing = data["routing"]
        tokenization = data["tokenization"]
        raw_backends = data["backends"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid control-plane Router config: {exc}") from exc
    asset_id, asset_origin, tokenizer_path = _resolve_tokenizer(data)
    legacy_threshold: int | None = None
    legacy_thinking = ""
    backends: tuple[BackendConfig, ...]
    if version == 1:
        try:
            threshold = int(routing["short_threshold_tokens"])
            legacy_thinking = str(routing.get("thinking_policy", "long"))
            backends = (
                _backend(raw_backends["short"], "short"),
                _backend(raw_backends["long"], "long"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid V1 Router config: {exc}") from exc
        rules = _legacy_rules(threshold, legacy_thinking)
        default_action = RouteAction(type="reject", reason="context_length_exceeded")
        legacy_threshold = threshold
        strategy = "token_budget"
    elif version == 2:
        if not isinstance(raw_backends, list):
            raise ConfigError("V2 backends must be a list")
        backends = tuple(_backend(value) for value in raw_backends)
        rules = tuple(_typed_rule(value) for value in routing.get("rules", []))
        default_action = _action(routing.get("default_action", {}))
        strategy = str(data.get("strategy", "typed_rules"))
    else:
        raise ConfigError(f"Unsupported config_version: {version}")
    cfg = RouterConfig(
        listen_host=listen_host,
        listen_port=listen_port,
        backend_url=str(data["backend_url"]).rstrip("/"),
        backend_api_key="",
        require_auth=False,
        logical_model=str(data["virtual_model_uid"]),
        model_aliases=tuple(str(v) for v in data.get("model_aliases", [])),
        tokenizer_path=tokenizer_path,
        context_reserve_tokens=int(routing.get("context_reserve_tokens", 64)),
        default_output_tokens=int(routing.get("default_output_tokens", 512)),
        request_timeout_seconds=float(data.get("request_timeout_seconds", 10800)),
        connect_timeout_seconds=float(data.get("connect_timeout_seconds", 10)),
        tokenization=_tokenization(tokenization),
        backends=backends,
        rules=tuple(sorted(rules, key=lambda item: item.priority, reverse=True)),
        default_action=default_action,
        log_level=log_level,
        config_version=version,
        route_profile=str(data.get("route_profile", "llm_chat")),
        strategy=strategy,
        enabled=bool(data.get("enabled", False)),
        revision=int(data.get("revision", 0)),
        router_uid=str(data.get("router_uid", "")),
        tokenizer_asset_id=asset_id,
        tokenizer_asset_origin=asset_origin,
        tokenizer_asset_revision=str(data.get("tokenizer_asset_revision", "")),
        tokenizer_asset_fingerprint=str(data.get("tokenizer_asset_fingerprint", "")),
        tokenizer_asset_files=tuple(
            str(item)
            for item in (
                data.get("tokenizer_asset_files") or DEFAULT_TOKENIZER_ASSET_FILES
            )
        ),
        tokenizer_asset_capabilities=_control_plane_capabilities(data),
        legacy_short_threshold_tokens=legacy_threshold,
        legacy_thinking_pool=legacy_thinking,
    )
    return _validate_config(cfg)
