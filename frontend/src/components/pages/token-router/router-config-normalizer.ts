import type {
  TokenizerAssetItem,
  TokenRouterAssignment,
  TokenRouterDynamicBackendConfig,
  TokenRouterItem,
  TokenRouterRoutingAction,
  TokenRouterRoutingRule,
  TokenRouterRuntimeInstance,
  TokenRouterTypedItem,
} from '@/types/services';
import { isTypedTokenRouter } from '@/types/services';

export type RouterMode = 'simple' | 'advanced';

export interface TypedRouterDraft {
  backends: TokenRouterDynamicBackendConfig[];
  rules: TokenRouterRoutingRule[];
  defaultAction: TokenRouterRoutingAction;
}

const admission = (maxActive: number, maxQueue: number) => ({
  max_active: maxActive,
  max_queue: maxQueue,
  queue_timeout_seconds: 5,
  retry_after_seconds: 1,
});

const defaultTokenization = () => ({
  executor: 'process' as const,
  multiprocessing_start_method: 'spawn' as const,
  max_workers: 2,
  max_active: 2,
  max_queue: 8,
  queue_timeout_seconds: 5,
  retry_after_seconds: 1,
});

const defaultDeployment = () => ({
  router_uid: '',
  management_mode: 'external' as const,
  desired_replicas: 0,
  desired_state: 'running' as const,
  placement: {},
  rollout: {},
  deployment_generation: 1,
  observed_ready_assignments: 0,
  effective_ready_runtimes: 0,
  controllable_ready_runtimes: 0,
  ready_replicas: 0,
  pending_replicas: 0,
  assignments: 0,
});

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const asString = (value: unknown, fallback = '') => (typeof value === 'string' ? value : fallback);

const asNumber = (value: unknown, fallback = 0) =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback;

const asArray = <T>(value: unknown): T[] => (Array.isArray(value) ? value : []);

function normalizeAdmission(value: unknown) {
  const source = isRecord(value) ? value : {};
  return {
    max_active: asNumber(source.max_active),
    max_queue: asNumber(source.max_queue),
    queue_timeout_seconds: asNumber(source.queue_timeout_seconds, 5),
    retry_after_seconds: asNumber(source.retry_after_seconds, 1),
  };
}

function normalizeBackend(value: unknown, id: string): TokenRouterDynamicBackendConfig {
  const source = isRecord(value) ? value : {};
  return {
    id,
    model_uid: asString(source.model_uid),
    max_context_tokens: asNumber(source.max_context_tokens),
    admission: normalizeAdmission(source.admission),
  };
}

function normalizeAction(value: unknown): TokenRouterRoutingAction {
  const source = isRecord(value) ? value : {};
  if (source.type === 'route') {
    return { type: 'route', backend_id: asString(source.backend_id) };
  }
  return { type: 'reject', reason: asString(source.reason, 'configuration_error') };
}

function normalizeRule(value: unknown, index: number): TokenRouterRoutingRule {
  const source = isRecord(value) ? value : {};
  const match = isRecord(source.match) ? source.match : {};
  return {
    id: asString(source.id, `rule-${index + 1}`),
    priority: asNumber(source.priority, 0),
    match: {
      ...(typeof match.total_tokens_gte === 'number'
        ? { total_tokens_gte: match.total_tokens_gte }
        : {}),
      ...(typeof match.total_tokens_lte === 'number'
        ? { total_tokens_lte: match.total_tokens_lte }
        : {}),
      ...(typeof match.thinking === 'boolean' ? { thinking: match.thinking } : {}),
      ...(typeof match.tools_present === 'boolean' ? { tools_present: match.tools_present } : {}),
      ...(typeof match.stream === 'boolean' ? { stream: match.stream } : {}),
    },
    action: normalizeAction(source.action),
  };
}

/**
 * Normalize API payloads at the UI boundary. The API is backed by a
 * persistent store and may contain Router records written by an older
 * version, so the detail drawer must not render raw unknown JSON directly.
 */
export function normalizeTokenRouterItem(value: unknown): TokenRouterItem | null {
  if (!isRecord(value)) return null;

  const rawBackends = value.backends;
  const typed = value.config_version === 2 || Array.isArray(rawBackends);
  const rawRouting = isRecord(value.routing) ? value.routing : {};
  const rawTokenization = isRecord(value.tokenization) ? value.tokenization : {};
  const rawDeployment = isRecord(value.deployment) ? value.deployment : {};
  const tokenizerAssetOrigin =
    value.tokenizer_asset_origin === 'builtin' || value.tokenizer_asset_origin === 'external'
      ? (value.tokenizer_asset_origin as 'builtin' | 'external')
      : undefined;
  const base = {
    router_uid: asString(value.router_uid),
    virtual_model_uid: asString(value.virtual_model_uid),
    model_type: 'LLM' as const,
    route_profile: 'llm_chat' as const,
    tokenizer_asset_id: asString(value.tokenizer_asset_id) || undefined,
    tokenizer_asset_origin: tokenizerAssetOrigin,
    tokenizer_path: asString(value.tokenizer_path),
    tokenizer_asset_revision: asString(value.tokenizer_asset_revision) || undefined,
    tokenizer_asset_fingerprint: asString(value.tokenizer_asset_fingerprint) || undefined,
    backend_url: asString(value.backend_url),
    model_aliases: asArray<unknown>(value.model_aliases).filter(
      (item): item is string => typeof item === 'string'
    ),
    request_timeout_seconds: asNumber(value.request_timeout_seconds, 10800),
    connect_timeout_seconds: asNumber(value.connect_timeout_seconds, 10),
    tokenization: {
      ...defaultTokenization(),
      executor: rawTokenization.executor === 'process' ? 'process' : ('process' as const),
      multiprocessing_start_method:
        rawTokenization.multiprocessing_start_method === 'spawn' ? 'spawn' : ('spawn' as const),
      max_workers: asNumber(rawTokenization.max_workers, 2),
      max_active: asNumber(rawTokenization.max_active, 2),
      max_queue: asNumber(rawTokenization.max_queue, 8),
      queue_timeout_seconds: asNumber(rawTokenization.queue_timeout_seconds, 5),
      retry_after_seconds: asNumber(rawTokenization.retry_after_seconds, 1),
    },
    enabled: Boolean(value.enabled),
    revision: asNumber(value.revision, 1),
    status: asString(value.status, 'unavailable') as TokenRouterItem['status'],
    runtime_instances: asNumber(value.runtime_instances),
    online_instances: asNumber(value.online_instances),
    deployment: {
      ...defaultDeployment(),
      ...rawDeployment,
      router_uid: asString(rawDeployment.router_uid, asString(value.router_uid)),
      desired_replicas: asNumber(rawDeployment.desired_replicas),
      pending_replicas: asNumber(rawDeployment.pending_replicas),
      ready_replicas: asNumber(rawDeployment.ready_replicas),
      assignments: asNumber(rawDeployment.assignments),
    },
    created_at: asString(value.created_at),
    updated_at: asString(value.updated_at),
  };

  if (typed) {
    const backends = asArray<unknown>(rawBackends).map((backend, index) => {
      const source = isRecord(backend) ? backend : {};
      return normalizeBackend(source, asString(source.id, `backend-${index + 1}`));
    });
    const rules = asArray<unknown>(rawRouting.rules).map(normalizeRule);
    return {
      ...base,
      config_version: 2,
      strategy: 'typed_rules',
      backends,
      routing: {
        evaluation_mode: 'first_match' as const,
        context_reserve_tokens: asNumber(rawRouting.context_reserve_tokens),
        default_output_tokens: asNumber(rawRouting.default_output_tokens),
        rules,
        default_action: normalizeAction(rawRouting.default_action),
      },
    };
  }

  const rawLegacyBackends = isRecord(rawBackends) ? rawBackends : {};
  return {
    ...base,
    config_version: 1,
    strategy: 'token_budget',
    backends: {
      short: normalizeBackend(rawLegacyBackends.short, 'short'),
      long: normalizeBackend(rawLegacyBackends.long, 'long'),
    },
    routing: {
      short_threshold_tokens: asNumber(rawRouting.short_threshold_tokens),
      context_reserve_tokens: asNumber(rawRouting.context_reserve_tokens),
      default_output_tokens: asNumber(rawRouting.default_output_tokens),
      thinking_policy:
        rawRouting.thinking_policy === 'long' || rawRouting.thinking_policy === 'reject'
          ? rawRouting.thinking_policy
          : ('short' as const),
      overflow_policy: 'reject' as const,
    },
  };
}

export function normalizeTokenRouterList(value: unknown): TokenRouterItem[] {
  return asArray<unknown>(value)
    .map(normalizeTokenRouterItem)
    .filter((item): item is TokenRouterItem => item !== null);
}

export function normalizeTokenRouterRuntimeInstances(value: unknown): TokenRouterRuntimeInstance[] {
  return asArray<unknown>(value)
    .map(normalizeTokenRouterRuntimeInstance)
    .filter((item): item is TokenRouterRuntimeInstance => item !== null);
}

export function normalizeTokenRouterRuntimeInstance(
  value: unknown
): TokenRouterRuntimeInstance | null {
  if (!isRecord(value)) return null;
  const process = isRecord(value.process) ? value.process : undefined;
  return {
    router_uid: asString(value.router_uid),
    assignment_id: asString(value.assignment_id) || undefined,
    assignment_generation: asNumber(value.assignment_generation),
    node_id: asString(value.node_id) || undefined,
    node_address: asString(value.node_address) || undefined,
    instance_id: asString(value.instance_id, 'unknown-instance'),
    endpoint: asString(value.endpoint),
    status: asString(value.status) || undefined,
    online: Boolean(value.online),
    acked_revision: asNumber(value.acked_revision),
    heartbeat_age_seconds: asNumber(value.heartbeat_age_seconds),
    last_heartbeat: asNumber(value.last_heartbeat),
    config_error: asString(value.config_error) || undefined,
    metrics: isRecord(value.metrics) ? value.metrics : undefined,
    backend_health: isRecord(value.backend_health) ? value.backend_health : undefined,
    process: process
      ? {
          ...process,
          tokenizer_asset: isRecord(process.tokenizer_asset)
            ? {
                asset_id: asString(process.tokenizer_asset.asset_id) || undefined,
                origin:
                  process.tokenizer_asset.origin === 'builtin' ||
                  process.tokenizer_asset.origin === 'external'
                    ? process.tokenizer_asset.origin
                    : undefined,
                revision: asString(process.tokenizer_asset.revision) || undefined,
                fingerprint: asString(process.tokenizer_asset.fingerprint) || undefined,
              }
            : undefined,
        }
      : undefined,
  };
}

export function normalizeTokenRouterAssignments(value: unknown): TokenRouterAssignment[] {
  return asArray<unknown>(value)
    .filter(isRecord)
    .map((item, index) => ({
      assignment_id: asString(item.assignment_id, `assignment-${index + 1}`),
      router_uid: asString(item.router_uid),
      replica_index: asNumber(item.replica_index, index),
      node_id: asString(item.node_id),
      listen_host: asString(item.listen_host),
      listen_port: asNumber(item.listen_port),
      public_endpoint: asString(item.public_endpoint),
      desired_state: item.desired_state === 'stopped' ? 'stopped' : ('running' as const),
      observed_state: asString(item.observed_state, 'unknown'),
      assignment_generation: asNumber(item.assignment_generation),
      config_revision: asNumber(item.config_revision),
      pid: typeof item.pid === 'number' ? item.pid : null,
      instance_id: asString(item.instance_id) || null,
      last_error: asString(item.last_error) || undefined,
      management_state:
        item.management_state === 'node_suspected' || item.management_state === 'node_lost'
          ? item.management_state
          : item.management_state === 'manageable'
            ? 'manageable'
            : undefined,
      failure_reason: asString(item.failure_reason) || undefined,
      observed: isRecord(item.observed) ? item.observed : undefined,
      created_at: asString(item.created_at) || undefined,
      updated_at: asString(item.updated_at) || undefined,
    }));
}

export function normalizeTokenizerAsset(value: unknown): TokenizerAssetItem | null {
  if (!isRecord(value)) return null;
  const origin = ['builtin', 'artifact', 'shared_fs', 'local', 'external'].includes(
    asString(value.origin)
  )
    ? (asString(value.origin) as TokenizerAssetItem['origin'])
    : 'external';
  const status = ['available', 'invalid', 'disabled', 'missing', 'validating'].includes(
    asString(value.status)
  )
    ? (asString(value.status) as TokenizerAssetItem['status'])
    : 'invalid';
  return {
    asset_id: asString(value.asset_id),
    origin,
    display_name: asString(value.display_name),
    model_family: asString(value.model_family),
    model_name: asString(value.model_name),
    revision: asString(value.revision),
    encoding_type: asString(value.encoding_type),
    compatible_models: asArray<unknown>(value.compatible_models).filter(
      (item): item is string => typeof item === 'string'
    ),
    capabilities: isRecord(value.capabilities) ? value.capabilities : {},
    enabled: Boolean(value.enabled),
    status,
    valid: Boolean(value.valid),
    fingerprint: asString(value.fingerprint),
    errors: asArray<unknown>(value.errors).map(String),
    checks: isRecord(value.checks)
      ? Object.fromEntries(Object.entries(value.checks).map(([key, item]) => [key, String(item)]))
      : undefined,
    validated_at: asString(value.validated_at) || undefined,
    source: isRecord(value.source) ? value.source : undefined,
    metadata: isRecord(value.metadata) ? value.metadata : undefined,
    bindings: asNumber(value.bindings),
    ready_bindings: asNumber(value.ready_bindings),
    binding_states: isRecord(value.binding_states)
      ? Object.fromEntries(
          Object.entries(value.binding_states).map(([key, item]) => [key, asNumber(item)])
        )
      : undefined,
    router_references: asArray<unknown>(value.router_references).filter(
      (item): item is string => typeof item === 'string'
    ),
  };
}

export function createTypedDraft(): TypedRouterDraft {
  return {
    backends: [
      {
        id: 'short',
        model_uid: '',
        max_context_tokens: 131072,
        admission: admission(8, 32),
      },
      {
        id: 'long',
        model_uid: '',
        max_context_tokens: 1048576,
        admission: admission(1, 2),
      },
    ],
    rules: [
      {
        id: 'short-budget',
        priority: 100,
        match: { total_tokens_lte: 32768 },
        action: { type: 'route', backend_id: 'short' },
      },
      {
        id: 'long-budget',
        priority: 90,
        match: { total_tokens_gte: 32769 },
        action: { type: 'route', backend_id: 'long' },
      },
    ],
    defaultAction: { type: 'reject', reason: 'context_length_exceeded' },
  };
}

export function typedDraftFromRouter(router?: TokenRouterItem | null): TypedRouterDraft {
  if (!router) return createTypedDraft();
  if (isTypedTokenRouter(router)) {
    return {
      backends: router.backends.map((backend) => ({
        ...backend,
        admission: { ...backend.admission },
      })),
      rules: router.routing.rules.map((rule) => ({
        ...rule,
        match: { ...rule.match },
        action: { ...rule.action },
      })),
      defaultAction: { ...router.routing.default_action },
    };
  }

  const thinkingRule: TokenRouterRoutingRule = {
    id: 'thinking-policy',
    priority: 110,
    match: { thinking: true },
    action:
      router.routing.thinking_policy === 'reject'
        ? { type: 'reject', reason: 'thinking_not_supported' }
        : { type: 'route', backend_id: router.routing.thinking_policy },
  };
  return {
    backends: [
      { id: 'short', ...router.backends.short, admission: { ...router.backends.short.admission } },
      { id: 'long', ...router.backends.long, admission: { ...router.backends.long.admission } },
    ],
    rules: [
      thinkingRule,
      {
        id: 'short-budget',
        priority: 100,
        match: { total_tokens_lte: router.routing.short_threshold_tokens },
        action: { type: 'route', backend_id: 'short' },
      },
      {
        id: 'long-budget',
        priority: 90,
        match: { total_tokens_gte: router.routing.short_threshold_tokens + 1 },
        action: { type: 'route', backend_id: 'long' },
      },
    ],
    defaultAction: { type: 'reject', reason: 'context_length_exceeded' },
  };
}

export function routerBackendList(router: TokenRouterItem): TokenRouterDynamicBackendConfig[] {
  if (isTypedTokenRouter(router)) return router.backends;
  return [
    { id: 'short', ...router.backends.short },
    { id: 'long', ...router.backends.long },
  ];
}

export function routerRuleList(router: TokenRouterItem): TokenRouterRoutingRule[] {
  return typedDraftFromRouter(router).rules;
}

export function routerContextReserve(router: TokenRouterItem): number {
  return router.routing.context_reserve_tokens;
}

export function routerDefaultOutput(router: TokenRouterItem): number {
  return router.routing.default_output_tokens;
}

export function routerMode(router?: TokenRouterItem | null): RouterMode {
  return router && isTypedTokenRouter(router) ? 'advanced' : 'simple';
}

export function asTypedRouter(router: TokenRouterItem): TokenRouterTypedItem | null {
  return isTypedTokenRouter(router) ? router : null;
}
