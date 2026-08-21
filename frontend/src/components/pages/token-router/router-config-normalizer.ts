import type {
  TokenRouterDynamicBackendConfig,
  TokenRouterItem,
  TokenRouterRoutingAction,
  TokenRouterRoutingRule,
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
