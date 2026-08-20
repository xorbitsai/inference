'use client';

import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Form } from '@/components/ui/form';
import { FormField } from '@/components/ui/form-field';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { useI18n, type TFunc } from '@/contexts/i18n-context';
import { useForm } from '@/hooks/use-form';
import request from '@/lib/request';
import type {
  TokenizerAssetItem,
  TokenizerAssetListResponse,
  TokenRouterBackendCandidate,
  TokenRouterBackendCandidateResponse,
  TokenRouterDefaultsResponse,
  TokenRouterItem,
} from '@/types/services';
import { isTypedTokenRouter } from '@/types/services';

import { AdvancedRoutingEditor } from './advanced-routing-editor';
import { BackendModelSelect } from './backend-model-select';
import {
  routerMode,
  typedDraftFromRouter,
  type RouterMode,
  type TypedRouterDraft,
} from './router-config-normalizer';

interface Props {
  open: boolean;
  router?: TokenRouterItem | null;
  onOpenChange: (open: boolean) => void;
  onSaved: () => void;
}

type FormState = {
  router_uid: string;
  virtual_model_uid: string;
  tokenizer_source: 'asset' | 'custom';
  tokenizer_asset_id: string;
  tokenizer_path: string;
  backend_url: string;
  model_aliases: string;
  short_model_uid: string;
  long_model_uid: string;
  short_threshold_tokens: number;
  short_max_context: number;
  long_max_context: number;
  context_reserve_tokens: number;
  default_output_tokens: number;
  thinking_policy: 'short' | 'long' | 'reject';
  request_timeout_seconds: number;
  connect_timeout_seconds: number;
  short_max_active: number;
  short_max_queue: number;
  long_max_active: number;
  long_max_queue: number;
  tokenization_workers: number;
  tokenization_max_active: number;
  tokenization_max_queue: number;
  management_mode: 'external' | 'managed';
  desired_replicas: number;
  placement_mode: 'auto' | 'node';
  placement_node_id: string;
  auto_failover: 'enabled' | 'disabled';
  drain_timeout_seconds: number;
};

const EMPTY: FormState = {
  router_uid: '',
  virtual_model_uid: '',
  tokenizer_source: 'asset',
  tokenizer_asset_id: '',
  tokenizer_path: '',
  backend_url: '',
  model_aliases: '',
  short_model_uid: '',
  long_model_uid: '',
  short_threshold_tokens: 32768,
  short_max_context: 131072,
  long_max_context: 1048576,
  context_reserve_tokens: 64,
  default_output_tokens: 512,
  thinking_policy: 'long',
  request_timeout_seconds: 10800,
  connect_timeout_seconds: 10,
  short_max_active: 8,
  short_max_queue: 32,
  long_max_active: 1,
  long_max_queue: 2,
  tokenization_workers: 2,
  tokenization_max_active: 2,
  tokenization_max_queue: 8,
  management_mode: 'external',
  desired_replicas: 1,
  placement_mode: 'auto',
  placement_node_id: '',
  auto_failover: 'disabled',
  drain_timeout_seconds: 7200,
};

function fromRouter(router: TokenRouterItem): FormState {
  const typed = isTypedTokenRouter(router);
  const short = typed ? router.backends[0] : router.backends.short;
  const long = typed ? router.backends[1] || router.backends[0] : router.backends.long;
  const configuredNodeIds = router.deployment?.placement?.node_ids;
  const configuredNodeId =
    (Array.isArray(configuredNodeIds) ? configuredNodeIds[0] : undefined) ||
    router.deployment?.placement?.node_id;
  return {
    router_uid: router.router_uid,
    virtual_model_uid: router.virtual_model_uid,
    tokenizer_source: router.tokenizer_asset_id ? 'asset' : 'custom',
    tokenizer_asset_id: router.tokenizer_asset_id || '',
    tokenizer_path: router.tokenizer_path,
    backend_url: router.backend_url,
    model_aliases: router.model_aliases.join(', '),
    short_model_uid: short?.model_uid || '',
    long_model_uid: long?.model_uid || '',
    short_threshold_tokens: typed
      ? router.routing.rules.find((rule) => rule.match.total_tokens_lte !== undefined)?.match
          .total_tokens_lte || 32768
      : router.routing.short_threshold_tokens,
    short_max_context: short?.max_context_tokens || 131072,
    long_max_context: long?.max_context_tokens || 1048576,
    context_reserve_tokens: router.routing.context_reserve_tokens,
    default_output_tokens: router.routing.default_output_tokens,
    thinking_policy: 'long',
    request_timeout_seconds: router.request_timeout_seconds,
    connect_timeout_seconds: router.connect_timeout_seconds,
    short_max_active: short?.admission.max_active || 1,
    short_max_queue: short?.admission.max_queue || 0,
    long_max_active: long?.admission.max_active || 1,
    long_max_queue: long?.admission.max_queue || 0,
    tokenization_workers: router.tokenization.max_workers,
    tokenization_max_active: router.tokenization.max_active,
    tokenization_max_queue: router.tokenization.max_queue,
    management_mode: router.deployment?.management_mode || 'external',
    desired_replicas: router.deployment?.desired_replicas ?? 1,
    placement_mode: configuredNodeId ? 'node' : 'auto',
    placement_node_id: String(configuredNodeId || ''),
    auto_failover: router.deployment?.rollout?.auto_failover ? 'enabled' : 'disabled',
    drain_timeout_seconds: Number(router.deployment?.rollout?.drain_timeout_seconds ?? 7200),
  };
}

const normalizeNumber = (value: unknown) => (value === '' ? '' : Number(value));
const isPositiveInteger = (value: unknown) => Number.isInteger(Number(value)) && Number(value) > 0;
const isNonNegativeInteger = (value: unknown) =>
  Number.isInteger(Number(value)) && Number(value) >= 0;

function normalizeBackendUrl(value: string) {
  return value.trim().replace(/\/+$/, '');
}

function isBackendUrl(value: unknown) {
  if (typeof value !== 'string') return false;
  try {
    const url = new URL(value.trim());
    return (
      (url.protocol === 'http:' || url.protocol === 'https:') &&
      Boolean(url.hostname) &&
      !url.username &&
      !url.password &&
      (url.pathname === '' || url.pathname === '/') &&
      !url.search &&
      !url.hash
    );
  } catch {
    return false;
  }
}

export function RouterFormDialog({ open, router, onOpenChange, onSaved }: Props) {
  const { t } = useI18n();
  const [saving, setSaving] = useState(false);
  const [candidates, setCandidates] = useState<TokenRouterBackendCandidate[]>([]);
  const [candidatesLoading, setCandidatesLoading] = useState(false);
  const [candidatesLoadFailed, setCandidatesLoadFailed] = useState(false);
  const [candidateErrors, setCandidateErrors] = useState<string[]>([]);
  const [simpleBackendUids, setSimpleBackendUids] = useState({ short: '', long: '' });
  const [mode, setMode] = useState<RouterMode>(routerMode(router));
  const [typedDraft, setTypedDraft] = useState<TypedRouterDraft>(() =>
    typedDraftFromRouter(router)
  );
  const [assets, setAssets] = useState<TokenizerAssetItem[]>([]);
  const [allowCustomPath, setAllowCustomPath] = useState(false);
  const [assetsLoadFailed, setAssetsLoadFailed] = useState(false);
  const [backendDefaults, setBackendDefaults] = useState<
    TokenRouterDefaultsResponse['backend'] | null
  >(null);
  const [backendDefaultsLoadFailed, setBackendDefaultsLoadFailed] = useState(false);
  const [useCustomBackend, setUseCustomBackend] = useState(false);
  const [tokenizerSource, setTokenizerSource] = useState<'asset' | 'custom'>('asset');
  const [selectedAssetId, setSelectedAssetId] = useState('');
  const [form] = useForm();
  const managementMode = form.getFieldValue('management_mode') as
    | FormState['management_mode']
    | undefined;
  const placementMode = form.getFieldValue('placement_mode') as
    | FormState['placement_mode']
    | undefined;
  const editing = Boolean(router);
  const canUseCustomPath = allowCustomPath || Boolean(router && !router.tokenizer_asset_id);
  const assetOptions = useMemo(() => {
    if (!router?.tokenizer_asset_id) return assets;
    if (assets.some((asset) => asset.asset_id === router.tokenizer_asset_id)) return assets;
    return [
      ...assets,
      {
        asset_id: router.tokenizer_asset_id,
        origin: router.tokenizer_asset_origin || 'external',
        display_name: router.tokenizer_asset_id,
        model_family: '',
        model_name: '',
        revision: router.tokenizer_asset_revision || '',
        encoding_type: '',
        compatible_models: [],
        capabilities: {},
        enabled: true,
        status: 'invalid' as const,
        valid: false,
        fingerprint: router.tokenizer_asset_fingerprint || '',
        errors: [],
      },
    ];
  }, [assets, router]);
  const selectedAsset = assetOptions.find((asset) => asset.asset_id === selectedAssetId);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;

    setAssets([]);
    setAllowCustomPath(false);
    setAssetsLoadFailed(false);
    setBackendDefaults(null);
    setBackendDefaultsLoadFailed(false);
    setUseCustomBackend(false);

    const load = async () => {
      const [assetsResult, defaultsResult] = await Promise.allSettled([
        request.get<TokenizerAssetListResponse>('/v1/tokenizer_assets'),
        request.get<TokenRouterDefaultsResponse>('/v1/token_routers/defaults'),
      ]);
      if (cancelled) return;

      const assetResponse =
        assetsResult.status === 'fulfilled'
          ? assetsResult.value
          : { items: [], allow_custom_path: false };
      const availableAssets = assetResponse.items || [];
      setAssets(availableAssets);
      setAllowCustomPath(assetResponse.allow_custom_path);
      setAssetsLoadFailed(
        assetsResult.status === 'rejected' || Boolean(assetResponse.config_error)
      );

      const defaults = defaultsResult.status === 'fulfilled' ? defaultsResult.value.backend : null;
      const defaultBackendUrl =
        defaults?.available && defaults.backend_url
          ? normalizeBackendUrl(defaults.backend_url)
          : '';
      setBackendDefaults(defaults);
      setBackendDefaultsLoadFailed(defaultsResult.status === 'rejected');

      const values: FormState = router
        ? fromRouter(router)
        : { ...EMPTY, backend_url: defaultBackendUrl };
      const customBackend = router
        ? !defaultBackendUrl || normalizeBackendUrl(router.backend_url) !== defaultBackendUrl
        : !defaultBackendUrl;
      setUseCustomBackend(customBackend);
      if (!router) {
        const defaultAsset = availableAssets.find((asset) => asset.status === 'available');
        if (defaultAsset) {
          values.tokenizer_source = 'asset';
          values.tokenizer_asset_id = defaultAsset.asset_id;
        } else {
          values.tokenizer_source = assetResponse.allow_custom_path ? 'custom' : 'asset';
        }
      }
      setMode(routerMode(router));
      setTypedDraft(typedDraftFromRouter(router));
      setTokenizerSource(values.tokenizer_source);
      setSelectedAssetId(values.tokenizer_asset_id);
      setSimpleBackendUids({
        short: values.short_model_uid,
        long: values.long_model_uid,
      });
      form.initialValues.current = values;
      form.resetFields();
      form.setFieldsValue(values);
    };

    void load();
    return () => {
      cancelled = true;
    };
  }, [form, open, router]);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    const assetQuery =
      tokenizerSource === 'asset' && selectedAssetId
        ? `?tokenizer_asset_id=${encodeURIComponent(selectedAssetId)}`
        : '';

    setCandidates([]);
    setCandidatesLoading(true);
    setCandidatesLoadFailed(false);
    setCandidateErrors([]);
    void request
      .get<TokenRouterBackendCandidateResponse>(`/v1/token_routers/backend-candidates${assetQuery}`)
      .then((response) => {
        if (cancelled) return;
        setCandidates(response.items || []);
        setCandidateErrors(response.errors || []);
      })
      .catch(() => {
        if (cancelled) return;
        setCandidates([]);
        setCandidatesLoadFailed(true);
      })
      .finally(() => {
        if (!cancelled) setCandidatesLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [open, selectedAssetId, tokenizerSource]);

  const requiredRule = { required: true, message: t('tokenRouter.validation.required') } as const;
  const positiveIntegerRule = {
    validator: isPositiveInteger,
    message: t('tokenRouter.validation.positiveInteger'),
  } as const;
  const nonNegativeIntegerRule = {
    validator: isNonNegativeInteger,
    message: t('tokenRouter.validation.nonNegativeInteger'),
  } as const;

  const handleSave = async (rawValues: Record<string, unknown>) => {
    const values = rawValues as FormState;
    if (router) {
      const sourceChanged = values.tokenizer_source !== fromRouter(router).tokenizer_source;
      const tokenizerChanged =
        sourceChanged ||
        (values.tokenizer_source === 'asset'
          ? values.tokenizer_asset_id.trim() !== (router.tokenizer_asset_id || '')
          : values.tokenizer_path.trim() !== router.tokenizer_path);
      if (tokenizerChanged && !window.confirm(t('tokenRouter.assetChangeWarning'))) return;
    }

    setSaving(true);

    const admission = (max_active: number, max_queue: number) => ({
      max_active,
      max_queue,
      queue_timeout_seconds: 5,
      retry_after_seconds: 1,
    });
    if (mode === 'advanced') {
      const backendIds = typedDraft.backends.map((backend) => backend.id.trim());
      const ruleIds = typedDraft.rules.map((rule) => rule.id.trim());
      const priorities = typedDraft.rules.map((rule) => rule.priority);
      const invalidRule = typedDraft.rules.some((rule) => {
        const hasCondition = Object.values(rule.match).some(
          (value) => value !== undefined && value !== null
        );
        const invalidTokenRange =
          rule.match.total_tokens_gte !== undefined &&
          rule.match.total_tokens_lte !== undefined &&
          rule.match.total_tokens_gte > rule.match.total_tokens_lte;
        const invalidAction =
          rule.action.type === 'route'
            ? !backendIds.includes(rule.action.backend_id)
            : !rule.action.reason.trim();
        return !hasCondition || invalidTokenRange || invalidAction;
      });
      if (
        typedDraft.backends.length < 1 ||
        typedDraft.backends.length > 16 ||
        typedDraft.rules.length < 1 ||
        typedDraft.rules.length > 64 ||
        typedDraft.backends.some(
          (backend) =>
            !/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(backend.id.trim()) ||
            !backend.model_uid.trim() ||
            backend.max_context_tokens < 1 ||
            backend.admission.max_active < 1 ||
            backend.admission.max_queue < 0
        ) ||
        new Set(backendIds).size !== backendIds.length ||
        typedDraft.rules.some(
          (rule) =>
            !/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(rule.id.trim()) ||
            rule.priority < 1 ||
            rule.priority > 10000
        ) ||
        new Set(ruleIds).size !== ruleIds.length ||
        new Set(priorities).size !== priorities.length ||
        invalidRule ||
        (typedDraft.defaultAction.type === 'route'
          ? !backendIds.includes(typedDraft.defaultAction.backend_id)
          : !typedDraft.defaultAction.reason.trim())
      ) {
        toast.error(t('tokenRouter.validation.invalidAdvancedConfig'));
        setSaving(false);
        return;
      }
    }

    const commonPayload = {
      virtual_model_uid: values.virtual_model_uid.trim(),
      model_type: 'LLM' as const,
      route_profile: 'llm_chat' as const,
      ...(values.tokenizer_source === 'asset'
        ? { tokenizer_asset_id: values.tokenizer_asset_id.trim() }
        : { tokenizer_path: values.tokenizer_path.trim() }),
      backend_url: values.backend_url.trim(),
      model_aliases: values.model_aliases
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean),
      request_timeout_seconds: values.request_timeout_seconds,
      connect_timeout_seconds: values.connect_timeout_seconds,
      tokenization: {
        executor: 'process' as const,
        multiprocessing_start_method: 'spawn' as const,
        max_workers: values.tokenization_workers,
        max_active: values.tokenization_max_active,
        max_queue: values.tokenization_max_queue,
        queue_timeout_seconds: 5,
        retry_after_seconds: 1,
      },
    };
    const payload =
      mode === 'advanced'
        ? {
            ...commonPayload,
            config_version: 2 as const,
            strategy: 'typed_rules' as const,
            backends: typedDraft.backends.map((backend) => ({
              ...backend,
              id: backend.id.trim(),
              model_uid: backend.model_uid.trim(),
            })),
            routing: {
              evaluation_mode: 'first_match' as const,
              context_reserve_tokens: values.context_reserve_tokens,
              default_output_tokens: values.default_output_tokens,
              rules: typedDraft.rules.map((rule) => ({ ...rule, id: rule.id.trim() })),
              default_action: typedDraft.defaultAction,
            },
          }
        : {
            ...commonPayload,
            config_version: 1 as const,
            strategy: 'token_budget' as const,
            backends: {
              short: {
                model_uid: values.short_model_uid.trim(),
                max_context_tokens: values.short_max_context,
                admission: admission(values.short_max_active, values.short_max_queue),
              },
              long: {
                model_uid: values.long_model_uid.trim(),
                max_context_tokens: values.long_max_context,
                admission: admission(values.long_max_active, values.long_max_queue),
              },
            },
            routing: {
              short_threshold_tokens: values.short_threshold_tokens,
              context_reserve_tokens: values.context_reserve_tokens,
              default_output_tokens: values.default_output_tokens,
              thinking_policy: values.thinking_policy,
              overflow_policy: 'reject' as const,
            },
          };

    const placement: Record<string, unknown> = {
      ...(router?.deployment?.placement || {}),
    };
    delete placement.node_id;
    delete placement.node_ids;
    if (values.placement_mode === 'node') {
      placement.node_ids = [values.placement_node_id.trim()];
    }

    try {
      const routerUid = router?.router_uid || values.router_uid.trim();
      if (router) {
        await request.put(`/v1/token_routers/${router.router_uid}`, {
          ...payload,
          revision: router.revision,
        });
      } else {
        await request.post('/v1/token_routers', {
          ...payload,
          router_uid: routerUid,
        });
      }
      await request.put(`/v1/token_routers/${routerUid}/deployment`, {
        management_mode: values.management_mode,
        desired_replicas: values.desired_replicas,
        placement,
        rollout: {
          ...(router?.deployment?.rollout || {}),
          auto_failover: values.auto_failover === 'enabled',
          drain_timeout_seconds: values.drain_timeout_seconds,
        },
        ...(router?.deployment?.deployment_generation
          ? { deployment_generation: router.deployment.deployment_generation }
          : {}),
      });
      toast.success(t(router ? 'tokenRouter.updateSuccess' : 'tokenRouter.createSuccess'));
      onOpenChange(false);
      onSaved();
    } finally {
      setSaving(false);
    }
  };

  const numberField = (
    name: keyof FormState,
    label: string,
    allowZero = false,
    extraRules: Array<{ validator: (value: unknown) => boolean; message: string }> = []
  ) => (
    <FormField
      name={name}
      label={label}
      normalize={normalizeNumber}
      rules={[allowZero ? nonNegativeIntegerRule : positiveIntegerRule, ...extraRules]}
    >
      <Input type="number" step="1" />
    </FormField>
  );

  return (
    <Dialog open={open} onOpenChange={(nextOpen) => !saving && onOpenChange(nextOpen)}>
      <DialogContent className="max-h-[90vh] w-[calc(100%-2rem)] overflow-y-auto sm:max-w-4xl">
        <DialogHeader>
          <DialogTitle>{editing ? t('tokenRouter.edit') : t('tokenRouter.create')}</DialogTitle>
        </DialogHeader>

        <Form form={form} onFinish={handleSave} className="space-y-6">
          <FormSection title={t('tokenRouter.sections.basic')}>
            <FormField
              name="router_uid"
              label={t('tokenRouter.routerUid')}
              extra={t('tokenRouter.routerUidHint')}
              disabled={editing}
              rules={[requiredRule]}
            >
              <Input />
            </FormField>
            <FormField
              name="virtual_model_uid"
              label={t('tokenRouter.virtualModelUid')}
              extra={t('tokenRouter.virtualModelUidHint')}
              rules={[requiredRule]}
            >
              <Input />
            </FormField>
            <FormField
              name="tokenizer_source"
              label={t('tokenRouter.tokenizerSource')}
              rules={[requiredRule]}
            >
              <Select<FormState['tokenizer_source']>
                allowClear={false}
                options={[
                  { value: 'asset', label: t('tokenRouter.tokenizerAsset') },
                  {
                    value: 'custom',
                    label: t('tokenRouter.customTokenizerPath'),
                    disabled: !canUseCustomPath,
                  },
                ]}
                onChange={(value) => setTokenizerSource(value || 'asset')}
              />
            </FormField>
            {tokenizerSource === 'asset' ? (
              <FormField
                name="tokenizer_asset_id"
                label={t('tokenRouter.tokenizerAsset')}
                rules={[requiredRule]}
              >
                <Select
                  showSearch
                  allowClear={false}
                  options={assetOptions.map((asset) => ({
                    value: asset.asset_id,
                    label: asset.display_name || asset.asset_id,
                    description: `${asset.asset_id} · ${t(
                      `tokenRouter.assetOrigins.${asset.origin}`
                    )} · ${asset.revision || '-'} · ${t(
                      `tokenRouter.assetStatuses.${asset.status}`
                    )}`,
                    disabled:
                      asset.status !== 'available' && asset.asset_id !== router?.tokenizer_asset_id,
                  }))}
                  onChange={(value) => setSelectedAssetId(value || '')}
                />
              </FormField>
            ) : (
              <FormField
                name="tokenizer_path"
                label={t('tokenRouter.tokenizerPath')}
                rules={[requiredRule]}
                tooltip={t('tokenRouter.customTokenizerPathHint')}
              >
                <Input />
              </FormField>
            )}
            {assetsLoadFailed && tokenizerSource === 'asset' && (
              <div className="md:col-span-2 rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
                {t('tokenRouter.tokenizerAssetsLoadFailed')}
              </div>
            )}
            {assets.length === 0 && tokenizerSource === 'asset' && (
              <div className="md:col-span-2 rounded-lg border border-dashed p-3 text-sm text-muted-foreground">
                {t('tokenRouter.noTokenizerAssets')}
              </div>
            )}
            {tokenizerSource === 'asset' && selectedAsset && (
              <TokenizerAssetSummary asset={selectedAsset} t={t} />
            )}
            {tokenizerSource === 'custom' && (
              <div className="md:col-span-2 rounded-lg border border-amber-500/40 bg-amber-500/5 p-3 text-sm text-amber-700 dark:text-amber-300">
                <div className="font-medium">{t('tokenRouter.legacyCustomTokenizerPath')}</div>
                <div className="mt-1 text-xs">{t('tokenRouter.customTokenizerPathRisk')}</div>
              </div>
            )}
            <div className="md:col-span-2 space-y-3 rounded-lg border bg-muted/20 p-3">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-sm font-medium">{t('tokenRouter.currentSupervisor')}</div>
                  <div className="mt-1 text-xs text-muted-foreground">
                    {t('tokenRouter.currentSupervisorDescription')}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">
                    {t('tokenRouter.useCustomBackend')}
                  </span>
                  <Switch
                    checked={useCustomBackend}
                    disabled={!backendDefaults?.available}
                    onChange={(checked) => {
                      setUseCustomBackend(checked);
                      if (!checked && backendDefaults?.backend_url) {
                        form.setFieldValue(
                          'backend_url',
                          normalizeBackendUrl(backendDefaults.backend_url)
                        );
                      }
                    }}
                  />
                </div>
              </div>
              {backendDefaults?.available && backendDefaults.backend_url && (
                <div>
                  <div className="text-xs text-muted-foreground">
                    {t('tokenRouter.backendAddress')}
                  </div>
                  <div className="mt-1 break-all font-mono text-xs">
                    {backendDefaults.backend_url}
                  </div>
                </div>
              )}
              {!useCustomBackend && !backendDefaults && !backendDefaultsLoadFailed && (
                <div className="text-xs text-muted-foreground">
                  {t('tokenRouter.backendDefaultsLoading')}
                </div>
              )}
              {backendDefaultsLoadFailed && (
                <div className="text-xs text-destructive">
                  {t('tokenRouter.backendDefaultsLoadFailed')}
                </div>
              )}
              {backendDefaults && !backendDefaults.available && (
                <div className="text-xs text-destructive">
                  {t('tokenRouter.backendDefaultsUnavailable')}
                </div>
              )}
              <div className="text-xs text-muted-foreground">
                {t('tokenRouter.customBackendHint')}
              </div>
            </div>
            {useCustomBackend && (
              <FormField
                name="backend_url"
                label={t('tokenRouter.backendUrl')}
                className="md:col-span-2"
                rules={[
                  requiredRule,
                  {
                    validator: isBackendUrl,
                    message: t('tokenRouter.validation.invalidUrl'),
                  },
                ]}
              >
                <Input placeholder="http://xinference-supervisor:9997" />
              </FormField>
            )}
            <FormField
              name="model_aliases"
              label={t('tokenRouter.aliases')}
              className="md:col-span-2"
            >
              <Input />
            </FormField>
          </FormSection>

          <FormSection title={t('tokenRouter.configurationMode')}>
            <div className="md:col-span-2 grid gap-3 sm:grid-cols-2">
              <button
                type="button"
                className={`rounded-lg border p-4 text-left transition-colors ${
                  mode === 'simple' ? 'border-primary bg-primary/5' : 'hover:bg-muted/40'
                } ${router && isTypedTokenRouter(router) ? 'cursor-not-allowed opacity-60' : ''}`}
                disabled={Boolean(router && isTypedTokenRouter(router))}
                onClick={() => setMode('simple')}
              >
                <div className="font-medium">{t('tokenRouter.simpleMode')}</div>
                <div className="mt-1 text-xs text-muted-foreground">
                  {t('tokenRouter.simpleModeDescription')}
                </div>
              </button>
              <button
                type="button"
                className={`rounded-lg border p-4 text-left transition-colors ${
                  mode === 'advanced' ? 'border-primary bg-primary/5' : 'hover:bg-muted/40'
                }`}
                onClick={() => {
                  if (mode === 'simple' && !router) {
                    const next = typedDraftFromRouter(null);
                    next.backends[0].model_uid = String(
                      form.getFieldValue('short_model_uid') || ''
                    );
                    next.backends[0].max_context_tokens = Number(
                      form.getFieldValue('short_max_context') || 131072
                    );
                    next.backends[0].admission.max_active = Number(
                      form.getFieldValue('short_max_active') || 8
                    );
                    next.backends[0].admission.max_queue = Number(
                      form.getFieldValue('short_max_queue') || 0
                    );
                    next.backends[1].model_uid = String(form.getFieldValue('long_model_uid') || '');
                    next.backends[1].max_context_tokens = Number(
                      form.getFieldValue('long_max_context') || 1048576
                    );
                    next.backends[1].admission.max_active = Number(
                      form.getFieldValue('long_max_active') || 1
                    );
                    next.backends[1].admission.max_queue = Number(
                      form.getFieldValue('long_max_queue') || 0
                    );
                    const threshold = Number(form.getFieldValue('short_threshold_tokens') || 32768);
                    const shortRule = next.rules.find((rule) => rule.id === 'short-budget');
                    const longRule = next.rules.find((rule) => rule.id === 'long-budget');
                    if (shortRule) shortRule.match.total_tokens_lte = threshold;
                    if (longRule) longRule.match.total_tokens_gte = threshold + 1;
                    const thinkingPolicy = String(form.getFieldValue('thinking_policy') || 'long');
                    next.rules.push({
                      id: 'thinking-policy',
                      priority: 110,
                      match: { thinking: true },
                      action:
                        thinkingPolicy === 'reject'
                          ? { type: 'reject', reason: 'thinking_not_supported' }
                          : { type: 'route', backend_id: thinkingPolicy },
                    });
                    setTypedDraft(next);
                  }
                  setMode('advanced');
                }}
              >
                <div className="font-medium">{t('tokenRouter.advancedMode')}</div>
                <div className="mt-1 text-xs text-muted-foreground">
                  {t('tokenRouter.advancedModeDescription')}
                </div>
              </button>
            </div>
          </FormSection>

          {mode === 'simple' ? (
            <>
              <FormSection title={t('tokenRouter.sections.backends')}>
                <FormField
                  name="short_model_uid"
                  label={t('tokenRouter.shortBackend')}
                  rules={[requiredRule]}
                >
                  <BackendModelSelect
                    candidates={candidates}
                    loading={candidatesLoading}
                    loadFailed={candidatesLoadFailed}
                    candidateErrors={candidateErrors}
                    tokenizerCompatibilityVerified={tokenizerSource === 'asset'}
                    excludedModelUids={[simpleBackendUids.long]}
                    onChange={(modelUid) =>
                      setSimpleBackendUids((current) => ({
                        ...current,
                        short: String(modelUid || ''),
                      }))
                    }
                  />
                </FormField>
                <FormField
                  name="long_model_uid"
                  label={t('tokenRouter.longBackend')}
                  rules={[
                    requiredRule,
                    {
                      validator: (value) =>
                        String(value).trim() !==
                        String(form.getFieldValue('short_model_uid') || '').trim(),
                      message: t('tokenRouter.validation.backendsMustDiffer'),
                    },
                  ]}
                >
                  <BackendModelSelect
                    candidates={candidates}
                    loading={candidatesLoading}
                    loadFailed={candidatesLoadFailed}
                    candidateErrors={candidateErrors}
                    tokenizerCompatibilityVerified={tokenizerSource === 'asset'}
                    excludedModelUids={[simpleBackendUids.short]}
                    onChange={(modelUid) =>
                      setSimpleBackendUids((current) => ({
                        ...current,
                        long: String(modelUid || ''),
                      }))
                    }
                  />
                </FormField>
                {numberField('short_max_context', t('tokenRouter.shortContext'))}
                {numberField('long_max_context', t('tokenRouter.longContext'), false, [
                  {
                    validator: (value) =>
                      Number(value) >= Number(form.getFieldValue('short_max_context')),
                    message: t('tokenRouter.validation.shortContextExceedsLongContext'),
                  },
                ])}
              </FormSection>

              <FormSection title={t('tokenRouter.sections.routing')}>
                {numberField('short_threshold_tokens', t('tokenRouter.threshold'), false, [
                  {
                    validator: (value) =>
                      Number(value) <= Number(form.getFieldValue('short_max_context')),
                    message: t('tokenRouter.validation.thresholdExceedsShortContext'),
                  },
                ])}
                {numberField('context_reserve_tokens', t('tokenRouter.reserve'), true)}
                {numberField('default_output_tokens', t('tokenRouter.defaultOutput'))}
                <FormField
                  name="thinking_policy"
                  label={t('tokenRouter.thinkingPolicy')}
                  rules={[requiredRule]}
                >
                  <Select<FormState['thinking_policy']>
                    allowClear={false}
                    options={[
                      { value: 'long', label: t('tokenRouter.thinkingPolicies.long') },
                      { value: 'short', label: t('tokenRouter.thinkingPolicies.short') },
                      { value: 'reject', label: t('tokenRouter.thinkingPolicies.reject') },
                    ]}
                  />
                </FormField>
              </FormSection>

              <FormSection title={t('tokenRouter.sections.admission')}>
                {numberField('short_max_active', t('tokenRouter.shortActive'))}
                {numberField('short_max_queue', t('tokenRouter.shortQueue'), true)}
                {numberField('long_max_active', t('tokenRouter.longActive'))}
                {numberField('long_max_queue', t('tokenRouter.longQueue'), true)}
              </FormSection>
            </>
          ) : (
            <FormSection title={t('tokenRouter.advancedConfiguration')}>
              {numberField('context_reserve_tokens', t('tokenRouter.reserve'), true)}
              {numberField('default_output_tokens', t('tokenRouter.defaultOutput'))}
              <AdvancedRoutingEditor
                value={typedDraft}
                candidates={candidates}
                candidatesLoading={candidatesLoading}
                candidatesLoadFailed={candidatesLoadFailed}
                candidateErrors={candidateErrors}
                tokenizerCompatibilityVerified={tokenizerSource === 'asset'}
                onChange={setTypedDraft}
              />
            </FormSection>
          )}

          <FormSection title={t('tokenRouter.sections.deployment')}>
            <FormField
              name="management_mode"
              label={t('tokenRouter.managementMode')}
              rules={[requiredRule]}
            >
              <Select<FormState['management_mode']>
                allowClear={false}
                options={[
                  { value: 'external', label: t('tokenRouter.managementModes.external') },
                  { value: 'managed', label: t('tokenRouter.managementModes.managed') },
                ]}
              />
            </FormField>
            {numberField('desired_replicas', t('tokenRouter.desiredReplicas'), true)}
            <FormField
              name="placement_mode"
              label={t('tokenRouter.placementMode')}
              rules={[requiredRule]}
            >
              <Select<FormState['placement_mode']>
                allowClear={false}
                disabled={managementMode !== 'managed'}
                options={[
                  { value: 'auto', label: t('tokenRouter.placementModes.auto') },
                  { value: 'node', label: t('tokenRouter.placementModes.node') },
                ]}
              />
            </FormField>
            {placementMode === 'node' && (
              <FormField
                name="placement_node_id"
                label={t('tokenRouter.placementNodeId')}
                rules={[requiredRule]}
              >
                <Input placeholder="t-xinference-router-001" />
              </FormField>
            )}
            <FormField
              name="auto_failover"
              label={t('tokenRouter.autoFailover')}
              rules={[requiredRule]}
            >
              <Select<FormState['auto_failover']>
                allowClear={false}
                options={[
                  { value: 'disabled', label: t('tokenRouter.failoverModes.disabled') },
                  { value: 'enabled', label: t('tokenRouter.failoverModes.enabled') },
                ]}
              />
            </FormField>
            {numberField('drain_timeout_seconds', t('tokenRouter.drainTimeout'), false)}
          </FormSection>

          <FormSection title={t('tokenRouter.sections.tokenization')}>
            {numberField('tokenization_workers', t('tokenRouter.tokenWorkers'))}
            {numberField('tokenization_max_active', t('tokenRouter.tokenActive'), false, [
              {
                validator: (value) =>
                  Number(value) >= Number(form.getFieldValue('tokenization_workers')),
                message: t('tokenRouter.validation.tokenActiveBelowWorkers'),
              },
            ])}
            {numberField('tokenization_max_queue', t('tokenRouter.tokenQueue'), true)}
          </FormSection>

          <FormSection title={t('tokenRouter.sections.timeouts')}>
            {numberField('request_timeout_seconds', t('tokenRouter.requestTimeout'))}
            {numberField('connect_timeout_seconds', t('tokenRouter.connectTimeout'))}
          </FormSection>

          <DialogFooter>
            <Button variant="outline" type="button" onClick={() => onOpenChange(false)}>
              {t('common.cancel')}
            </Button>
            <Button type="submit" loading={saving}>
              {t('common.save')}
            </Button>
          </DialogFooter>
        </Form>
      </DialogContent>
    </Dialog>
  );
}

function FormSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-4 rounded-xl border border-border p-4">
      <h3 className="text-sm font-semibold">{title}</h3>
      <div className="grid gap-4 md:grid-cols-2">{children}</div>
    </section>
  );
}

function TokenizerAssetSummary({ asset, t }: { asset: TokenizerAssetItem; t: TFunc }) {
  const capabilityNames = Object.entries(asset.capabilities || {})
    .filter(([, enabled]) => enabled)
    .map(([name]) => t(`tokenRouter.capabilities.${name}`));

  return (
    <div className="md:col-span-2 grid gap-3 rounded-lg border bg-muted/20 p-3 text-sm sm:grid-cols-2">
      <AssetSummaryItem
        label={t('tokenRouter.tokenizerAssetStatus')}
        value={t(`tokenRouter.assetStatuses.${asset.status}`)}
      />
      <AssetSummaryItem
        label={t('tokenRouter.tokenizerAssetOrigin')}
        value={t(`tokenRouter.assetOrigins.${asset.origin}`)}
      />
      <AssetSummaryItem label={t('tokenRouter.revision')} value={asset.revision || '—'} mono />
      <AssetSummaryItem
        label={t('tokenRouter.compatibleModels')}
        value={asset.compatible_models.length ? asset.compatible_models.join(', ') : '—'}
      />
      <AssetSummaryItem
        label={t('tokenRouter.tokenizerCapabilities')}
        value={capabilityNames.length ? capabilityNames.join(', ') : '—'}
      />
    </div>
  );
}

function AssetSummaryItem({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="min-w-0">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={mono ? 'mt-1 break-all font-mono text-xs' : 'mt-1 break-words'}>{value}</div>
    </div>
  );
}
