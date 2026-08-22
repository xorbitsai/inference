'use client';

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  Activity,
  BarChart3,
  Copy,
  Gauge,
  Loader2,
  PackageCheck,
  RefreshCw,
  Server,
  Settings2,
  X,
} from 'lucide-react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import { CollapsiblePanel } from '@/components/ui/collapsible';
import { Sheet, SheetClose, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useI18n, type TFunc } from '@/contexts/i18n-context';
import request from '@/lib/request';
import { cn } from '@/lib/utils';
import type {
  TokenizerAssetItem,
  TokenRouterAssignment,
  TokenRouterItem,
  TokenRouterRoutingAction,
  TokenRouterRuleMatch,
  TokenRouterRuntimeInstance,
} from '@/types/services';
import { isTypedTokenRouter } from '@/types/services';
import { routerBackendList, routerRuleList } from './router-config-normalizer';
import { RouterStatusBadge } from './router-status-badge';

interface Props {
  router: TokenRouterItem | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

type DetailTab = 'overview' | 'instances' | 'metrics';
type LoadMode = 'initial' | 'manual' | 'silent';
type MetricsResponse = Record<string, unknown>;

interface CapacitySummary {
  active?: number;
  waiting?: number;
  maxActive?: number;
  maxQueue?: number;
}

interface MetricsSummary {
  completedRequests?: number;
  activeRequests?: number;
  waitingRequests?: number;
  onlineInstances?: number;
  backendPools: Record<string, CapacitySummary>;
  tokenization: CapacitySummary;
}

const POLL_INTERVAL_MS = 10000;

export function RouterDetailDrawer({ router, open, onOpenChange }: Props) {
  const { locale, t } = useI18n();
  const routerUid = router?.router_uid;
  const tokenizerAssetId = router?.tokenizer_asset_id;
  const [activeTab, setActiveTab] = useState<DetailTab>('overview');
  const [instances, setInstances] = useState<TokenRouterRuntimeInstance[]>([]);
  const [assignments, setAssignments] = useState<TokenRouterAssignment[]>([]);
  const [metrics, setMetrics] = useState<MetricsResponse>({});
  const [tokenizerAsset, setTokenizerAsset] = useState<TokenizerAssetItem | null>(null);
  const [tokenizerAssetLoading, setTokenizerAssetLoading] = useState(false);
  const [tokenizerAssetLoadFailed, setTokenizerAssetLoadFailed] = useState(false);
  const [instancesInitialLoading, setInstancesInitialLoading] = useState(false);
  const [instancesRefreshing, setInstancesRefreshing] = useState(false);
  const [metricsInitialLoading, setMetricsInitialLoading] = useState(false);
  const [metricsRefreshing, setMetricsRefreshing] = useState(false);
  const [instancesLoaded, setInstancesLoaded] = useState(false);
  const [metricsLoaded, setMetricsLoaded] = useState(false);
  const [instancesUpdatedAt, setInstancesUpdatedAt] = useState<Date | null>(null);
  const [metricsUpdatedAt, setMetricsUpdatedAt] = useState<Date | null>(null);
  const generationRef = useRef(0);
  const instancesRequestRef = useRef<string | null>(null);
  const metricsRequestRef = useRef<string | null>(null);
  const activeRouterUidRef = useRef<string | null>(open && routerUid ? routerUid : null);

  activeRouterUidRef.current = open && routerUid ? routerUid : null;

  const isCurrentRequest = useCallback((generation: number, uid: string) => {
    return generationRef.current === generation && activeRouterUidRef.current === uid;
  }, []);

  const loadInstances = useCallback(
    async (mode: LoadMode = 'silent') => {
      if (!open || !routerUid) return;

      const generation = generationRef.current;
      const requestKey = `${generation}:${routerUid}`;
      if (instancesRequestRef.current === requestKey) return;
      instancesRequestRef.current = requestKey;

      if (mode === 'initial') setInstancesInitialLoading(true);
      if (mode === 'manual') setInstancesRefreshing(true);

      try {
        const [runtimeData, assignmentData] = await Promise.all([
          request.get<TokenRouterRuntimeInstance[]>(`/v1/token_routers/${routerUid}/instances`),
          request.get<TokenRouterAssignment[]>(`/v1/token_routers/${routerUid}/assignments`),
        ]);
        if (!isCurrentRequest(generation, routerUid)) return;

        setInstances(Array.isArray(runtimeData) ? runtimeData : []);
        setAssignments(Array.isArray(assignmentData) ? assignmentData : []);
        setInstancesLoaded(true);
        setInstancesUpdatedAt(new Date());
      } catch {
        // The shared request interceptor displays the request error. Keep the last successful data.
      } finally {
        if (instancesRequestRef.current === requestKey) {
          instancesRequestRef.current = null;
        }
        if (isCurrentRequest(generation, routerUid)) {
          if (mode === 'initial') setInstancesInitialLoading(false);
          if (mode === 'manual') setInstancesRefreshing(false);
        }
      }
    },
    [isCurrentRequest, open, routerUid]
  );

  const loadMetrics = useCallback(
    async (mode: LoadMode = 'silent') => {
      if (!open || !routerUid) return;

      const generation = generationRef.current;
      const requestKey = `${generation}:${routerUid}`;
      if (metricsRequestRef.current === requestKey) return;
      metricsRequestRef.current = requestKey;

      if (mode === 'initial') setMetricsInitialLoading(true);
      if (mode === 'manual') setMetricsRefreshing(true);

      try {
        const metricData = await request.get<MetricsResponse>(
          `/v1/token_routers/${routerUid}/metrics-summary`
        );
        if (!isCurrentRequest(generation, routerUid)) return;

        setMetrics(isRecord(metricData) ? metricData : {});
        setMetricsLoaded(true);
        setMetricsUpdatedAt(new Date());
      } catch {
        // The shared request interceptor displays the request error. Keep the last successful data.
      } finally {
        if (metricsRequestRef.current === requestKey) {
          metricsRequestRef.current = null;
        }
        if (isCurrentRequest(generation, routerUid)) {
          if (mode === 'initial') setMetricsInitialLoading(false);
          if (mode === 'manual') setMetricsRefreshing(false);
        }
      }
    },
    [isCurrentRequest, open, routerUid]
  );

  const loadTokenizerAsset = useCallback(async () => {
    if (!open || !routerUid || !tokenizerAssetId) return;

    const generation = generationRef.current;
    setTokenizerAssetLoading(true);
    setTokenizerAssetLoadFailed(false);
    try {
      const asset = await request.get<TokenizerAssetItem>(
        `/v1/tokenizer_assets/${encodeURIComponent(tokenizerAssetId)}`
      );
      if (!isCurrentRequest(generation, routerUid)) return;
      setTokenizerAsset(asset);
    } catch {
      if (!isCurrentRequest(generation, routerUid)) return;
      setTokenizerAssetLoadFailed(true);
    } finally {
      if (isCurrentRequest(generation, routerUid)) {
        setTokenizerAssetLoading(false);
      }
    }
  }, [isCurrentRequest, open, routerUid, tokenizerAssetId]);

  useEffect(() => {
    generationRef.current += 1;
    instancesRequestRef.current = null;
    metricsRequestRef.current = null;
    setActiveTab('overview');
    setInstances([]);
    setAssignments([]);
    setMetrics({});
    setTokenizerAsset(null);
    setTokenizerAssetLoading(false);
    setTokenizerAssetLoadFailed(false);
    setInstancesInitialLoading(false);
    setInstancesRefreshing(false);
    setMetricsInitialLoading(false);
    setMetricsRefreshing(false);
    setInstancesLoaded(false);
    setMetricsLoaded(false);
    setInstancesUpdatedAt(null);
    setMetricsUpdatedAt(null);

    if (open && routerUid) {
      void loadInstances('initial');
      if (tokenizerAssetId) void loadTokenizerAsset();
    }

    return () => {
      generationRef.current += 1;
    };
  }, [loadInstances, loadTokenizerAsset, open, routerUid, tokenizerAssetId]);

  useEffect(() => {
    if (open && routerUid && activeTab === 'metrics' && !metricsLoaded) {
      void loadMetrics('initial');
    }
  }, [activeTab, loadMetrics, metricsLoaded, open, routerUid]);

  useEffect(() => {
    if (!open || !routerUid) return;

    const timer = window.setInterval(() => {
      if (activeTab === 'metrics') {
        void loadMetrics('silent');
      } else {
        void loadInstances('silent');
      }
    }, POLL_INTERVAL_MS);

    return () => window.clearInterval(timer);
  }, [activeTab, loadInstances, loadMetrics, open, routerUid]);

  const metricsSummary = useMemo(() => summarizeMetrics(metrics), [metrics]);
  const hasMetrics = useMemo(() => containsMetricsData(metrics), [metrics]);
  const hasKnownMetrics = hasMetricsSummary(metricsSummary);
  const displayedInstanceCount = instancesLoaded
    ? instances.length
    : router?.runtime_instances || 0;
  const displayedOnlineCount = instancesLoaded
    ? instances.filter((item) => item.online).length
    : router?.online_instances || 0;
  const currentInitialLoading =
    activeTab === 'metrics' ? metricsInitialLoading : instancesInitialLoading;
  const currentRefreshing = activeTab === 'metrics' ? metricsRefreshing : instancesRefreshing;
  const currentUpdatedAt = activeTab === 'metrics' ? metricsUpdatedAt : instancesUpdatedAt;

  const handleRefresh = () => {
    if (activeTab === 'metrics') {
      void loadMetrics('manual');
    } else {
      void loadInstances('manual');
      if (activeTab === 'overview' && tokenizerAssetId) {
        void loadTokenizerAsset();
      }
    }
  };

  const copyValue = async (value: string) => {
    try {
      await writeClipboard(value);
      toast.success(t('common.copySuccess'));
    } catch {
      toast.error(t('tokenRouter.copyFailed'));
    }
  };

  const updatedAtText = currentUpdatedAt
    ? formatDateTime(currentUpdatedAt, locale)
    : t('tokenRouter.notUpdated');

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        showClose={false}
        className="w-full gap-0 p-0 sm:w-[min(56vw,960px)] sm:max-w-none"
      >
        <SheetHeader className="shrink-0 gap-3 border-b px-4 py-4 sm:px-6">
          <div className="flex items-center justify-between gap-3">
            <SheetTitle>{t('tokenRouter.details')}</SheetTitle>
            <div className="flex shrink-0 items-center gap-2">
              <Button
                size="sm"
                variant="outline"
                loading={currentRefreshing}
                disabled={!router || currentInitialLoading}
                onClick={handleRefresh}
                title={t('common.refresh')}
              >
                {!currentRefreshing && <RefreshCw className="size-4" />}
                <span className="hidden sm:inline">{t('common.refresh')}</span>
              </Button>
              <SheetClose asChild>
                <Button
                  size="icon"
                  variant="ghost"
                  className="size-8"
                  title={t('tokenRouter.closeDetails')}
                  aria-label={t('tokenRouter.closeDetails')}
                >
                  <X className="size-4" />
                </Button>
              </SheetClose>
            </div>
          </div>

          {router && (
            <div className="flex min-w-0 items-start gap-3">
              <div className="min-w-0 flex-1 space-y-1">
                <CopyableValue
                  value={router.router_uid}
                  label={t('tokenRouter.routerUid')}
                  copyLabel={t('tokenRouter.copyValue')}
                  onCopy={copyValue}
                  className="font-mono text-sm font-medium"
                />
                <CopyableValue
                  value={router.virtual_model_uid}
                  label={t('tokenRouter.virtualModelUid')}
                  copyLabel={t('tokenRouter.copyValue')}
                  onCopy={copyValue}
                  className="text-xs text-muted-foreground"
                />
              </div>
              <RouterStatusBadge status={router.status} />
            </div>
          )}
        </SheetHeader>

        <Tabs
          value={activeTab}
          onValueChange={(value) => setActiveTab(value as DetailTab)}
          className="min-h-0 flex-1 gap-0"
        >
          <div className="shrink-0 border-b px-4 pt-3 sm:px-6">
            <TabsList className="grid h-10 w-full grid-cols-3 rounded-b-none bg-transparent p-0">
              <TabsTrigger
                value="overview"
                className="rounded-b-none border-b-2 border-transparent shadow-none data-[state=active]:border-primary data-[state=active]:shadow-none"
              >
                {t('tokenRouter.tabs.overview')}
              </TabsTrigger>
              <TabsTrigger
                value="instances"
                className="rounded-b-none border-b-2 border-transparent shadow-none data-[state=active]:border-primary data-[state=active]:shadow-none"
              >
                {t('tokenRouter.tabs.instances')}
              </TabsTrigger>
              <TabsTrigger
                value="metrics"
                className="rounded-b-none border-b-2 border-transparent shadow-none data-[state=active]:border-primary data-[state=active]:shadow-none"
              >
                {t('tokenRouter.tabs.metrics')}
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="overview" className="mt-0 min-h-0 overflow-y-auto p-4 sm:p-6">
            {router && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                  <SummaryCard label={t('tokenRouter.status')}>
                    <RouterStatusBadge status={router.status} />
                  </SummaryCard>
                  <SummaryCard label={t('tokenRouter.revision')} value={router.revision} />
                  <SummaryCard
                    label={t('tokenRouter.runtimeInstanceCount')}
                    value={displayedInstanceCount}
                    loading={instancesInitialLoading && !instancesLoaded}
                  />
                  <SummaryCard
                    label={t('tokenRouter.onlineInstanceCount')}
                    value={displayedOnlineCount}
                    loading={instancesInitialLoading && !instancesLoaded}
                  />
                </div>

                <section className="space-y-3">
                  <SectionTitle icon={<PackageCheck className="size-4" />}>
                    {t('tokenRouter.tokenizerAssetDetails')}
                  </SectionTitle>
                  <div className="rounded-xl border bg-muted/20 p-4">
                    {router.tokenizer_asset_id ? (
                      <div className="space-y-4">
                        <div className="flex min-w-0 flex-wrap items-start justify-between gap-3">
                          <CopyableValue
                            value={router.tokenizer_asset_id}
                            label={t('tokenRouter.tokenizerAsset')}
                            copyLabel={t('tokenRouter.copyValue')}
                            onCopy={copyValue}
                            className="font-mono text-xs font-medium"
                          />
                          {tokenizerAsset && (
                            <span className="rounded-full border bg-background px-2.5 py-1 text-xs font-medium">
                              {t(`tokenRouter.assetStatuses.${tokenizerAsset.status}`)}
                            </span>
                          )}
                        </div>

                        {tokenizerAssetLoading && !tokenizerAsset ? (
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <Loader2 className="size-4 animate-spin" />
                            {t('tokenRouter.loading')}
                          </div>
                        ) : (
                          <div className="grid gap-3 sm:grid-cols-2">
                            <DetailItem
                              label={t('tokenRouter.tokenizerAssetOrigin')}
                              value={
                                tokenizerAsset?.origin
                                  ? t(`tokenRouter.assetOrigins.${tokenizerAsset.origin}`)
                                  : router.tokenizer_asset_origin
                                    ? t(`tokenRouter.assetOrigins.${router.tokenizer_asset_origin}`)
                                    : '—'
                              }
                            />
                            <DetailItem
                              label={t('tokenRouter.revision')}
                              value={
                                tokenizerAsset?.revision || router.tokenizer_asset_revision || '—'
                              }
                              mono
                            />
                            <DetailItem
                              label={t('tokenRouter.compatibleModels')}
                              value={
                                tokenizerAsset?.compatible_models.length
                                  ? tokenizerAsset.compatible_models.join(', ')
                                  : '—'
                              }
                            />
                            <DetailItem
                              label={t('tokenRouter.tokenizerCapabilities')}
                              value={formatCapabilities(tokenizerAsset, t)}
                            />
                            <DetailItem
                              label={t('tokenRouter.tokenizerAssetValidatedAt')}
                              value={
                                tokenizerAsset?.validated_at
                                  ? formatDateTime(tokenizerAsset.validated_at, locale)
                                  : '—'
                              }
                            />
                            <div className="sm:col-span-2">
                              <DetailItem
                                label={t('tokenRouter.tokenizerAssetFingerprint')}
                                value={
                                  tokenizerAsset?.fingerprint ||
                                  router.tokenizer_asset_fingerprint ||
                                  '—'
                                }
                                mono
                              />
                            </div>
                          </div>
                        )}

                        {tokenizerAssetLoadFailed && (
                          <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
                            {t('tokenRouter.tokenizerAssetDetailsLoadFailed')}
                          </div>
                        )}
                        {Boolean(tokenizerAsset?.errors.length) && (
                          <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
                            <div className="font-medium">
                              {t('tokenRouter.tokenizerAssetErrors')}
                            </div>
                            <ul className="mt-1 list-disc space-y-1 pl-5">
                              {tokenizerAsset?.errors.map((error) => (
                                <li key={error} className="break-words">
                                  {error}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <div className="font-medium">
                          {t('tokenRouter.legacyCustomTokenizerPath')}
                        </div>
                        <CopyableValue
                          value={router.tokenizer_path}
                          label={t('tokenRouter.tokenizerPath')}
                          copyLabel={t('tokenRouter.copyValue')}
                          onCopy={copyValue}
                          className="font-mono text-xs text-muted-foreground"
                        />
                      </div>
                    )}
                  </div>
                </section>

                <section className="space-y-3">
                  <SectionTitle icon={<Gauge className="size-4" />}>
                    {t('tokenRouter.routingSummary')}
                  </SectionTitle>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="rounded-xl border bg-muted/20 p-4 sm:col-span-2">
                      <div className="grid gap-3 sm:grid-cols-4">
                        <DetailItem label={t('tokenRouter.modelType')} value="LLM" />
                        <DetailItem label={t('tokenRouter.profile.task')} value="Chat" />
                        <DetailItem
                          label={t('tokenRouter.routeProfile')}
                          value={router.route_profile || 'llm_chat'}
                          mono
                        />
                        <DetailItem
                          label={t('tokenRouter.strategy')}
                          value={
                            isTypedTokenRouter(router)
                              ? t('tokenRouter.strategies.typedRules')
                              : t('tokenRouter.strategies.tokenBudget')
                          }
                        />
                      </div>
                    </div>
                    {routerBackendList(router).map((backend) => (
                      <BackendCard
                        key={backend.id}
                        title={backend.id}
                        modelUid={backend.model_uid}
                        contextTokens={backend.max_context_tokens}
                        contextLabel={t('tokenRouter.maxContext')}
                        copyLabel={t('tokenRouter.copyValue')}
                        onCopy={copyValue}
                      />
                    ))}
                    <div className="rounded-xl border bg-muted/20 p-4 sm:col-span-2">
                      {isTypedTokenRouter(router) ? (
                        <div className="space-y-2">
                          <div className="text-xs font-medium text-muted-foreground">
                            {t('tokenRouter.orderedRules')}
                          </div>
                          {[...routerRuleList(router)]
                            .sort((left, right) => right.priority - left.priority)
                            .map((rule) => (
                              <div
                                key={rule.id}
                                className="grid gap-1 rounded-lg border bg-background p-3 text-xs sm:grid-cols-[8rem_5rem_1fr_1fr]"
                              >
                                <span className="font-mono">{rule.id}</span>
                                <span>P{rule.priority}</span>
                                <span className="break-words text-muted-foreground">
                                  {formatRuleMatch(rule.match)}
                                </span>
                                <span className="break-words font-mono">
                                  {formatRoutingAction(rule.action)}
                                </span>
                              </div>
                            ))}
                          <div className="text-xs text-muted-foreground">
                            {t('tokenRouter.defaultAction')}:{' '}
                            {formatRoutingAction(router.routing.default_action)}
                          </div>
                        </div>
                      ) : (
                        <DetailItem
                          label={t('tokenRouter.threshold')}
                          value={router.routing.short_threshold_tokens}
                        />
                      )}
                    </div>
                  </div>
                </section>

                <CollapsiblePanel
                  title={t('tokenRouter.advancedConfiguration')}
                  description={t('tokenRouter.advancedConfigurationDescription')}
                  icon={<Settings2 className="size-4" />}
                  contentClassName="space-y-5"
                >
                  <ConfigurationGroup title={t('tokenRouter.sections.basic')}>
                    <DetailItem label={t('tokenRouter.modelType')} value={router.model_type} />
                    <DetailItem
                      label={t('tokenRouter.strategy')}
                      value={
                        isTypedTokenRouter(router)
                          ? t('tokenRouter.strategies.typedRules')
                          : t('tokenRouter.strategies.tokenBudget')
                      }
                    />
                    <DetailItem
                      label={t('tokenRouter.backendUrl')}
                      value={router.backend_url}
                      mono
                    />
                    <DetailItem
                      label={t('tokenRouter.tokenizerAsset')}
                      value={router.tokenizer_asset_id || t('tokenRouter.customTokenizerPath')}
                      mono
                    />
                    <DetailItem
                      label={t('tokenRouter.tokenizerAssetRevision')}
                      value={router.tokenizer_asset_revision || '—'}
                      mono
                    />
                    <DetailItem
                      label={t('tokenRouter.tokenizerAssetFingerprint')}
                      value={router.tokenizer_asset_fingerprint || '—'}
                      mono
                    />
                    <DetailItem
                      label={t('tokenRouter.tokenizerPath')}
                      value={router.tokenizer_path}
                      mono
                    />
                    <DetailItem
                      label={t('tokenRouter.aliases')}
                      value={router.model_aliases.length ? router.model_aliases.join(', ') : '—'}
                      mono
                    />
                    <DetailItem
                      label={t('tokenRouter.createdAt')}
                      value={formatDateTime(router.created_at, locale)}
                    />
                    <DetailItem
                      label={t('tokenRouter.updatedAt')}
                      value={formatDateTime(router.updated_at, locale)}
                    />
                  </ConfigurationGroup>

                  <ConfigurationGroup title={t('tokenRouter.sections.routing')}>
                    <DetailItem
                      label={t('tokenRouter.reserve')}
                      value={router.routing.context_reserve_tokens}
                    />
                    <DetailItem
                      label={t('tokenRouter.defaultOutput')}
                      value={router.routing.default_output_tokens}
                    />
                    {isTypedTokenRouter(router) ? (
                      <>
                        <DetailItem
                          label={t('tokenRouter.evaluationMode')}
                          value={router.routing.evaluation_mode}
                        />
                        <DetailItem
                          label={t('tokenRouter.defaultAction')}
                          value={formatRoutingAction(router.routing.default_action)}
                          mono
                        />
                      </>
                    ) : (
                      <>
                        <DetailItem
                          label={t('tokenRouter.thinkingPolicy')}
                          value={t(
                            `tokenRouter.thinkingPolicies.${router.routing.thinking_policy}`
                          )}
                        />
                        <DetailItem
                          label={t('tokenRouter.overflowPolicy')}
                          value={t(
                            `tokenRouter.overflowPolicies.${router.routing.overflow_policy}`
                          )}
                        />
                      </>
                    )}
                  </ConfigurationGroup>

                  {routerBackendList(router).map((backend) => (
                    <ConfigurationGroup
                      key={backend.id}
                      title={`${t('tokenRouter.backendAdmission')}: ${backend.id}`}
                    >
                      <DetailItem
                        label={t('tokenRouter.backendModel')}
                        value={backend.model_uid}
                        mono
                      />
                      <DetailItem
                        label={t('tokenRouter.maxContext')}
                        value={backend.max_context_tokens}
                      />
                      <DetailItem
                        label={t('tokenRouter.maxActive')}
                        value={backend.admission.max_active}
                      />
                      <DetailItem
                        label={t('tokenRouter.maxQueue')}
                        value={backend.admission.max_queue}
                      />
                      <DetailItem
                        label={t('tokenRouter.queueTimeout')}
                        value={t('tokenRouter.secondsValue', {
                          seconds: backend.admission.queue_timeout_seconds,
                        })}
                      />
                      <DetailItem
                        label={t('tokenRouter.retryAfter')}
                        value={t('tokenRouter.secondsValue', {
                          seconds: backend.admission.retry_after_seconds,
                        })}
                      />
                    </ConfigurationGroup>
                  ))}

                  <ConfigurationGroup title={t('tokenRouter.sections.tokenization')}>
                    <DetailItem
                      label={t('tokenRouter.tokenizationExecutor')}
                      value={router.tokenization.executor}
                    />
                    <DetailItem
                      label={t('tokenRouter.multiprocessingStartMethod')}
                      value={router.tokenization.multiprocessing_start_method}
                    />
                    <DetailItem
                      label={t('tokenRouter.tokenWorkers')}
                      value={router.tokenization.max_workers}
                    />
                    <DetailItem
                      label={t('tokenRouter.tokenActive')}
                      value={router.tokenization.max_active}
                    />
                    <DetailItem
                      label={t('tokenRouter.tokenQueue')}
                      value={router.tokenization.max_queue}
                    />
                    <DetailItem
                      label={t('tokenRouter.queueTimeout')}
                      value={t('tokenRouter.secondsValue', {
                        seconds: router.tokenization.queue_timeout_seconds,
                      })}
                    />
                    <DetailItem
                      label={t('tokenRouter.retryAfter')}
                      value={t('tokenRouter.secondsValue', {
                        seconds: router.tokenization.retry_after_seconds,
                      })}
                    />
                  </ConfigurationGroup>

                  <ConfigurationGroup title={t('tokenRouter.sections.timeouts')}>
                    <DetailItem
                      label={t('tokenRouter.requestTimeout')}
                      value={router.request_timeout_seconds}
                    />
                    <DetailItem
                      label={t('tokenRouter.connectTimeout')}
                      value={router.connect_timeout_seconds}
                    />
                  </ConfigurationGroup>
                </CollapsiblePanel>
              </div>
            )}
          </TabsContent>

          <TabsContent value="instances" className="mt-0 min-h-0 overflow-y-auto p-4 sm:p-6">
            <div className="space-y-4">
              <TabMeta
                count={displayedInstanceCount}
                updatedAt={instancesUpdatedAt}
                locale={locale}
                autoRefreshingLabel={t('tokenRouter.autoRefreshing')}
                lastUpdatedLabel={t('tokenRouter.lastUpdated')}
                notUpdatedLabel={t('tokenRouter.notUpdated')}
              />

              {(router?.deployment.management_mode === 'managed' || assignments.length > 0) && (
                <section className="space-y-3">
                  <div className="flex items-center justify-between gap-3">
                    <h3 className="text-sm font-semibold">{t('tokenRouter.assignments')}</h3>
                    <span className="text-xs text-muted-foreground">
                      {t('tokenRouter.assignmentCount', { count: assignments.length })}
                    </span>
                  </div>
                  {assignments.length === 0 ? (
                    <div className="rounded-xl border border-dashed p-6 text-center text-sm text-muted-foreground">
                      {t('tokenRouter.noAssignments')}
                    </div>
                  ) : (
                    <div className="grid gap-3 xl:grid-cols-2">
                      {assignments.map((assignment) => (
                        <article
                          key={assignment.assignment_id}
                          className="min-w-0 rounded-xl border p-4 text-sm"
                        >
                          <div className="flex min-w-0 items-start justify-between gap-3">
                            <CopyableValue
                              value={assignment.assignment_id}
                              label={t('tokenRouter.assignmentId')}
                              copyLabel={t('tokenRouter.copyValue')}
                              onCopy={copyValue}
                              className="font-mono text-xs font-medium"
                            />
                            <RouterStatusBadge status={assignment.observed_state} />
                          </div>
                          <div className="mt-4 grid grid-cols-2 gap-3 rounded-lg bg-muted/30 p-3">
                            <DetailItem
                              label={t('tokenRouter.nodeId')}
                              value={assignment.node_id}
                              mono
                            />
                            <DetailItem
                              label={t('tokenRouter.processId')}
                              value={assignment.pid ?? '—'}
                            />
                            <DetailItem
                              label={t('tokenRouter.assignmentGeneration')}
                              value={assignment.assignment_generation}
                            />
                            <DetailItem
                              label={t('tokenRouter.configRevision')}
                              value={assignment.config_revision}
                            />
                            <DetailItem
                              label={t('tokenRouter.desiredState')}
                              value={assignment.desired_state}
                            />
                            <DetailItem
                              label={t('tokenRouter.listenPort')}
                              value={assignment.listen_port}
                            />
                          </div>
                          <div className="mt-4">
                            <CopyableValue
                              value={assignment.public_endpoint}
                              label={t('tokenRouter.endpoint')}
                              copyLabel={t('tokenRouter.copyValue')}
                              onCopy={copyValue}
                              className="font-mono text-xs text-muted-foreground"
                            />
                          </div>
                          {assignment.last_error && (
                            <div className="mt-4 whitespace-pre-wrap break-words rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
                              <div className="mb-1 font-medium">{t('tokenRouter.lastError')}</div>
                              {assignment.last_error}
                            </div>
                          )}
                        </article>
                      ))}
                    </div>
                  )}
                </section>
              )}

              {instancesInitialLoading && !instancesLoaded ? (
                <LoadingState label={t('tokenRouter.loading')} />
              ) : instances.length === 0 ? (
                <EmptyState icon={<Server className="size-6 opacity-50" />}>
                  {t('tokenRouter.noInstances')}
                </EmptyState>
              ) : (
                <div className="grid gap-3 xl:grid-cols-2">
                  {instances.map((item) => (
                    <article
                      key={item.instance_id}
                      className="min-w-0 rounded-xl border p-4 text-sm"
                    >
                      <div className="flex min-w-0 items-start gap-3">
                        <RouterStatusBadge
                          status={item.online ? item.status || 'ready' : 'offline'}
                        />
                        <CopyableValue
                          value={item.instance_id}
                          label={t('tokenRouter.instanceId')}
                          copyLabel={t('tokenRouter.copyValue')}
                          onCopy={copyValue}
                          className="font-mono text-xs font-medium"
                        />
                      </div>

                      <div className="mt-4 grid grid-cols-2 gap-3 rounded-lg bg-muted/30 p-3">
                        <DetailItem
                          label={t('tokenRouter.ackedRevision')}
                          value={item.acked_revision}
                        />
                        <DetailItem
                          label={t('tokenRouter.heartbeatAgeLabel')}
                          value={t('tokenRouter.secondsValue', {
                            seconds: item.heartbeat_age_seconds.toFixed(1),
                          })}
                        />
                      </div>

                      <div className="mt-4">
                        <CopyableValue
                          value={item.endpoint}
                          label={t('tokenRouter.endpoint')}
                          copyLabel={t('tokenRouter.copyValue')}
                          onCopy={copyValue}
                          className="font-mono text-xs text-muted-foreground"
                        />
                      </div>

                      {item.process?.tokenizer_asset && (
                        <div className="mt-4 rounded-lg border bg-muted/20 p-3">
                          <div className="mb-3 text-xs font-medium text-muted-foreground">
                            {t('tokenRouter.runtimeTokenizerAsset')}
                          </div>
                          <div className="grid gap-3 sm:grid-cols-2">
                            <DetailItem
                              label={t('tokenRouter.tokenizerAsset')}
                              value={item.process.tokenizer_asset.asset_id || '—'}
                              mono
                            />
                            <DetailItem
                              label={t('tokenRouter.tokenizerAssetOrigin')}
                              value={
                                item.process.tokenizer_asset.origin
                                  ? t(
                                      `tokenRouter.assetOrigins.${item.process.tokenizer_asset.origin}`
                                    )
                                  : '—'
                              }
                            />
                            <DetailItem
                              label={t('tokenRouter.revision')}
                              value={item.process.tokenizer_asset.revision || '—'}
                              mono
                            />
                            <div className="sm:col-span-2">
                              <DetailItem
                                label={t('tokenRouter.tokenizerAssetFingerprint')}
                                value={item.process.tokenizer_asset.fingerprint || '—'}
                                mono
                              />
                            </div>
                          </div>
                        </div>
                      )}

                      {item.config_error && (
                        <div className="mt-4 whitespace-pre-wrap break-words rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
                          <div className="mb-1 font-medium">{t('tokenRouter.configError')}</div>
                          {item.config_error}
                        </div>
                      )}
                    </article>
                  ))}
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="metrics" className="mt-0 min-h-0 overflow-y-auto p-4 sm:p-6">
            <div className="space-y-4">
              <TabMeta
                updatedAt={metricsUpdatedAt}
                locale={locale}
                autoRefreshingLabel={t('tokenRouter.autoRefreshing')}
                lastUpdatedLabel={t('tokenRouter.lastUpdated')}
                notUpdatedLabel={t('tokenRouter.notUpdated')}
              />

              {metricsInitialLoading && !metricsLoaded ? (
                <LoadingState label={t('tokenRouter.loading')} />
              ) : !hasMetrics ? (
                <EmptyState icon={<Activity className="size-6 opacity-50" />}>
                  {t('tokenRouter.noMetrics')}
                </EmptyState>
              ) : (
                <>
                  {hasKnownMetrics ? (
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                        {metricsSummary.completedRequests !== undefined && (
                          <MetricCard
                            label={t('tokenRouter.completedRequests')}
                            value={metricsSummary.completedRequests}
                          />
                        )}
                        {metricsSummary.activeRequests !== undefined && (
                          <MetricCard
                            label={t('tokenRouter.activeRequests')}
                            value={metricsSummary.activeRequests}
                          />
                        )}
                        {metricsSummary.waitingRequests !== undefined && (
                          <MetricCard
                            label={t('tokenRouter.waitingRequests')}
                            value={metricsSummary.waitingRequests}
                          />
                        )}
                        {metricsSummary.onlineInstances !== undefined && (
                          <MetricCard
                            label={t('tokenRouter.onlineInstanceCount')}
                            value={metricsSummary.onlineInstances}
                          />
                        )}
                      </div>

                      <div className="grid gap-3 lg:grid-cols-3">
                        {Object.entries(metricsSummary.backendPools).map(
                          ([backendId, summary]) =>
                            hasCapacityData(summary) && (
                              <CapacityCard
                                key={backendId}
                                title={backendId}
                                summary={summary}
                                activeLabel={t('tokenRouter.activeCapacity')}
                                waitingLabel={t('tokenRouter.waitingCapacity')}
                              />
                            )
                        )}
                        {hasCapacityData(metricsSummary.tokenization) && (
                          <CapacityCard
                            title={t('tokenRouter.tokenizationPool')}
                            summary={metricsSummary.tokenization}
                            activeLabel={t('tokenRouter.activeCapacity')}
                            waitingLabel={t('tokenRouter.waitingCapacity')}
                          />
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="rounded-xl border border-dashed p-4 text-sm text-muted-foreground">
                      {t('tokenRouter.noKnownMetrics')}
                    </div>
                  )}

                  <CollapsiblePanel
                    title={t('tokenRouter.advancedDiagnostics')}
                    description={t('tokenRouter.rawMetrics')}
                    icon={<BarChart3 className="size-4" />}
                    contentClassName="space-y-3"
                  >
                    <div className="flex justify-end">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => void copyValue(JSON.stringify(metrics, null, 2))}
                      >
                        <Copy className="size-4" />
                        {t('tokenRouter.copyMetrics')}
                      </Button>
                    </div>
                    <pre className="max-h-96 overflow-auto rounded-lg bg-muted/50 p-4 text-xs leading-5">
                      {JSON.stringify(metrics, null, 2)}
                    </pre>
                  </CollapsiblePanel>
                </>
              )}
            </div>
          </TabsContent>
        </Tabs>

        <span className="sr-only" aria-live="polite">
          {t('tokenRouter.lastUpdated')}: {updatedAtText}
        </span>
      </SheetContent>
    </Sheet>
  );
}

function SummaryCard({
  label,
  value,
  loading = false,
  children,
}: {
  label: string;
  value?: string | number;
  loading?: boolean;
  children?: ReactNode;
}) {
  return (
    <div className="min-w-0 rounded-xl border bg-muted/20 p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-2 flex min-h-6 items-center text-lg font-semibold">
        {loading ? (
          <Loader2 className="size-4 animate-spin text-muted-foreground" />
        ) : (
          children || value
        )}
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-xl border bg-muted/20 p-4">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-2 text-2xl font-semibold tabular-nums">{value}</div>
    </div>
  );
}

function CapacityCard({
  title,
  summary,
  activeLabel,
  waitingLabel,
}: {
  title: string;
  summary: CapacitySummary;
  activeLabel: string;
  waitingLabel: string;
}) {
  return (
    <div className="rounded-xl border p-4">
      <div className="mb-3 flex items-center gap-2 text-sm font-semibold">
        <Gauge className="size-4 text-muted-foreground" />
        {title}
      </div>
      <div className="space-y-3">
        {(summary.active !== undefined || summary.maxActive !== undefined) && (
          <CapacityRow label={activeLabel} value={summary.active} maximum={summary.maxActive} />
        )}
        {(summary.waiting !== undefined || summary.maxQueue !== undefined) && (
          <CapacityRow label={waitingLabel} value={summary.waiting} maximum={summary.maxQueue} />
        )}
      </div>
    </div>
  );
}

function CapacityRow({
  label,
  value,
  maximum,
}: {
  label: string;
  value?: number;
  maximum?: number;
}) {
  const displayValue =
    value !== undefined && maximum !== undefined
      ? `${value} / ${maximum}`
      : String(value ?? maximum ?? '—');

  return (
    <div className="flex items-center justify-between gap-3 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium tabular-nums">{displayValue}</span>
    </div>
  );
}

function BackendCard({
  title,
  modelUid,
  contextTokens,
  contextLabel,
  copyLabel,
  onCopy,
}: {
  title: string;
  modelUid: string;
  contextTokens: number;
  contextLabel: string;
  copyLabel: string;
  onCopy: (value: string) => Promise<void>;
}) {
  return (
    <div className="min-w-0 rounded-xl border p-4">
      <div className="mb-3 text-sm font-semibold">{title}</div>
      <CopyableValue
        value={modelUid}
        label={title}
        copyLabel={copyLabel}
        onCopy={onCopy}
        className="font-mono text-xs"
      />
      <div className="mt-3 border-t pt-3">
        <DetailItem label={contextLabel} value={contextTokens} />
      </div>
    </div>
  );
}

function CopyableValue({
  value,
  label,
  copyLabel,
  onCopy,
  className,
}: {
  value: string;
  label: string;
  copyLabel: string;
  onCopy: (value: string) => Promise<void>;
  className?: string;
}) {
  return (
    <div className="flex min-w-0 flex-1 items-center gap-1.5">
      <span className={cn('min-w-0 flex-1 truncate', className)} title={`${label}: ${value}`}>
        {value}
      </span>
      <Button
        size="icon"
        variant="ghost"
        className="size-7 shrink-0 text-muted-foreground"
        onClick={() => void onCopy(value)}
        title={`${copyLabel}: ${label}`}
        aria-label={`${copyLabel}: ${label}`}
      >
        <Copy className="size-3.5" />
      </Button>
    </div>
  );
}

function ConfigurationGroup({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-3">
      <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </h4>
      <div className="grid gap-x-6 gap-y-4 sm:grid-cols-2">{children}</div>
    </section>
  );
}

function SectionTitle({ icon, children }: { icon: ReactNode; children: ReactNode }) {
  return (
    <h3 className="flex items-center gap-2 text-sm font-semibold">
      <span className="text-muted-foreground">{icon}</span>
      {children}
    </h3>
  );
}

function DetailItem({
  label,
  value,
  mono = false,
  className,
}: {
  label: string;
  value: ReactNode;
  mono?: boolean;
  className?: string;
}) {
  return (
    <div className={cn('min-w-0', className)}>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={cn('mt-1 break-words text-sm font-medium', mono && 'font-mono text-xs')}>
        {value}
      </div>
    </div>
  );
}

function TabMeta({
  count,
  updatedAt,
  locale,
  autoRefreshingLabel,
  lastUpdatedLabel,
  notUpdatedLabel,
}: {
  count?: number;
  updatedAt: Date | null;
  locale: string;
  autoRefreshingLabel: string;
  lastUpdatedLabel: string;
  notUpdatedLabel: string;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
      <div className="flex items-center gap-2">
        <span className="inline-flex items-center gap-1.5">
          <span className="size-1.5 rounded-full bg-emerald-500" />
          {autoRefreshingLabel}
        </span>
        {count !== undefined && <span>· {count}</span>}
      </div>
      <span>
        {lastUpdatedLabel}: {updatedAt ? formatDateTime(updatedAt, locale) : notUpdatedLabel}
      </span>
    </div>
  );
}

function LoadingState({ label }: { label: string }) {
  return (
    <div className="flex min-h-40 items-center justify-center gap-2 rounded-xl border text-sm text-muted-foreground">
      <Loader2 className="size-4 animate-spin" />
      {label}
    </div>
  );
}

function EmptyState({ icon, children }: { icon: ReactNode; children: ReactNode }) {
  return (
    <div className="flex min-h-40 flex-col items-center justify-center gap-2 rounded-xl border border-dashed text-sm text-muted-foreground">
      {icon}
      {children}
    </div>
  );
}

function formatRuleMatch(match: TokenRouterRuleMatch): string {
  const entries = Object.entries(match).filter(([, value]) => value !== undefined);
  return entries.length
    ? entries.map(([key, value]) => `${key}=${String(value)}`).join(' AND ')
    : '—';
}

function formatRoutingAction(action: TokenRouterRoutingAction): string {
  return action.type === 'route'
    ? `route -> ${action.backend_id || '—'}`
    : `reject: ${action.reason || '—'}`;
}

function summarizeMetrics(metrics: MetricsResponse): MetricsSummary {
  const summary: MetricsSummary = {
    backendPools: {},
    tokenization: {},
  };
  const instances = Array.isArray(metrics.instances) ? metrics.instances : [];
  let onlineSeen = false;

  for (const rawInstance of instances) {
    const instance = asRecord(rawInstance);
    if (!instance) continue;

    if (typeof instance.online === 'boolean') {
      onlineSeen = true;
      summary.onlineInstances = (summary.onlineInstances || 0) + (instance.online ? 1 : 0);
    }

    const instanceMetrics = asRecord(instance.metrics);
    const requests = asRecord(instanceMetrics?.requests);
    if (requests) {
      for (const [key, value] of Object.entries(requests)) {
        if (key.split('/')[0] === 'completed') {
          summary.completedRequests = addKnownNumber(summary.completedRequests, value);
        }
      }
    }

    const process = asRecord(instance.process);
    summary.activeRequests = addKnownNumber(summary.activeRequests, process?.active_requests);

    const pools = asRecord(process?.backends) || asRecord(process?.pools);
    if (pools) {
      for (const [backendId, rawPool] of Object.entries(pools)) {
        const pool = asRecord(rawPool);
        const capacity = (summary.backendPools[backendId] ||= {});
        addCapacity(capacity, pool);
        summary.waitingRequests = addKnownNumber(summary.waitingRequests, pool?.waiting);
      }
    }
    addCapacity(summary.tokenization, asRecord(process?.tokenization));
  }

  if (!onlineSeen) delete summary.onlineInstances;
  return summary;
}

function addCapacity(summary: CapacitySummary, source: Record<string, unknown> | null) {
  if (!source) return;
  summary.active = addKnownNumber(summary.active, source.active);
  summary.waiting = addKnownNumber(summary.waiting, source.waiting);
  summary.maxActive = addKnownNumber(summary.maxActive, source.max_active);
  summary.maxQueue = addKnownNumber(summary.maxQueue, source.max_queue);
}

function addKnownNumber(current: number | undefined, value: unknown): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) return current;
  return (current || 0) + value;
}

function hasMetricsSummary(summary: MetricsSummary): boolean {
  return (
    summary.completedRequests !== undefined ||
    summary.activeRequests !== undefined ||
    summary.waitingRequests !== undefined ||
    summary.onlineInstances !== undefined ||
    Object.values(summary.backendPools).some(hasCapacityData) ||
    hasCapacityData(summary.tokenization)
  );
}

function hasCapacityData(summary: CapacitySummary): boolean {
  return Object.values(summary).some((value) => value !== undefined);
}

function containsMetricsData(metrics: MetricsResponse): boolean {
  const instances = Array.isArray(metrics.instances) ? metrics.instances : [];
  return instances.some((rawInstance) => {
    const instance = asRecord(rawInstance);
    return (
      hasObjectEntries(instance?.metrics) ||
      hasObjectEntries(instance?.backend_health) ||
      hasObjectEntries(instance?.process)
    );
  });
}

function formatCapabilities(asset: TokenizerAssetItem | null, t: TFunc): string {
  if (!asset) return '—';
  const capabilities = Object.entries(asset.capabilities || {})
    .filter(([, enabled]) => enabled)
    .map(([name]) => t(`tokenRouter.capabilities.${name}`));
  return capabilities.length ? capabilities.join(', ') : '—';
}

function hasObjectEntries(value: unknown): boolean {
  const record = asRecord(value);
  return Boolean(record && Object.keys(record).length > 0);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function formatDateTime(value: string | Date, locale: string): string {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);

  return new Intl.DateTimeFormat(locale, {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(date);
}

async function writeClipboard(value: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }

  const textarea = document.createElement('textarea');
  textarea.value = value;
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  const copied = document.execCommand('copy');
  document.body.removeChild(textarea);
  if (!copied) throw new Error('Clipboard copy failed');
}
