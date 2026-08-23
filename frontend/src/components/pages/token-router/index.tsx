'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Eye,
  Loader2,
  Pencil,
  Plus,
  Power,
  PowerOff,
  RefreshCw,
  Route,
  ShieldCheck,
  Trash2,
} from 'lucide-react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import PageContainer from '@/components/ui/page-container';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useGlobal } from '@/contexts/global-context';
import { useI18n } from '@/contexts/i18n-context';
import { useMenuAuth } from '@/hooks/use-menu-auth';
import request from '@/lib/request';
import type {
  TokenizerAssetItem,
  TokenizerAssetListResponse,
  TokenRouterItem,
  TokenRouterNode,
} from '@/types/services';
import { isTypedTokenRouter } from '@/types/services';
import { RouterDetailDrawer } from './router-detail-drawer';
import { RouterFormDialog } from './router-form-dialog';
import { RouterNodeSection } from './router-node-section';
import { resolveRouterCapabilities } from './router-capabilities.mjs';
import { normalizeTokenRouterList, routerBackendList } from './router-config-normalizer';
import { RouterStatusBadge } from './router-status-badge';
import { TokenizerAssetSection } from './tokenizer-asset-section';

type RouterAction = 'enable' | 'disable' | 'validate' | 'delete';

type BusyAction = {
  routerUid: string;
  action: RouterAction;
} | null;

export default function TokenRouterPage() {
  const { t } = useI18n();
  const { clusterUIConfig, globalReady } = useGlobal();
  const auth = useMenuAuth();
  const { canWriteRouters, canOperateRouters } = resolveRouterCapabilities({
    globalReady,
    authAdvanced: clusterUIConfig?.auth_advanced,
    canWriteRouters: auth.canWriteRouters,
    canOperateRouters: auth.canOperateRouters,
  });
  const [routers, setRouters] = useState<TokenRouterItem[]>([]);
  const [routerNodes, setRouterNodes] = useState<TokenRouterNode[]>([]);
  const [tokenizerAssets, setTokenizerAssets] = useState<TokenizerAssetItem[]>([]);
  const [initialLoading, setInitialLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [formOpen, setFormOpen] = useState(false);
  const [editing, setEditing] = useState<TokenRouterItem | null>(null);
  const [details, setDetails] = useState<TokenRouterItem | null>(null);
  const [deleting, setDeleting] = useState<TokenRouterItem | null>(null);
  const [busyAction, setBusyAction] = useState<BusyAction>(null);
  const refreshRequestId = useRef(0);

  const refreshAll = useCallback(async (mode: 'initial' | 'manual' | 'silent' = 'silent') => {
    const requestId = ++refreshRequestId.current;
    if (mode === 'initial') setInitialLoading(true);
    if (mode === 'manual') setRefreshing(true);

    try {
      const [nodeResult, assetResult, routerResult] = await Promise.allSettled([
        request.get<TokenRouterNode[]>('/v1/token_router_nodes?include_offline=false'),
        request.get<TokenizerAssetListResponse>('/v1/tokenizer_assets'),
        request.get<TokenRouterItem[]>('/v1/token_routers'),
      ] as const);

      if (requestId !== refreshRequestId.current) return;

      if (nodeResult.status === 'fulfilled') {
        setRouterNodes(Array.isArray(nodeResult.value) ? nodeResult.value : []);
      }
      if (assetResult.status === 'fulfilled') {
        setTokenizerAssets(Array.isArray(assetResult.value.items) ? assetResult.value.items : []);
      }
      if (routerResult.status === 'fulfilled') {
        setRouters(normalizeTokenRouterList(routerResult.value));
      }
    } finally {
      if (requestId === refreshRequestId.current) {
        setInitialLoading(false);
        setRefreshing(false);
      }
    }
  }, []);

  useEffect(() => {
    void refreshAll('initial');
    const timer = window.setInterval(() => void refreshAll('silent'), 10000);
    return () => {
      window.clearInterval(timer);
      refreshRequestId.current += 1;
    };
  }, [refreshAll]);

  const isBusy = useCallback(
    (routerUid: string, action: RouterAction) =>
      busyAction?.routerUid === routerUid && busyAction.action === action,
    [busyAction]
  );

  const runAction = async (router: TokenRouterItem, action: Exclude<RouterAction, 'delete'>) => {
    setBusyAction({ routerUid: router.router_uid, action });
    try {
      const result = await request.post<{ valid?: boolean; errors?: string[] }>(
        `/v1/token_routers/${router.router_uid}/${action}`
      );

      if (action === 'validate') {
        if (result.valid) {
          toast.success(t('tokenRouter.validationPassed'));
        } else {
          toast.error(
            result.errors?.length ? result.errors.join('; ') : t('tokenRouter.validationFailed')
          );
        }
      } else {
        toast.success(
          t(action === 'enable' ? 'tokenRouter.enableSuccess' : 'tokenRouter.disableSuccess')
        );
      }

      await refreshAll('silent');
    } finally {
      setBusyAction(null);
    }
  };

  const handleDelete = async () => {
    if (!deleting) return;

    setBusyAction({ routerUid: deleting.router_uid, action: 'delete' });
    try {
      await request.delete(`/v1/token_routers/${deleting.router_uid}`);
      toast.success(t('tokenRouter.deleteSuccess'));
      setDeleting(null);
      await refreshAll('silent');
    } finally {
      setBusyAction(null);
    }
  };

  const deletingBusy = useMemo(
    () => Boolean(deleting && isBusy(deleting.router_uid, 'delete')),
    [deleting, isBusy]
  );

  return (
    <PageContainer
      title={t('tokenRouter.title')}
      subTitle={t('tokenRouter.description')}
      extraContent={
        <div className="flex flex-wrap justify-end gap-2">
          <Button variant="outline" loading={refreshing} onClick={() => void refreshAll('manual')}>
            {!refreshing && <RefreshCw className="size-4" />}
            {t('common.refresh')}
          </Button>
          {canWriteRouters && (
            <Button
              onClick={() => {
                setEditing(null);
                setFormOpen(true);
              }}
            >
              <Plus className="size-4" />
              {t('tokenRouter.create')}
            </Button>
          )}
        </div>
      }
    >
      <RouterNodeSection
        nodes={routerNodes}
        initialLoading={initialLoading}
        canOperate={canOperateRouters}
        onChanged={() => refreshAll('silent')}
      />
      <TokenizerAssetSection
        assets={tokenizerAssets}
        nodes={routerNodes}
        initialLoading={initialLoading}
        canWrite={canWriteRouters}
        canOperate={canOperateRouters}
        onChanged={() => refreshAll('silent')}
      />

      <section className="mt-6 space-y-3">
        <div>
          <h2 className="flex items-center gap-2 text-base font-semibold">
            <Route className="size-4" />
            {t('tokenRouter.routingPolicies')}
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {t('tokenRouter.routingPoliciesDescription')}
          </p>
        </div>

        <div className="overflow-hidden rounded-xl border border-border">
          <Table className="min-w-[1370px] table-fixed">
            <colgroup>
              <col className="w-[190px]" />
              <col className="w-[190px]" />
              <col className="w-[260px]" />
              <col className="w-[140px]" />
              <col className="w-[120px]" />
              <col className="w-[180px]" />
              <col className="w-[80px]" />
              <col className="w-[210px]" />
            </colgroup>
            <TableHeader>
              <TableRow>
                <TableHead className="whitespace-nowrap">{t('tokenRouter.routerUid')}</TableHead>
                <TableHead className="whitespace-nowrap">
                  {t('tokenRouter.virtualModelUid')}
                </TableHead>
                <TableHead className="whitespace-nowrap">{t('tokenRouter.backends')}</TableHead>
                <TableHead className="whitespace-nowrap">{t('tokenRouter.routeProfile')}</TableHead>
                <TableHead className="whitespace-nowrap">{t('tokenRouter.status')}</TableHead>
                <TableHead className="whitespace-nowrap">{t('tokenRouter.deployment')}</TableHead>
                <TableHead className="whitespace-nowrap">{t('tokenRouter.revision')}</TableHead>
                <TableHead className="whitespace-nowrap text-right">
                  {t('common.operation')}
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {initialLoading ? (
                <TableRow>
                  <TableCell colSpan={8} className="h-40 text-center text-muted-foreground">
                    <div className="flex items-center justify-center gap-2">
                      <Loader2 className="size-4 animate-spin" />
                      {t('tokenRouter.loading')}
                    </div>
                  </TableCell>
                </TableRow>
              ) : routers.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={8} className="h-48 text-center text-muted-foreground">
                    <div className="flex flex-col items-center justify-center gap-3">
                      <Route className="size-8 opacity-50" />
                      <span>{t('tokenRouter.empty')}</span>
                    </div>
                  </TableCell>
                </TableRow>
              ) : (
                routers.map((router) => {
                  const rowBusy = busyAction?.routerUid === router.router_uid;
                  const validating = isBusy(router.router_uid, 'validate');
                  const toggling = isBusy(router.router_uid, router.enabled ? 'disable' : 'enable');

                  return (
                    <TableRow key={router.router_uid}>
                      <TableCell className="min-w-0 font-mono text-xs" title={router.router_uid}>
                        <div className="min-w-0 truncate">{router.router_uid}</div>
                      </TableCell>
                      <TableCell
                        className="min-w-0 font-mono text-xs"
                        title={router.virtual_model_uid}
                      >
                        <div className="min-w-0 truncate">{router.virtual_model_uid}</div>
                      </TableCell>
                      <TableCell className="min-w-0">
                        {routerBackendList(router)
                          .slice(0, 3)
                          .map((backend, index) => (
                            <div
                              key={backend.id}
                              className={`min-w-0 truncate text-xs ${
                                index ? 'text-muted-foreground' : ''
                              }`}
                              title={`${backend.id}: ${backend.model_uid}`}
                            >
                              {backend.id}: {backend.model_uid}
                            </div>
                          ))}
                        {routerBackendList(router).length > 3 && (
                          <div className="text-xs text-muted-foreground">
                            +{routerBackendList(router).length - 3}
                          </div>
                        )}
                      </TableCell>
                      <TableCell>
                        <div className="text-xs font-medium">LLM / Chat</div>
                        <div className="text-xs text-muted-foreground">
                          {isTypedTokenRouter(router)
                            ? t('tokenRouter.typedRulesSummary', {
                                count: router.routing.rules.length,
                              })
                            : t('tokenRouter.tokenBudgetSummary', {
                                threshold: router.routing.short_threshold_tokens.toLocaleString(),
                              })}
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-col items-start gap-1">
                          <RouterStatusBadge status={router.status} />
                          <div className="whitespace-nowrap text-xs text-muted-foreground">
                            {t('tokenRouter.onlineSummary', {
                              online: router.online_instances,
                              total: router.runtime_instances,
                            })}
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="text-xs font-medium">
                          {t(`tokenRouter.managementModes.${router.deployment.management_mode}`)}
                        </div>
                        <div className="whitespace-nowrap text-xs text-muted-foreground">
                          {t('tokenRouter.replicaSummary', {
                            ready: router.deployment.ready_replicas,
                            desired: router.deployment.desired_replicas,
                            pending: router.deployment.pending_replicas,
                          })}
                        </div>
                      </TableCell>
                      <TableCell className="whitespace-nowrap">
                        <div>{router.revision}</div>
                        <div className="text-xs text-muted-foreground">
                          D{router.deployment.deployment_generation}
                        </div>
                      </TableCell>
                      <TableCell className="whitespace-nowrap text-right">
                        <div className="flex shrink-0 items-center justify-end gap-1 whitespace-nowrap">
                          <Button
                            size="icon"
                            variant="ghost"
                            title={t('tokenRouter.details')}
                            onClick={() => setDetails(router)}
                          >
                            <Eye className="size-4" />
                          </Button>
                          {canWriteRouters && (
                            <Button
                              size="icon"
                              variant="ghost"
                              title={t('tokenRouter.edit')}
                              onClick={() => {
                                setEditing(router);
                                setFormOpen(true);
                              }}
                            >
                              <Pencil className="size-4" />
                            </Button>
                          )}
                          {canOperateRouters && (
                            <Button
                              size="icon"
                              variant="ghost"
                              title={t('tokenRouter.validate')}
                              loading={validating}
                              disabled={rowBusy}
                              onClick={() => void runAction(router, 'validate')}
                            >
                              {!validating && <ShieldCheck className="size-4" />}
                            </Button>
                          )}
                          {canOperateRouters && (
                            <Button
                              size="icon"
                              variant="ghost"
                              title={
                                router.enabled ? t('tokenRouter.disable') : t('tokenRouter.enable')
                              }
                              loading={toggling}
                              disabled={rowBusy}
                              onClick={() =>
                                void runAction(router, router.enabled ? 'disable' : 'enable')
                              }
                            >
                              {!toggling &&
                                (router.enabled ? (
                                  <PowerOff className="size-4" />
                                ) : (
                                  <Power className="size-4" />
                                ))}
                            </Button>
                          )}
                          {canWriteRouters && (
                            <Button
                              size="icon"
                              variant="ghost"
                              className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                              title={
                                router.enabled
                                  ? t('tokenRouter.deleteDisabledHint')
                                  : t('common.delete')
                              }
                              disabled={router.enabled || rowBusy}
                              onClick={() => setDeleting(router)}
                            >
                              <Trash2 className="size-4" />
                            </Button>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })
              )}
            </TableBody>
          </Table>
        </div>
      </section>

      <RouterFormDialog
        open={formOpen}
        router={editing}
        onOpenChange={setFormOpen}
        onSaved={() => void refreshAll('silent')}
      />
      <RouterDetailDrawer
        open={Boolean(details)}
        router={details}
        onOpenChange={(open) => !open && setDetails(null)}
      />
      <ConfirmDialog
        isOpen={Boolean(deleting)}
        onOpenChange={(open) => {
          if (!open && !deletingBusy) setDeleting(null);
        }}
        onConfirm={handleDelete}
        title={t('tokenRouter.deleteTitle')}
        description={t('tokenRouter.confirmDelete', {
          routerUid: deleting?.router_uid || '',
        })}
        confirmText={t('common.delete')}
        confirmClassName="bg-destructive text-white hover:bg-destructive/90"
        isLoading={deletingBusy}
      />
    </PageContainer>
  );
}
