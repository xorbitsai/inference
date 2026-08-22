'use client';

import { useMemo, useState } from 'react';
import { Database, Loader2 } from 'lucide-react';
import { toast } from 'sonner';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useI18n } from '@/contexts/i18n-context';
import request from '@/lib/request';
import type { TokenizerAssetBinding, TokenizerAssetItem, TokenRouterNode } from '@/types/services';
import {
  getBindingObservedStateBadgeVariant,
  getBindingStatusLabel,
  getBindingStatusTitle,
} from './tokenizer-asset-binding-status';

interface Props {
  assets: TokenizerAssetItem[];
  nodes: TokenRouterNode[];
  initialLoading: boolean;
  canWrite: boolean;
  canOperate: boolean;
  onChanged: () => Promise<void>;
}

export function TokenizerAssetSection({
  assets,
  nodes,
  initialLoading,
  canWrite,
  canOperate,
  onChanged,
}: Props) {
  const { t } = useI18n();
  const [selectedNodes, setSelectedNodes] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState('');

  const bindings = useMemo(
    () =>
      new Map(
        nodes.flatMap((node) =>
          (node.tokenizer_asset_bindings || []).map((binding) => [
            `${binding.asset_id}:${binding.node_id}`,
            binding,
          ])
        )
      ),
    [nodes]
  );

  const bind = async (asset: TokenizerAssetItem) => {
    const nodeId = selectedNodes[asset.asset_id];
    if (!nodeId) return;
    setBusy(`${asset.asset_id}:${nodeId}:bind`);
    try {
      await request.post(`/v1/tokenizer_assets/${encodeURIComponent(asset.asset_id)}/bindings`, {
        node_ids: [nodeId],
        desired_state: 'present',
        binding_mode: 'manual',
      });
      toast.success(t('tokenRouter.assetBindingUpdated'));
      setSelectedNodes((current) => ({ ...current, [asset.asset_id]: '' }));
      await onChanged();
    } finally {
      setBusy('');
    }
  };

  const updateBinding = async (
    binding: TokenizerAssetBinding,
    action: 'revalidate' | 'absent' | 'delete'
  ) => {
    const key = `${binding.asset_id}:${binding.node_id}:${action}`;
    setBusy(key);
    const base = `/v1/tokenizer_assets/${encodeURIComponent(binding.asset_id)}/bindings/${encodeURIComponent(binding.node_id)}`;
    try {
      if (action === 'revalidate') await request.post(`${base}/revalidate`);
      else if (action === 'absent') await request.patch(base, { desired_state: 'absent' });
      else await request.delete(base);
      toast.success(t('tokenRouter.assetBindingUpdated'));
      await onChanged();
    } finally {
      setBusy('');
    }
  };

  return (
    <section className="mt-6 space-y-3">
      <div>
        <h2 className="flex items-center gap-2 text-base font-semibold">
          <Database className="size-4" />
          {t('tokenRouter.assetCatalog')}
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {t('tokenRouter.assetCatalogDescription')}
        </p>
      </div>
      <div className="overflow-hidden rounded-xl border border-border">
        <Table className="min-w-[1100px] table-fixed">
          <TableHeader>
            <TableRow>
              <TableHead>{t('tokenRouter.tokenizerAsset')}</TableHead>
              <TableHead>{t('tokenRouter.tokenizerAssetOrigin')}</TableHead>
              <TableHead>{t('tokenRouter.tokenizerAssetRevision')}</TableHead>
              <TableHead>{t('tokenRouter.tokenizerAssetFingerprint')}</TableHead>
              <TableHead>{t('tokenRouter.assetBindings')}</TableHead>
              <TableHead>{t('common.actions')}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {initialLoading ? (
              <TableRow>
                <TableCell colSpan={6} className="h-24 text-center text-muted-foreground">
                  <Loader2 className="mx-auto mb-2 size-5 animate-spin" />
                  {t('tokenRouter.loading')}
                </TableCell>
              </TableRow>
            ) : assets.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} className="h-24 text-center text-muted-foreground">
                  {t('tokenRouter.noTokenizerAssets')}
                </TableCell>
              </TableRow>
            ) : (
              assets.map((asset) => {
                const assetBindings = nodes
                  .flatMap((node) => node.tokenizer_asset_bindings || [])
                  .filter((item) => item.asset_id === asset.asset_id);
                return (
                  <TableRow key={asset.asset_id}>
                    <TableCell>
                      <div className="font-mono text-xs">{asset.asset_id}</div>
                      <Badge variant={asset.enabled ? 'default' : 'secondary'} className="mt-1">
                        {asset.enabled
                          ? t('tokenRouter.assetEnabled')
                          : t('tokenRouter.assetDisabled')}
                      </Badge>
                    </TableCell>
                    <TableCell>{asset.origin}</TableCell>
                    <TableCell className="font-mono text-xs">{asset.revision}</TableCell>
                    <TableCell className="font-mono text-xs" title={asset.fingerprint}>
                      {asset.fingerprint.slice(0, 20)}…
                    </TableCell>
                    <TableCell>
                      <div className="space-y-2">
                        {assetBindings.length === 0 ? (
                          <span className="text-xs text-muted-foreground">—</span>
                        ) : (
                          assetBindings.map((binding) => (
                            <div key={binding.node_id} className="rounded border p-2 text-xs">
                              <div className="flex items-center justify-between gap-2">
                                <span className="font-mono">{binding.node_id}</span>
                                <Badge
                                  variant={getBindingObservedStateBadgeVariant(
                                    binding.observed_state
                                  )}
                                  title={getBindingStatusTitle(
                                    t,
                                    binding.desired_state,
                                    binding.observed_state
                                  )}
                                >
                                  {getBindingStatusLabel(
                                    t,
                                    binding.desired_state,
                                    binding.observed_state
                                  )}
                                </Badge>
                              </div>
                              {binding.last_error && (
                                <div className="mt-1 text-destructive">{binding.last_error}</div>
                              )}
                              {(canWrite || canOperate) && (
                                <div className="mt-2 flex flex-wrap gap-1">
                                  {canOperate && (
                                    <Button
                                      size="sm"
                                      variant="outline"
                                      disabled={Boolean(busy)}
                                      onClick={() => void updateBinding(binding, 'revalidate')}
                                    >
                                      {t('tokenRouter.revalidateAsset')}
                                    </Button>
                                  )}
                                  {canWrite && (
                                    <>
                                      <Button
                                        size="sm"
                                        variant="outline"
                                        disabled={
                                          Boolean(busy) || binding.desired_state === 'absent'
                                        }
                                        onClick={() => void updateBinding(binding, 'absent')}
                                      >
                                        {t('tokenRouter.removeAsset')}
                                      </Button>
                                      <Button
                                        size="sm"
                                        variant="destructive"
                                        disabled={
                                          Boolean(busy) || binding.observed_state !== 'absent'
                                        }
                                        onClick={() => void updateBinding(binding, 'delete')}
                                      >
                                        {t('common.delete')}
                                      </Button>
                                    </>
                                  )}
                                </div>
                              )}
                            </div>
                          ))
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      {canWrite && (
                        <div className="flex gap-2">
                          <select
                            className="h-9 min-w-44 rounded-md border bg-background px-2 text-sm"
                            value={selectedNodes[asset.asset_id] || ''}
                            onChange={(event) =>
                              setSelectedNodes((current) => ({
                                ...current,
                                [asset.asset_id]: event.target.value,
                              }))
                            }
                          >
                            <option value="">{t('tokenRouter.selectRouterNode')}</option>
                            {nodes
                              .filter(
                                (node) =>
                                  node.online &&
                                  node.connectivity_status === 'online' &&
                                  node.desired_state === 'active' &&
                                  !bindings.has(`${asset.asset_id}:${node.node_id}`)
                              )
                              .map((node) => (
                                <option key={node.node_id} value={node.node_id}>
                                  {node.node_id}
                                </option>
                              ))}
                          </select>
                          <Button
                            size="sm"
                            disabled={!selectedNodes[asset.asset_id] || Boolean(busy)}
                            onClick={() => void bind(asset)}
                          >
                            {t('tokenRouter.bindAsset')}
                          </Button>
                        </div>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </div>
    </section>
  );
}
