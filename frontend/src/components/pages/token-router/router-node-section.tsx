'use client';

import { useState } from 'react';
import { Loader2, ServerCog } from 'lucide-react';
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
import type { TokenRouterNode } from '@/types/services';
import { getBindingStatusLabel, getBindingStatusTitle } from './tokenizer-asset-binding-status';

interface Props {
  nodes: TokenRouterNode[];
  initialLoading: boolean;
  canOperate: boolean;
  onChanged: () => Promise<void>;
}

type NodeState = TokenRouterNode['desired_state'];

export function RouterNodeSection({ nodes, initialLoading, canOperate, onChanged }: Props) {
  const { t } = useI18n();
  const [busyNodeId, setBusyNodeId] = useState<string | null>(null);

  const setState = async (node: TokenRouterNode, desiredState: NodeState) => {
    setBusyNodeId(node.node_id);
    try {
      await request.put(`/v1/token_router_nodes/${encodeURIComponent(node.node_id)}/state`, {
        desired_state: desiredState,
      });
      toast.success(t('tokenRouter.nodeStateUpdated'));
      await onChanged();
    } finally {
      setBusyNodeId(null);
    }
  };

  return (
    <section className="space-y-3">
      <div>
        <h2 className="flex items-center gap-2 text-base font-semibold">
          <ServerCog className="size-4" />
          {t('tokenRouter.routerNodes')}
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {t('tokenRouter.routerNodesDescription')}
        </p>
      </div>

      <div className="overflow-hidden rounded-xl border border-border">
        <Table className="min-w-[1260px] table-fixed">
          <colgroup>
            <col className="w-[210px]" />
            <col className="w-[130px]" />
            <col className="w-[170px]" />
            <col className="w-[190px]" />
            <col className="w-[140px]" />
            <col className="w-[220px]" />
            <col className="w-[190px]" />
            <col className="w-[260px]" />
          </colgroup>
          <TableHeader>
            <TableRow>
              <TableHead>{t('tokenRouter.nodeId')}</TableHead>
              <TableHead>{t('tokenRouter.status')}</TableHead>
              <TableHead>{t('tokenRouter.advertiseHost')}</TableHead>
              <TableHead>{t('tokenRouter.portPool')}</TableHead>
              <TableHead>{t('tokenRouter.nodeCapacity')}</TableHead>
              <TableHead>{t('tokenRouter.tokenizerAssets')}</TableHead>
              <TableHead>{t('tokenRouter.lastHeartbeat')}</TableHead>
              <TableHead className="text-right">{t('common.actions')}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {initialLoading ? (
              <TableRow>
                <TableCell colSpan={8} className="h-28 text-center text-muted-foreground">
                  <Loader2 className="mx-auto mb-2 size-5 animate-spin" />
                  {t('tokenRouter.loading')}
                </TableCell>
              </TableRow>
            ) : nodes.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} className="h-28 text-center text-muted-foreground">
                  {t('tokenRouter.noRouterNodes')}
                </TableCell>
              </TableRow>
            ) : (
              nodes.map((node) => {
                const bindings = node.tokenizer_asset_bindings || [];
                const busy = busyNodeId === node.node_id;
                const connectivity =
                  node.connectivity_status || (node.online ? 'online' : 'offline');
                const heartbeatAge = Number.isFinite(node.heartbeat_age_seconds)
                  ? Math.round(node.heartbeat_age_seconds)
                  : null;
                return (
                  <TableRow key={node.node_id}>
                    <TableCell>
                      <div className="break-all font-mono text-xs font-medium">{node.node_id}</div>
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-col items-start gap-1">
                        <Badge variant={connectivity === 'online' ? 'default' : 'secondary'}>
                          {t(
                            connectivity === 'online'
                              ? 'tokenRouter.nodeOnline'
                              : connectivity === 'suspected'
                                ? 'tokenRouter.nodeSuspected'
                                : 'tokenRouter.nodeOffline'
                          )}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {t(`tokenRouter.nodeStates.${node.desired_state}`)}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell className="font-mono text-xs">{node.advertise_host}</TableCell>
                    <TableCell>
                      <div className="font-mono text-xs">
                        {node.port_range_start}-{node.port_range_end}
                      </div>
                      <div className="mt-1 break-words text-xs text-muted-foreground">
                        {t('tokenRouter.usedPorts')}: {node.used_ports.join(', ') || '—'}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="text-sm">
                        {node.assignments}/{node.max_instances}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {t('tokenRouter.availableSlots')}: {node.available_slots}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="space-y-1 text-xs text-muted-foreground">
                        {bindings.length === 0
                          ? '—'
                          : bindings.map((binding) => (
                              <div
                                key={binding.asset_id}
                                className="break-words"
                                title={getBindingStatusTitle(
                                  t,
                                  binding.desired_state,
                                  binding.observed_state
                                )}
                              >
                                <span className="font-mono">{binding.asset_id}</span>:{' '}
                                {getBindingStatusLabel(
                                  t,
                                  binding.desired_state,
                                  binding.observed_state
                                )}
                              </div>
                            ))}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="text-xs">
                        {node.last_seen_at ? new Date(node.last_seen_at).toLocaleString() : '—'}
                      </div>
                      <div className="mt-1 text-xs text-muted-foreground">
                        {heartbeatAge === null
                          ? '—'
                          : t('tokenRouter.heartbeatAge', { seconds: heartbeatAge })}
                      </div>
                    </TableCell>
                    <TableCell className="text-right">
                      {canOperate ? (
                        <div className="flex flex-wrap justify-end gap-1">
                          {node.desired_state === 'active' ? (
                            <Button
                              size="sm"
                              variant="outline"
                              disabled={busy}
                              onClick={() => void setState(node, 'cordoned')}
                            >
                              {t('tokenRouter.cordon')}
                            </Button>
                          ) : (
                            <Button
                              size="sm"
                              variant="outline"
                              disabled={busy}
                              onClick={() => void setState(node, 'active')}
                            >
                              {t('tokenRouter.uncordon')}
                            </Button>
                          )}
                          <Button
                            size="sm"
                            variant="outline"
                            disabled={busy || node.desired_state === 'draining'}
                            onClick={() => void setState(node, 'draining')}
                          >
                            {t('tokenRouter.drain')}
                          </Button>
                          <Button
                            size="sm"
                            variant="destructive"
                            disabled={busy || node.desired_state === 'disabled'}
                            onClick={() => void setState(node, 'disabled')}
                          >
                            {t('tokenRouter.disableNode')}
                          </Button>
                        </div>
                      ) : (
                        '—'
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
