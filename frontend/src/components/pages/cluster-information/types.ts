import type { ReactNode } from 'react';
import type { ClusterInfo, ClusterInformationItem, ClusterGPUInfo } from '@/types/services';

/**
 * Supported read-side extension contract for edition adapters.
 * Supplying this object disables the page's community snapshot polling. The adapter
 * owns polling, errors, timestamps and conversion to ClusterInformationItem[].
 * refresh() is called by manual refresh and observed API/Supervisor/Worker completion.
 *
 * This does not replace action transports or authorization: relaunch controls still
 * use the shared request client and /v1 API, Supervisor and Worker relaunch routes.
 * Adapters must read the same cluster and provide those compatible action endpoints;
 * administrator visibility continues to follow the shared authorization context.
 */
export interface ClusterDataSource {
  nodes: ClusterInformationItem[];
  loaded: boolean;
  updating: boolean;
  failed: boolean;
  updatedAt: string;
  refresh: () => void;
  notice?: ReactNode;
}
/**
 * Presentation-only annotations; these callbacks do not authorize or change actions.
 * Counts may display licensed/total values without changing the underlying metrics.
 * Badges are additive: GPU utilization indicators remain visible. GPU badges render
 * outside the card's selection button so a badge can provide its own tooltip.
 */
export interface ClusterDecorations {
  summaryCount?: (kind: 'workers' | 'gpus', total: number) => ReactNode;
  workerBadge?: (worker: ClusterInfo) => ReactNode;
  gpuBadge?: (worker: ClusterInfo, gpu: ClusterGPUInfo) => ReactNode;
  workerGpuCount?: (worker: ClusterInfo) => ReactNode;
}
