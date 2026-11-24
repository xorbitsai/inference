import type { ReactNode } from 'react';
import type { ClusterInfo, ClusterInformationItem, ClusterGPUInfo } from '@/types/services';

/** Edition adapters provide data and optional annotations; the layout stays shared. */
export interface ClusterDataSource {
  nodes: ClusterInformationItem[];
  loaded: boolean;
  updating: boolean;
  failed: boolean;
  updatedAt: string;
  refresh: () => void;
  notice?: ReactNode;
}
export interface ClusterDecorations {
  summaryCount?: (kind: 'workers' | 'gpus', total: number) => ReactNode;
  workerBadge?: (worker: ClusterInfo) => ReactNode;
  gpuBadge?: (worker: ClusterInfo, gpu: ClusterGPUInfo) => ReactNode;
  workerGpuCount?: (worker: ClusterInfo) => ReactNode;
}
