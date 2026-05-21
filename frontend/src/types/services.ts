export interface ClusterAuth {
  auth: boolean;
}
export interface ClusterVersion {
  date: string;
  dirty: boolean;
  error: any;
  'full-revisionid': string;
  version: string;
}
export interface ClusterInfo {
  node_type: 'Supervisor' | 'Worker';
  ip_address: string;
  gpu_count: number;
  gpu_vram_total: number;
  cpu_available: number;
  cpu_count: number;
  mem_used: number;
  mem_available: number;
  mem_total: number;
  gpu_utilization: number;
  gpu_vram_available: number;
}

export type ModelFamily = Record<string, string[]>;