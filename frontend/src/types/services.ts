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

interface PromptsItem {
  chat_template: string;
  stop: string[];
  stop_token_ids: string[];
  reasoning_start_tag: string;
  reasoning_end_tag: string;
  tool_parser: string;
}
export type ModelPrompts = Record<string, PromptsItem>;
export type ModelFamily = Record<string, string[]>;

export interface ModelCachedItem {
  model_name: string;
  model_size_in_billions: number;
  model_format: string;
  quantization: string;
  model_version: string;
  path: string;
  real_path: string;
  actor_ip_address: string;
}

export interface ModelEnvItem {
  model_name: string;
  model_engine: string;
  path: string;
  real_path: string;
  python_version: string;
  actor_ip_address: string;
}

export type ModelEngineItem = {
  model_format: string;
  model_name: string;
  model_size_in_billions: string | number;
  quantizations: string[];
};
export type ModelEngine = Record<string, string | ModelEngineItem[]>;
