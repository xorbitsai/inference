import { ModelAbility } from '@/constants';
export interface ClusterAuth {
  auth: boolean;
}
export interface ClusterVersion {
  date: string;
  dirty: boolean;
  error: unknown;
  'full-revisionid': string;
  version: string;
}
export interface ClusterUIConfig {
  grafana_url: string;
  grafana_datasource: string;
  grafana_alert_datasource: string;
  grafana_dashboard_uid: string;
  cluster_name: string;
  es_enabled: boolean;
  auth_advanced: boolean;
  oidc_enabled: boolean;
}
export interface ClusterInfo {
  node_type: 'Supervisor' | 'Worker';
  ip_address: string;
  ip?: string;
  gpu_count: number;
  gpu_vram_total: number;
  cpu_available: number;
  cpu_count: number;
  mem_used: number;
  mem_available: number;
  mem_total: number;
  gpu_utilization: number | null;
  gpu_vram_available: number;
}

export type ClusterInfoResponse =
  | ClusterInfo[]
  | {
      supervisors?: ClusterInfo[];
      workers?: ClusterInfo[];
    };

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

export interface VirtualEnv {
  model_name?: string;
  model_engine?: string;
  python_version?: string;
  worker_ip?: string;
  env_path?: string;
  path?: string;
  real_path?: string;
  [key: string]: unknown;
}
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
  multimodal_projectors?: string[];
  quantization?: string;
  quantizations?: string[];
};
export type ModelEngine = Record<string, string | ModelEngineItem[]>;

export interface ReplicaItem {
  created_ts: number;
  error_message: string;
  replica_id: number;
  replica_model_uid: string;
  status: string;
  worker_address: string;
}

export interface RunningModelItem {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  model_type: string;
  address: string;
  accelerators: string[];
  model_name: string;
  model_lang: string[];
  model_ability: ModelAbility[];
  model_description: string;
  model_engine?: string;
  model_format: string;
  model_size_in_billions: number;
  model_family: string;
  quantization: string;
  multimodal_projector: null;
  model_hub: string;
  revision: string | null;
  context_length: number;
  replica: number;
  // Real-time per-model GPU memory usage in bytes, keyed by worker address
  // and then by (worker-local) GPU index. Only present for NVIDIA GPUs;
  // absent otherwise.
  gpu_memory?: Record<string, Record<string, number>>;
}

export interface RunningModelDetail {
  model_type: string;
  address: string;
  accelerators: string[];
  model_name: string;
  model_lang: string[];
  model_ability: ModelAbility[];
  model_description: string;
  model_format: string;
  model_size_in_billions: number;
  model_family: string;
  quantization: string;
  multimodal_projector: string;
  model_hub: string;
  revision: string;
  context_length: number;
  replica: number;
}

interface MessageFileType {
  data: string;
  expires_at: number;
  id: string;
  transcript: string;
}

export interface ChatChoicesMessage {
  content: string;
  role: string;
  audio?: MessageFileType;
  image?: MessageFileType;
  video?: MessageFileType;
}

export interface ChatStreamResult {
  created: number;
  id: string;
  model: string;
  object: string;
  choices: {
    index: number;
    finish_reason: string;
    delta: {
      content: string;
      reasoning_content?: string;
    };
    message?: ChatChoicesMessage;
  }[];
  usage: {
    completion_tokens: number;
    prompt_tokens: number;
    total_tokens: number;
  };
}

interface CompletionChoice {
  text?: string;
  index?: number;
  logprobs?: unknown;
  finish_reason?: string | null;
  [key: string]: unknown;
}

interface TokenUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  [key: string]: unknown;
}

export interface CompletionResponse {
  id?: string;
  object?: string;
  created?: number;
  model?: string;
  choices: CompletionChoice[];
  usage?: TokenUsage;
  [key: string]: unknown;
}

interface RerankMeta {
  api_version: string | null;
  billed_units: string | null;
  tokens: string | null;
  warnings: string | null;
}
interface RerankResult {
  index: number;
  relevance_score: number;
  document: string;
}
export interface RerankResponse {
  id: string;
  meta: RerankMeta;
  results: RerankResult[];
}

interface EmbeddingsData {
  embedding: number[];
  index: number;
  object: string;
}
export interface EmbeddingsResponse {
  data: EmbeddingsData[];
  model: string;
  model_replica: string;
  object: string;
  usage: TokenUsage;
}

export interface UserItem {
  id: number;
  username: string;
  source: string;
  enabled: boolean;
  must_change_password: boolean;
  permissions: string[];
  created_at: string | null;
}
