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
  grafana_dashboards?: Record<string, string>;
  grafana_dashboards_configured?: string[];
  cluster_name: string;
  es_enabled: boolean;
  auth_advanced: boolean;
  oidc_enabled: boolean;
  token_router_enabled?: boolean;
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
  /** whether this engine can run a drafter for speculative decoding */
  support_draft_model?: boolean;
};
export type ModelEngine = Record<string, string | ModelEngineItem[]>;

export interface ReplicaItem {
  created_ts: number;
  error_message: string;
  replica_id: number;
  replica_model_uid: string;
  status: string;
  worker_address: string;
  // Address of the model subprocess/subpool, not the Worker's service port.
  model_address?: string | null;
  // Actual accelerators assigned to this replica after scheduling.
  accelerators?: string[] | null;
  replica_uid?: string;
  gpu_idx?: number[];
}

export interface AddReplicaRequest {
  replica_config?: {
    replica_uid?: string;
    devices: Array<{
      worker_ip?: string;
      gpu_idx?: number[];
    }>;
  };
}

export interface AddReplicaResponse {
  replica_id: number;
  replica_model_uid: string;
  worker_address: string;
}

export interface RunningModelDeploymentSummary {
  management_mode?: 'external' | 'managed' | string;
  desired_state?: 'running' | 'stopped' | string;
  desired_replicas?: number;
  ready_replicas?: number;
  pending_replicas?: number;
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
  model_kind?: string;
  virtual_model_type?: string;
  router_uid?: string;
  router_status?: string;
  route_profile?: string;
  runtime_instances?: number;
  online_instances?: number;
  ready_instances?: number;
  backend_count?: number;
  deployment?: RunningModelDeploymentSummary;
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
  model_engine?: string;
  model_kind?: string;
  virtual_model_type?: string;
  router_uid?: string;
  router_status?: string;
  route_profile?: string;
  runtime_instances?: number;
  online_instances?: number;
  ready_instances?: number;
  backend_count?: number;
  deployment?: RunningModelDeploymentSummary;
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

export interface AudioEmbeddingResponse {
  object: 'embedding';
  model: string;
  dimensions: number;
  embedding: number[];
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

export interface TokenizerAssetItem {
  asset_id: string;
  origin: 'builtin' | 'artifact' | 'shared_fs' | 'local' | 'external';
  display_name: string;
  model_family: string;
  model_name: string;
  revision: string;
  encoding_type: string;
  compatible_models: string[];
  capabilities: Record<string, unknown>;
  enabled: boolean;
  status: 'available' | 'invalid' | 'disabled' | 'missing' | 'validating';
  valid: boolean;
  fingerprint: string;
  errors: string[];
  checks?: Record<string, string>;
  validated_at?: string;
  source?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  bindings?: number;
  ready_bindings?: number;
  binding_states?: Record<string, number>;
  router_references?: string[];
  binding_items?: TokenizerAssetBinding[];
}

export interface TokenizerAssetListResponse {
  items: TokenizerAssetItem[];
  allow_custom_path: boolean;
  config_error?: string;
}

export interface TokenRouterAdmissionConfig {
  max_active: number;
  max_queue: number;
  queue_timeout_seconds: number;
  retry_after_seconds: number;
}

export interface TokenRouterBackendConfig {
  model_uid: string;
  max_context_tokens: number;
  admission: TokenRouterAdmissionConfig;
}

export interface TokenRouterDynamicBackendConfig extends TokenRouterBackendConfig {
  id: string;
}

export interface TokenRouterRuleMatch {
  total_tokens_gte?: number;
  total_tokens_lte?: number;
  thinking?: boolean;
  tools_present?: boolean;
  stream?: boolean;
}

export type TokenRouterRouteAction = {
  type: 'route';
  backend_id: string;
};

export type TokenRouterRejectAction = {
  type: 'reject';
  reason: string;
};

export type TokenRouterRoutingAction = TokenRouterRouteAction | TokenRouterRejectAction;

export interface TokenRouterRoutingRule {
  id: string;
  priority: number;
  match: TokenRouterRuleMatch;
  action: TokenRouterRoutingAction;
}

export interface TokenRouterDeployment {
  router_uid: string;
  management_mode: 'external' | 'managed';
  desired_replicas: number;
  desired_state: 'running' | 'stopped';
  placement: Record<string, unknown>;
  rollout: {
    auto_failover?: boolean;
    drain_timeout_seconds?: number;
    [key: string]: unknown;
  };
  deployment_generation: number;
  observed_ready_assignments: number;
  effective_ready_runtimes: number;
  controllable_ready_runtimes: number;
  ready_replicas: number;
  pending_replicas: number;
  assignments: number;
  failure_reason?: string;
  recovery_state?: 'stable' | 'degraded' | 'recovering' | 'blocked';
}

export interface TokenRouterAssignment {
  assignment_id: string;
  router_uid: string;
  replica_index: number;
  node_id: string;
  listen_host: string;
  listen_port: number;
  public_endpoint: string;
  desired_state: 'running' | 'stopped';
  observed_state: string;
  assignment_generation: number;
  config_revision: number;
  pid?: number | null;
  instance_id?: string | null;
  last_error?: string;
  management_state?: 'manageable' | 'node_suspected' | 'node_lost';
  failure_reason?: string;
  observed?: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
}

export interface TokenizerAssetBinding {
  asset_id: string;
  node_id: string;
  desired_state: 'present' | 'absent';
  observed_state: string;
  desired_revision: string;
  desired_fingerprint: string;
  observed_revision: string;
  observed_fingerprint: string;
  local_path: string;
  binding_mode: 'manual' | 'on_demand';
  owner_type: string;
  owner_id: string;
  generation: number;
  last_error_code: string;
  last_error: string;
  last_seen_at?: string | null;
}

export interface TokenRouterNode {
  node_id: string;
  advertise_host: string;
  port_range_start: number;
  port_range_end: number;
  max_instances: number;
  labels: Record<string, unknown>;
  reported_labels: Record<string, unknown>;
  managed_labels: Record<string, unknown>;
  tokenizer_asset_bindings: TokenizerAssetBinding[];
  capabilities: {
    [key: string]: unknown;
  };
  software_version: string;
  software_revision?: string | null;
  desired_state: 'active' | 'cordoned' | 'draining' | 'disabled';
  connectivity_status: 'online' | 'suspected' | 'offline';
  management_state: 'active' | 'cordoned' | 'draining' | 'disabled';
  suspected_at?: string | null;
  offline_at?: string | null;
  observed: Record<string, unknown>;
  last_seen_at?: string | null;
  heartbeat_age_seconds: number;
  can_schedule: boolean;
  failure_reason?: string;
  online: boolean;
  assignments: number;
  used_ports: number[];
  available_slots: number;
  created_at: string;
  updated_at: string;
}

interface TokenRouterItemBase {
  router_uid: string;
  virtual_model_uid: string;
  model_type: 'LLM';
  route_profile?: 'llm_chat';
  tokenizer_asset_id?: string;
  tokenizer_asset_origin?: 'builtin' | 'external';
  tokenizer_path: string;
  tokenizer_asset_revision?: string;
  tokenizer_asset_fingerprint?: string;
  backend_url: string;
  model_aliases: string[];
  request_timeout_seconds: number;
  connect_timeout_seconds: number;
  tokenization: {
    executor: 'process';
    multiprocessing_start_method: 'spawn';
    max_workers: number;
    max_active: number;
    max_queue: number;
    queue_timeout_seconds: number;
    retry_after_seconds: number;
  };
  enabled: boolean;
  revision: number;
  status:
    | 'draft'
    | 'disabled'
    | 'pending'
    | 'starting'
    | 'syncing'
    | 'ready'
    | 'degraded'
    | 'offline'
    | 'unavailable'
    | 'draining'
    | 'error';
  runtime_instances: number;
  online_instances: number;
  deployment: TokenRouterDeployment;
  created_at: string;
  updated_at: string;
}

export interface TokenRouterLegacyItem extends TokenRouterItemBase {
  config_version?: 1;
  strategy: 'token_budget';
  backends: {
    short: TokenRouterBackendConfig;
    long: TokenRouterBackendConfig;
  };
  routing: {
    short_threshold_tokens: number;
    context_reserve_tokens: number;
    default_output_tokens: number;
    thinking_policy: 'short' | 'long' | 'reject';
    overflow_policy: 'reject';
  };
}

export interface TokenRouterTypedItem extends TokenRouterItemBase {
  config_version: 2;
  route_profile: 'llm_chat';
  strategy: 'typed_rules';
  backends: TokenRouterDynamicBackendConfig[];
  routing: {
    evaluation_mode: 'first_match';
    context_reserve_tokens: number;
    default_output_tokens: number;
    rules: TokenRouterRoutingRule[];
    default_action: TokenRouterRoutingAction;
  };
}

export type TokenRouterItem = TokenRouterLegacyItem | TokenRouterTypedItem;

export interface TokenRouterBackendCandidate {
  model_uid: string;
  model_name: string;
  model_type?: string;
  model_engine: string;
  model_format: string;
  model_ability: string[];
  context_length?: number;
  compatibility_status: 'Verified' | 'Unsupported' | 'Unknown';
  compatibility_reason: string;
  eligible: boolean;
  ineligible_reasons: string[];
}

export interface TokenRouterBackendCandidateResponse {
  items: TokenRouterBackendCandidate[];
  errors: string[];
}

export interface TokenRouterDefaultsResponse {
  backend: {
    mode: 'current_supervisor';
    display_name: string;
    backend_url: string | null;
    source: 'server_config' | 'rest_endpoint' | 'unavailable';
    available: boolean;
    error?: string;
  };
}

export function isTypedTokenRouter(router: TokenRouterItem): router is TokenRouterTypedItem {
  return router.config_version === 2 || Array.isArray(router.backends);
}

export type RouterClusterStatus =
  | 'ready'
  | 'syncing'
  | 'degraded'
  | 'config_error'
  | 'heartbeat_timeout'
  | 'not_running'
  | 'disabled';

export interface RouterRuntimeMetadata {
  assignment_id?: string;
  assignment_generation?: number;
  node_id?: string;
  hostname?: string;
  python_version?: string;
  platform?: string;
  started_at?: number;
}

export interface RouterProcessResources {
  cpu_percent?: number;
  cpu_cores?: number;
  cpu_count?: number;
  rss_bytes?: number;
  memory_total_bytes?: number;
  main_process_rss_bytes?: number;
  child_process_rss_bytes?: number;
  child_process_count?: number;
  thread_count?: number;
  started_at?: number;
  uptime_seconds?: number;
  sampled_at?: number;
}

export interface RouterClusterInfo {
  node_type: 'Router';
  router_uid: string;
  router_status: RouterClusterStatus;
  instance_id: string;
  endpoint: string;
  online: boolean;
  instance_status?: string | null;
  acked_revision?: number | null;
  config_revision?: number | null;
  registered_at?: number | null;
  last_heartbeat?: number | null;
  heartbeat_age_seconds?: number | null;
  version?: string;
  protocol_version?: string;
  software_version?: string;
  software_revision?: string | null;
  metadata?: RouterRuntimeMetadata;
  config_error?: string;
  metrics?: Record<string, unknown>;
  backend_health?: Record<string, unknown>;
  process?: {
    pid?: number;
    revision?: number;
    resources?: RouterProcessResources;
    tokenization?: {
      worker_pids?: number[];
      [key: string]: unknown;
    };
    [key: string]: unknown;
  };
}

export interface RouterNodeClusterInfo {
  node_type: 'Router';
  node_id: string;
  ip_address: string;
  online: boolean;
  connectivity_status: 'online';
  gpu_count: number;
  gpu_vram_total: number;
  cpu_available: number | null;
  cpu_count: number | null;
  mem_used: number | null;
  mem_available: number | null;
  mem_total: number | null;
  software_version?: string;
  software_revision?: string | null;
}

export type ClusterInformationItem = ClusterInfo | RouterNodeClusterInfo;

export interface TokenRouterRuntimeInstance {
  router_uid: string;
  assignment_id?: string;
  assignment_generation?: number;
  node_id?: string;
  node_address?: string | null;
  instance_id: string;
  endpoint: string;
  status?: string;
  online: boolean;
  acked_revision: number;
  heartbeat_age_seconds: number;
  last_heartbeat: number;
  config_error?: string;
  metrics?: Record<string, unknown>;
  backend_health?: Record<string, unknown>;
  process?: {
    tokenizer_asset?: {
      asset_id?: string;
      origin?: 'builtin' | 'external';
      revision?: string;
      fingerprint?: string;
    };
    [key: string]: unknown;
  };
}
