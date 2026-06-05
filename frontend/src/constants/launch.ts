import { ModelType } from '@/constants';

export const LAUNCH_MODEL_ROUTE_TABS = [
  { key: ModelType.LLM, path: 'llm', labelKey: 'model.languageModels' },
  { key: ModelType.Embedding, path: 'embedding', labelKey: 'model.embeddingModels' },
  { key: ModelType.Rerank, path: 'rerank', labelKey: 'model.rerankModels' },
  { key: ModelType.Image, path: 'image', labelKey: 'model.imageModels' },
  { key: ModelType.Audio, path: 'audio', labelKey: 'model.audioModels' },
  { key: ModelType.Video, path: 'video', labelKey: 'model.videoModels' },
  { key: ModelType.Custom, path: 'custom', labelKey: 'model.customModels' },
] as const;

export const LAUNCH_MODEL_UPDATE_OPTIONS = [
  { label: 'LLM', value: ModelType.LLM },
  { label: 'Embedding', value: ModelType.Embedding },
  { label: 'Rerank', value: ModelType.Rerank },
  { label: 'Image', value: ModelType.Image },
  { label: 'Audio', value: ModelType.Audio },
  { label: 'Video', value: ModelType.Video },
];

export const COLLECTION_STORAGE_KEY = 'collectionArr';

export const ENGINES_WITH_WORKER = ['SGLang', 'vLLM', 'MLX'];

export const VIRTUAL_ENV_OPTIONS = [
  { label: 'Unset', value: 'unset' },
  { label: 'False', value: false },
  { label: 'True', value: true },
];

export const KWARGS_OPTIONS_FOR_ENGINES: Record<string, Array<{ label: string; value: string }>> = {
  transformers: [
    { label: 'torch_dtype', value: 'torch_dtype' },
    { label: 'device', value: 'device' },
    { label: 'enable_flash_attn', value: 'enable_flash_attn' },
  ],
  'llama.cpp': [
    { label: 'n_ctx', value: 'n_ctx' },
    { label: 'use_mmap', value: 'use_mmap' },
    { label: 'use_mlock', value: 'use_mlock' },
  ],
  vllm: [
    { label: 'block_size', value: 'block_size' },
    { label: 'gpu_memory_utilization', value: 'gpu_memory_utilization' },
    { label: 'max_num_seqs', value: 'max_num_seqs' },
    { label: 'max_model_len', value: 'max_model_len' },
    { label: 'guided_decoding_backend', value: 'guided_decoding_backend' },
    { label: 'scheduling_policy', value: 'scheduling_policy' },
    { label: 'tensor_parallel_size', value: 'tensor_parallel_size' },
    { label: 'pipeline_parallel_size', value: 'pipeline_parallel_size' },
    { label: 'enable_prefix_caching', value: 'enable_prefix_caching' },
    { label: 'enable_chunked_prefill', value: 'enable_chunked_prefill' },
    { label: 'enable_expert_parallel', value: 'enable_expert_parallel' },
    { label: 'enforce_eager', value: 'enforce_eager' },
    { label: 'cpu_offload_gb', value: 'cpu_offload_gb' },
    { label: 'disable_custom_all_reduce', value: 'disable_custom_all_reduce' },
    { label: 'limit_mm_per_prompt', value: 'limit_mm_per_prompt' },
    { label: 'model_quantization', value: 'model_quantization' },
    { label: 'mm_processor_kwargs', value: 'mm_processor_kwargs' },
    { label: 'min_pixels', value: 'min_pixels' },
    { label: 'max_pixels', value: 'max_pixels' },
  ],
  sglang: [
    { label: 'mem_fraction_static', value: 'mem_fraction_static' },
    { label: 'attention_reduce_in_fp32', value: 'attention_reduce_in_fp32' },
    { label: 'tp_size', value: 'tp_size' },
    { label: 'dp_size', value: 'dp_size' },
    { label: 'chunked_prefill_size', value: 'chunked_prefill_size' },
    { label: 'cpu_offload_gb', value: 'cpu_offload_gb' },
    { label: 'enable_dp_attention', value: 'enable_dp_attention' },
    { label: 'enable_ep_moe', value: 'enable_ep_moe' },
  ],
  mlx: [
    { label: 'cache_limit_gb', value: 'cache_limit_gb' },
    { label: 'max_kv_size', value: 'max_kv_size' },
  ],
};
export const QUANTIZATION_OPTIONS = [
  { label: 'load_in_8bit', value: 'load_in_8bit' },
  { label: 'load_in_4bit', value: 'load_in_4bit' },
  { label: 'llm_int8_threshold', value: 'llm_int8_threshold' },
  { label: 'llm_int8_skip_modules', value: 'llm_int8_skip_modules' },
  { label: 'llm_int8_enable_fp32_cpu_offload', value: 'llm_int8_enable_fp32_cpu_offload' },
  { label: 'llm_int8_has_fp16_weight', value: 'llm_int8_has_fp16_weight' },
  { label: 'bnb_4bit_compute_dtype', value: 'bnb_4bit_compute_dtype' },
  { label: 'bnb_4bit_quant_type', value: 'bnb_4bit_quant_type' },
  { label: 'bnb_4bit_use_double_quant', value: 'bnb_4bit_use_double_quant' },
  { label: 'bnb_4bit_quant_storage', value: 'bnb_4bit_quant_storage' },
];
/**
 * Keys owned by the launch form.
 * Extra top-level fields are created from kwargs during submit, so cache restore
 * moves keys outside this list back into the kwargs form-list.
 * Add new launch form fields here when the dialog gains new submit fields.
 */
export const ALL_FORM_KEYS = [
  'model_uid',
  'model_name',
  'model_type',
  'model_engine',
  'model_format',
  'model_size_in_billions',
  'quantization',
  'n_worker',
  'n_gpu',
  'n_gpu_layers',
  'replica',
  'request_limits',
  'worker_ip',
  'gpu_idx',
  'download_hub',
  'model_path',
  'reasoning_content',
  'gguf_quantization',
  'gguf_model_path',
  'lightning_version',
  'lightning_model_path',
  'cpu_offload',
  'peft_model_config',
  'quantization_config',
  'enable_thinking',
  'multimodal_projector',
  'enable_virtual_env',
  'virtual_env_packages',
  'envs',
];
