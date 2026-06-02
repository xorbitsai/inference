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
