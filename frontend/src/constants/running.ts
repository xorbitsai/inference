import { ModelType, ModelAbility } from '@/constants';

export type CodeLanguage = 'python' | 'typescript' | 'java' | 'go' | 'shell';

export type CodeExampleContentType = 'json' | 'form';

export type CodeExampleFieldType = 'text' | 'file';

export interface CodeExampleField {
  key: string;
  required?: boolean;
  value?: unknown;
  comment?: string;
  type?: CodeExampleFieldType;
  stringify?: boolean;
}

export interface CodeExampleConfig {
  method: 'POST';
  contentType: CodeExampleContentType;
  fields: CodeExampleField[];
}

export const CODE_LANGUAGE_OPTIONS: {
  label: string;
  value: CodeLanguage;
  highlight: string;
}[] = [
  { label: 'Python', value: 'python', highlight: 'python' },
  { label: 'TypeScript', value: 'typescript', highlight: 'typescript' },
  { label: 'Java', value: 'java', highlight: 'java' },
  { label: 'Go', value: 'go', highlight: 'go' },
  { label: 'Shell', value: 'shell', highlight: 'bash' },
];

export const CHAT_CODE_EXAMPLE: CodeExampleConfig = {
  method: 'POST',
  contentType: 'json',
  fields: [
    { key: 'model', required: true },
    {
      key: 'messages',
      required: true,
      value: [{ role: 'user', content: 'Hello, what can you do?' }],
    },
    { key: 'stream', value: false, comment: 'Optional' },
    { key: 'max_tokens', value: 4000 },
    { key: 'top_p', value: 1 },
    { key: 'top_k', value: 40 },
    { key: 'presence_penalty', value: 0 },
    { key: 'frequency_penalty', value: 0 },
    { key: 'temperature', value: 0.6 },
  ],
};

export const CODE_EXAMPLE_DEFAULT_VALUES: Record<string, unknown> = {
  model: '{MODEL_UID}',
  prompt: 'Hello, what can you do?',
  input: 'Hello, can you read this text aloud?',
  voice: 'default',
  image: '/path/to/image.png',
  mask_image: '/path/to/mask.png',
  first_frame: '/path/to/first-frame.png',
  last_frame: '/path/to/last-frame.png',
  file: './audio.wav',
  negative_prompt: '',
  language: 'en',
  response_format: 'json',
  speed: 1,
  n: 1,
  size: '1024*1024',
  stream: false,
  temperature: 1,
  max_tokens: 256,
  kwargs: {},
};

export const SAMPLING_METHOD_OPTIONS = [
  'default',
  'DPM++ 2M',
  'DPM++ 2M Karras',
  'DPM++ 2M SDE',
  'DPM++ 2M SDE Karras',
  'DPM++ SDE',
  'DPM++ SDE Karras',
  'DPM2',
  'DPM2 Karras',
  'DPM2 a',
  'DPM2 a Karras',
  'Euler',
  'Euler a',
  'Heun',
  'LMS',
  'LMS Karras',
].map((value) => ({
  label: value === 'default' ? 'Default' : value,
  value,
}));

export const OCR_TYPE_OPTIONS = [
  { label: 'Ocr', value: 'ocr' },
  { label: 'Markdown', value: 'markdown' },
  { label: 'Format', value: 'format' },
];

export const OCR_MODEL_SIZE_OPTIONS = [
  { label: 'Tiny', value: 'tiny' },
  { label: 'Small', value: 'small' },
  { label: 'Base', value: 'base' },
  { label: 'Large', value: 'large' },
  { label: 'Gundam', value: 'gundam' },
];

export const MODEL_TYPE_ABILITY_MAP: Record<string, ModelAbility[]> = {
  [ModelType.Rerank]: [ModelAbility.Rerank],
  [ModelType.Embedding]: [ModelAbility.Embed],
};
