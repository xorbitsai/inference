export const XINFERENCE_DOCS_URL = 'https://inference.readthedocs.io';
export const XINFERENCE_BASE_URL = 'https://xinference.io';
export const XINFERENCE_CN_URL = 'https://xinference.cn';
export const XINFERENCE_GITHUB = 'https://github.com/xorbitsai';
export const XINFERENCE_IO = 'https://model.xinference.io';

export const LOGIN_PATH = '/login';
export const SETUP_PATH = '/setup';
export const HIDE_SIDEBAR_PATHS = [LOGIN_PATH, SETUP_PATH];
export const NO_AUTH = 'no_auth';
// sessionStorage flag: set by the setup page right before it navigates to
// /login, so the login page can show a one-time "account created" toast
// without threading router state through a full page navigation.
export const SETUP_COMPLETE_FLAG = 'xinference_setup_complete';

export const LANGUAGES = [
  { label: '🇺🇸 English', value: 'en' },
  { label: '🇨🇳 中文', value: 'zh' },
  { label: '🇰🇷 한국어', value: 'ko' },
  { label: '🇯🇵 日本語', value: 'ja' },
] as const;
export const LANGUAGES_KEYS = LANGUAGES.map((lang) => lang.value);
export const DEFAULT_LANGUAGE = 'en';

export enum RequestEvents {
  // 401
  UNAUTHORIZED = 'UNAUTHORIZED',
  // 403
  FORBIDDEN = 'FORBIDDEN',
  SERVER_ERROR = 'SERVER_ERROR',
  SHOW_LOGIN_MODAL = 'SHOW_LOGIN_MODAL',
}

export enum ModelType {
  LLM = 'LLM',
  Embedding = 'embedding',
  Rerank = 'rerank',
  Image = 'image',
  Audio = 'audio',
  Video = 'video',
  Custom = 'custom',
  Flexible = 'flexible',
}

export enum ModelAbility {
  Generate = 'generate',
  Chat = 'chat',
  Vision = 'vision',
  Tools = 'tools',
  Reasoning = 'reasoning',
  Audio = 'audio',
  Omni = 'omni',
  Hybrid = 'hybrid',
  Embed = 'embed',
  Rerank = 'rerank',
  Text2image = 'text2image',
  Image2image = 'image2image',
  Inpainting = 'inpainting',
  Audio2text = 'audio2text',
  Text2audio = 'text2audio',
  Audio2audio = 'audio2audio',
  Text2video = 'text2video',
  Image2video = 'image2video',
  Ocr = 'ocr',
  Firstlastframe2video = 'firstlastframe2video',
  Text2audioVoiceCloning = 'text2audio_voice_cloning',
  Text2audioZeroShot = 'text2audio_zero_shot',
  Text2audioEmotionControl = 'text2audio_emotion_control',
  Docanalyze = 'docanalyze',
}

export const CUSTOM_MODEL_OPTIONS = [
  { value: ModelType.LLM, labelKey: 'model.languageModels' },
  { value: ModelType.Embedding, labelKey: 'model.embeddingModels' },
  { value: ModelType.Rerank, labelKey: 'model.rerankModels' },
  { value: ModelType.Image, labelKey: 'model.imageModels' },
  { value: ModelType.Audio, labelKey: 'model.audioModels' },
  { value: ModelType.Flexible, labelKey: 'model.flexibleModels' },
];
