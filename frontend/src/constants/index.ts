export const XINFERENCE_DOCS_URL = 'https://inference.readthedocs.io';
export const XINFERENCE_BASE_URL = 'https://xinference.io';
export const XINFERENCE_CN_URL = 'https://xinference.cn';
export const XINFERENCE_GITHUB = 'https://github.com/xorbitsai';
export const XINFERENCE_IO = 'https://model.xinference.io';

export const LOGIN_PATH = '/login';
export const HIDE_SIDEBAR_PATHS = [LOGIN_PATH];
export const NO_AUTH = 'no_auth';

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
}
