export const XINFERENCE_DOCS_URL = 'https://inference.readthedocs.io';
export const XINFERENCE_BASE_URL = 'https://xinference.io';
export const XINFERENCE_CN_URL = 'https://xinference.cn';
export const XINFERENCE_GITHUB = 'https://github.com/xorbitsai';

export const languages = [
  { label: '🇺🇸 English', value: 'en' },
  { label: '🇨🇳 中文', value: 'zh' },
  { label: '🇰🇷 한국어', value: 'ko' },
  { label: '🇯🇵 日本語', value: 'ja' },
] as const;
export const languageKeys = languages.map(lang => lang.value);
export const defaultLanguage = 'en';

export enum RequestEvents {
  // 401
  UNAUTHORIZED = 'UNAUTHORIZED',
  // 403
  FORBIDDEN = 'FORBIDDEN',
  SERVER_ERROR = 'SERVER_ERROR',
  SHOW_LOGIN_MODAL = 'SHOW_LOGIN_MODAL',
}
