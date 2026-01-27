import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

import en from './locales/en.json'
import ja from './locales/ja.json'
import ko from './locales/ko.json'
import zh from './locales/zh.json'

const supportedLangs = ['en', 'zh', 'ja', 'ko']
const detectBrowserLanguage = () => {
  try {
    const candidate =
      (typeof navigator !== 'undefined' &&
        (navigator.languages?.[0] || navigator.language)) ||
      ''
    const normalized = String(candidate).toLowerCase()
    const prefix = normalized.split('-')[0]
    return supportedLangs.includes(prefix) ? prefix : 'en'
  } catch {
    return 'en'
  }
}

i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: localStorage.getItem('language') || detectBrowserLanguage(),
  debug: true,
  interpolation: {
    escapeValue: false,
  },
  resources: {
    en: { translation: en },
    zh: { translation: zh },
    ja: { translation: ja },
    ko: { translation: ko },
  },
})

i18n.on('languageChanged', (lng) => {
  localStorage.setItem('language', lng)
})

export default i18n
