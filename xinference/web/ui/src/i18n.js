import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

import en from './locales/en.json'
import zh from './locales/zh.json'

i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: localStorage.getItem('language') || 'en',
  debug: true,
  interpolation: {
    escapeValue: false,
  },
  resources: {
    en: { translation: en },
    zh: { translation: zh },
  },
})

i18n.on('languageChanged', (lng) => {
  localStorage.setItem('language', lng)
})

export default i18n
