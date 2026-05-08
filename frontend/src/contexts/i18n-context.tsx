"use client"

import React, { createContext, useContext, useEffect, useMemo, useState } from "react"
import { translations } from "@/i18n/translations"

export type Locale = "en" | "zh"

type TFunc = (key: string, vars?: Record<string, string | number>) => string

interface I18nContextValue {
  locale: Locale
  setLocale: (l: Locale) => void
  t: TFunc
}

const I18nContext = createContext<I18nContextValue | undefined>(undefined)

function interpolate(str: string, vars?: Record<string, string | number>) {
  if (!vars) return str
  return Object.entries(vars).reduce((s, [k, v]) => s.replace(new RegExp(`\\{${k}\\}`, "g"), String(v)), str)
}

export function I18nProvider({
  children,
  initialLocale = "en",
}: {
  children: React.ReactNode
  initialLocale?: Locale
}) {
  const [locale, setLocaleState] = useState<Locale>(initialLocale)

  useEffect(() => {
    try {
      const stored = typeof window !== "undefined" ? localStorage.getItem("app_locale") : null
      if (stored === "en" || stored === "zh") {
        if (stored !== locale) {
          setLocaleState(stored as Locale)
        }
      }
    } catch {
      // ignore
    }
  }, [locale])

  const setLocale = (l: Locale) => {
    setLocaleState(l)
    try {
      localStorage.setItem("app_locale", l)
      document.cookie = `app_locale=${l}; path=/; max-age=31536000; samesite=lax`
    } catch {
      // ignore
    }
  }

  // Sync <html lang> attribute
  useEffect(() => {
    if (typeof document !== "undefined") {
      document.documentElement.lang = locale
    }
  }, [locale])

  const t: TFunc = useMemo(() => {
    return (key, vars) => {
      const dict: any = translations[locale] || {}
      const value = key.split(".").reduce((acc: any, part: string) => (acc && acc[part] !== undefined ? acc[part] : undefined), dict)
      const str = typeof value === "string" ? value : key
      return interpolate(str, vars)
    }
  }, [locale])

  const value = useMemo(() => ({ locale, setLocale, t }), [locale, t])

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n() {
  const ctx = useContext(I18nContext)
  if (!ctx) throw new Error("useI18n must be used within I18nProvider")
  return ctx
}
