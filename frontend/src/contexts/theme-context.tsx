'use client';

import React, { createContext, useContext, useEffect, useState } from 'react';
import { themes, getThemeFromEnv, applyTheme, Theme } from '@/lib/theme';

interface ThemeContextType {
  theme: Theme;
  themeName: string;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  // Get initial theme from environment variable
  const initialTheme = getThemeFromEnv();
  const [themeName, setThemeNameState] = useState<string>(initialTheme);
  const [theme, setThemeState] = useState<Theme>(themes[initialTheme] || themes.dark);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    // Apply theme only on client side
    setIsClient(true);
    applyTheme(initialTheme);
  }, []);

  return (
    <ThemeContext.Provider
      value={{
        theme,
        themeName,
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
