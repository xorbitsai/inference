import { ThemeProvider as MuiThemeProvider } from '@mui/material'
import { createContext, useContext, useState } from 'react'

import { useMode } from '../theme'

const ThemeContext = createContext()

export function useThemeContext() {
  return useContext(ThemeContext)
}

export const ThemeProvider = ({ children }) => {
  const themeKey = 'theme'
  const systemPreference = window.matchMedia('(prefers-color-scheme: dark)')
    .matches
    ? 'dark'
    : 'light'
  const initialMode = localStorage.getItem(themeKey) || systemPreference

  const [themeMode, setThemeMode] = useState(initialMode)
  const theme = useMode(themeMode)[0]

  const switchTheme = () => {
    const nextTheme = themeMode === 'light' ? 'dark' : 'light'
    setThemeMode(nextTheme)
    localStorage.setItem(themeKey, nextTheme)
  }

  return (
    <MuiThemeProvider theme={theme}>
      <ThemeContext.Provider value={{ themeMode, toggleTheme: switchTheme }}>
        {children}
      </ThemeContext.Provider>
    </MuiThemeProvider>
  )
}
