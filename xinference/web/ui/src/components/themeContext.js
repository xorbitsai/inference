import { createTheme, ThemeProvider } from '@mui/material'
import React, { createContext, useContext, useEffect, useState } from 'react'

export const ThemeContext = createContext()
export const useThemeSwitch = () => useContext(ThemeContext)

export const ThemeContextProvider = ({ children }) => {
  const systemPreference = window.matchMedia('(prefers-color-scheme: dark)')
    .matches
    ? 'dark'
    : 'light'
  const initialMode = localStorage.getItem('theme') || systemPreference
  const [mode, setMode] = useState(initialMode)

  useEffect(() => {
    localStorage.setItem('theme', mode)
  }, [mode])

  let theme = createTheme({
    palette: {
      mode: mode,
    },
  })

  const toggleMode = () => {
    setMode((prevMode) => (prevMode === 'light' ? 'dark' : 'light'))
  }

  return (
    <ThemeProvider theme={theme}>
      <ThemeContext.Provider value={{ mode, toggleMode }}>
        {children}
      </ThemeContext.Provider>
    </ThemeProvider>
  )
}
