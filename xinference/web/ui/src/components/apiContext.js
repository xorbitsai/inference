import { createTheme, ThemeProvider } from '@mui/material'
import React, { createContext, useContext, useEffect, useState } from 'react'

import { getEndpoint } from './utils'

export const ApiContext = createContext()
export const useThemeSwitch = () => useContext(ApiContext)

export const ApiContextProvider = ({ children }) => {
  const [isCallingApi, setIsCallingApi] = useState(false)
  const [isUpdatingModel, setIsUpdatingModel] = useState(false)
  const [errorMsg, setErrorMsg] = useState('')
  const endPoint = getEndpoint()

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
      <ApiContext.Provider
        value={{
          isCallingApi,
          setIsCallingApi,
          isUpdatingModel,
          setIsUpdatingModel,
          endPoint,
          errorMsg,
          setErrorMsg,
          mode,
          toggleMode,
        }}
      >
        {children}
      </ApiContext.Provider>
    </ThemeProvider>
  )
}
