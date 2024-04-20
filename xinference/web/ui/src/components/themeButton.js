import DarkModeIcon from '@mui/icons-material/DarkMode'
import LightModeIcon from '@mui/icons-material/LightMode'
import { Box, IconButton } from '@mui/material'
import React from 'react'

import { useThemeContext } from './themeContext'

const ThemeButton = ({ sx }) => {
  const { themeMode, toggleTheme } = useThemeContext()

  console.log('主题：', useThemeContext)
  return (
    <Box sx={sx}>
      <IconButton size="large" onClick={toggleTheme}>
        {themeMode === 'light' ? <LightModeIcon /> : <DarkModeIcon />}
      </IconButton>
    </Box>
  )
}

export default ThemeButton
