import { Box } from '@mui/material'
import React from 'react'
import { Outlet } from 'react-router-dom'

import { useThemeSwitch } from '../../components/apiContext';
import MenuSide from '../../components/MenuSide'
import ThemeSwitch from '../../components/themeSwitch';

const Layout = () => {
  const { toggleMode, mode } = useThemeSwitch();

  return (
    <Box display="flex" width="100%" height="100%">
      <MenuSide />
      <Box flexGrow={1}>
        <Outlet />
      </Box>
      <Box position='absolute' right='20px' top='20px'>
        <ThemeSwitch onChange={toggleMode} checked={mode === 'dark'} />
      </Box>
    </Box>
  )
}

export default Layout