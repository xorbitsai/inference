import { Box } from '@mui/material'
import React from 'react'
import { Outlet } from 'react-router-dom'

import MenuSide from '../../components/MenuSide'

const Layout = () => {
  return (
    <Box display="flex" width="100%" height="100%">
      <MenuSide />
      <Box flexGrow={1}>
        <Outlet />
      </Box>
    </Box>
  )
}

export default Layout
