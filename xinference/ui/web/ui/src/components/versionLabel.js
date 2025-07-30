import { Box, Typography } from '@mui/material'
import React, { useEffect, useState } from 'react'

import fetchWrapper from '../components/fetchWrapper'

const VersionLabel = ({ sx }) => {
  const [version, setVersion] = useState('')

  useEffect(() => {
    fetchWrapper
      .get('/v1/cluster/version')
      .then((data) => {
        setVersion('v' + data['version'])
      })
      .catch((error) => {
        console.error('Error:', error)
      })
  }, [])

  return (
    <Box sx={sx}>
      <Typography variant="h5">{version}</Typography>
    </Box>
  )
}

export default VersionLabel
