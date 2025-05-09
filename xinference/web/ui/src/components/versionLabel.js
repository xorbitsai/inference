import { Box, Typography } from '@mui/material'
import React, { useEffect, useState } from 'react'

import fetchWrapper from '../components/fetchWrapper'

const VersionLabel = ({ sx }) => {
  // 版本号状态变量，方便后续通过网络请求更新
  const [version, setVersion] = useState('')

  // 组件加载后获取版本信息
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
