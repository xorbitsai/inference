import { Box, Typography } from '@mui/material'
import React, { useContext, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { ApiContext } from '../../components/apiContext'
import { buildGrafanaUrl } from '../../components/grafanaUtils'
import { useThemeContext } from '../../components/themeContext'

const Monitoring = () => {
  const { endPoint } = useContext(ApiContext)
  const { themeMode } = useThemeContext()
  const [config, setConfig] = useState(null)
  const { t } = useTranslation()

  useEffect(() => {
    fetch(endPoint + '/v1/cluster/ui_config')
      .then((res) => res.json())
      .then((data) => setConfig(data))
      .catch(() => setConfig(null))
  }, [endPoint])

  if (!config || !config.grafana_url) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        height="calc(100vh - 64px)"
      >
        <Typography variant="h6" color="text.secondary">
          {t('monitoring.notConfigured')}
        </Typography>
      </Box>
    )
  }

  const src = buildGrafanaUrl(config, themeMode)

  return (
    <Box sx={{ width: '100%', height: 'calc(100vh - 64px)' }}>
      <iframe
        src={src}
        width="100%"
        height="100%"
        frameBorder="0"
        title="Xinference Monitoring"
        style={{ border: 'none' }}
      />
    </Box>
  )
}

export default Monitoring
