import { Box } from '@mui/material'
import Paper from '@mui/material/Paper'
import Grid from '@mui/material/Unstable_Grid2'
import React, { useContext, useEffect } from 'react'
import { useCookies } from 'react-cookie'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'

import { ApiContext } from '../../components/apiContext'
import TableTitle from '../../components/tableTitle'
import Title from '../../components/Title'
import { isValidBearerToken } from '../../components/utils'
import NodeInfo from './nodeInfo'

const ClusterInfo = () => {
  const endPoint = useContext(ApiContext).endPoint
  const [cookie] = useCookies(['token'])
  const navigate = useNavigate()
  const { t } = useTranslation()

  useEffect(() => {
    if (
      sessionStorage.getItem('auth') === 'true' &&
      !isValidBearerToken(sessionStorage.getItem('token')) &&
      !isValidBearerToken(cookie.token)
    ) {
      navigate('/login', { replace: true })
    }
  }, [cookie.token])

  const handleGoBack = () => {
    const lastUrl = sessionStorage.getItem('lastActiveUrl')
    if (lastUrl === 'launch_model') {
      navigate(sessionStorage.getItem('modelType'))
    } else if (lastUrl === 'running_models') {
      navigate(sessionStorage.getItem('runningModelType'))
    } else if (lastUrl === 'register_model') {
      navigate(sessionStorage.getItem('registerModelType'))
    } else {
      navigate('/launch_model/llm')
    }
  }

  return (
    <Box
      sx={{
        height: '100%',
        width: '100%',
        padding: '20px 20px 0 20px',
      }}
    >
      <Title title={t('menu.clusterInfo')} />
      <Grid container spacing={3} style={{ width: '100%' }}>
        <Grid item xs={12}>
          <Paper
            sx={{
              padding: 2,
              display: 'flex',
              overflow: 'auto',
              flexDirection: 'column',
            }}
          >
            <TableTitle>{t('clusterInfo.supervisor')}</TableTitle>
            <NodeInfo
              nodeRole="Supervisor"
              endpoint={endPoint}
              cookie={cookie}
              handleGoBack={handleGoBack}
              t={t}
            />
          </Paper>
        </Grid>
        <Grid item xs={12}>
          <Paper
            sx={{
              padding: 2,
              display: 'flex',
              overflow: 'auto',
              flexDirection: 'column',
            }}
          >
            <TableTitle>{t('clusterInfo.workers')}</TableTitle>
            <NodeInfo
              nodeRole="Worker"
              endpoint={endPoint}
              cookie={cookie}
              handleGoBack={handleGoBack}
              t={t}
            />
          </Paper>
        </Grid>
        <Grid item xs={12}>
          <Paper
            sx={{
              padding: 2,
              display: 'flex',
              overflow: 'auto',
              flexDirection: 'column',
            }}
          >
            <TableTitle>{t('clusterInfo.workerDetails')}</TableTitle>
            <NodeInfo
              nodeRole="Worker-Details"
              endpoint={endPoint}
              cookie={cookie}
              handleGoBack={handleGoBack}
              t={t}
            />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  )
}

export default ClusterInfo
