import { TabContext, TabList, TabPanel } from '@mui/lab'
import { Box, Tab } from '@mui/material'
import React, { useContext, useEffect, useState } from 'react'
import { useCookies } from 'react-cookie'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'

import { ApiContext } from '../../components/apiContext'
import ErrorMessageSnackBar from '../../components/errorMessageSnackBar'
import fetchWrapper from '../../components/fetchWrapper'
import SuccessMessageSnackBar from '../../components/successMessageSnackBar'
import Title from '../../components/Title'
import { isValidBearerToken } from '../../components/utils'
import { featureModels } from './data/data'
import LaunchCustom from './launchCustom'
import LaunchModelComponent from './LaunchModel'

const LaunchModel = () => {
  const [value, setValue] = React.useState(
    sessionStorage.getItem('modelType')
      ? sessionStorage.getItem('modelType')
      : '/launch_model/llm'
  )
  const [gpuAvailable, setGPUAvailable] = useState(-1)

  const { setErrorMsg } = useContext(ApiContext)
  const [cookie] = useCookies(['token'])
  const navigate = useNavigate()
  const { t } = useTranslation()

  const handleTabChange = (event, newValue) => {
    setValue(newValue)
    navigate(newValue)
    sessionStorage.setItem('modelType', newValue)
    newValue === '/launch_model/custom/llm'
      ? sessionStorage.setItem('subType', newValue)
      : ''
  }

  useEffect(() => {
    if (
      sessionStorage.getItem('auth') === 'true' &&
      !isValidBearerToken(sessionStorage.getItem('token')) &&
      !isValidBearerToken(cookie.token)
    ) {
      navigate('/login', { replace: true })
    }

    if (gpuAvailable === -1) {
      fetchWrapper
        .get('/v1/cluster/devices')
        .then((data) => setGPUAvailable(parseInt(data, 10)))
        .catch((error) => {
          console.error('Error:', error)
          if (error.response.status !== 403 && error.response.status !== 401) {
            setErrorMsg(error.message)
          }
        })
    }
  }, [cookie.token])

  return (
    <Box m="20px">
      <Title title={t('menu.launchModel')} />
      <ErrorMessageSnackBar />
      <SuccessMessageSnackBar />
      <TabContext value={value}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <TabList value={value} onChange={handleTabChange} aria-label="tabs">
            <Tab label={t('model.languageModels')} value="/launch_model/llm" />
            <Tab
              label={t('model.embeddingModels')}
              value="/launch_model/embedding"
            />
            <Tab label={t('model.rerankModels')} value="/launch_model/rerank" />
            <Tab label={t('model.imageModels')} value="/launch_model/image" />
            <Tab label={t('model.audioModels')} value="/launch_model/audio" />
            <Tab label={t('model.videoModels')} value="/launch_model/video" />
            <Tab
              label={t('model.customModels')}
              value="/launch_model/custom/llm"
            />
          </TabList>
        </Box>
        <TabPanel value="/launch_model/llm" sx={{ padding: 0 }}>
          <LaunchModelComponent
            modelType={'LLM'}
            gpuAvailable={gpuAvailable}
            featureModels={
              featureModels.find((item) => item.type === 'llm').feature_models
            }
          />
        </TabPanel>
        <TabPanel value="/launch_model/embedding" sx={{ padding: 0 }}>
          <LaunchModelComponent
            modelType={'embedding'}
            gpuAvailable={gpuAvailable}
            featureModels={
              featureModels.find((item) => item.type === 'embedding')
                .feature_models
            }
          />
        </TabPanel>
        <TabPanel value="/launch_model/rerank" sx={{ padding: 0 }}>
          <LaunchModelComponent
            modelType={'rerank'}
            gpuAvailable={gpuAvailable}
            featureModels={
              featureModels.find((item) => item.type === 'rerank')
                .feature_models
            }
          />
        </TabPanel>
        <TabPanel value="/launch_model/image" sx={{ padding: 0 }}>
          <LaunchModelComponent
            modelType={'image'}
            featureModels={
              featureModels.find((item) => item.type === 'image').feature_models
            }
          />
        </TabPanel>
        <TabPanel value="/launch_model/audio" sx={{ padding: 0 }}>
          <LaunchModelComponent
            modelType={'audio'}
            featureModels={
              featureModels.find((item) => item.type === 'audio').feature_models
            }
          />
        </TabPanel>
        <TabPanel value="/launch_model/video" sx={{ padding: 0 }}>
          <LaunchModelComponent
            modelType={'video'}
            featureModels={
              featureModels.find((item) => item.type === 'video').feature_models
            }
          />
        </TabPanel>
        <TabPanel value="/launch_model/custom/llm" sx={{ padding: 0 }}>
          <LaunchCustom gpuAvailable={gpuAvailable} />
        </TabPanel>
      </TabContext>
    </Box>
  )
}

export default LaunchModel
