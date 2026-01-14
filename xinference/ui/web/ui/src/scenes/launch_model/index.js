import { LoadingButton, TabContext, TabList, TabPanel } from '@mui/lab'
import { Box, MenuItem, Select, Tab } from '@mui/material'
import React, { useContext, useEffect, useRef, useState } from 'react'
import { useCookies } from 'react-cookie'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'

import { ApiContext } from '../../components/apiContext'
import ErrorMessageSnackBar from '../../components/errorMessageSnackBar'
import fetchWrapper from '../../components/fetchWrapper'
import SuccessMessageSnackBar from '../../components/successMessageSnackBar'
import Title from '../../components/Title'
import { isValidBearerToken } from '../../components/utils'
import LaunchCustom from './launchCustom'
import LaunchModelComponent from './LaunchModel'

const LaunchModel = () => {
  const [value, setValue] = React.useState(
    sessionStorage.getItem('modelType')
      ? sessionStorage.getItem('modelType')
      : '/launch_model/llm'
  )
  const [gpuAvailable, setGPUAvailable] = useState(-1)
  const [modelType, setModelType] = useState('llm')
  const [loading, setLoading] = useState(false)

  const { setErrorMsg } = useContext(ApiContext)
  const [cookie] = useCookies(['token'])
  const navigate = useNavigate()
  const { t } = useTranslation()
  const LaunchModelRefs = useRef({})

  const handleTabChange = (newValue) => {
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

  const updateList = (modelType) => {
    LaunchModelRefs.current[modelType]?.update()
  }

  const updateModels = () => {
    setLoading(true)
    fetchWrapper
      .post('/v1/models/update_type', { model_type: modelType })
      .then(() => {
        handleTabChange(`/launch_model/${modelType}`)
        updateList(modelType)
      })
      .catch((error) => {
        console.error('Error:', error)
        if (error.response.status !== 403 && error.response.status !== 401) {
          setErrorMsg(error.message)
        }
      })
      .finally(() => {
        setLoading(false)
      })
  }

  return (
    <Box m="20px">
      <Title title={t('menu.launchModel')} />
      <ErrorMessageSnackBar />
      <SuccessMessageSnackBar />
      <TabContext value={value}>
        <Box
          sx={{
            borderBottom: 1,
            borderColor: 'divider',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <TabList
            value={value}
            onChange={(_, value) => handleTabChange(value)}
            aria-label="tabs"
          >
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
          <Box sx={{ display: 'flex', gap: 0 }}>
            <Select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              size="small"
              sx={{
                borderTopRightRadius: 0,
                borderBottomRightRadius: 0,
                minWidth: 100,
              }}
            >
              <MenuItem value="llm">LLM</MenuItem>
              <MenuItem value="embedding">Embedding</MenuItem>
              <MenuItem value="rerank">Rerank</MenuItem>
              <MenuItem value="image">Image</MenuItem>
              <MenuItem value="audio">Audio</MenuItem>
              <MenuItem value="video">Video</MenuItem>
            </Select>

            <LoadingButton
              variant="contained"
              onClick={updateModels}
              loading={loading}
              sx={{
                borderTopLeftRadius: 0,
                borderBottomLeftRadius: 0,
                whiteSpace: 'nowrap',
              }}
            >
              {t('launchModel.update')}
            </LoadingButton>
          </Box>
        </Box>
        <TabPanel value="/launch_model/llm" sx={{ padding: 0 }}>
          <LaunchModelComponent
            modelType={'LLM'}
            gpuAvailable={gpuAvailable}
            ref={(ref) => (LaunchModelRefs.current.llm = ref)}
          />
        </TabPanel>
        <TabPanel value="/launch_model/embedding" sx={{ padding: 0 }}>
          <LaunchModelComponent
            modelType={'embedding'}
            gpuAvailable={gpuAvailable}
            ref={(ref) => (LaunchModelRefs.current.embedding = ref)}
          />
        </TabPanel>
        <TabPanel value="/launch_model/rerank" sx={{ padding: 0 }}>
          <LaunchModelComponent
            modelType={'rerank'}
            gpuAvailable={gpuAvailable}
            ref={(ref) => (LaunchModelRefs.current.rerank = ref)}
          />
        </TabPanel>
        <TabPanel value="/launch_model/image" sx={{ padding: 0 }}>
          <LaunchModelComponent
            modelType={'image'}
            gpuAvailable={gpuAvailable}
            ref={(ref) => (LaunchModelRefs.current.image = ref)}
          />
        </TabPanel>
        <TabPanel value="/launch_model/audio" sx={{ padding: 0 }}>
          <LaunchModelComponent
            modelType={'audio'}
            gpuAvailable={gpuAvailable}
            ref={(ref) => (LaunchModelRefs.current.audio = ref)}
          />
        </TabPanel>
        <TabPanel value="/launch_model/video" sx={{ padding: 0 }}>
          <LaunchModelComponent
            modelType={'video'}
            gpuAvailable={gpuAvailable}
            ref={(ref) => (LaunchModelRefs.current.video = ref)}
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
