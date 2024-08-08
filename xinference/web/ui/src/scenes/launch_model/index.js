import { TabContext, TabList, TabPanel } from '@mui/lab'
import { Box, Tab } from '@mui/material'
import React, { useContext, useEffect, useState } from 'react'
import { useCookies } from 'react-cookie'
import { useNavigate } from 'react-router-dom'

import { ApiContext } from '../../components/apiContext'
import ErrorMessageSnackBar from '../../components/errorMessageSnackBar'
import fetchWrapper from '../../components/fetchWrapper'
import Title from '../../components/Title'
import LaunchCustom from './launchCustom'
import LaunchLLM from './launchLLM'
import LaunchModelComponent from './LaunchModelComponent'

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
      !sessionStorage.getItem('auth') &&
      sessionStorage.getItem('token') !== 'no_auth' &&
      cookie.token !== 'no_auth'
    ) {
      navigate('/login', { replace: true })
      return
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

  useEffect(() => {})
  return (
    <Box m="20px">
      <Title title="Launch Model" />
      <ErrorMessageSnackBar />
      <TabContext value={value}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <TabList value={value} onChange={handleTabChange} aria-label="tabs">
            <Tab label="Language Models" value="/launch_model/llm" />
            <Tab label="Embedding Models" value="/launch_model/embedding" />
            <Tab label="Rerank Models" value="/launch_model/rerank" />
            <Tab label="Image Models" value="/launch_model/image" />
            <Tab label="Audio Models" value="/launch_model/audio" />
            <Tab label="Custom Models" value="/launch_model/custom/llm" />
          </TabList>
        </Box>
        <TabPanel value="/launch_model/llm" sx={{ padding: 0 }}>
          <LaunchLLM gpuAvailable={gpuAvailable} />
        </TabPanel>
        <TabPanel value="/launch_model/embedding" sx={{ padding: 0 }}>
          <LaunchModelComponent
            modelType={'embedding'}
            gpuAvailable={gpuAvailable}
          />
        </TabPanel>
        <TabPanel value="/launch_model/rerank" sx={{ padding: 0 }}>
          <LaunchModelComponent
            modelType={'rerank'}
            gpuAvailable={gpuAvailable}
          />
        </TabPanel>
        <TabPanel value="/launch_model/image" sx={{ padding: 0 }}>
          <LaunchModelComponent modelType={'image'} />
        </TabPanel>
        <TabPanel value="/launch_model/audio" sx={{ padding: 0 }}>
          <LaunchModelComponent modelType={'audio'} />
        </TabPanel>
        <TabPanel value="/launch_model/custom/llm" sx={{ padding: 0 }}>
          <LaunchCustom gpuAvailable={gpuAvailable} />
        </TabPanel>
      </TabContext>
    </Box>
  )
}

export default LaunchModel
