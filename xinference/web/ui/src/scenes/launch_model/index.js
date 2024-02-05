import { TabContext, TabList, TabPanel } from '@mui/lab'
import { Box, Tab } from '@mui/material'
import React, { useContext, useEffect, useState } from 'react'
import { useCookies } from 'react-cookie'
import { useNavigate } from 'react-router-dom'

import { ApiContext } from '../../components/apiContext'
import ErrorMessageSnackBar from '../../components/errorMessageSnackBar'
import Title from '../../components/Title'
import LaunchCustom from './launchCustom'
import LaunchEmbedding from './launchEmbedding'
import LaunchImage from './launchImage'
import LaunchLLM from './launchLLM'
import LaunchRerank from './launchRerank'

const LaunchModel = () => {
  let endPoint = useContext(ApiContext).endPoint
  const [value, setValue] = React.useState('1')
  const [gpuAvailable, setGPUAvailable] = useState(-1)

  const { setErrorMsg } = useContext(ApiContext)
  const [cookie] = useCookies(['token'])
  const navigate = useNavigate()

  const handleTabChange = (event, newValue) => {
    setValue(newValue)
  }

  useEffect(() => {
    if (cookie.token === '' || cookie.token === undefined) {
      return
    }
    if (cookie.token === 'need_auth') {
      navigate('/login', { replace: true })
      return
    }

    if (gpuAvailable === -1) {
      fetch(endPoint + '/v1/cluster/devices', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }).then((res) => {
        if (!res.ok) {
          // Usually, if some errors happen here, check if the cluster is available
          res.json().then((errorData) => {
            setErrorMsg(
              `Server error: ${res.status} - ${
                errorData.detail || 'Unknown error'
              }`
            )
          })
        } else {
          res.json().then((data) => {
            setGPUAvailable(parseInt(data, 10))
          })
        }
      })
    }
  }, [cookie.token])

  return (
    <Box m="20px">
      <Title title="Launch Model" />
      <ErrorMessageSnackBar />
      <TabContext value={value}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <TabList value={value} onChange={handleTabChange} aria-label="tabs">
            <Tab label="Language Models" value="1" />
            <Tab label="Embedding Models" value="2" />
            <Tab label="Rerank Models" value="3" />
            <Tab label="Custom Models" value="4" />
            <Tab label="Image Models" value="5" />
          </TabList>
        </Box>
        <TabPanel value="1" sx={{ padding: 0 }}>
          <LaunchLLM gpuAvailable={gpuAvailable} />
        </TabPanel>
        <TabPanel value="2" sx={{ padding: 0 }}>
          <LaunchEmbedding />
        </TabPanel>
        <TabPanel value="3" sx={{ padding: 0 }}>
          <LaunchRerank />
        </TabPanel>
        <TabPanel value="4" sx={{ padding: 0 }}>
          <LaunchCustom gpuAvailable={gpuAvailable} />
        </TabPanel>
        <TabPanel value="5" sx={{ paddding: 0 }}>
          <LaunchImage />
        </TabPanel>
      </TabContext>
    </Box>
  )
}

export default LaunchModel
