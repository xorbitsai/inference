import { TabContext, TabList, TabPanel } from '@mui/lab'
import {
  Box,
  Tab,
} from '@mui/material'
import React, { useEffect } from 'react'
import { useCookies } from 'react-cookie'
import { useNavigate } from 'react-router-dom'

import ErrorMessageSnackBar from '../../components/errorMessageSnackBar'
import Title from '../../components/Title'
import RegisterEmbeddingModel from './register_embedding'
import RegisterLanguageModel from './register_language'
import RegisterRerankModel from './register_rerank'


const RegisterModel = () => {
  const [tabValue, setTabValue] = React.useState('1')
  const [cookie] = useCookies(['token'])
  const navigate = useNavigate()

  useEffect(() => {
    if (cookie.token === '' || cookie.token === undefined) {
      return
    }
    if (cookie.token === 'need_auth') {
      navigate('/login', { replace: true })
    }

  }, [cookie.token])

  return (
    <Box m="20px">
      <Title title="Register Model" />
      <ErrorMessageSnackBar />
      <TabContext value={tabValue}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <TabList
            value={tabValue}
            onChange={(e, v) => {
              setTabValue(v)
            }}
            aria-label="tabs"
          >
            <Tab label="Language Model" value="1" />
            <Tab label="Embedding Model" value="2" />
            <Tab label="Rerank Model" value="3" />
          </TabList>
        </Box>
        <TabPanel value="1" sx={{ padding: 0 }}>
          <RegisterLanguageModel />
        </TabPanel>
        <TabPanel value="2" sx={{ padding: 0 }}>
          <RegisterEmbeddingModel />
        </TabPanel>
        <TabPanel value="3" sx={{ padding: 0 }}>
          <RegisterRerankModel />
        </TabPanel>
      </TabContext>
    </Box>
  )
}

export default RegisterModel
