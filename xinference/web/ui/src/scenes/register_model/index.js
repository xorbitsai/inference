import { TabContext, TabList, TabPanel } from '@mui/lab'
import { Box, Tab } from '@mui/material'
import React, { useEffect } from 'react'
import { useCookies } from 'react-cookie'
import { useNavigate } from 'react-router-dom'

import ErrorMessageSnackBar from '../../components/errorMessageSnackBar'
import Title from '../../components/Title'
import RegisterEmbeddingModel from './register_embedding'
import RegisterLLM from './register_llm'
import RegisterRerankModel from './register_rerank'

const RegisterModel = () => {
  const [tabValue, setTabValue] = React.useState('/register_model/llm')
  const [cookie] = useCookies(['token'])
  const navigate = useNavigate()

  useEffect(() => {
    if (cookie.token === '' || cookie.token === undefined) {
      return
    }
    if (cookie.token !== 'no_auth' && !sessionStorage.getItem('token')) {
      navigate('/login', { replace: true })
      return
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
            <Tab label="Language Model" value="/register_model/llm" />
            <Tab label="Embedding Model" value="/register_model/embedding" />
            <Tab label="Rerank Model" value="/register_model/rerank" />
          </TabList>
        </Box>
        <TabPanel value="/register_model/llm" sx={{ padding: 0 }}>
          <RegisterLLM  />
        </TabPanel>
        <TabPanel value="/register_model/embedding" sx={{ padding: 0 }}>
          <RegisterEmbeddingModel />
        </TabPanel>
        <TabPanel value="/register_model/rerank" sx={{ padding: 0 }}>
          <RegisterRerankModel />
        </TabPanel>
      </TabContext>
    </Box>
  )
}

export default RegisterModel
