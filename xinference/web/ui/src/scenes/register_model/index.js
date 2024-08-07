import { TabContext, TabList, TabPanel } from '@mui/lab'
import { Box, Tab } from '@mui/material'
import React, { useEffect } from 'react'
import { useCookies } from 'react-cookie'
import { useNavigate } from 'react-router-dom'

import ErrorMessageSnackBar from '../../components/errorMessageSnackBar'
import Title from '../../components/Title'
import RegisterModelComponent from './registerModel'

const RegisterModel = () => {
  const [tabValue, setTabValue] = React.useState(
    sessionStorage.getItem('registerModelType')
      ? sessionStorage.getItem('registerModelType')
      : '/register_model/llm'
  )
  const [cookie] = useCookies(['token'])
  const navigate = useNavigate()

  useEffect(() => {
    if (!sessionStorage.getItem('auth') &&
        (sessionStorage.getItem('token') !== 'no_auth' && cookie.token !== 'no_auth')
    ){
        navigate('/login', { replace: true })
    }
  }, [cookie.token])

  const handleTabChange = (_, newValue) => {
    setTabValue(newValue)
    navigate(newValue)
    sessionStorage.setItem('registerModelType', newValue)
  }

  return (
    <Box m="20px" style={{ overflow: 'hidden' }}>
      <Title title="Register Model" />
      <ErrorMessageSnackBar />
      <TabContext value={tabValue}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <TabList
            value={tabValue}
            onChange={handleTabChange}
            aria-label="tabs"
          >
            <Tab label="Language Model" value="/register_model/llm" />
            <Tab label="Embedding Model" value="/register_model/embedding" />
            <Tab label="Rerank Model" value="/register_model/rerank" />
            <Tab label="Image Model" value="/register_model/image" />
            <Tab label="Audio Model" value="/register_model/audio" />
            <Tab label="Flexible Model" value="/register_model/flexible" />
          </TabList>
        </Box>
        <TabPanel value="/register_model/llm" sx={{ padding: 0 }}>
          <RegisterModelComponent
            modelType="LLM"
            customData={{
              version: 1,
              model_name: 'custom-llm',
              model_description: 'This is a custom model description.',
              context_length: 2048,
              model_lang: ['en'],
              model_ability: ['generate'],
              model_family: '',
              model_specs: [
                {
                  model_uri: '/path/to/llama-1',
                  model_size_in_billions: 7,
                  model_format: 'pytorch',
                  quantizations: ['none'],
                },
              ],
              prompt_style: undefined,
            }}
          />
        </TabPanel>
        <TabPanel value="/register_model/embedding" sx={{ padding: 0 }}>
          <RegisterModelComponent
            modelType="embedding"
            customData={{
              model_name: 'custom-embedding',
              dimensions: 768,
              max_tokens: 512,
              model_uri: '/path/to/embedding-model',
              language: ['en'],
            }}
          />
        </TabPanel>
        <TabPanel value="/register_model/rerank" sx={{ padding: 0 }}>
          <RegisterModelComponent
            modelType="rerank"
            customData={{
              model_name: 'custom-rerank',
              model_uri: '/path/to/rerank-model',
              language: ['en'],
            }}
          />
        </TabPanel>
        <TabPanel value="/register_model/image" sx={{ padding: 0 }}>
          <RegisterModelComponent
            modelType="image"
            customData={{
              model_name: 'custom-image',
              model_uri: '/path/to/image-model',
              model_family: 'stable_diffusion',
              controlnet: [],
            }}
          />
        </TabPanel>
        <TabPanel value="/register_model/audio" sx={{ padding: 0 }}>
          <RegisterModelComponent
            modelType="audio"
            customData={{
              model_name: 'custom-audio',
              model_uri: '/path/to/audio-model',
              multilingual: false,
              model_family: 'whisper',
            }}
          />
        </TabPanel>
        <TabPanel value="/register_model/flexible" sx={{ padding: 0 }}>
          <RegisterModelComponent
            modelType="flexible"
            customData={{
              model_name: 'flexible-model',
              model_description: 'This is a model description.',
              model_uri: '/path/to/model',
              launcher: 'xinference.model.flexible.launchers.transformers',
              launcher_args: '{}',
            }}
          />
        </TabPanel>
      </TabContext>
    </Box>
  )
}

export default RegisterModel
