import { TabContext, TabList, TabPanel } from '@mui/lab'
import { Box, FormControl, Tab, TextField } from '@mui/material'
import React, { useContext, useEffect, useState } from 'react'

import { ApiContext } from '../../components/apiContext'
import fetcher from '../../components/fetcher'
import ModelCard from './modelCard'

const LaunchCustom = ({ gpuAvailable }) => {
  let endPoint = useContext(ApiContext).endPoint
  const [registrationData, setRegistrationData] = useState([])
  const { isCallingApi, setIsCallingApi } = useContext(ApiContext)
  const { isUpdatingModel } = useContext(ApiContext)

  // States used for filtering
  const [searchTerm, setSearchTerm] = useState('')
  const [value, setValue] = useState('1')

  const handleTabChange = (event, newValue) => {
    setValue(newValue)
    update()
  }

  const handleChange = (event) => {
    setSearchTerm(event.target.value)
  }

  const filter = (registration) => {
    if (!registration || typeof searchTerm !== 'string') return false
    const modelName = registration.model_name
      ? registration.model_name.toLowerCase()
      : ''
    return modelName.includes(searchTerm.toLowerCase())
  }

  const update = async () => {
    if (isCallingApi || isUpdatingModel) return

    try {
      setIsCallingApi(true)

      const rerankResponse = await fetcher(
        `${endPoint}/v1/model_registrations/rerank`,
        {
          method: 'GET',
        }
      )
      const rerankRegistrations = await rerankResponse.json()
      const customRerankRegistrations = rerankRegistrations.filter(
        (data) => !data.is_builtin
      )

      const embeddingResponse = await fetcher(
        `${endPoint}/v1/model_registrations/embedding`,
        {
          method: 'GET',
        }
      )

      const embeddingRegistrations = await embeddingResponse.json()
      const customEmbeddingRegistrations = embeddingRegistrations.filter(
        (data) => !data.is_builtin
      )

      const llmResponse = await fetcher(
        `${endPoint}/v1/model_registrations/LLM`,
        {
          method: 'GET',
        }
      )
      const llmRegistrations = await llmResponse.json()
      const customLLMRegistrations = llmRegistrations.filter(
        (data) => !data.is_builtin
      )

      const newEmbeddingData = await Promise.all(
        customEmbeddingRegistrations.map(async (registration) => {
          const desc = await fetcher(
            `${endPoint}/v1/model_registrations/embedding/${registration.model_name}`,
            {
              method: 'GET',
            }
          )

          return {
            ...(await desc.json()),
            is_builtin: registration.is_builtin,
          }
        })
      )

      const newLLMData = await Promise.all(
        customLLMRegistrations.map(async (registration) => {
          const desc = await fetcher(
            `${endPoint}/v1/model_registrations/LLM/${registration.model_name}`,
            {
              method: 'GET',
            }
          )

          return {
            ...(await desc.json()),
            is_builtin: registration.is_builtin,
          }
        })
      )

      const newRerankData = await Promise.all(
        customRerankRegistrations.map(async (registration) => {
          const desc = await fetcher(
            `${endPoint}/v1/model_registrations/rerank/${registration.model_name}`,
            {
              method: 'GET',
            }
          )

          return {
            ...(await desc.json()),
            is_builtin: registration.is_builtin,
          }
        })
      )

      setRegistrationData(
        newLLMData.concat(newEmbeddingData).concat(newRerankData)
      )
    } catch (error) {
      console.error('Error:', error)
    } finally {
      setIsCallingApi(false)
    }
  }

  useEffect(() => {
    update()
  }, [])

  const style = {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
    paddingLeft: '2rem',
    paddingTop: '2rem',
    gridGap: '2rem 0rem',
  }

  return (
    <Box m="20px">
      <TabContext value={value}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <TabList value={value} onChange={handleTabChange} aria-label="tabs">
            <Tab label="Language Models" value="1" />
            <Tab label="Embedding Models" value="2" />
            <Tab label="Rerank Models" value="3" />
          </TabList>
        </Box>
        <TabPanel value="1" sx={{ padding: 0 }}>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr',
              margin: '30px 2rem',
            }}
          >
            <FormControl variant="outlined" margin="normal">
              <TextField
                id="search"
                type="search"
                label="Search for custom model name"
                value={searchTerm}
                onChange={handleChange}
                size="small"
              />
            </FormControl>
          </div>
          <div style={style}>
            {registrationData
              .filter((registration) => filter(registration))
              .map((filteredRegistration) => {
                if (
                  !(
                    filteredRegistration.max_tokens &&
                    filteredRegistration.dimensions
                  ) &&
                  !(
                    filteredRegistration.model_type &&
                    filteredRegistration.model_type === 'rerank'
                  )
                ) {
                  return (
                    <ModelCard
                      url={endPoint}
                      modelData={filteredRegistration}
                      gpuAvailable={gpuAvailable}
                      is_custom={true}
                    />
                  )
                }
              })}
          </div>
        </TabPanel>
        <TabPanel value="2" sx={{ padding: 0 }}>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr',
              margin: '30px 2rem',
            }}
          >
            <FormControl variant="outlined" margin="normal">
              <TextField
                id="search"
                type="search"
                label="Search for custom model name"
                value={searchTerm}
                onChange={handleChange}
                size="small"
              />
            </FormControl>
          </div>
          <div style={style}>
            {registrationData
              .filter((registration) => filter(registration))
              .map((filteredRegistration) => {
                if (
                  filteredRegistration.max_tokens &&
                  filteredRegistration.dimensions
                ) {
                  return (
                    <ModelCard
                      url={endPoint}
                      modelData={filteredRegistration}
                      is_custom={true}
                    />
                  )
                }
              })}
          </div>
        </TabPanel>
        <TabPanel value="3" sx={{ padding: 0 }}>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr',
              margin: '30px 2rem',
            }}
          >
            <FormControl variant="outlined" margin="normal">
              <TextField
                id="search"
                type="search"
                label="Search for custom model name"
                value={searchTerm}
                onChange={handleChange}
                size="small"
              />
            </FormControl>
          </div>
          <div style={style}>
            {registrationData
              .filter((registration) => filter(registration))
              .map((filteredRegistration) => {
                if (
                  filteredRegistration.model_type &&
                  filteredRegistration.model_type === 'rerank'
                ) {
                  return (
                    <ModelCard
                      url={endPoint}
                      modelData={filteredRegistration}
                      is_custom={true}
                    />
                  )
                }
              })}
          </div>
        </TabPanel>
      </TabContext>
    </Box>
  )
}

export default LaunchCustom
