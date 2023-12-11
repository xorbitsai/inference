import { Box, FormControl, TextField } from '@mui/material'
import React, { useContext, useEffect, useState } from 'react'

import { ApiContext } from '../../components/apiContext'
import EmbeddingCard from './embeddingCard'
import ModelCard from './modelCard'

const LaunchCustom = ({ gpuAvailable }) => {
  let endPoint = useContext(ApiContext).endPoint
  const [registrationData, setRegistrationData] = useState([])
  const { isCallingApi, setIsCallingApi } = useContext(ApiContext)
  const { isUpdatingModel } = useContext(ApiContext)

  // States used for filtering
  const [searchTerm, setSearchTerm] = useState('')

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

      const embeddingResponse = await fetch(
        `${endPoint}/v1/model_registrations/embedding`,
        {
          method: 'GET',
        }
      )

      const embeddingRegistrations = await embeddingResponse.json()
      const customEmbeddingRegistrations = embeddingRegistrations.filter(
        (data) => !data.is_builtin
      )

      const llmResponse = await fetch(
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
          const desc = await fetch(
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
          const desc = await fetch(
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

      setRegistrationData(newLLMData.concat(newEmbeddingData))
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
    gridGap: '2rem 0rem',
  }

  return (
    <Box m="20px">
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
                <EmbeddingCard
                  url={endPoint}
                  modelData={filteredRegistration}
                  cardHeight={380}
                  is_custom={true}
                />
              )
            } else {
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
    </Box>
  )
}

export default LaunchCustom
