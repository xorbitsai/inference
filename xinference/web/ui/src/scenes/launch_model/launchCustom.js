import { Box, FormControl, TextField } from '@mui/material'
import React, { useContext, useEffect, useState } from 'react'

import { ApiContext } from '../../components/apiContext'
import fetcher from '../../components/fetcher'
import EmbeddingCard from './cards/embeddingCard'
import ModelCard from './cards/modelCard'
import RerankCard from './cards/rerankCard'
import PanelStyle from './panelStyles'

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

  return (
    <Box style={PanelStyle.boxStyle}>
      <div style={PanelStyle.boxDivStyle}>
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
      <div style={PanelStyle.cardsGridStyle}>
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
            } else if (
              filteredRegistration.model_type &&
              filteredRegistration.model_type === 'rerank'
            ) {
              return (
                <RerankCard
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
