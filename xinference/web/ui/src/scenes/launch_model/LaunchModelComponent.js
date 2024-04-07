import { Box, FormControl, TextField } from '@mui/material'
import React, { useContext, useEffect, useState } from 'react'

import { ApiContext } from '../../components/apiContext'
import fetcher from '../../components/fetcher'
import ModelCard from './modelCard'
import style from './styles/launchModelStyle'

const LaunchModelComponent = ({modelType}) => {
  let endPoint = useContext(ApiContext).endPoint
  const [registrationData, setRegistrationData] = useState([])
  const [searchTerm, setSearchTerm] = useState('')

  const { isCallingApi, setIsCallingApi } = useContext(ApiContext)
  const { isUpdatingModel } = useContext(ApiContext)

  const handleChange = (e) => {
    setSearchTerm(e.target.value)
  }

  const filter = (registration) => {
    if (!registration || typeof searchTerm !== 'string') return false
    const modelName = registration.model_name ? registration.model_name.toLowerCase() : ''
    return modelName.includes(searchTerm.toLowerCase())
  }

  const update = async () => {
    if (isCallingApi || isUpdatingModel) return

    try {
      setIsCallingApi(true)

      const response = await fetcher(
        `${endPoint}/v1/model_registrations/${modelType}?detailed=true`,
        {
          method: 'GET',
        }
      )

      const registrations = await response.json()

      const builtinModels = registrations.filter((v) => {
        return v.is_builtin
      })
      setRegistrationData(builtinModels)
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
            label={`Search for ${modelType} model name`}
            value={searchTerm}
            onChange={handleChange}
            size="small"
          />
        </FormControl>
      </div>
      <div style={style}>
        {registrationData
          .filter((registration) => filter(registration))
          .map((filteredRegistration) => (
            <ModelCard
              key={filteredRegistration.model_name}
              url={endPoint}
              modelData={filteredRegistration}
              modelType={modelType}
            />
          ))}
      </div>
    </Box>
  )
}

export default LaunchModelComponent
