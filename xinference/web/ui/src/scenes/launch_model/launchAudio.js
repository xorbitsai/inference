import { Box, FormControl, TextField } from '@mui/material'
import React, { useContext, useEffect, useState } from 'react'

import { ApiContext } from '../../components/apiContext'
import fetcher from '../../components/fetcher'
import AudioCard from './audioCard'

const LaunchAudio = () => {
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

      const response = await fetcher(
        `${endPoint}/v1/model_registrations/audio?detailed=true`,
        {
          method: 'GET',
        }
      )

      const registrations = await response.json()

      const builtinAudioModels = registrations.filter((v) => {
        return v.is_builtin
      })
      setRegistrationData(builtinAudioModels)
    } catch (error) {
      console.error('Error:', error)
    } finally {
      setIsCallingApi(false)
    }
  }

  useEffect(() => {
    update()
    // eslint-disable-next-line
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
            label="Search for audio model name"
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
            <AudioCard url={endPoint} modelData={filteredRegistration} />
          ))}
      </div>
    </Box>
  )
}

export default LaunchAudio
