import {
  Box,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  TextField,
} from '@mui/material'
import React, { useContext, useEffect, useState } from 'react'
import { useCookies } from 'react-cookie'

import { ApiContext } from '../../components/apiContext'
import fetcher from '../../components/fetcher'
import ModelCard from './cards/modelCard'
import PanelStyle from './panelStyles'

const LaunchLLM = ({ gpuAvailable }) => {
  let endPoint = useContext(ApiContext).endPoint
  const { isCallingApi, setIsCallingApi } = useContext(ApiContext)
  const { isUpdatingModel } = useContext(ApiContext)
  const { setErrorMsg } = useContext(ApiContext)
  const [cookie] = useCookies(['token'])

  const [registrationData, setRegistrationData] = useState([])
  // States used for filtering
  const [searchTerm, setSearchTerm] = useState('')
  const [modelAbility, setModelAbility] = useState('all')

  const handleChange = (event) => {
    setSearchTerm(event.target.value)
  }

  const handleAbilityChange = (event) => {
    setModelAbility(event.target.value)
  }

  const filter = (registration) => {
    if (!registration || typeof searchTerm !== 'string') return false
    const modelName = registration.model_name
      ? registration.model_name.toLowerCase()
      : ''
    const modelDescription = registration.model_description
      ? registration.model_description.toLowerCase()
      : ''

    if (
      !modelName.includes(searchTerm.toLowerCase()) &&
      !modelDescription.includes(searchTerm.toLowerCase())
    ) {
      return false
    }
    if (modelAbility && modelAbility !== 'all') {
      if (registration.model_ability.indexOf(modelAbility) < 0) {
        return false
      }
    }
    return true
  }

  const update = () => {
    if (
      isCallingApi ||
      isUpdatingModel ||
      cookie.token === '' ||
      cookie.token === undefined ||
      cookie.token === 'need_auth'
    )
      return

    try {
      setIsCallingApi(true)

      fetcher(`${endPoint}/v1/model_registrations/LLM?detailed=true`, {
        method: 'GET',
      }).then((response) => {
        if (!response.ok) {
          response
            .json()
            .then((errData) =>
              setErrorMsg(
                `Server error: ${response.status} - ${
                  errData.detail || 'Unknown error'
                }`
              )
            )
        } else {
          response.json().then((data) => {
            const builtinRegistrations = data.filter((v) => v.is_builtin)
            setRegistrationData(builtinRegistrations)
          })
        }
      })
    } catch (error) {
      console.error('Error:', error)
    } finally {
      setIsCallingApi(false)
    }
  }

  useEffect(() => {
    update()
  }, [cookie.token])

  return (
    <Box style={PanelStyle.boxStyle}>
      <div style={PanelStyle.boxDivStyle}>
        <FormControl variant="outlined" margin="normal">
          <InputLabel id="ability-select-label">Model Ability</InputLabel>
          <Select
            id="ability"
            labelId="ability-select-label"
            label="Model Ability"
            onChange={handleAbilityChange}
            value={modelAbility}
            size="small"
            sx={{ width: '150px' }}
          >
            <MenuItem value="all">all</MenuItem>
            <MenuItem value="generate">generate</MenuItem>
            <MenuItem value="chat">chat</MenuItem>
            <MenuItem value="vision">vl-chat</MenuItem>
          </Select>
        </FormControl>
        <FormControl variant="outlined" margin="normal">
          <TextField
            id="search"
            type="search"
            label="Search for model name and description"
            value={searchTerm}
            onChange={handleChange}
            size="small"
          />
        </FormControl>
      </div>
      <div style={PanelStyle.cardsGridStyle}>
        {registrationData
          .filter((registration) => filter(registration))
          .map((filteredRegistration) => (
            <ModelCard
              url={endPoint}
              modelData={filteredRegistration}
              gpuAvailable={gpuAvailable}
            />
          ))}
      </div>
    </Box>
  )
}

export default LaunchLLM
