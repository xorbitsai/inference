import { Box, FormControl, InputLabel, MenuItem, Select } from '@mui/material'
import React, { useContext, useEffect, useState } from 'react'
import { useCookies } from 'react-cookie'

import { ApiContext } from '../../components/apiContext'
import fetcher from '../../components/fetcher'
import HotkeyFocusTextField from '../../components/hotkeyFocusTextField'
import ModelCard from './modelCard'

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
  const [status, setStatus] = useState('all')
  const [completeDeleteArr, setCompleteDeleteArr] = useState([])

  const filter = (registration) => {
    if (searchTerm !== '') {
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
    }

    if (
      modelAbility !== 'all' &&
      registration.model_ability.indexOf(modelAbility) < 0
    )
      return false

    if (completeDeleteArr.includes(registration.model_name)) {
      registration.model_specs.forEach((item) => {
        item.cache_status = Array.isArray(item) ? [false] : false
      })
    }
    if (status !== 'all') {
      const judge = registration.model_specs.some((spec) => filterCache(spec))
      return judge && !completeDeleteArr.includes(registration.model_name)
    }

    return true
  }

  const filterCache = (spec) => {
    if (spec.model_format === 'pytorch') {
      return spec.cache_status && spec.cache_status === true
    } else {
      return spec.cache_status && spec.cache_status.some((cs) => cs)
    }
  }

  const handleCompleteDelete = (model_name) => {
    setCompleteDeleteArr([...completeDeleteArr, model_name])
  }

  const update = () => {
    if (
      isCallingApi ||
      isUpdatingModel ||
      (cookie.token !== 'no_auth' && !sessionStorage.getItem('token'))
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
          gridTemplateColumns: '150px 150px 1fr',
          columnGap: '20px',
          margin: '30px 2rem',
        }}
      >
        <FormControl variant="outlined" margin="normal">
          <InputLabel id="ability-select-label">Model Ability</InputLabel>
          <Select
            id="ability"
            labelId="ability-select-label"
            label="Model Ability"
            onChange={(e) => setModelAbility(e.target.value)}
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
          <InputLabel id="select-status">Status</InputLabel>
          <Select
            id="status"
            labelId="select-status"
            label="Status"
            onChange={(e) => setStatus(e.target.value)}
            value={status}
            size="small"
            sx={{ width: '150px' }}
          >
            <MenuItem value="all">all</MenuItem>
            <MenuItem value="cached">cached</MenuItem>
          </Select>
        </FormControl>
        <FormControl variant="outlined" margin="normal">
          <HotkeyFocusTextField
            id="search"
            type="search"
            label="Search for model name and description"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            size="small"
            hotkey="/"
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
              gpuAvailable={gpuAvailable}
              modelType={'LLM'}
              onHandleCompleteDelete={handleCompleteDelete}
            />
          ))}
      </div>
    </Box>
  )
}

export default LaunchLLM
