import { Box, FormControl, InputLabel, MenuItem, Select } from '@mui/material'
import React, { useContext, useEffect, useState } from 'react'

import { ApiContext } from '../../components/apiContext'
import fetcher from '../../components/fetcher'
import HotkeyFocusTextField from '../../components/hotkeyFocusTextField'
import ModelCard from './modelCard'

const LaunchModelComponent = ({ modelType, gpuAvailable }) => {
  let endPoint = useContext(ApiContext).endPoint
  const [registrationData, setRegistrationData] = useState([])
  const [listData, setListData] = useState([])
  const [searchTerm, setSearchTerm] = useState('')
  const [status, setStatus] = useState('all')
  const [completeDeleteArr, setCompleteDeleteArr] = useState([])
  const [collectionArr, setCollectionArr] = useState([])

  const { isCallingApi, setIsCallingApi } = useContext(ApiContext)
  const { isUpdatingModel } = useContext(ApiContext)

  const filter = (registration) => {
    if (searchTerm !== '') {
      if (!registration || typeof searchTerm !== 'string') return false
      const modelName = registration.model_name
        ? registration.model_name.toLowerCase()
        : ''
      if (!modelName.includes(searchTerm.toLowerCase())) {
        return false
      }
    }

    if (completeDeleteArr.includes(registration.model_name)) {
      registration.cache_status = Array.isArray(registration.cache_status)
        ? [false]
        : false
    }

    if (status === 'cached') {
      return (
        registration.cache_status &&
        !completeDeleteArr.includes(registration.model_name)
      )
    } else if(status === 'collection') {
      return collectionArr.includes(registration.model_name)
    }

    return true
  }

  const handleCompleteDelete = (model_name) => {
    setCompleteDeleteArr([...completeDeleteArr, model_name])
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
      setListData(builtinModels)

      const collectionData = JSON.parse(localStorage.getItem('collectionArr'))
      setCollectionArr(collectionData)
      if(collectionData?.length) {
        const collection = builtinModels.filter(item => {
          return collectionData.includes(item.model_name)
        })
        const notCollection = builtinModels.filter(item => {
          return !collectionData.includes(item.model_name)
        })
        setRegistrationData([...collection, ...notCollection])
      } else {
        setRegistrationData(builtinModels)
      }
    } catch (error) {
      console.error('Error:', error)
    } finally {
      setIsCallingApi(false)
    }
  }

  useEffect(() => {
    update()
  }, [])

  const getCollectionArr = (data) => {
    setCollectionArr(data)
    const collection = listData.filter(item => {
      return data.includes(item.model_name)
    })
    const notCollection = listData.filter(item => {
      return !data.includes(item.model_name)
    })
    setRegistrationData([...collection, ...notCollection])
  }

  return (
    <Box m="20px">
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '150px 1fr',
          columnGap: '20px',
          margin: '30px 2rem',
        }}
      >
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
            <MenuItem value="collection">collection</MenuItem>
          </Select>
        </FormControl>
        <FormControl variant="outlined" margin="normal">
          <HotkeyFocusTextField
            id="search"
            type="search"
            label={`Search for ${modelType} model name`}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            size="small"
            hotkey="/"
          />
        </FormControl>
      </div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
          paddingLeft: '2rem',
          gridGap: '2rem 0rem',
        }}
      >
        {registrationData
          .filter((registration) => filter(registration))
          .map((filteredRegistration) => (
            <ModelCard
              key={filteredRegistration.model_name}
              url={endPoint}
              modelData={filteredRegistration}
              modelType={modelType}
              gpuAvailable={gpuAvailable}
              onHandleCompleteDelete={handleCompleteDelete}
              onGetCollectionArr={getCollectionArr}
            />
          ))}
      </div>
    </Box>
  )
}

export default LaunchModelComponent
