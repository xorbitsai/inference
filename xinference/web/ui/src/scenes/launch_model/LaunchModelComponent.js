import {
  Box,
  Chip,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
} from '@mui/material'
import React, { useContext, useEffect, useState } from 'react'

import { ApiContext } from '../../components/apiContext'
import fetchWrapper from '../../components/fetchWrapper'
import HotkeyFocusTextField from '../../components/hotkeyFocusTextField'
import ModelCard from './modelCard'

const LaunchModelComponent = ({ modelType, gpuAvailable }) => {
  let endPoint = useContext(ApiContext).endPoint
  const [registrationData, setRegistrationData] = useState([])
  const [searchTerm, setSearchTerm] = useState('')
  const [status, setStatus] = useState('')
  const [completeDeleteArr, setCompleteDeleteArr] = useState([])
  const [collectionArr, setCollectionArr] = useState([])
  const [filterArr, setFilterArr] = useState([])

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

    if (filterArr.length === 1) {
      if (filterArr[0] === 'cached') {
        return (
          registration.cache_status &&
          !completeDeleteArr.includes(registration.model_name)
        )
      } else {
        return collectionArr?.includes(registration.model_name)
      }
    } else if (filterArr.length > 1) {
      return (
        registration.cache_status &&
        !completeDeleteArr.includes(registration.model_name) &&
        collectionArr?.includes(registration.model_name)
      )
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

      fetchWrapper
        .get(`/v1/model_registrations/${modelType}?detailed=true`)
        .then((data) => {
          const builtinModels = data.filter((v) => {
            return v.is_builtin
          })
          setRegistrationData(builtinModels)
          const collectionData = JSON.parse(
            localStorage.getItem('collectionArr')
          )
          setCollectionArr(collectionData)
        })
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
  }

  const handleChangeFilter = (value) => {
    setStatus(value)
    const arr = [
      ...filterArr.filter((item) => {
        return item !== value
      }),
      value,
    ]
    setFilterArr(arr)
  }

  const handleDeleteChip = (item) => {
    setFilterArr(
      filterArr.filter((subItem) => {
        return subItem !== item
      })
    )

    if (item === status) setStatus('')
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
        <FormControl sx={{ marginTop: 2, minWidth: 120 }} size="small">
          <InputLabel id="select-status">Status</InputLabel>
          <Select
            id="status"
            labelId="select-status"
            label="Status"
            onChange={(e) => handleChangeFilter(e.target.value)}
            value={status}
            size="small"
            sx={{ width: '150px' }}
          >
            <MenuItem value="cached">cached</MenuItem>
            <MenuItem value="favorite">favorite</MenuItem>
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
      <div style={{ margin: '0 0 30px 30px' }}>
        {filterArr.map((item, index) => (
          <Chip
            key={index}
            label={item}
            variant="outlined"
            size="small"
            color="primary"
            style={{ marginRight: 10 }}
            onDelete={() => handleDeleteChip(item)}
          />
        ))}
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
