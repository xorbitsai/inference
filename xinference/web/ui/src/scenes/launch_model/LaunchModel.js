import {
  Box,
  Button,
  ButtonGroup,
  Chip,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
} from '@mui/material'
import React, { useContext, useEffect, useState } from 'react'
import { useCookies } from 'react-cookie'
import { useTranslation } from 'react-i18next'

import { ApiContext } from '../../components/apiContext'
import fetchWrapper from '../../components/fetchWrapper'
import HotkeyFocusTextField from '../../components/hotkeyFocusTextField'
import ModelCard from './modelCard'

const modelAbilityArr = ['generate', 'chat', 'vision', 'reasoning']
const audioModelTypeArr = ['audio2text', 'text2audio', 'audio2audio']

const LaunchModelComponent = ({ modelType, gpuAvailable, featureModels }) => {
  const { isCallingApi, setIsCallingApi, endPoint } = useContext(ApiContext)
  const { isUpdatingModel } = useContext(ApiContext)
  const { setErrorMsg } = useContext(ApiContext)
  const [cookie] = useCookies(['token'])

  const [registrationData, setRegistrationData] = useState([])
  // States used for filtering
  const [searchTerm, setSearchTerm] = useState('')
  const [LLMModelAbility, setLLMModelAbility] = useState('')
  const [status, setStatus] = useState('')
  const [statusArr, setStatusArr] = useState([])
  const [completeDeleteArr, setCompleteDeleteArr] = useState([])
  const [collectionArr, setCollectionArr] = useState([])
  const [filterArr, setFilterArr] = useState([])
  const { t } = useTranslation()
  const [modelListType, setModelListType] = useState('featured')
  const [audioModelAbility, setAudioModelAbility] = useState('')

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

    if (modelListType === 'featured') {
      if (
        featureModels.length &&
        !featureModels.includes(registration.model_name)
      ) {
        return false
      }
    }

    if (
      LLMModelAbility &&
      ((Array.isArray(registration.model_ability) &&
        registration.model_ability.indexOf(LLMModelAbility) < 0) ||
        (typeof registration.model_ability === 'string' &&
          registration.model_ability !== LLMModelAbility))
    )
      return false

    if (
      audioModelAbility &&
      ((Array.isArray(registration.model_ability) &&
        registration.model_ability.indexOf(audioModelAbility) < 0) ||
        (typeof registration.model_ability === 'string' &&
          registration.model_ability !== audioModelAbility))
    )
      return false

    if (completeDeleteArr.includes(registration.model_name)) {
      registration.model_specs.forEach((item) => {
        item.cache_status = Array.isArray(item) ? [false] : false
      })
    }

    if (statusArr.length === 1) {
      if (statusArr[0] === 'cached') {
        const judge =
          registration.model_specs?.some((spec) => filterCache(spec)) ||
          registration?.cache_status
        return judge && !completeDeleteArr.includes(registration.model_name)
      } else {
        return collectionArr?.includes(registration.model_name)
      }
    } else if (statusArr.length > 1) {
      const judge =
        registration.model_specs?.some((spec) => filterCache(spec)) ||
        registration?.cache_status
      return (
        judge &&
        !completeDeleteArr.includes(registration.model_name) &&
        collectionArr?.includes(registration.model_name)
      )
    }

    return true
  }

  const filterCache = (spec) => {
    if (Array.isArray(spec.cache_status)) {
      return spec.cache_status?.some((cs) => cs)
    } else {
      return spec.cache_status === true
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

      fetchWrapper
        .get(`/v1/model_registrations/${modelType}?detailed=true`)
        .then((data) => {
          const builtinRegistrations = data.filter((v) => v.is_builtin)
          setRegistrationData(builtinRegistrations)
          const collectionData = JSON.parse(
            localStorage.getItem('collectionArr')
          )
          setCollectionArr(collectionData)
        })
        .catch((error) => {
          console.error('Error:', error)
          if (error.response.status !== 403 && error.response.status !== 401) {
            setErrorMsg(error.message)
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

  const getCollectionArr = (data) => {
    setCollectionArr(data)
  }

  const handleChangeFilter = (type, value) => {
    const typeMap = {
      LLMModelAbility: {
        setter: setLLMModelAbility,
        filterArr: modelAbilityArr,
      },
      audioModelAbility: {
        setter: setAudioModelAbility,
        filterArr: audioModelTypeArr,
      },
      status: { setter: setStatus, filterArr: [] },
    }

    const { setter, filterArr: excludeArr } = typeMap[type] || {}
    if (!setter) return

    setter(value)

    const updatedFilterArr = [
      ...filterArr.filter((item) => !excludeArr.includes(item)),
      value,
    ]

    setFilterArr(updatedFilterArr)

    if (type === 'status') {
      setStatusArr(
        updatedFilterArr.filter(
          (item) => ![...modelAbilityArr, ...audioModelTypeArr].includes(item)
        )
      )
    }
  }

  const handleDeleteChip = (item) => {
    setFilterArr(
      filterArr.filter((subItem) => {
        return subItem !== item
      })
    )
    if (item === LLMModelAbility) {
      setLLMModelAbility('')
    } else if (item === audioModelAbility) {
      setAudioModelAbility('')
    } else {
      setStatusArr(
        statusArr.filter((subItem) => {
          return subItem !== item
        })
      )
      if (item === status) setStatus('')
    }
  }

  const handleModelType = (newModelType) => {
    if (newModelType !== null) {
      setModelListType(newModelType)
    }
  }

  return (
    <Box m="20px">
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: (() => {
            const baseColumns = ['LLM', 'audio'].includes(modelType)
              ? ['150px', '150px']
              : ['150px']
            return featureModels.length
              ? [...baseColumns, '150px', '1fr'].join(' ')
              : [...baseColumns, '1fr'].join(' ')
          })(),
          columnGap: '20px',
          margin: '30px 2rem',
          alignItems: 'center',
        }}
      >
        {featureModels.length > 0 && (
          <FormControl sx={{ minWidth: 120 }} size="small">
            <ButtonGroup>
              <Button
                fullWidth
                onClick={() => handleModelType('featured')}
                variant={
                  modelListType === 'featured' ? 'contained' : 'outlined'
                }
              >
                {t('launchModel.featured')}
              </Button>
              <Button
                fullWidth
                onClick={() => handleModelType('all')}
                variant={modelListType === 'all' ? 'contained' : 'outlined'}
              >
                {t('launchModel.all')}
              </Button>
            </ButtonGroup>
          </FormControl>
        )}
        {modelType === 'LLM' && (
          <FormControl sx={{ minWidth: 120 }} size="small">
            <InputLabel id="ability-select-label">
              {t('launchModel.modelAbility')}
            </InputLabel>
            <Select
              id="ability"
              labelId="ability-select-label"
              label="Model Ability"
              onChange={(e) =>
                handleChangeFilter('LLMModelAbility', e.target.value)
              }
              value={LLMModelAbility}
              size="small"
              sx={{ width: '150px' }}
            >
              {modelAbilityArr.map((item) => (
                <MenuItem key={item} value={item}>
                  {t(`launchModel.${item}`)}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        )}
        {modelType === 'audio' && (
          <FormControl sx={{ minWidth: 120 }} size="small">
            <InputLabel id="ability-select-label">
              {t('launchModel.modelAbility')}
            </InputLabel>
            <Select
              id="ability"
              labelId="ability-select-label"
              label="Model Ability"
              onChange={(e) =>
                handleChangeFilter('audioModelAbility', e.target.value)
              }
              value={audioModelAbility}
              size="small"
              sx={{ width: '150px' }}
            >
              {audioModelTypeArr.map((item) => (
                <MenuItem key={item} value={item}>
                  {item}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        )}
        <FormControl sx={{ minWidth: 120 }} size="small">
          <InputLabel id="select-status">{t('launchModel.status')}</InputLabel>
          <Select
            id="status"
            labelId="select-status"
            label={t('launchModel.status')}
            onChange={(e) => handleChangeFilter('status', e.target.value)}
            value={status}
            size="small"
            sx={{ width: '150px' }}
          >
            <MenuItem value="cached">{t('launchModel.cached')}</MenuItem>
            <MenuItem value="favorite">{t('launchModel.favorite')}</MenuItem>
          </Select>
        </FormControl>

        <FormControl sx={{ marginTop: 1 }} variant="outlined" margin="normal">
          <HotkeyFocusTextField
            id="search"
            type="search"
            label={t('launchModel.search')}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            size="small"
            hotkey="Enter"
            t={t}
          />
        </FormControl>
      </div>
      <div style={{ margin: '0 0 30px 30px' }}>
        {filterArr.map((item, index) => (
          <Chip
            key={index}
            label={t(`launchModel.${item}`)}
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
          .sort((a, b) => {
            if (modelListType === 'featured') {
              const indexA = featureModels.indexOf(a.model_name)
              const indexB = featureModels.indexOf(b.model_name)
              return (
                (indexA !== -1 ? indexA : Infinity) -
                (indexB !== -1 ? indexB : Infinity)
              )
            }
            return 0
          })
          .map((filteredRegistration) => (
            <ModelCard
              key={filteredRegistration.model_name}
              url={endPoint}
              modelData={filteredRegistration}
              gpuAvailable={gpuAvailable}
              modelType={modelType}
              onHandleCompleteDelete={handleCompleteDelete}
              onGetCollectionArr={getCollectionArr}
            />
          ))}
      </div>
    </Box>
  )
}

export default LaunchModelComponent
