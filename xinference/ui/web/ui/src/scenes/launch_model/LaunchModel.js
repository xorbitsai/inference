import {
  Box,
  Button,
  ButtonGroup,
  Chip,
  CircularProgress,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
} from '@mui/material'
import React, {
  forwardRef,
  useCallback,
  useContext,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from 'react'
import { useCookies } from 'react-cookie'
import { useTranslation } from 'react-i18next'

import { ApiContext } from '../../components/apiContext'
import fetchWrapper from '../../components/fetchWrapper'
import HotkeyFocusTextField from '../../components/hotkeyFocusTextField'
import LaunchModelDrawer from './components/launchModelDrawer'
import ModelCard from './modelCard'

// Toggle pagination globally for this page. Set to false to disable pagination and load all items.
const ENABLE_PAGINATION = false

const LaunchModelComponent = forwardRef(({ modelType, gpuAvailable }, ref) => {
  const { isCallingApi, setIsCallingApi, endPoint } = useContext(ApiContext)
  const { isUpdatingModel } = useContext(ApiContext)
  const { setErrorMsg } = useContext(ApiContext)
  const [cookie] = useCookies(['token'])

  const [registrationData, setRegistrationData] = useState([])
  // States used for filtering
  const [searchTerm, setSearchTerm] = useState('')
  const [status, setStatus] = useState('')
  const [statusArr, setStatusArr] = useState([])
  const [collectionArr, setCollectionArr] = useState([])
  const [filterArr, setFilterArr] = useState([])
  const { t } = useTranslation()
  const [modelListType, setModelListType] = useState('featured')
  const [modelAbilityData, setModelAbilityData] = useState({
    type: modelType,
    modelAbility: '',
    options: [],
  })
  const [selectedModel, setSelectedModel] = useState(null)
  const [isOpenLaunchModelDrawer, setIsOpenLaunchModelDrawer] = useState(false)

  // Pagination status
  const [displayedData, setDisplayedData] = useState([])
  // Virtual environments data
  const [virtualEnvs, setVirtualEnvs] = useState([])
  const [currentPage, setCurrentPage] = useState(1)
  const [hasMore, setHasMore] = useState(true)
  const itemsPerPage = 20
  const loaderRef = useRef(null)

  const filter = useCallback(
    (registration) => {
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
        const isFavorite =
          Array.isArray(collectionArr) &&
          collectionArr.includes(registration.model_name)
        if (registration?.featured !== true && !isFavorite) {
          return false
        }
      }

      if (
        modelAbilityData.modelAbility &&
        ((Array.isArray(registration.model_ability) &&
          registration.model_ability.indexOf(modelAbilityData.modelAbility) <
            0) ||
          (typeof registration.model_ability === 'string' &&
            registration.model_ability !== modelAbilityData.modelAbility))
      )
        return false

      if (statusArr.length === 1) {
        if (statusArr[0] === 'cached') {
          const judge =
            registration.model_specs?.some((spec) => filterCache(spec)) ||
            registration?.cache_status
          return judge
        } else {
          return collectionArr?.includes(registration.model_name)
        }
      } else if (statusArr.length > 1) {
        const judge =
          registration.model_specs?.some((spec) => filterCache(spec)) ||
          registration?.cache_status
        return judge && collectionArr?.includes(registration.model_name)
      }

      return true
    },
    [
      searchTerm,
      modelListType,
      collectionArr,
      modelAbilityData.modelAbility,
      statusArr,
    ]
  )

  const filterCache = useCallback((spec) => {
    if (Array.isArray(spec.cache_status)) {
      return spec.cache_status?.some((cs) => cs)
    } else {
      return spec.cache_status === true
    }
  }, [])

  function getUniqueModelAbilities(arr) {
    const uniqueAbilities = new Set()

    arr.forEach((item) => {
      if (Array.isArray(item.model_ability)) {
        item.model_ability.forEach((ability) => {
          uniqueAbilities.add(ability)
        })
      }
    })

    return Array.from(uniqueAbilities)
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

      // Fetch both model registrations and virtual environments in parallel
      Promise.all([
        fetchWrapper.get(`/v1/model_registrations/${modelType}?detailed=true`),
        fetchWrapper.get('/v1/virtualenvs').catch(() => ({ list: [] })), // Fallback for virtual env API
      ])
        .then(([modelData, virtualEnvData]) => {
          const builtinRegistrations = modelData.filter((v) => v.is_builtin)
          setModelAbilityData({
            ...modelAbilityData,
            options: getUniqueModelAbilities(builtinRegistrations),
          })
          setRegistrationData(builtinRegistrations)
          setVirtualEnvs(virtualEnvData.list || [])
          const collectionData = JSON.parse(
            localStorage.getItem('collectionArr')
          )
          setCollectionArr(collectionData)

          // If no featured models in backend or favorites, default to 'all'
          const hasFeaturedOrFavorite = builtinRegistrations.some(
            (r) =>
              r.featured === true ||
              (Array.isArray(collectionData) &&
                collectionData.includes(r.model_name))
          )
          if (!hasFeaturedOrFavorite && modelListType === 'featured') {
            setModelListType('all')
          }

          // Reset pagination status
          setCurrentPage(1)
          setHasMore(true)
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

  // Update pagination data
  const updateDisplayedData = useCallback(() => {
    const filteredData = registrationData.filter((registration) =>
      filter(registration)
    )

    const sortedData = [...filteredData].sort((a, b) => {
      if (modelListType === 'featured') {
        const isFavA =
          Array.isArray(collectionArr) && collectionArr.includes(a.model_name)
        const isFavB =
          Array.isArray(collectionArr) && collectionArr.includes(b.model_name)
        const rankA = a.featured === true ? 0 : isFavA ? 1 : 2
        const rankB = b.featured === true ? 0 : isFavB ? 1 : 2
        return rankA - rankB
      }
      return 0
    })

    // If pagination is disabled, show all data at once
    if (!ENABLE_PAGINATION) {
      setDisplayedData(sortedData)
      setHasMore(false)
      return
    }

    const startIndex = (currentPage - 1) * itemsPerPage
    const endIndex = currentPage * itemsPerPage
    const newData = sortedData.slice(startIndex, endIndex)

    if (currentPage === 1) {
      setDisplayedData(newData)
    } else {
      setDisplayedData((prev) => [...prev, ...newData])
    }
    setHasMore(endIndex < sortedData.length)
  }, [registrationData, filter, modelListType, currentPage, itemsPerPage])

  useEffect(() => {
    updateDisplayedData()
  }, [updateDisplayedData])

  // Reset pagination when filters change
  useEffect(() => {
    setCurrentPage(1)
    setHasMore(true)
  }, [searchTerm, modelAbilityData.modelAbility, status, modelListType])

  // Infinite scroll observer
  useEffect(() => {
    if (!ENABLE_PAGINATION) return

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore && !isCallingApi) {
          setCurrentPage((prev) => prev + 1)
        }
      },
      { threshold: 1.0 }
    )

    if (loaderRef.current) {
      observer.observe(loaderRef.current)
    }

    return () => {
      if (loaderRef.current) {
        observer.unobserve(loaderRef.current)
      }
    }
  }, [hasMore, isCallingApi, currentPage])

  const getCollectionArr = (data) => {
    setCollectionArr(data)
  }

  const handleChangeFilter = (type, value) => {
    const typeMap = {
      modelAbility: {
        setter: (value) => {
          setModelAbilityData({
            ...modelAbilityData,
            modelAbility: value,
          })
        },
        filterArr: modelAbilityData.options,
      },
      status: { setter: setStatus, filterArr: [] },
    }

    const { setter, filterArr: excludeArr } = typeMap[type] || {}
    if (!setter) return

    setter(value)

    const updatedFilterArr = Array.from(
      new Set([
        ...filterArr.filter((item) => !excludeArr.includes(item)),
        value,
      ])
    )

    setFilterArr(updatedFilterArr)

    if (type === 'status') {
      setStatusArr(
        updatedFilterArr.filter(
          (item) => ![...modelAbilityData.options].includes(item)
        )
      )
    }

    // Reset pagination status
    setDisplayedData([])
    setCurrentPage(1)
    setHasMore(true)
  }

  const handleDeleteChip = (item) => {
    setFilterArr(
      filterArr.filter((subItem) => {
        return subItem !== item
      })
    )
    if (item === modelAbilityData.modelAbility) {
      setModelAbilityData({
        ...modelAbilityData,
        modelAbility: '',
      })
    } else {
      setStatusArr(
        statusArr.filter((subItem) => {
          return subItem !== item
        })
      )
      if (item === status) setStatus('')
    }

    // Reset pagination status
    setCurrentPage(1)
    setHasMore(true)
  }

  const handleModelType = (newModelType) => {
    if (newModelType !== null) {
      setModelListType(newModelType)

      // Reset pagination status
      setDisplayedData([])
      setCurrentPage(1)
      setHasMore(true)
    }
  }

  function getLabel(item) {
    const translation = t(`launchModel.${item}`)
    return translation === `launchModel.${item}` ? item : translation
  }

  useImperativeHandle(ref, () => ({
    update,
  }))

  const hasFeatured = registrationData?.some(
    (r) =>
      r.featured === true ||
      (Array.isArray(collectionArr) && collectionArr.includes(r.model_name))
  )

  useEffect(() => {
    if (modelListType === 'featured' && !hasFeatured) {
      setModelListType('all')
    }
  }, [modelListType, hasFeatured])

  useEffect(() => {
    setModelListType('featured')
  }, [modelType])

  return (
    <Box m="20px">
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: (() => {
            const hasAbility = modelAbilityData.options.length > 0
            const baseColumns = hasAbility ? ['200px', '150px'] : ['200px']
            const altColumns = hasAbility ? ['150px', '150px'] : ['150px']
            const columns = hasFeatured
              ? [...baseColumns, '150px', '1fr']
              : [...altColumns, '1fr']
            return columns.join(' ')
          })(),
          columnGap: '20px',
          margin: '30px 2rem',
          alignItems: 'center',
        }}
      >
        {hasFeatured && (
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
        {modelAbilityData.options.length > 0 && (
          <FormControl sx={{ minWidth: 120 }} size="small">
            <InputLabel id="ability-select-label">
              {t('launchModel.modelAbility')}
            </InputLabel>
            <Select
              id="ability"
              labelId="ability-select-label"
              label="Model Ability"
              onChange={(e) =>
                handleChangeFilter('modelAbility', e.target.value)
              }
              value={modelAbilityData.modelAbility}
              size="small"
              sx={{ width: '150px' }}
            >
              {modelAbilityData.options.map((item) => (
                <MenuItem key={item} value={item}>
                  {getLabel(item)}
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
            onChange={(e) => {
              setSearchTerm(e.target.value)
            }}
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
            label={getLabel(item)}
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
        {displayedData.map((filteredRegistration) => (
          <ModelCard
            key={filteredRegistration.model_name}
            url={endPoint}
            modelData={filteredRegistration}
            gpuAvailable={gpuAvailable}
            modelType={modelType}
            onGetCollectionArr={getCollectionArr}
            onUpdate={update}
            virtualEnvs={virtualEnvs}
            onClick={() => {
              setSelectedModel(filteredRegistration)
              setIsOpenLaunchModelDrawer(true)
            }}
          />
        ))}
      </div>

      <div ref={loaderRef} style={{ height: '20px', margin: '20px 0' }}>
        {ENABLE_PAGINATION && hasMore && !isCallingApi && (
          <div style={{ textAlign: 'center', padding: '10px' }}>
            <CircularProgress />
          </div>
        )}
      </div>

      {selectedModel && (
        <LaunchModelDrawer
          key={selectedModel.model_name}
          modelData={selectedModel}
          modelType={modelType}
          gpuAvailable={gpuAvailable}
          open={isOpenLaunchModelDrawer}
          onClose={() => setIsOpenLaunchModelDrawer(false)}
        />
      )}
    </Box>
  )
})

LaunchModelComponent.displayName = 'LaunchModelComponent'

export default LaunchModelComponent
