import {
  ContentPasteGo,
  ExpandLess,
  ExpandMore,
  RocketLaunchOutlined,
  StopCircle,
  UndoOutlined,
} from '@mui/icons-material'
import {
  Box,
  Button,
  Chip,
  CircularProgress,
  Collapse,
  Drawer,
  FormControl,
  FormControlLabel,
  InputLabel,
  ListItemButton,
  ListItemText,
  MenuItem,
  Radio,
  RadioGroup,
  Select,
  Switch,
  TextField,
  Tooltip,
} from '@mui/material'
import React, { useContext, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'

import { ApiContext } from '../../../components/apiContext'
import fetchWrapper from '../../../components/fetchWrapper'
import TitleTypography from '../../../components/titleTypography'
import { llmAllDataKey } from '../data/data'
import CopyToCommandLine from './commandBuilder'
import DynamicFieldList from './dynamicFieldList'
import getModelFormConfig from './modelFormConfig'
import PasteDialog from './pasteDialog'
import Progress from './progress'

const enginesWithNWorker = ['SGLang', 'vLLM', 'MLX']
const modelEngineType = ['LLM', 'embedding', 'rerank']

const SelectField = ({
  label,
  labelId,
  name,
  value,
  onChange,
  options = [],
  disabled = false,
  required = false,
}) => (
  <FormControl
    variant="outlined"
    margin="normal"
    disabled={disabled}
    required={required}
    fullWidth
  >
    <InputLabel id={labelId}>{label}</InputLabel>
    <Select
      labelId={labelId}
      name={name}
      value={value}
      onChange={onChange}
      label={label}
      className="textHighlight"
    >
      {options.map((item) => (
        <MenuItem key={item.value || item} value={item.value || item}>
          {item.label || item}
        </MenuItem>
      ))}
    </Select>
  </FormControl>
)

const LaunchModelDrawer = ({
  modelData,
  modelType,
  gpuAvailable,
  open,
  onClose,
}) => {
  const navigate = useNavigate()
  const { t } = useTranslation()
  const {
    isCallingApi,
    isUpdatingModel,
    setErrorMsg,
    setSuccessMsg,
    setIsCallingApi,
  } = useContext(ApiContext)
  const [formData, setFormData] = useState({})
  const [hasHistory, setHasHistory] = useState(false)

  const [enginesObj, setEnginesObj] = useState({})
  const [engineOptions, setEngineOptions] = useState([])
  const [formatOptions, setFormatOptions] = useState([])
  const [sizeOptions, setSizeOptions] = useState([])
  const [quantizationOptions, setQuantizationOptions] = useState([])
  const [multimodalProjectorOptions, setMultimodalProjectorOptions] = useState(
    []
  )
  const [multimodalProjector, setMultimodalProjector] = useState([])
  const [isOpenPasteDialog, setIsOpenPasteDialog] = useState(false)
  const [collapseState, setCollapseState] = useState({})
  const [isShowProgress, setIsShowProgress] = useState(false)
  const [isShowCancel, setIsShowCancel] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [checkDynamicFieldComplete, setCheckDynamicFieldComplete] = useState([])

  const intervalRef = useRef(null)

  const downloadHubOptions = useMemo(() => {
    let hubs = []

    if (
      Array.isArray(modelData?.model_specs) &&
      modelData.model_specs.length > 0
    ) {
      hubs = modelData.model_specs.map((spec) => spec.model_hub).filter(Boolean)
    } else if (modelData?.model_hub) {
      hubs = [modelData.model_hub]
    }

    return ['none', ...new Set(hubs)]
  }, [modelData])

  const isCached = (spec) => {
    if (Array.isArray(spec.cache_status)) {
      return spec.cache_status.some((cs) => cs)
    } else {
      return spec.cache_status === true
    }
  }

  const convertModelSize = (size) => {
    return size.toString().includes('_') ? size : parseInt(size, 10)
  }

  const range = (start, end) => {
    return new Array(end - start + 1).fill(undefined).map((_, i) => i + start)
  }

  const getNGPURange = (modelType) => {
    if (['LLM', 'image'].includes(modelType)) {
      return gpuAvailable > 0
        ? ['auto', 'CPU', ...range(1, gpuAvailable)]
        : ['auto', 'CPU']
    } else {
      return gpuAvailable === 0 ? ['CPU'] : ['GPU', 'CPU']
    }
  }

  const normalizeNGPU = (value) => {
    if (!value) return null

    if (value === 'CPU') return null
    if (value === 'auto' || value === 'GPU') return 'auto'

    const num = parseInt(value, 10)
    return isNaN(num) || num === 0 ? null : num
  }

  const handleValueType = (str) => {
    str = String(str)
    if (str.toLowerCase() === 'none') {
      return null
    } else if (str.toLowerCase() === 'true') {
      return true
    } else if (str.toLowerCase() === 'false') {
      return false
    } else if (str.includes(',')) {
      return str.split(',')
    } else if (str.includes('，')) {
      return str.split('，')
    } else if (Number(str) || (str !== '' && Number(str) === 0)) {
      return Number(str)
    } else {
      return str
    }
  }

  const arrayToObject = (arr, transformValue = (v) => v) =>
    Object.fromEntries(
      arr.map(({ key, value }) => [key, transformValue(value)])
    )

  const handleGetHistory = () => {
    const historyArr = JSON.parse(localStorage.getItem('historyArr')) || []
    return historyArr.find((item) => item.model_name === modelData.model_name)
  }

  const deleteHistory = () => {
    const arr = JSON.parse(localStorage.getItem('historyArr'))
    const newArr = arr.filter(
      (item) => item.model_name !== modelData.model_name
    )
    localStorage.setItem('historyArr', JSON.stringify(newArr))
    setHasHistory(false)
    setFormData({})
    setCollapseState({})
  }

  const objectToArray = (obj) => {
    if (!obj || typeof obj !== 'object') return []
    return Object.entries(obj).map(([key, value]) => ({ key, value }))
  }

  const restoreNGPU = (value) => {
    if (value === null) return 'CPU'
    if (value === 'auto') {
      return ['LLM', 'image'].includes(modelType) ? 'auto' : 'GPU'
    }
    if (typeof value === 'number') return String(value)
    return value || 'CPU'
  }

  const restoreFormDataFormat = (finalData) => {
    const result = { ...finalData }

    let customData = []
    for (let key in result) {
      !llmAllDataKey.includes(key) &&
        customData.push({
          key: key,
          value:
            result[key] === null
              ? 'none'
              : result[key] === false
              ? false
              : result[key],
        })
    }
    if (customData.length) result.custom = customData

    result.n_gpu = restoreNGPU(result.n_gpu)
    if (result?.gpu_idx && Array.isArray(result.gpu_idx)) {
      result.gpu_idx = result.gpu_idx.join(',')
    }

    if (result?.peft_model_config) {
      const pmc = result.peft_model_config

      if (pmc.image_lora_load_kwargs) {
        result.image_lora_load_kwargs = objectToArray(
          pmc.image_lora_load_kwargs
        )
      }

      if (pmc.image_lora_fuse_kwargs) {
        result.image_lora_fuse_kwargs = objectToArray(
          pmc.image_lora_fuse_kwargs
        )
      }

      if (pmc.lora_list) {
        result.lora_list = pmc.lora_list
      }

      delete result.peft_model_config
    }

    if (
      result?.envs &&
      typeof result.envs === 'object' &&
      !Array.isArray(result.envs)
    ) {
      result.envs = objectToArray(result.envs)
    }

    if (
      result?.quantization_config &&
      typeof result.quantization_config === 'object' &&
      !Array.isArray(result.quantization_config)
    ) {
      result.quantization_config = objectToArray(result.quantization_config)
    }

    return result
  }

  const getCollapseStateFromData = (result) => {
    const newState = {}

    if (
      result.model_uid ||
      result.request_limits ||
      result.worker_ip ||
      result.gpu_idx ||
      result.download_hub ||
      result.model_path
    ) {
      newState.nested_section_optional = true
    }

    if (
      result.image_lora_load_kwargs?.length ||
      result.image_lora_fuse_kwargs?.length ||
      result.lora_list?.length
    ) {
      newState.nested_section_optional = true
      newState.nested_section_lora = true
    }

    if (result.envs?.length) {
      newState.nested_section_optional = true
      newState.nested_section_env = true
    }

    if (
      (result.virtual_env_packages?.length ?? 0) > 0 ||
      result.enable_virtual_env !== undefined
    ) {
      newState.nested_section_optional = true
      newState.nested_section_virtualEnv = true
    }

    return newState
  }

  const handleCommandLine = (data) => {
    if (data.model_name === modelData.model_name) {
      const restoredData = restoreFormDataFormat(data)
      setFormData(restoredData)

      const collapseFromData = getCollapseStateFromData(restoredData)
      setCollapseState((prev) => ({ ...prev, ...collapseFromData }))
    } else {
      setErrorMsg(t('launchModel.commandLineTip'))
    }
  }

  const fetchModelEngine = (model_name, model_type) => {
    fetchWrapper
      .get(
        model_type === 'LLM'
          ? `/v1/engines/${model_name}`
          : `/v1/engines/${model_type}/${model_name}`
      )
      .then((data) => {
        setEnginesObj(data)
        setEngineOptions(Object.keys(data))
      })
      .catch((error) => {
        console.error('Error:', error)
        if (error.response.status !== 403) {
          setErrorMsg(error.message)
        }
      })
      .finally(() => {
        setIsCallingApi(false)
      })
  }

  const fetchCancelModel = async () => {
    try {
      await fetchWrapper.post(`/v1/models/${modelData.model_name}/cancel`)
      setIsLoading(true)
    } catch (error) {
      console.error('Error:', error)
      if (error.response.status !== 403) {
        setErrorMsg(error.message)
      }
    } finally {
      setProgress(0)
      stopPolling()
      setIsShowProgress(false)
      setIsShowCancel(false)
    }
  }

  const fetchProgress = async () => {
    try {
      const res = await fetchWrapper.get(
        `/v1/models/${modelData.model_name}/progress`
      )
      if (res.progress !== 1.0) setProgress(Number(res.progress))
    } catch (error) {
      console.error('Error:', error)
      if (error.response.status !== 403) {
        setErrorMsg(error.message)
      }
    } finally {
      stopPolling()
      setIsCallingApi(false)
    }
  }

  const startPolling = () => {
    if (intervalRef.current) return
    intervalRef.current = setInterval(fetchProgress, 500)
  }

  const stopPolling = () => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }

  useEffect(() => {
    const data = handleGetHistory()
    if (data) {
      setHasHistory(true)
      const restoredData = restoreFormDataFormat(data)
      setFormData(
        modelEngineType.includes(modelType)
          ? { ...restoredData, __isInitializing: true }
          : restoredData
      )
      const collapseFromData = getCollapseStateFromData(restoredData)
      setCollapseState((prev) => ({ ...prev, ...collapseFromData }))
    }
  }, [])

  useEffect(() => {
    if (modelEngineType.includes(modelType))
      fetchModelEngine(modelData.model_name, modelType)
  }, [modelData.model_name, modelType])

  useEffect(() => {
    if (formData.__isInitializing) {
      setFormData((prev) => {
        const { __isInitializing, ...rest } = prev
        console.log('__isInitializing', __isInitializing)
        return rest
      })
      return
    }

    if (formData.model_engine && modelEngineType.includes(modelType)) {
      const format = [
        ...new Set(
          enginesObj[formData.model_engine]?.map((item) => item.model_format)
        ),
      ]
      setFormatOptions(format)

      if (!format.includes(formData.model_format)) {
        setFormData((prev) => ({
          ...prev,
          model_format: '',
        }))
      }
      if (format.length === 1) {
        setFormData((prev) => ({
          ...prev,
          model_format: format[0],
        }))
      }
    }
  }, [formData.model_engine, enginesObj])

  useEffect(() => {
    if (formData.__isInitializing) return

    if (!formData.model_engine || !formData.model_format) return

    const configMap = {
      LLM: {
        field: 'model_size_in_billions',
        optionSetter: setSizeOptions,
        extractor: (item) => item.model_size_in_billions,
      },
      embedding: {
        field: 'quantization',
        optionSetter: setQuantizationOptions,
        extractor: (item) => item.quantization,
      },
      rerank: {
        field: 'quantization',
        optionSetter: setQuantizationOptions,
        extractor: (item) => item.quantization,
      },
    }

    const config = configMap[modelType]
    if (!config) return

    const options = [
      ...new Set(
        enginesObj[formData.model_engine]
          ?.filter((item) => item.model_format === formData.model_format)
          ?.map(config.extractor)
      ),
    ]

    config.optionSetter(options)
    if (!options.includes(formData[config.field])) {
      setFormData((prev) => ({ ...prev, [config.field]: '' }))
    }

    if (options.length === 1) {
      setFormData((prev) => ({ ...prev, [config.field]: options[0] }))
    }
  }, [formData.model_engine, formData.model_format, enginesObj])

  useEffect(() => {
    if (formData.__isInitializing) return
    if (
      formData.model_engine &&
      formData.model_format &&
      formData.model_size_in_billions
    ) {
      const quants = [
        ...new Set(
          enginesObj[formData.model_engine]
            ?.filter(
              (item) =>
                item.model_format === formData.model_format &&
                item.model_size_in_billions ===
                  convertModelSize(formData.model_size_in_billions)
            )
            .flatMap((item) => item.quantizations)
        ),
      ]
      const multimodal_projectors = [
        ...new Set(
          enginesObj[formData.model_engine]
            ?.filter(
              (item) =>
                item.model_format === formData.model_format &&
                item.model_size_in_billions ===
                  convertModelSize(formData.model_size_in_billions)
            )
            .flatMap((item) => item.multimodal_projectors || [])
        ),
      ]
      setQuantizationOptions(quants)
      setMultimodalProjectorOptions(multimodal_projectors || [])
      if (!quants.includes(formData.quantization)) {
        setFormData((prev) => ({
          ...prev,
          quantization: '',
        }))
      }
      if (quants.length === 1) {
        setFormData((prev) => ({
          ...prev,
          quantization: quants[0],
        }))
      }
      if (!multimodal_projectors.includes(multimodalProjector)) {
        setMultimodalProjector('')
      }
      if (multimodal_projectors.length > 0 && !multimodalProjector) {
        setMultimodalProjector(multimodal_projectors[0])
      }
    }
  }, [
    formData.model_engine,
    formData.model_format,
    formData.model_size_in_billions,
    enginesObj,
  ])

  const engineItems = useMemo(() => {
    return engineOptions.map((engine) => {
      const modelFormats = Array.from(
        new Set(enginesObj[engine]?.map((item) => item.model_format))
      )

      const relevantSpecs = modelData.model_specs.filter((spec) =>
        modelFormats.includes(spec.model_format)
      )

      const cached = relevantSpecs.some((spec) => isCached(spec))

      return {
        value: engine,
        label: cached ? `${engine} ${t('launchModel.cached')}` : engine,
      }
    })
  }, [engineOptions, enginesObj, modelData])

  const formatItems = useMemo(() => {
    return formatOptions.map((format) => {
      const specs = modelData.model_specs.filter(
        (spec) => spec.model_format === format
      )

      const cached = specs.some((spec) => isCached(spec))

      return {
        value: format,
        label: cached ? `${format} ${t('launchModel.cached')}` : format,
      }
    })
  }, [formatOptions, modelData])

  const sizeItems = useMemo(() => {
    return sizeOptions.map((size) => {
      const specs = modelData.model_specs
        .filter((spec) => spec.model_format === formData.model_format)
        .filter((spec) => spec.model_size_in_billions === size)
      const cached = specs.some((spec) => isCached(spec))

      return {
        value: size,
        label: cached ? `${size} ${t('launchModel.cached')}` : size,
      }
    })
  }, [sizeOptions, modelData])

  const quantizationItems = useMemo(() => {
    return quantizationOptions.map((quant) => {
      const specs = modelData.model_specs
        .filter((spec) => spec.model_format === formData.model_format)
        .filter((spec) =>
          modelType === 'LLM'
            ? spec.model_size_in_billions ===
              convertModelSize(formData.model_size_in_billions)
            : true
        )

      const spec = specs.find((s) => {
        return s.quantizations === quant
      })
      const cached = Array.isArray(spec?.cache_status)
        ? spec?.cache_status[spec?.quantizations.indexOf(quant)]
        : spec?.cache_status

      return {
        value: quant,
        label: cached ? `${quant} ${t('launchModel.cached')}` : quant,
      }
    })
  }, [quantizationOptions, modelData])

  const checkRequiredFields = (fields, data) => {
    return fields.every((field) => {
      if (field.type === 'collapse' && field.children) {
        return checkRequiredFields(field.children, data)
      }
      if (field.visible && field.required) {
        const value = data[field.name]
        return (
          value !== undefined && value !== null && String(value).trim() !== ''
        )
      }
      return true
    })
  }

  const isDynamicFieldComplete = (val) => {
    if (!val) return false
    if (Array.isArray(val) && typeof val[0] === 'string') {
      return val.every((item) => item?.trim())
    }
    if (Array.isArray(val) && typeof val[0] === 'object') {
      return val.every((obj) => {
        return Object.values(obj).every(
          (v) => typeof v !== 'string' || v.trim()
        )
      })
    }
    return true
  }

  const handleDynamicField = (name, val) => {
    setCheckDynamicFieldComplete((prev) => {
      const filtered = prev.filter((item) => item.name !== name)
      return [...filtered, { name, isComplete: isDynamicFieldComplete(val) }]
    })
    setFormData((prev) => ({
      ...prev,
      [name]: val,
    }))
  }

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }))
  }

  const handleGPUIdx = (data) => {
    const arr = []
    data?.split(',').forEach((item) => {
      arr.push(Number(item))
    })
    return arr
  }

  const processFieldsRecursively = (fields, result) => {
    fields.forEach((field) => {
      if (result[field.name] === undefined && field.default !== undefined) {
        result[field.name] = field.default
      }
      if (
        (result[field.name] || result[field.name] === 0) &&
        field.type === 'number' &&
        typeof result[field.name] !== 'number'
      ) {
        result[field.name] = parseInt(result[field.name], 10)
      }
      if (field.name === 'gpu_idx' && result[field.name]) {
        result[field.name] = handleGPUIdx(result[field.name])
      }
      if (
        field.visible === false ||
        result[field.name] === '' ||
        result[field.name] == null
      ) {
        delete result[field.name]
      }
      if (field.type === 'dynamicField' && Array.isArray(result[field.name])) {
        result[field.name] = result[field.name].filter((item) => {
          if (typeof item === 'string') {
            return item.trim() !== ''
          } else if (typeof item === 'object' && item !== null) {
            return Object.values(item).every((val) => {
              if (typeof val === 'string') {
                return val.trim() !== ''
              }
              return val !== undefined && val !== null
            })
          }
          return false
        })
        if (result[field.name].length === 0) {
          delete result[field.name]
        }
      }
      if (field.type === 'collapse' && Array.isArray(field.children)) {
        processFieldsRecursively(field.children, result)
      }
    })
  }

  const getFinalFormData = () => {
    const fields = modelFormConfig[modelType] || []
    let result = {
      model_name: modelData.model_name,
      model_type: modelType,
      ...formData,
    }

    processFieldsRecursively(fields, result)

    if (result.n_gpu) {
      result.n_gpu = normalizeNGPU(result.n_gpu)
    }

    if (result.n_gpu_layers < 0) {
      delete result.n_gpu_layers
    }
    if (result.download_hub === 'none') {
      delete result.download_hub
    }
    if (result.gguf_quantization === 'none') {
      delete result.gguf_quantization
    }

    const peft_model_config = {}
    ;['image_lora_load_kwargs', 'image_lora_fuse_kwargs'].forEach((key) => {
      if (result[key]?.length) {
        peft_model_config[key] = arrayToObject(result[key], handleValueType)
        delete result[key]
      }
    })
    if (result.lora_list?.length) {
      peft_model_config.lora_list = result.lora_list
      delete result.lora_list
    }
    if (Object.keys(peft_model_config).length) {
      result.peft_model_config = peft_model_config
    }

    if (result.enable_virtual_env === 'unset') {
      delete result.enable_virtual_env
    } else if (result.enable_virtual_env === 'true') {
      result.enable_virtual_env = true
    } else if (result.enable_virtual_env === 'false') {
      result.enable_virtual_env = false
    }

    if (result.envs?.length) {
      result.envs = arrayToObject(result.envs)
    }

    if (result.quantization_config?.length) {
      result.quantization_config = arrayToObject(
        result.quantization_config,
        handleValueType
      )
    }

    for (const key in result) {
      if (![...llmAllDataKey, 'custom'].includes(key)) {
        delete result[key]
      }
    }
    if (result.custom?.length) {
      Object.assign(result, arrayToObject(result.custom, handleValueType))
      delete result.custom
    }

    return result
  }

  const handleSubmit = () => {
    if (isCallingApi || isUpdatingModel) {
      return
    }

    setIsCallingApi(true)
    setProgress(0)
    setIsShowProgress(true)
    setIsShowCancel(true)

    try {
      const data = getFinalFormData()
      // First fetcher request to initiate the model
      fetchWrapper
        .post('/v1/models', data)
        .then(() => {
          navigate(`/running_models/${modelType}`)
          sessionStorage.setItem(
            'runningModelType',
            `/running_models/${modelType}`
          )
          let historyArr = JSON.parse(localStorage.getItem('historyArr')) || []
          const historyModelNameArr = historyArr.map((item) => item.model_name)
          if (historyModelNameArr.includes(data.model_name)) {
            historyArr = historyArr.map((item) => {
              if (item.model_name === data.model_name) {
                return data
              }
              return item
            })
          } else {
            historyArr.push(data)
          }
          localStorage.setItem('historyArr', JSON.stringify(historyArr))
        })
        .catch((error) => {
          console.error('Error:', error)
          if (error.response?.status === 499) {
            setSuccessMsg(t('launchModel.cancelledSuccessfully'))
          } else if (error.response?.status !== 403) {
            setErrorMsg(error.message)
          }
        })
        .finally(() => {
          setIsCallingApi(false)
          stopPolling()
          setIsShowProgress(false)
          setIsShowCancel(false)
          setIsLoading(false)
        })
      startPolling()
    } catch (error) {
      setErrorMsg(`${error}`)
      setIsCallingApi(false)
    }
  }

  const modelFormConfig = getModelFormConfig({
    t,
    formData,
    modelData,
    gpuAvailable,
    engineItems,
    formatItems,
    sizeItems,
    quantizationItems,
    getNGPURange,
    downloadHubOptions,
    enginesWithNWorker,
    multimodalProjectorOptions,
  })

  const areRequiredFieldsFilled = useMemo(() => {
    const data = getFinalFormData()
    const fields = modelFormConfig[modelType] || []
    return checkRequiredFields(fields, data)
  }, [formData, modelType])

  const renderFormFields = (fields = []) => {
    const enhancedFields = fields.map((field) => {
      if (field.name === 'reasoning_content') {
        const enable_thinking_default = fields.find(
          (item) => item.name === 'enable_thinking'
        ).default
        return {
          ...field,
          visible:
            !!modelData.model_ability?.includes('reasoning') &&
            (formData.enable_thinking ??
              enable_thinking_default ??
              field.default ??
              false),
        }
      }
      if (field.name === 'gpu_idx' && field.visible !== true) {
        const n_gpu_default = fields.find(
          (item) => item.name === 'n_gpu'
        ).default

        return {
          ...field,
          visible:
            formData.n_gpu === 'GPU' ||
            ((formData.n_gpu === undefined ||
              formData.n_gpu === null ||
              formData.n_gpu === '') &&
              n_gpu_default === 'GPU'),
        }
      }
      return field
    })

    return enhancedFields
      .filter((field) => field.visible)
      .map((field) => {
        const fieldKey = field.name
        switch (field.type) {
          case 'collapse': {
            const open = collapseState[fieldKey] ?? false

            const toggleCollapse = () => {
              setCollapseState((prev) => ({
                ...prev,
                [fieldKey]: !prev[fieldKey],
              }))
            }

            return (
              <Box key={fieldKey} sx={{ mt: 2 }}>
                <ListItemButton onClick={toggleCollapse}>
                  <div
                    style={{ display: 'flex', alignItems: 'center', gap: 10 }}
                  >
                    <ListItemText primary={field.label} />
                    {open ? <ExpandLess /> : <ExpandMore />}
                  </div>
                </ListItemButton>
                <Collapse in={open} timeout="auto" unmountOnExit>
                  <Box sx={{ pl: 2 }}>
                    {renderFormFields(field.children || [])}
                  </Box>
                </Collapse>
              </Box>
            )
          }
          case 'select':
            return (
              <SelectField
                key={fieldKey}
                label={field.label}
                labelId={`${field.name}-label`}
                name={field.name}
                disabled={field.disabled}
                value={formData[field.name] ?? field.default ?? ''}
                onChange={handleChange}
                options={field.options}
                required={field.required}
              />
            )
          case 'number':
            return (
              <TextField
                key={fieldKey}
                name={field.name}
                label={field.label}
                type="number"
                disabled={field.disabled}
                InputProps={field.inputProps}
                value={formData[field.name] ?? field.default ?? ''}
                onChange={handleChange}
                required={field.required}
                error={field.error}
                helperText={field.error && field.helperText}
                fullWidth
                margin="normal"
                className="textHighlight"
              />
            )
          case 'input':
            return (
              <TextField
                key={fieldKey}
                name={field.name}
                label={field.label}
                disabled={field.disabled}
                InputProps={field.inputProps}
                value={formData[field.name] ?? field.default ?? ''}
                onChange={handleChange}
                required={field.required}
                error={field.error}
                helperText={field.error && field.helperText}
                fullWidth
                margin="normal"
                className="textHighlight"
              />
            )
          case 'switch':
            return (
              <div key={fieldKey}>
                <Tooltip title={field.tip} placement="top">
                  <FormControlLabel
                    name={field.name}
                    label={field.label}
                    labelPlacement="start"
                    control={
                      <Switch
                        checked={formData[field.name] ?? field.default ?? false}
                      />
                    }
                    onChange={handleChange}
                    required={field.required}
                  />
                </Tooltip>
              </div>
            )
          case 'radio':
            return (
              <div
                key={fieldKey}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  marginLeft: 15,
                }}
              >
                <span>{field.label}</span>
                <RadioGroup
                  row
                  name={field.name}
                  value={
                    formData[field.name] ??
                    field.default ??
                    field.options?.[0]?.value ??
                    ''
                  }
                  onChange={handleChange}
                >
                  {field.options.map((item) => (
                    <FormControlLabel
                      key={item.label}
                      value={item.value}
                      control={<Radio />}
                      label={item.label}
                    />
                  ))}
                </RadioGroup>
              </div>
            )
          case 'dynamicField':
            return (
              <DynamicFieldList
                key={fieldKey}
                name={field.name}
                label={field.label}
                mode={field.mode}
                keyPlaceholder={field.keyPlaceholder}
                valuePlaceholder={field.valuePlaceholder}
                value={formData[field.name]}
                onChange={handleDynamicField}
                keyOptions={field.keyOptions}
              />
            )
          default:
            return null
        }
      })
  }

  const renderButtonContent = () => {
    if (isShowCancel) {
      return <StopCircle sx={{ fontSize: 26 }} />
    }
    if (isLoading) {
      return <CircularProgress size={26} />
    }

    return <RocketLaunchOutlined sx={{ fontSize: 26 }} />
  }

  return (
    <Drawer open={open} onClose={onClose} anchor="right">
      <Box className="drawerCard">
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box display="flex" alignItems="center">
            <TitleTypography value={modelData.model_name} />
            {hasHistory && (
              <Chip
                label={t('launchModel.lastConfig')}
                variant="outlined"
                size="small"
                color="primary"
                onDelete={deleteHistory}
              />
            )}
          </Box>
          <Box display="flex" alignItems="center" gap={1}>
            <Tooltip
              title={t('launchModel.commandLineParsing')}
              placement="top"
            >
              <ContentPasteGo
                className="pasteText"
                onClick={() => setIsOpenPasteDialog(true)}
              />
            </Tooltip>

            {areRequiredFieldsFilled && (
              <CopyToCommandLine
                getData={getFinalFormData}
                predefinedKeys={llmAllDataKey}
              />
            )}
          </Box>
        </Box>

        <Box
          sx={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
          }}
          component="form"
          onSubmit={handleSubmit}
        >
          <Box>{renderFormFields(modelFormConfig[modelType])}</Box>

          <Box marginTop={4}>
            <Box height={16}>
              {isShowProgress && (
                <Progress style={{ marginBottom: 20 }} progress={progress} />
              )}
            </Box>
            <Box display="flex" gap={2}>
              <Button
                style={{ flex: 1 }}
                variant="outlined"
                color="primary"
                title={t(
                  isShowCancel ? 'launchModel.cancel' : 'launchModel.launch'
                )}
                disabled={
                  !areRequiredFieldsFilled ||
                  isLoading ||
                  isCallingApi ||
                  checkDynamicFieldComplete.some((item) => !item.isComplete)
                }
                onClick={() => {
                  if (isShowCancel) {
                    fetchCancelModel()
                  } else {
                    handleSubmit()
                  }
                }}
              >
                {renderButtonContent()}
              </Button>
              <Button
                style={{ flex: 1 }}
                variant="outlined"
                color="primary"
                onClick={onClose}
                title={t('launchModel.goBack')}
              >
                <UndoOutlined sx={{ fontSize: 26 }} />
              </Button>
            </Box>
          </Box>
        </Box>

        <PasteDialog
          open={isOpenPasteDialog}
          onHandleClose={() => setIsOpenPasteDialog(false)}
          onHandleCommandLine={handleCommandLine}
        />
      </Box>
    </Drawer>
  )
}

export default LaunchModelDrawer
