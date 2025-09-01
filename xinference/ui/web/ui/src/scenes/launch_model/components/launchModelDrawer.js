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
import {
  additionalParameterTipList,
  llmAllDataKey,
  quantizationParametersTipList,
} from '../data/data'
import CopyToCommandLine from './commandBuilder'
import DynamicFieldList from './dynamicFieldList'
import PasteDialog from './pasteDialog'
import Progress from './progress'

const csghubArr = ['qwen2-instruct']
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

  const intervalRef = useRef(null)

  const downloadHubOptions = [
    'none',
    'huggingface',
    'modelscope',
    'openmind_hub',
    ...(csghubArr.includes(modelData.model_name) ? ['csghub'] : []),
  ]

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

  const handleDynamicField = (name, val) => {
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
      if (field.visible === false || result[field.name] === '') {
        delete result[field.name]
      }
      if (field.type === 'collapse') {
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

  const modelFormConfig = {
    LLM: [
      {
        name: 'model_engine',
        label: t('launchModel.modelEngine'),
        type: 'select',
        options: engineItems,
        visible: true,
        required: true,
      },
      {
        name: 'model_format',
        label: t('launchModel.modelFormat'),
        type: 'select',
        options: formatItems,
        visible: true,
        disabled: !formData.model_engine,
        required: true,
      },
      {
        name: 'model_size_in_billions',
        label: t('launchModel.modelSize'),
        type: 'select',
        options: sizeItems,
        visible: true,
        disabled: !formData.model_format,
        required: true,
      },
      {
        name: 'quantization',
        label: t('launchModel.quantization'),
        type: 'select',
        options: quantizationItems,
        visible: true,
        disabled: !formData.model_size_in_billions,
        required: true,
      },
      {
        name: 'multimodal_projector',
        label: t('launchModel.multimodelProjector'),
        type: 'select',
        default: multimodalProjectorOptions[0],
        options: multimodalProjectorOptions,
        disabled: !formData.model_size_in_billions,
        required: true,
        visible: !!multimodalProjectorOptions.length,
      },
      {
        name: 'n_gpu',
        label: t(
          enginesWithNWorker.includes(formData.model_engine)
            ? 'launchModel.nGPUPerWorker'
            : 'launchModel.nGPU'
        ),
        type: 'select',
        default: 'auto',
        options: getNGPURange('LLM'),
        disabled: !formData.quantization,
        visible: true,
      },
      {
        name: 'n_gpu_layers',
        label: t('launchModel.nGpuLayers'),
        type: 'number',
        default: -1,
        inputProps: {
          inputProps: {
            min: -1,
          },
        },
        disabled: !formData.quantization,
        required: true,
        visible: !!['ggufv2', 'ggmlv3'].includes(formData.model_format),
      },
      {
        name: 'replica',
        label: t('launchModel.replica'),
        type: 'number',
        default: 1,
        inputProps: {
          inputProps: {
            min: 1,
          },
        },
        disabled: !formData.quantization,
        visible: true,
      },
      {
        name: 'enable_thinking',
        label: t('launchModel.enableThinking'),
        type: 'switch',
        visible: !!modelData.model_ability?.includes('hybrid'),
        default: true,
      },
      {
        name: 'reasoning_content',
        label: t('launchModel.parsingReasoningContent'),
        type: 'switch',
        default: false,
      },
      {
        name: 'nested_section_optional',
        label: t('launchModel.optionalConfigurations'),
        type: 'collapse',
        visible: true,
        children: [
          {
            name: 'model_uid',
            label: t('launchModel.modelUID.optional'),
            type: 'input',
            visible: true,
          },
          {
            name: 'request_limits',
            label: t('launchModel.requestLimits.optional'),
            type: 'number',
            error:
              !!formData.request_limits &&
              !/^[1-9]\d*$/.test(formData.request_limits),
            helperText: t('launchModel.enterIntegerGreaterThanZero'),
            visible: true,
          },
          {
            name: 'n_worker',
            label: t('launchModel.workerCount.optional'),
            type: 'input',
            visible:
              !!formData.model_engine &&
              enginesWithNWorker.includes(formData.model_engine),
          },
          {
            name: 'worker_ip',
            label: t('launchModel.workerIp.optional'),
            type: 'input',
            visible: true,
          },
          {
            name: 'gpu_idx',
            label: t('launchModel.GPUIdx'),
            type: 'input',
            error:
              !!formData.gpu_idx && !/^\d+(?:,\d+)*$/.test(formData.gpu_idx),
            helperText: t('launchModel.enterCommaSeparatedNumbers'),
            visible: true,
          },
          {
            name: 'download_hub',
            label: t('launchModel.downloadHub.optional'),
            type: 'select',
            options: downloadHubOptions,
            visible: true,
          },
          {
            name: 'model_path',
            label: t('launchModel.modelPath.optional'),
            type: 'input',
            visible: true,
          },
          {
            name: 'nested_section_lora',
            label: t('launchModel.loraConfig'),
            type: 'collapse',
            visible: true,
            children: [
              {
                name: 'lora_list',
                label: t('launchModel.loraModelConfig'),
                type: 'dynamicField',
                mode: 'key-value',
                keyPlaceholder: 'lora_name',
                valuePlaceholder: 'local_path',
                visible: true,
              },
            ],
          },
          {
            name: 'nested_section_virtualEnv',
            label: t('launchModel.virtualEnvConfig'),
            type: 'collapse',
            visible: true,
            children: [
              {
                name: 'enable_virtual_env',
                label: t('launchModel.modelVirtualEnv'),
                type: 'radio',
                options: [
                  { label: 'Unset', value: 'unset' },
                  { label: 'False', value: false },
                  { label: 'True', value: true },
                ],
                default: 'unset',
                visible: true,
              },
              {
                name: 'virtual_env_packages',
                label: t('launchModel.virtualEnvPackage'),
                type: 'dynamicField',
                mode: 'value',
                valuePlaceholder: 'value',
                visible: true,
              },
            ],
          },
          {
            name: 'nested_section_env',
            label: t('launchModel.envVariableConfig'),
            type: 'collapse',
            visible: true,
            children: [
              {
                name: 'envs',
                label: t('launchModel.envVariable'),
                type: 'dynamicField',
                mode: 'key-value',
                keyPlaceholder: 'key',
                valuePlaceholder: 'value',
                visible: true,
              },
            ],
          },
        ],
      },
      {
        name: 'quantization_config',
        label: t(
          'launchModel.additionalQuantizationParametersForInferenceEngine'
        ),
        type: 'dynamicField',
        mode: 'key-value',
        keyPlaceholder: 'key',
        valuePlaceholder: 'value',
        keyOptions: quantizationParametersTipList,
        visible:
          !!formData.model_engine && formData.model_engine === 'Transformers',
      },
      {
        name: 'custom',
        label: `${t('launchModel.additionalParametersForInferenceEngine')}${
          formData.model_engine ? ': ' + formData.model_engine : ''
        }`,
        type: 'dynamicField',
        mode: 'key-value',
        keyPlaceholder: 'key',
        valuePlaceholder: 'value',
        keyOptions:
          additionalParameterTipList[
            formData.model_engine?.toLocaleLowerCase()
          ],
        visible: true,
      },
    ],
    embedding: [
      {
        name: 'model_engine',
        label: t('launchModel.modelEngine'),
        type: 'select',
        options: engineItems,
        visible: true,
      },
      {
        name: 'model_format',
        label: t('launchModel.modelFormat'),
        type: 'select',
        options: formatItems,
        visible: true,
        disabled: !formData.model_engine,
      },
      {
        name: 'quantization',
        label: t('launchModel.quantization'),
        type: 'select',
        options: quantizationItems,
        visible: true,
        disabled: !formData.model_format,
      },
      {
        name: 'replica',
        label: t('launchModel.replica'),
        type: 'number',
        visible: true,
        default: 1,
        inputProps: {
          inputProps: {
            min: 1,
          },
        },
        disabled: !formData.quantization,
      },
      {
        name: 'n_gpu',
        label: t('launchModel.device'),
        type: 'select',
        default: gpuAvailable === 0 ? 'CPU' : 'GPU',
        options: getNGPURange('embedding'),
        visible: true,
      },
      {
        name: 'gpu_idx',
        label: t('launchModel.GPUIdx'),
        type: 'input',
        error: !!formData.gpu_idx && !/^\d+(?:,\d+)*$/.test(formData.gpu_idx),
        helperText: t('launchModel.enterCommaSeparatedNumbers'),
        visible: !!formData.n_gpu && formData.n_gpu === 'GPU',
      },
      {
        name: 'model_uid',
        label: t('launchModel.modelUID.optional'),
        type: 'input',
        visible: true,
      },
      {
        name: 'worker_ip',
        label: t('launchModel.workerIp.optional'),
        type: 'input',
        visible: true,
      },
      {
        name: 'download_hub',
        label: t('launchModel.downloadHub.optional'),
        type: 'select',
        options: downloadHubOptions,
        visible: true,
      },
      {
        name: 'model_path',
        label: t('launchModel.modelPath.optional'),
        type: 'input',
        visible: true,
      },
      {
        name: 'nested_section_virtualEnv',
        label: t('launchModel.virtualEnvConfig'),
        type: 'collapse',
        visible: true,
        children: [
          {
            name: 'enable_virtual_env',
            label: t('launchModel.modelVirtualEnv'),
            type: 'radio',
            options: [
              { label: 'Unset', value: 'unset' },
              { label: 'False', value: false },
              { label: 'True', value: true },
            ],
            default: 'unset',
            visible: true,
          },
          {
            name: 'virtual_env_packages',
            label: t('launchModel.virtualEnvPackage'),
            type: 'dynamicField',
            mode: 'value',
            valuePlaceholder: 'value',
            visible: true,
          },
        ],
      },
      {
        name: 'nested_section_env',
        label: t('launchModel.envVariableConfig'),
        type: 'collapse',
        visible: true,
        children: [
          {
            name: 'envs',
            label: t('launchModel.envVariable'),
            type: 'dynamicField',
            mode: 'key-value',
            keyPlaceholder: 'key',
            valuePlaceholder: 'value',
            visible: true,
          },
        ],
      },
      {
        name: 'custom',
        label: `${t('launchModel.additionalParametersForInferenceEngine')}${
          formData.model_engine ? ': ' + formData.model_engine : ''
        }`,
        type: 'dynamicField',
        mode: 'key-value',
        keyPlaceholder: 'key',
        valuePlaceholder: 'value',
        visible: true,
      },
    ],
    rerank: [
      {
        name: 'model_engine',
        label: t('launchModel.modelEngine'),
        type: 'select',
        options: engineItems,
        visible: true,
      },
      {
        name: 'model_format',
        label: t('launchModel.modelFormat'),
        type: 'select',
        options: formatItems,
        visible: true,
        disabled: !formData.model_engine,
      },
      {
        name: 'quantization',
        label: t('launchModel.quantization'),
        type: 'select',
        options: quantizationItems,
        visible: true,
        disabled: !formData.model_format,
      },
      {
        name: 'replica',
        label: t('launchModel.replica'),
        type: 'number',
        visible: true,
        default: 1,
        inputProps: {
          inputProps: {
            min: 1,
          },
        },
      },
      {
        name: 'n_gpu',
        label: t('launchModel.device'),
        type: 'select',
        default: gpuAvailable === 0 ? 'CPU' : 'GPU',
        options: getNGPURange('rerank'),
        visible: true,
      },
      {
        name: 'gpu_idx',
        label: t('launchModel.GPUIdx'),
        type: 'input',
        error: !!formData.gpu_idx && !/^\d+(?:,\d+)*$/.test(formData.gpu_idx),
        helperText: t('launchModel.enterCommaSeparatedNumbers'),
        visible: !!formData.n_gpu && formData.n_gpu === 'GPU',
      },
      {
        name: 'model_uid',
        label: t('launchModel.modelUID.optional'),
        type: 'input',
        visible: true,
      },
      {
        name: 'worker_ip',
        label: t('launchModel.workerIp.optional'),
        type: 'input',
        visible: true,
      },
      {
        name: 'download_hub',
        label: t('launchModel.downloadHub.optional'),
        type: 'select',
        options: downloadHubOptions,
        visible: true,
      },
      {
        name: 'model_path',
        label: t('launchModel.modelPath.optional'),
        type: 'input',
        visible: true,
      },
      {
        name: 'nested_section_virtualEnv',
        label: t('launchModel.virtualEnvConfig'),
        type: 'collapse',
        visible: true,
        children: [
          {
            name: 'enable_virtual_env',
            label: t('launchModel.modelVirtualEnv'),
            type: 'radio',
            options: [
              { label: 'Unset', value: 'unset' },
              { label: 'False', value: false },
              { label: 'True', value: true },
            ],
            default: 'unset',
            visible: true,
          },
          {
            name: 'virtual_env_packages',
            label: t('launchModel.virtualEnvPackage'),
            type: 'dynamicField',
            mode: 'value',
            valuePlaceholder: 'value',
            visible: true,
          },
        ],
      },
      {
        name: 'nested_section_env',
        label: t('launchModel.envVariableConfig'),
        type: 'collapse',
        visible: true,
        children: [
          {
            name: 'envs',
            label: t('launchModel.envVariable'),
            type: 'dynamicField',
            mode: 'key-value',
            keyPlaceholder: 'key',
            valuePlaceholder: 'value',
            visible: true,
          },
        ],
      },
      {
        name: 'custom',
        label: t('launchModel.additionalParametersForInferenceEngine'),
        type: 'dynamicField',
        mode: 'key-value',
        keyPlaceholder: 'key',
        valuePlaceholder: 'value',
        visible: true,
      },
    ],
    image: [
      {
        name: 'model_uid',
        label: t('launchModel.modelUID.optional'),
        type: 'input',
        visible: true,
      },
      {
        name: 'replica',
        label: t('launchModel.replica'),
        type: 'number',
        visible: true,
        default: 1,
        inputProps: {
          inputProps: {
            min: 1,
          },
        },
      },
      {
        name: 'n_gpu',
        label: t('launchModel.nGPU'),
        type: 'select',
        default: 'auto',
        options: getNGPURange('image'),
        visible: true,
      },
      {
        name: 'gpu_idx',
        label: t('launchModel.GPUIdx'),
        type: 'input',
        error: !!formData.gpu_idx && !/^\d+(?:,\d+)*$/.test(formData.gpu_idx),
        helperText: t('launchModel.enterCommaSeparatedNumbers'),
        visible: !!formData.n_gpu && formData.n_gpu !== 'CPU',
      },
      {
        name: 'worker_ip',
        label: t('launchModel.workerIp.optional'),
        type: 'input',
        visible: true,
      },
      {
        name: 'download_hub',
        label: t('launchModel.downloadHub.optional'),
        type: 'select',
        options: downloadHubOptions,
        visible: true,
      },
      {
        name: 'model_path',
        label: t('launchModel.modelPath.optional'),
        type: 'input',
        visible: true,
      },
      {
        name: 'gguf_quantization',
        label: t('launchModel.GGUFQuantization.optional'),
        type: 'select',
        options: ['none', ...(modelData.gguf_quantizations || [])],
        visible: !!modelData.gguf_quantizations,
      },
      {
        name: 'gguf_model_path',
        label: t('launchModel.GGUFModelPath.optional'),
        type: 'input',
        visible: !!modelData.gguf_quantizations,
      },
      {
        name: 'lightning_version',
        label: t('launchModel.lightningVersions.optional'),
        type: 'select',
        options: ['none', ...(modelData.lightning_versions || [])],
        visible: !!modelData.lightning_versions,
      },
      {
        name: 'lightning_model_path',
        label: t('launchModel.lightningModelPath.optional'),
        type: 'input',
        visible: !!modelData.lightning_versions,
      },
      {
        name: 'cpu_offload',
        label: t('launchModel.CPUOffload'),
        type: 'switch',
        default: false,
        tip: t('launchModel.CPUOffload.tip'),
        visible: true,
      },
      {
        name: 'nested_section_lora',
        label: t('launchModel.loraConfig'),
        type: 'collapse',
        visible: true,
        children: [
          {
            name: 'lora_list',
            label: t('launchModel.loraModelConfig'),
            type: 'dynamicField',
            mode: 'key-value',
            keyPlaceholder: 'lora_name',
            valuePlaceholder: 'local_path',
            visible: true,
          },
          {
            name: 'image_lora_load_kwargs',
            label: t('launchModel.loraLoadKwargsForImageModel'),
            type: 'dynamicField',
            mode: 'key-value',
            visible: true,
          },
          {
            name: 'image_lora_fuse_kwargs',
            label: t('launchModel.loraFuseKwargsForImageModel'),
            type: 'dynamicField',
            mode: 'key-value',
            visible: true,
          },
        ],
      },
      {
        name: 'nested_section_virtualEnv',
        label: t('launchModel.virtualEnvConfig'),
        type: 'collapse',
        visible: true,
        children: [
          {
            name: 'enable_virtual_env',
            label: t('launchModel.modelVirtualEnv'),
            type: 'radio',
            options: [
              { label: 'Unset', value: 'unset' },
              { label: 'False', value: false },
              { label: 'True', value: true },
            ],
            default: 'unset',
            visible: true,
          },
          {
            name: 'virtual_env_packages',
            label: t('launchModel.virtualEnvPackage'),
            type: 'dynamicField',
            mode: 'value',
            valuePlaceholder: 'value',
            visible: true,
          },
        ],
      },
      {
        name: 'nested_section_env',
        label: t('launchModel.envVariableConfig'),
        type: 'collapse',
        visible: true,
        children: [
          {
            name: 'envs',
            label: t('launchModel.envVariable'),
            type: 'dynamicField',
            mode: 'key-value',
            keyPlaceholder: 'key',
            valuePlaceholder: 'value',
            visible: true,
          },
        ],
      },
      {
        name: 'custom',
        label: t('launchModel.additionalParametersForInferenceEngine'),
        type: 'dynamicField',
        mode: 'key-value',
        keyPlaceholder: 'key',
        valuePlaceholder: 'value',
        visible: true,
      },
    ],
    audio: [
      {
        name: 'model_uid',
        label: t('launchModel.modelUID.optional'),
        type: 'input',
        visible: true,
      },
      {
        name: 'replica',
        label: t('launchModel.replica'),
        type: 'number',
        visible: true,
        default: 1,
        inputProps: {
          inputProps: {
            min: 1,
          },
        },
      },
      {
        name: 'n_gpu',
        label: t('launchModel.device'),
        type: 'select',
        default: gpuAvailable === 0 ? 'CPU' : 'GPU',
        options: getNGPURange('audio'),
        visible: true,
      },
      {
        name: 'gpu_idx',
        label: t('launchModel.GPUIdx'),
        type: 'input',
        error: !!formData.gpu_idx && !/^\d+(?:,\d+)*$/.test(formData.gpu_idx),
        helperText: t('launchModel.enterCommaSeparatedNumbers'),
        visible: !!formData.n_gpu && formData.n_gpu === 'GPU',
      },
      {
        name: 'worker_ip',
        label: t('launchModel.workerIp.optional'),
        type: 'input',
        visible: true,
      },
      {
        name: 'download_hub',
        label: t('launchModel.downloadHub.optional'),
        type: 'select',
        options: downloadHubOptions,
        visible: true,
      },
      {
        name: 'model_path',
        label: t('launchModel.modelPath.optional'),
        type: 'input',
        visible: true,
      },
      {
        name: 'nested_section_virtualEnv',
        label: t('launchModel.virtualEnvConfig'),
        type: 'collapse',
        visible: true,
        children: [
          {
            name: 'enable_virtual_env',
            label: t('launchModel.modelVirtualEnv'),
            type: 'radio',
            options: [
              { label: 'Unset', value: 'unset' },
              { label: 'False', value: false },
              { label: 'True', value: true },
            ],
            default: 'unset',
            visible: true,
          },
          {
            name: 'virtual_env_packages',
            label: t('launchModel.virtualEnvPackage'),
            type: 'dynamicField',
            mode: 'value',
            valuePlaceholder: 'value',
            visible: true,
          },
        ],
      },
      {
        name: 'nested_section_env',
        label: t('launchModel.envVariableConfig'),
        type: 'collapse',
        visible: true,
        children: [
          {
            name: 'envs',
            label: t('launchModel.envVariable'),
            type: 'dynamicField',
            mode: 'key-value',
            keyPlaceholder: 'key',
            valuePlaceholder: 'value',
            visible: true,
          },
        ],
      },
      {
        name: 'custom',
        label: t('launchModel.additionalParametersForInferenceEngine'),
        type: 'dynamicField',
        mode: 'key-value',
        keyPlaceholder: 'key',
        valuePlaceholder: 'value',
        visible: true,
      },
    ],
    video: [
      {
        name: 'model_uid',
        label: t('launchModel.modelUID.optional'),
        type: 'input',
        visible: true,
      },
      {
        name: 'replica',
        label: t('launchModel.replica'),
        type: 'number',
        visible: true,
        default: 1,
        inputProps: {
          inputProps: {
            min: 1,
          },
        },
      },
      {
        name: 'n_gpu',
        label: t('launchModel.device'),
        type: 'select',
        default: gpuAvailable === 0 ? 'CPU' : 'GPU',
        options: getNGPURange('video'),
        visible: true,
      },
      {
        name: 'gpu_idx',
        label: t('launchModel.GPUIdx'),
        type: 'input',
        error: !!formData.gpu_idx && !/^\d+(?:,\d+)*$/.test(formData.gpu_idx),
        helperText: t('launchModel.enterCommaSeparatedNumbers'),
        visible: !!formData.n_gpu && formData.n_gpu === 'GPU',
      },
      {
        name: 'worker_ip',
        label: t('launchModel.workerIp.optional'),
        type: 'input',
        visible: true,
      },
      {
        name: 'download_hub',
        label: t('launchModel.downloadHub.optional'),
        type: 'select',
        options: downloadHubOptions,
        visible: true,
      },
      {
        name: 'model_path',
        label: t('launchModel.modelPath.optional'),
        type: 'input',
        visible: true,
      },
      {
        name: 'cpu_offload',
        label: t('launchModel.CPUOffload'),
        type: 'switch',
        default: false,
        tip: t('launchModel.CPUOffload.tip'),
        visible: true,
      },
      {
        name: 'nested_section_lora',
        label: t('launchModel.loraConfig'),
        type: 'collapse',
        visible: true,
        children: [
          {
            name: 'lora_list',
            label: t('launchModel.loraModelConfig'),
            type: 'dynamicField',
            mode: 'key-value',
            keyPlaceholder: 'lora_name',
            valuePlaceholder: 'local_path',
            visible: true,
          },
          {
            name: 'image_lora_load_kwargs',
            label: t('launchModel.loraLoadKwargsForImageModel'),
            type: 'dynamicField',
            mode: 'key-value',
            visible: true,
          },
          {
            name: 'image_lora_fuse_kwargs',
            label: t('launchModel.loraFuseKwargsForImageModel'),
            type: 'dynamicField',
            mode: 'key-value',
            visible: true,
          },
        ],
      },
      {
        name: 'nested_section_virtualEnv',
        label: t('launchModel.virtualEnvConfig'),
        type: 'collapse',
        visible: true,
        children: [
          {
            name: 'enable_virtual_env',
            label: t('launchModel.modelVirtualEnv'),
            type: 'radio',
            options: [
              { label: 'Unset', value: 'unset' },
              { label: 'False', value: false },
              { label: 'True', value: true },
            ],
            default: 'unset',
            visible: true,
          },
          {
            name: 'virtual_env_packages',
            label: t('launchModel.virtualEnvPackage'),
            type: 'dynamicField',
            mode: 'value',
            valuePlaceholder: 'value',
            visible: true,
          },
        ],
      },
      {
        name: 'nested_section_env',
        label: t('launchModel.envVariableConfig'),
        type: 'collapse',
        visible: true,
        children: [
          {
            name: 'envs',
            label: t('launchModel.envVariable'),
            type: 'dynamicField',
            mode: 'key-value',
            keyPlaceholder: 'key',
            valuePlaceholder: 'value',
            visible: true,
          },
        ],
      },
      {
        name: 'custom',
        label: t('launchModel.additionalParametersForInferenceEngine'),
        type: 'dynamicField',
        mode: 'key-value',
        keyPlaceholder: 'key',
        valuePlaceholder: 'value',
        visible: true,
      },
    ],
  }

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
                disabled={!areRequiredFieldsFilled || isLoading || isCallingApi}
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
