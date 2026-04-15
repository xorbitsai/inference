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
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Drawer,
  FormControlLabel,
  ListItemButton,
  ListItemText,
  Paper,
  Radio,
  RadioGroup,
  Switch,
  TextField,
  Tooltip,
  Typography,
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
import SelectField from './selectField'

const enginesWithNWorker = ['SGLang', 'vLLM', 'MLX']
const modelEngineType = ['LLM', 'embedding', 'rerank', 'image']
const historyStorageKey = 'historyArr'
const defaultHistoryUID = '__default_model_uid__'

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
  const [replicaStatuses, setReplicaStatuses] = useState([])
  const [pendingHistory, setPendingHistory] = useState(null)
  const [hasFetchedEngines, setHasFetchedEngines] = useState(false)
  const [historyEntries, setHistoryEntries] = useState([])
  const [selectedHistoryKey, setSelectedHistoryKey] = useState(null)
  const [isHistoryDialogOpen, setIsHistoryDialogOpen] = useState(false)
  const [historyToDelete, setHistoryToDelete] = useState(null)

  const intervalRef = useRef(null)

  const modelSpecs = modelData.model_specs || []

  const downloadHubOptions = useMemo(
    () => ['none', ...(modelData?.download_hubs || [])],
    [modelData?.download_hubs]
  )

  const hasHistory = historyEntries.length > 0

  const selectedHistoryEntry = useMemo(
    () =>
      historyEntries.find((item) => item.cache_key === selectedHistoryKey) ||
      historyEntries[0] ||
      null,
    [historyEntries, selectedHistoryKey]
  )

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
    let val = String(str).trim()

    const isBracketed = (s) =>
      (s.startsWith('{') && s.endsWith('}')) ||
      (s.startsWith('[') && s.endsWith(']'))

    const tryParseStructured = (s) => {
      try {
        const normalized = s
          .replace(/\bNone\b/g, 'null')
          .replace(/\bTrue\b/g, 'true')
          .replace(/\bFalse\b/g, 'false')
          .replace(/'([^']*)'/g, '"$1"')
        return JSON.parse(normalized)
      } catch (e) {
        return null
      }
    }

    if (val.toLowerCase() === 'none') {
      return null
    } else if (val.toLowerCase() === 'true') {
      return true
    } else if (val.toLowerCase() === 'false') {
      return false
    } else if (isBracketed(val)) {
      const parsed = tryParseStructured(val)
      if (parsed !== null) return parsed
      // parsing failed: keep original string without splitting by comma
      return val
    } else if (val.includes(',')) {
      return val.split(',').map((item) => item.trim())
    } else if (val.includes('，')) {
      return val.split('，').map((item) => item.trim())
    } else if (
      !Number.isNaN(Number(val)) &&
      (val !== '' || Number(val) === 0)
    ) {
      return Number(val)
    } else {
      return val
    }
  }

  const arrayToObject = (arr, transformValue = (v) => v) =>
    Object.fromEntries(
      arr.map(({ key, value }) => [key, transformValue(value)])
    )

  const buildHistoryKey = (modelName, modelUID) =>
    `${modelName}::${modelUID || defaultHistoryUID}`

  const normalizeHistoryEntry = (item) => {
    if (!item || typeof item !== 'object') return null

    const data =
      item.data && typeof item.data === 'object' && !Array.isArray(item.data)
        ? item.data
        : item
    const modelName = data.model_name || item.model_name

    if (!modelName) return null

    const modelUID = data.model_uid || item.model_uid || ''

    return {
      cache_key: buildHistoryKey(modelName, modelUID),
      model_name: modelName,
      model_uid: modelUID,
      updated_at: Number(item.updated_at) || 0,
      data,
    }
  }

  const readHistoryEntries = () => {
    try {
      const historyArr =
        JSON.parse(localStorage.getItem(historyStorageKey)) || []
      if (!Array.isArray(historyArr)) return []
      return historyArr.map(normalizeHistoryEntry).filter(Boolean)
    } catch (error) {
      console.error('Failed to read history cache:', error)
      return []
    }
  }

  const sortHistoryEntries = (entries) =>
    [...entries].sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0))

  const getHistoryEntriesForModel = (entries = readHistoryEntries()) =>
    sortHistoryEntries(
      entries.filter((item) => item.model_name === modelData.model_name)
    )

  const writeHistoryEntries = (entries) => {
    localStorage.setItem(
      historyStorageKey,
      JSON.stringify(
        entries.map((item) => ({
          model_name: item.model_name,
          model_uid: item.model_uid,
          updated_at: item.updated_at,
          data: item.data,
        }))
      )
    )
  }

  const upsertHistoryEntry = (data) => {
    const nextEntry = normalizeHistoryEntry({
      model_name: data.model_name,
      model_uid: data.model_uid,
      updated_at: Date.now(),
      data,
    })
    const entries = readHistoryEntries().filter(
      (item) => item.cache_key !== nextEntry.cache_key
    )
    const nextEntries = [...entries, nextEntry]
    writeHistoryEntries(nextEntries)
    return getHistoryEntriesForModel(nextEntries)
  }

  const removeHistoryEntry = (cacheKey) => {
    const nextEntries = readHistoryEntries().filter(
      (item) => item.cache_key !== cacheKey
    )
    writeHistoryEntries(nextEntries)
    return getHistoryEntriesForModel(nextEntries)
  }

  const isHistoryEngineValid = (historyData) => {
    if (!historyData?.model_engine) return false

    const engineData = enginesObj[historyData.model_engine]
    return (
      engineOptions.includes(historyData.model_engine) &&
      Array.isArray(engineData) &&
      engineData.length > 0
    )
  }

  const loadHistoryEntry = (entry, shouldCloseDialog = true) => {
    if (!entry) return

    setSelectedHistoryKey(entry.cache_key)
    if (shouldCloseDialog) {
      setIsHistoryDialogOpen(false)
    }

    if (modelEngineType.includes(modelType)) {
      if (open && hasFetchedEngines) {
        if (isHistoryEngineValid(entry.data)) {
          applyHistory(entry.data)
        } else {
          setErrorMsg(t('launchModel.invalidConfigCache'))
        }
      } else {
        setPendingHistory(entry.data)
      }
      return
    }

    applyHistory(entry.data)
  }

  const handleDeleteHistory = () => {
    if (!historyToDelete) return

    const wasSelected = historyToDelete.cache_key === selectedHistoryKey
    const nextEntries = removeHistoryEntry(historyToDelete.cache_key)

    setHistoryEntries(nextEntries)
    setHistoryToDelete(null)

    if (!nextEntries.length) {
      setSelectedHistoryKey(null)
      setPendingHistory(null)
      setFormData({})
      setCollapseState({})
      return
    }

    if (!wasSelected) return

    const latestEntry = nextEntries[0]
    setSelectedHistoryKey(latestEntry.cache_key)

    if (modelEngineType.includes(modelType)) {
      if (open && hasFetchedEngines) {
        if (isHistoryEngineValid(latestEntry.data)) {
          applyHistory(latestEntry.data)
        } else {
          setFormData({})
          setCollapseState({})
        }
      } else {
        setPendingHistory(latestEntry.data)
      }
      return
    }

    applyHistory(latestEntry.data)
  }

  const formatHistoryTime = (updatedAt) => {
    if (!updatedAt) return '—'

    const date = new Date(updatedAt)
    return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString()
  }

  const objectToArray = (obj) => {
    if (!obj || typeof obj !== 'object') return []
    return Object.entries(obj).map(([key, value]) => ({ key, value }))
  }

  const getReplicaWorkerAddress = (replica) => replica?.worker_address || '—'

  const restoreNGPU = (value) => {
    if (value === null) return 'CPU'
    if (value === 'auto') {
      return ['LLM', 'image'].includes(modelType) ? 'auto' : 'GPU'
    }
    if (typeof value === 'number') return String(value)
    return value || 'CPU'
  }

  const stringifyStructuredForInput = (val) => {
    if (val === null) return 'none'

    if (Array.isArray(val)) {
      const allPrimitives = val.every(
        (item) =>
          item === null || ['string', 'number', 'boolean'].includes(typeof item)
      )
      if (allPrimitives) {
        return val.map((v) => String(v)).join(',')
      }
    }

    try {
      const json = JSON.stringify(val)
      return json.replace(/"/g, "'").replace(/:/g, ': ').replace(/,/g, ', ')
    } catch (e) {
      return String(val)
    }
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
              : typeof result[key] === 'object'
              ? stringifyStructuredForInput(result[key])
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

  const applyHistory = (data) => {
    const restoredData = restoreFormDataFormat(data)
    setFormData(restoredData)
    const collapseFromData = getCollapseStateFromData(restoredData)
    setCollapseState((prev) => ({ ...prev, ...collapseFromData }))
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
    setHasFetchedEngines(false)
    const enableVirtualEnv =
      formData?.enable_virtual_env === true ||
      formData?.enable_virtual_env === 'true'
        ? 'true'
        : formData?.enable_virtual_env === false ||
          formData?.enable_virtual_env === 'false'
        ? 'false'
        : null
    const enableVirtualEnvQuery = enableVirtualEnv
      ? `?enable_virtual_env=${enableVirtualEnv}`
      : ''
    fetchWrapper
      .get(
        model_type === 'LLM'
          ? `/v1/engines/${model_name}${enableVirtualEnvQuery}`
          : `/v1/engines/${model_type}/${model_name}${enableVirtualEnvQuery}`
      )
      .then((data) => {
        if (!data) {
          setEnginesObj({})
          setEngineOptions([])
          return
        }
        setEnginesObj(data)
        setEngineOptions(Object.keys(data))
      })
      .catch((error) => {
        console.error('Error:', error)
        if (error?.response?.status !== 403) {
          setErrorMsg(error.message)
        }
      })
      .finally(() => {
        setIsCallingApi(false)
        setHasFetchedEngines(true)
      })
  }

  const fetchCancelModel = async () => {
    try {
      await fetchWrapper.post(`/v1/models/${modelData.model_name}/cancel`)
      setIsLoading(true)
    } catch (error) {
      console.error('Error:', error)
      if (error?.response?.status !== 403) {
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
      const data = getFinalFormData()
      const modelUid = data.model_uid || data.model_name

      // Fetch overall progress (existing logic)
      const res = await fetchWrapper.get(
        `/v1/models/${modelData.model_name}/progress`
      )
      if (res.progress !== 1.0) setProgress(Number(res.progress))

      // Fetch replica statuses (new logic)
      const replicaRes = await fetchWrapper.get(
        `/v1/models/${modelUid}/replicas`
      )
      setReplicaStatuses(replicaRes)
    } catch (error) {
      console.error('Error:', error)
      // Suppress 404 errors during early launch phase as model might not exist yet
      if (error?.response?.status !== 403 && error?.response?.status !== 404) {
        setErrorMsg(error.message)
      }
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
    const entries = getHistoryEntriesForModel()
    setHistoryEntries(entries)

    const latestEntry = entries[0]
    if (latestEntry) {
      setSelectedHistoryKey(latestEntry.cache_key)
      if (modelEngineType.includes(modelType)) {
        setPendingHistory(latestEntry.data)
      } else {
        applyHistory(latestEntry.data)
      }
    }
  }, [])

  useEffect(() => {
    if (!pendingHistory || !modelEngineType.includes(modelType)) return
    if (!open || !hasFetchedEngines) return

    if (!pendingHistory.model_engine) {
      setFormData({})
      setCollapseState({})
      setPendingHistory(null)
      return
    }

    if (isHistoryEngineValid(pendingHistory)) {
      applyHistory(pendingHistory)
    } else {
      setFormData({})
      setCollapseState({})
    }
    setPendingHistory(null)
  }, [pendingHistory, open, hasFetchedEngines, engineOptions, modelType])

  useEffect(() => {
    if (open && modelEngineType.includes(modelType))
      fetchModelEngine(modelData.model_name, modelType)
  }, [open, modelData.model_name, modelType])

  useEffect(() => {
    if (formData.model_engine && modelEngineType.includes(modelType)) {
      const format = [
        ...new Set(
          enginesObj[formData.model_engine]?.map((item) => item.model_format)
        ),
      ].filter((value) => value !== undefined && value !== null && value !== '')
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
      image: {
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
    ].filter(
      (option) => option !== undefined && option !== null && option !== ''
    )

    config.optionSetter(options)
    if (!options.includes(formData[config.field])) {
      setFormData((prev) => ({ ...prev, [config.field]: '' }))
    }

    if (options.length === 1) {
      setFormData((prev) => ({ ...prev, [config.field]: options[0] }))
    }
  }, [formData.model_engine, formData.model_format, enginesObj])

  useEffect(() => {
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
      const engineData = enginesObj[engine]
      let modelFormats = []
      let label = engine
      let disabled = false

      if (Array.isArray(engineData)) {
        modelFormats = Array.from(
          new Set(engineData.map((item) => item.model_format))
        )

        const relevantSpecs = modelSpecs.filter((spec) =>
          modelFormats.includes(spec.model_format)
        )

        const cached = relevantSpecs.some((spec) => isCached(spec))

        label = cached ? `${engine} ${t('launchModel.cached')}` : engine
      } else if (typeof engineData === 'string') {
        label = `${engine} (${engineData})`
        disabled = true
      }

      return {
        value: engine,
        label,
        disabled,
      }
    })
  }, [engineOptions, enginesObj, modelData])

  const formatItems = useMemo(() => {
    return formatOptions
      .filter(
        (format) => format !== undefined && format !== null && format !== ''
      )
      .map((format) => {
        const specs = modelSpecs.filter((spec) => spec.model_format === format)

        const cached = specs.some((spec) => isCached(spec))

        return {
          value: format,
          label: cached ? `${format} ${t('launchModel.cached')}` : format,
        }
      })
  }, [formatOptions, modelData])

  const sizeItems = useMemo(() => {
    return sizeOptions.map((size) => {
      const specs = modelSpecs
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
      const specs = modelSpecs
        .filter((spec) => spec.model_format === formData.model_format)
        .filter((spec) =>
          modelType === 'LLM'
            ? spec.model_size_in_billions ===
              convertModelSize(formData.model_size_in_billions)
            : true
        )

      const spec = specs.find((s) => {
        return modelType === 'LLM'
          ? s.quantizations === quant
          : s.quantization === quant
      })
      const cached =
        modelType === 'LLM' && Array.isArray(spec?.cache_status)
          ? spec?.cache_status[spec?.quantizations.indexOf(quant)]
          : spec?.cache_status

      return {
        value: quant,
        label: cached ? `${quant} ${t('launchModel.cached')}` : quant,
      }
    })
  }, [quantizationOptions, modelData])

  const modelFormConfig = useMemo(
    () =>
      getModelFormConfig({
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
      }),
    [
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
    ]
  )

  useEffect(() => {
    if (modelFormConfig[modelType]) {
      const defaults = {}
      const traverse = (fields) => {
        fields.forEach((field) => {
          if (
            field.default !== undefined &&
            formData[field.name] === undefined
          ) {
            defaults[field.name] = field.default
          }
          if (field.children) {
            traverse(field.children)
          }
        })
      }
      traverse(modelFormConfig[modelType])
      if (Object.keys(defaults).length > 0) {
        setFormData((prev) => ({ ...prev, ...defaults }))
      }
    }
  }, [modelFormConfig, modelType])

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
          const nextHistoryEntries = upsertHistoryEntry(data)
          setHistoryEntries(nextHistoryEntries)
          setSelectedHistoryKey(
            buildHistoryKey(data.model_name, data.model_uid)
          )
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
                label={`${t('launchModel.configCache')} (${
                  historyEntries.length
                })`}
                variant="outlined"
                size="small"
                color="primary"
                onClick={() => setIsHistoryDialogOpen(true)}
                onDelete={() => setHistoryToDelete(selectedHistoryEntry)}
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
              <Tooltip
                title={
                  isShowCancel ? (
                    <Box sx={{ minWidth: 200 }}>
                      <Typography
                        variant="subtitle2"
                        sx={{ mb: 1, fontWeight: 600, color: 'inherit' }}
                      >
                        {t('launchModel.launchProgress')}:
                      </Typography>
                      {replicaStatuses.length > 0 ? (
                        replicaStatuses.map((replica) => (
                          <Box
                            key={replica.replica_id}
                            sx={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              mb: 0.75,
                              gap: 1.5,
                            }}
                          >
                            <Box sx={{ minWidth: 0 }}>
                              <Typography
                                variant="caption"
                                display="block"
                                sx={{ fontWeight: 600, color: 'inherit' }}
                              >
                                {t('modelReplicaDetails.replica')}{' '}
                                {replica.replica_id}:
                              </Typography>
                              <Typography
                                variant="caption"
                                display="block"
                                sx={{
                                  fontFamily: 'monospace',
                                  color: 'rgba(255, 255, 255, 0.82)',
                                  lineHeight: 1.4,
                                  wordBreak: 'break-all',
                                }}
                              >
                                {getReplicaWorkerAddress(replica)}
                              </Typography>
                            </Box>
                            <Chip
                              label={replica.status}
                              color={
                                replica.status === 'READY'
                                  ? 'success'
                                  : replica.status === 'ERROR'
                                  ? 'error'
                                  : 'default'
                              }
                              size="small"
                              sx={{
                                'height': 22,
                                'fontSize': '0.7rem',
                                'fontWeight': 600,
                                '&.MuiChip-colorDefault': {
                                  bgcolor: 'rgba(255, 255, 255, 0.14)',
                                  color: '#fff',
                                },
                              }}
                            />
                          </Box>
                        ))
                      ) : (
                        <Typography variant="caption" sx={{ color: 'inherit' }}>
                          {t('launchModel.initializing')}
                        </Typography>
                      )}
                    </Box>
                  ) : (
                    t('launchModel.launch')
                  )
                }
                placement="top"
                arrow
                slotProps={{
                  tooltip: {
                    sx: {
                      bgcolor: 'rgba(17, 24, 39, 0.96)',
                      color: '#fff',
                      border: '1px solid rgba(255, 255, 255, 0.12)',
                      boxShadow: 6,
                      px: 1.5,
                      py: 1.25,
                      maxWidth: 360,
                    },
                  },
                  arrow: {
                    sx: {
                      color: 'rgba(17, 24, 39, 0.96)',
                    },
                  },
                }}
              >
                <Button
                  style={{ flex: 1 }}
                  variant="outlined"
                  color="primary"
                  disabled={
                    !isShowCancel &&
                    (!areRequiredFieldsFilled ||
                      isLoading ||
                      isCallingApi ||
                      checkDynamicFieldComplete.some(
                        (item) => !item.isComplete
                      ))
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
              </Tooltip>
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

        <Dialog
          open={isHistoryDialogOpen}
          onClose={() => setIsHistoryDialogOpen(false)}
          fullWidth
          maxWidth="sm"
        >
          <DialogTitle>{t('launchModel.configCache')}</DialogTitle>
          <DialogContent dividers>
            {historyEntries.length ? (
              historyEntries.map((entry, index) => {
                const isSelected = entry.cache_key === selectedHistoryKey
                return (
                  <Paper
                    key={entry.cache_key}
                    variant="outlined"
                    sx={{
                      p: 2,
                      mb: index === historyEntries.length - 1 ? 0 : 1.5,
                      borderColor: isSelected ? 'primary.main' : 'divider',
                    }}
                  >
                    <Box
                      sx={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        gap: 2,
                        flexWrap: 'wrap',
                      }}
                    >
                      <Box sx={{ minWidth: 0 }}>
                        <Box
                          sx={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 1,
                            flexWrap: 'wrap',
                            mb: 0.5,
                          }}
                        >
                          <Typography variant="subtitle2">
                            {entry.model_uid || t('launchModel.defaultConfig')}
                          </Typography>
                        </Box>
                        <Typography variant="body2" color="text.secondary">
                          {t('modelReplicaDetails.modelUid')}:{' '}
                          {entry.model_uid || '—'}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {t('launchModel.lastUpdated')}:{' '}
                          {formatHistoryTime(entry.updated_at)}
                        </Typography>
                      </Box>
                      <Box
                        sx={{
                          display: 'flex',
                          gap: 1,
                          alignItems: 'center',
                          flexWrap: 'wrap',
                        }}
                      >
                        <Button
                          size="small"
                          variant={isSelected ? 'contained' : 'outlined'}
                          onClick={() => loadHistoryEntry(entry)}
                        >
                          {t('launchModel.loadCache')}
                        </Button>
                        <Button
                          size="small"
                          color="error"
                          onClick={() => setHistoryToDelete(entry)}
                        >
                          {t('launchModel.deleteCache')}
                        </Button>
                      </Box>
                    </Box>
                  </Paper>
                )
              })
            ) : (
              <Typography color="text.secondary">
                {t('launchModel.noConfigCache')}
              </Typography>
            )}
          </DialogContent>
          <DialogActions style={{ justifyContent: 'space-between' }}>
            <Button
              onClick={() => {
                setFormData({})
                setCollapseState({})
                setSelectedHistoryKey(null)
                setPendingHistory(null)
                setIsHistoryDialogOpen(false)
              }}
            >
              {t('launchModel.newCache')}
            </Button>
            <Button onClick={() => setIsHistoryDialogOpen(false)}>
              {t('modelReplicaDetails.close')}
            </Button>
          </DialogActions>
        </Dialog>

        <Dialog
          open={!!historyToDelete}
          onClose={() => setHistoryToDelete(null)}
          aria-labelledby="delete-history-cache-title"
        >
          <DialogTitle id="delete-history-cache-title">
            {t('components.warning')}
          </DialogTitle>
          <DialogContent>
            {t('launchModel.confirmDeleteConfigCache')}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setHistoryToDelete(null)}>
              {t('components.cancel')}
            </Button>
            <Button color="error" onClick={handleDeleteHistory} autoFocus>
              {t('components.ok')}
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </Drawer>
  )
}

export default LaunchModelDrawer
