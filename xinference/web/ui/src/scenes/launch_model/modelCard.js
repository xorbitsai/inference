import './styles/modelCardStyle.css'

import {
  ChatOutlined,
  Close,
  ContentPasteGo,
  Delete,
  EditNote,
  EditNoteOutlined,
  ExpandLess,
  ExpandMore,
  Grade,
  HelpCenterOutlined,
  HelpOutline,
  RocketLaunchOutlined,
  StarBorder,
  UndoOutlined,
} from '@mui/icons-material'
import {
  Alert,
  Backdrop,
  Box,
  Button,
  Chip,
  CircularProgress,
  Collapse,
  FormControl,
  FormControlLabel,
  Grid,
  IconButton,
  InputLabel,
  ListItemButton,
  ListItemText,
  MenuItem,
  Paper,
  Select,
  Snackbar,
  Stack,
  Switch,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TablePagination,
  TableRow,
  TextField,
  Tooltip,
} from '@mui/material'
import { styled } from '@mui/material/styles'
import React, { useContext, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'

import { ApiContext } from '../../components/apiContext'
import CopyComponent from '../../components/copyComponent/copyComponent'
import DeleteDialog from '../../components/deleteDialog'
import fetchWrapper from '../../components/fetchWrapper'
import TitleTypography from '../../components/titleTypography'
import AddPair from './components/addPair'
import CopyToCommandLine from './components/copyComponent'
import Drawer from './components/drawer'
import PasteDialog from './components/pasteDialog'
import { additionalParameterTipList, llmAllDataKey } from './data/data'

const csghubArr = ['qwen2-instruct']
const enginesWithNWorker = ['SGLang']

const ModelCard = ({
  url,
  modelData,
  gpuAvailable,
  modelType,
  is_custom = false,
  onHandleCompleteDelete,
  onHandlecustomDelete,
  onGetCollectionArr,
}) => {
  const [hover, setHover] = useState(false)
  const [selected, setSelected] = useState(false)
  const [requestLimitsAlert, setRequestLimitsAlert] = useState(false)
  const [GPUIdxAlert, setGPUIdxAlert] = useState(false)
  const [isOther, setIsOther] = useState(false)
  const [isPeftModelConfig, setIsPeftModelConfig] = useState(false)
  const [openSnackbar, setOpenSnackbar] = useState(false)
  const [openErrorSnackbar, setOpenErrorSnackbar] = useState(false)
  const [errorSnackbarValue, setErrorSnackbarValue] = useState('')
  const { isCallingApi, setIsCallingApi } = useContext(ApiContext)
  const { isUpdatingModel } = useContext(ApiContext)
  const { setErrorMsg } = useContext(ApiContext)
  const navigate = useNavigate()

  // Model parameter selections
  const [modelUID, setModelUID] = useState('')
  const [modelEngine, setModelEngine] = useState('')
  const [modelFormat, setModelFormat] = useState('')
  const [modelSize, setModelSize] = useState('')
  const [quantization, setQuantization] = useState('')
  const [nWorker, setNWorker] = useState(1)
  const [nGPU, setNGPU] = useState('auto')
  const [nGpu, setNGpu] = useState(gpuAvailable === 0 ? 'CPU' : 'GPU')
  const [nGPULayers, setNGPULayers] = useState(-1)
  const [replica, setReplica] = useState(1)
  const [requestLimits, setRequestLimits] = useState('')
  const [workerIp, setWorkerIp] = useState('')
  const [GPUIdx, setGPUIdx] = useState('')
  const [downloadHub, setDownloadHub] = useState('')
  const [modelPath, setModelPath] = useState('')
  const [ggufQuantizations, setGgufQuantizations] = useState('')
  const [ggufModelPath, setGgufModelPath] = useState('')
  const [cpuOffload, setCpuOffload] = useState(false)

  const [enginesObj, setEnginesObj] = useState({})
  const [engineOptions, setEngineOptions] = useState([])
  const [formatOptions, setFormatOptions] = useState([])
  const [sizeOptions, setSizeOptions] = useState([])
  const [quantizationOptions, setQuantizationOptions] = useState([])
  const [customDeleted, setCustomDeleted] = useState(false)
  const [customParametersArr, setCustomParametersArr] = useState([])
  const [loraListArr, setLoraListArr] = useState([])
  const [imageLoraLoadKwargsArr, setImageLoraLoadKwargsArr] = useState([])
  const [imageLoraFuseKwargsArr, setImageLoraFuseKwargsArr] = useState([])
  const [isOpenCachedList, setIsOpenCachedList] = useState(false)
  const [isDeleteCached, setIsDeleteCached] = useState(false)
  const [cachedListArr, setCachedListArr] = useState([])
  const [cachedModelVersion, setCachedModelVersion] = useState('')
  const [cachedRealPath, setCachedRealPath] = useState('')
  const [page, setPage] = useState(0)
  const [isDeleteCustomModel, setIsDeleteCustomModel] = useState(false)
  const [isJsonShow, setIsJsonShow] = useState(false)
  const [isHistory, setIsHistory] = useState(false)
  const [customArr, setCustomArr] = useState([])
  const [loraArr, setLoraArr] = useState([])
  const [imageLoraLoadArr, setImageLoraLoadArr] = useState([])
  const [imageLoraFuseArr, setImageLoraFuseArr] = useState([])
  const [customParametersArrLength, setCustomParametersArrLength] = useState(0)
  const [isOpenPasteDialog, setIsOpenPasteDialog] = useState(false)

  const parentRef = useRef(null)
  const { t } = useTranslation()

  const range = (start, end) => {
    return new Array(end - start + 1).fill(undefined).map((_, i) => i + start)
  }

  const isCached = (spec) => {
    if (Array.isArray(spec.cache_status)) {
      return spec.cache_status.some((cs) => cs)
    } else {
      return spec.cache_status === true
    }
  }

  // model size can be int or string. For string style, "1_8" means 1.8 as an example.
  const convertModelSize = (size) => {
    return size.toString().includes('_') ? size : parseInt(size, 10)
  }

  useEffect(() => {
    let keyArr = []
    for (let key in enginesObj) {
      keyArr.push(key)
    }
    const data = handleGetHistory()
    if (keyArr.length && data.model_name) {
      handleLlmHistory(data)
    }
  }, [enginesObj])

  useEffect(() => {
    if (modelEngine) {
      const format = [
        ...new Set(enginesObj[modelEngine].map((item) => item.model_format)),
      ]
      setFormatOptions(format)
      if (!format.includes(modelFormat)) {
        setModelFormat('')
      }
      if (format.length === 1) {
        setModelFormat(format[0])
      }
    }
  }, [modelEngine])

  useEffect(() => {
    if (modelEngine && modelFormat) {
      const sizes = [
        ...new Set(
          enginesObj[modelEngine]
            .filter((item) => item.model_format === modelFormat)
            .map((item) => item.model_size_in_billions)
        ),
      ]
      setSizeOptions(sizes)
      if (
        sizeOptions.length &&
        JSON.stringify(sizes) !== JSON.stringify(sizeOptions)
      ) {
        setModelSize('')
      }
      if (sizes.length === 1) {
        setModelSize(sizes[0])
      }
    }
  }, [modelEngine, modelFormat])

  useEffect(() => {
    if (modelEngine && modelFormat && modelSize) {
      const quants = [
        ...new Set(
          enginesObj[modelEngine]
            .filter(
              (item) =>
                item.model_format === modelFormat &&
                item.model_size_in_billions === convertModelSize(modelSize)
            )
            .flatMap((item) => item.quantizations)
        ),
      ]
      setQuantizationOptions(quants)
      if (!quants.includes(quantization)) {
        setQuantization('')
      }
      if (quants.length === 1) {
        setQuantization(quants[0])
      }
    }
  }, [modelEngine, modelFormat, modelSize])

  useEffect(() => {
    setCustomParametersArrLength(customParametersArr.length)
    if (
      parentRef.current &&
      customParametersArr.length > customParametersArrLength
    ) {
      parentRef.current.scrollTo({
        top: parentRef.current.scrollHeight,
        behavior: 'smooth',
      })
    }
  }, [customParametersArr])

  const getNGPURange = () => {
    if (gpuAvailable === 0) {
      // remain 'auto' for distributed situation
      return ['auto', 'CPU']
    }
    return ['auto', 'CPU'].concat(range(1, gpuAvailable))
  }

  const getNewNGPURange = () => {
    if (gpuAvailable === 0) {
      return ['CPU']
    } else {
      return ['GPU', 'CPU']
    }
  }

  const getModelEngine = (model_name) => {
    fetchWrapper
      .get(`/v1/engines/${model_name}`)
      .then((data) => {
        setEnginesObj(data)
        setEngineOptions(Object.keys(data))
        setIsCallingApi(false)
      })
      .catch((error) => {
        console.error('Error:', error)
        if (error.response.status !== 403) {
          setErrorMsg(error.message)
        }
        setIsCallingApi(false)
      })
  }

  const handleModelData = () => {
    const modelDataWithID_LLM = {
      // If user does not fill model_uid, pass null (None) to server and server generates it.
      model_uid: modelUID?.trim() === '' ? null : modelUID?.trim(),
      model_name: modelData.model_name,
      model_type: modelType,
      model_engine: modelEngine,
      model_format: modelFormat,
      model_size_in_billions: convertModelSize(modelSize),
      quantization: quantization,
      n_gpu:
        parseInt(nGPU, 10) === 0 || nGPU === 'CPU'
          ? null
          : nGPU === 'auto'
          ? 'auto'
          : parseInt(nGPU, 10),
      replica: replica,
      request_limits:
        String(requestLimits)?.trim() === ''
          ? null
          : Number(String(requestLimits)?.trim()),
      n_worker: nWorker,
      worker_ip: workerIp?.trim() === '' ? null : workerIp?.trim(),
      gpu_idx: GPUIdx?.trim() === '' ? null : handleGPUIdx(GPUIdx?.trim()),
      download_hub: downloadHub === '' ? null : downloadHub,
      model_path: modelPath?.trim() === '' ? null : modelPath?.trim(),
    }

    const modelDataWithID_other = {
      model_uid: modelUID?.trim() === '' ? null : modelUID?.trim(),
      model_name: modelData.model_name,
      model_type: modelType,
      replica: replica,
      n_gpu: nGpu === 'GPU' ? 'auto' : null,
      worker_ip: workerIp?.trim() === '' ? null : workerIp?.trim(),
      gpu_idx: GPUIdx?.trim() === '' ? null : handleGPUIdx(GPUIdx?.trim()),
      download_hub: downloadHub === '' ? null : downloadHub,
      model_path: modelPath?.trim() === '' ? null : modelPath?.trim(),
    }

    if (nGPULayers >= 0) modelDataWithID_LLM.n_gpu_layers = nGPULayers
    if (ggufQuantizations)
      modelDataWithID_other.gguf_quantization = ggufQuantizations
    if (ggufModelPath) modelDataWithID_other.gguf_model_path = ggufModelPath
    if (modelType === 'image') modelDataWithID_other.cpu_offload = cpuOffload

    const modelDataWithID =
      modelType === 'LLM' ? modelDataWithID_LLM : modelDataWithID_other

    if (
      loraListArr.length ||
      imageLoraLoadKwargsArr.length ||
      imageLoraFuseKwargsArr.length
    ) {
      const peft_model_config = {}
      if (imageLoraLoadKwargsArr.length) {
        const image_lora_load_kwargs = {}
        imageLoraLoadKwargsArr.forEach((item) => {
          image_lora_load_kwargs[item.key] = handleValueType(item.value)
        })
        peft_model_config['image_lora_load_kwargs'] = image_lora_load_kwargs
      }
      if (imageLoraFuseKwargsArr.length) {
        const image_lora_fuse_kwargs = {}
        imageLoraFuseKwargsArr.forEach((item) => {
          image_lora_fuse_kwargs[item.key] = handleValueType(item.value)
        })
        peft_model_config['image_lora_fuse_kwargs'] = image_lora_fuse_kwargs
      }
      if (loraListArr.length) {
        const lora_list = loraListArr
        lora_list.map((item) => {
          delete item.id
        })
        peft_model_config['lora_list'] = lora_list
      }
      modelDataWithID['peft_model_config'] = peft_model_config
    }

    if (customParametersArr.length) {
      customParametersArr.forEach((item) => {
        modelDataWithID[item.key] = handleValueType(item.value)
      })
    }

    return modelDataWithID
  }

  const launchModel = () => {
    if (isCallingApi || isUpdatingModel) {
      return
    }

    setIsCallingApi(true)

    try {
      const modelDataWithID = handleModelData()
      // First fetcher request to initiate the model
      fetchWrapper
        .post('/v1/models', modelDataWithID)
        .then(() => {
          navigate(`/running_models/${modelType}`)
          sessionStorage.setItem(
            'runningModelType',
            `/running_models/${modelType}`
          )
          let historyArr = JSON.parse(localStorage.getItem('historyArr')) || []
          const historyModelNameArr = historyArr.map((item) => item.model_name)
          if (historyModelNameArr.includes(modelDataWithID.model_name)) {
            historyArr = historyArr.map((item) => {
              if (item.model_name === modelDataWithID.model_name) {
                return modelDataWithID
              }
              return item
            })
          } else {
            historyArr.push(modelDataWithID)
          }
          localStorage.setItem('historyArr', JSON.stringify(historyArr))

          setIsCallingApi(false)
        })
        .catch((error) => {
          console.error('Error:', error)
          if (error.response.status !== 403) {
            setErrorMsg(error.message)
          }
          setIsCallingApi(false)
        })
    } catch (error) {
      setOpenErrorSnackbar(true)
      setErrorSnackbarValue(`${error}`)
      setIsCallingApi(false)
    }
  }

  const handleGPUIdx = (data) => {
    const arr = []
    data?.split(',').forEach((item) => {
      arr.push(Number(item))
    })
    return arr
  }

  const handeCustomDelete = (e) => {
    e.stopPropagation()
    const subType = sessionStorage.getItem('subType').split('/')
    if (subType) {
      subType[3]
      fetchWrapper
        .delete(
          `/v1/model_registrations/${
            subType[3] === 'llm' ? 'LLM' : subType[3]
          }/${modelData.model_name}`
        )
        .then(() => {
          setCustomDeleted(true)
          onHandlecustomDelete(modelData.model_name)
          setIsDeleteCustomModel(false)
        })
        .catch((error) => {
          console.error(error)
          if (error.response.status !== 403) {
            setErrorMsg(error.message)
          }
        })
    }
  }

  const judgeArr = (arr, keysArr) => {
    if (
      arr.length &&
      arr[arr.length - 1][keysArr[0]] !== '' &&
      arr[arr.length - 1][keysArr[1]] !== ''
    ) {
      return true
    } else if (arr.length === 0) {
      return true
    } else {
      return false
    }
  }

  const handleValueType = (str) => {
    str = String(str)
    if (str.toLowerCase() === 'none') {
      return null
    } else if (str.toLowerCase() === 'true') {
      return true
    } else if (str.toLowerCase() === 'false') {
      return false
    } else if (Number(str) || (str !== '' && Number(str) === 0)) {
      return Number(str)
    } else {
      return str
    }
  }

  const StyledTableRow = styled(TableRow)(({ theme }) => ({
    '&:nth-of-type(odd)': {
      backgroundColor: theme.palette.action.hover,
    },
  }))

  const emptyRows =
    page >= 0 ? Math.max(0, (1 + page) * 5 - cachedListArr.length) : 0

  const handleChangePage = (_, newPage) => {
    setPage(newPage)
  }

  const handleOpenCachedList = () => {
    setIsOpenCachedList(true)
    getCachedList()
    document.body.style.overflow = 'hidden'
  }

  const handleCloseCachedList = () => {
    document.body.style.overflow = 'auto'
    setHover(false)
    setIsOpenCachedList(false)
    if (cachedListArr.length === 0) {
      onHandleCompleteDelete(modelData.model_name)
    }
  }

  const getCachedList = () => {
    fetchWrapper
      .get(`/v1/cache/models?model_name=${modelData.model_name}`)
      .then((data) => setCachedListArr(data.list))
      .catch((error) => {
        console.error(error)
        if (error.response.status !== 403) {
          setErrorMsg(error.message)
        }
      })
  }

  const handleOpenDeleteCachedDialog = (real_path, model_version) => {
    setCachedRealPath(real_path)
    setCachedModelVersion(model_version)
    setIsDeleteCached(true)
  }

  const handleDeleteCached = () => {
    fetchWrapper
      .delete(`/v1/cache/models?model_version=${cachedModelVersion}`)
      .then(() => {
        const cachedArr = cachedListArr.filter(
          (item) => item.real_path !== cachedRealPath
        )
        setCachedListArr(cachedArr)
        setIsDeleteCached(false)
        if (cachedArr.length) {
          if (
            (page + 1) * 5 >= cachedListArr.length &&
            cachedArr.length % 5 === 0
          ) {
            setPage(cachedArr.length / 5 - 1)
          }
        }
      })
      .catch((error) => {
        console.error(error)
        if (error.response.status !== 403) {
          setErrorMsg(error.message)
        }
      })
  }

  const handleJsonDataPresentation = () => {
    const arr = sessionStorage.getItem('subType').split('/')
    sessionStorage.setItem(
      'registerModelType',
      `/register_model/${arr[arr.length - 1]}`
    )
    sessionStorage.setItem('customJsonData', JSON.stringify(modelData))
    navigate(`/register_model/${arr[arr.length - 1]}/${modelData.model_name}`)
  }

  const handleGetHistory = () => {
    const historyArr = JSON.parse(localStorage.getItem('historyArr')) || []
    return (
      historyArr.find((item) => item.model_name === modelData.model_name) || {}
    )
  }

  const handleLlmHistory = (data) => {
    const {
      model_engine,
      model_format,
      model_size_in_billions,
      quantization,
      n_worker,
      n_gpu,
      n_gpu_layers,
      replica,
      model_uid,
      request_limits,
      worker_ip,
      gpu_idx,
      download_hub,
      model_path,
      peft_model_config,
    } = data

    if (!engineOptions.includes(model_engine)) {
      setModelEngine('')
    } else {
      setModelEngine(model_engine || '')
    }
    setModelFormat(model_format || '')
    setModelSize(String(model_size_in_billions) || '')
    setQuantization(quantization || '')
    setNWorker(Number(n_worker) || 1)
    setNGPU(n_gpu || 'auto')
    if (n_gpu_layers >= 0) {
      setNGPULayers(n_gpu_layers)
    } else {
      setNGPULayers(-1)
    }
    setReplica(Number(replica) || 1)
    setModelUID(model_uid || '')
    setRequestLimits(request_limits || '')
    setWorkerIp(worker_ip || '')
    setGPUIdx(gpu_idx?.join(',') || '')
    setDownloadHub(download_hub || '')
    setModelPath(model_path || '')

    let loraData = []
    peft_model_config?.lora_list?.forEach((item) => {
      loraData.push({
        lora_name: item.lora_name,
        local_path: item.local_path,
      })
    })
    setLoraArr(loraData)

    let customData = []
    for (let key in data) {
      !llmAllDataKey.includes(key) &&
        customData.push({ key: key, value: data[key] || 'none' })
    }
    setCustomArr(customData)

    if (
      model_uid ||
      request_limits ||
      worker_ip ||
      gpu_idx?.join(',') ||
      download_hub ||
      model_path
    )
      setIsOther(true)

    if (loraData.length) {
      setIsOther(true)
      setIsPeftModelConfig(true)
    }
  }

  const handleOtherHistory = (data) => {
    const {
      model_uid,
      replica,
      n_gpu,
      gpu_idx,
      worker_ip,
      download_hub,
      model_path,
      gguf_quantization,
      gguf_model_path,
      cpu_offload,
      model_type,
      peft_model_config,
    } = data
    setModelUID(model_uid || '')
    setReplica(replica || 1)
    setNGpu(n_gpu === 'auto' ? 'GPU' : 'CPU')
    setGPUIdx(gpu_idx?.join(',') || '')
    setWorkerIp(worker_ip || '')
    setDownloadHub(download_hub || '')
    setModelPath(model_path || '')
    setGgufQuantizations(gguf_quantization || '')
    setGgufModelPath(gguf_model_path || '')
    setCpuOffload(cpu_offload || false)

    if (model_type === 'image') {
      let loraData = []
      peft_model_config?.lora_list?.forEach((item) => {
        loraData.push({
          lora_name: item.lora_name,
          local_path: item.local_path,
        })
      })
      setLoraArr(loraData)

      let ImageLoraLoadData = []
      for (let key in peft_model_config?.image_lora_load_kwargs) {
        ImageLoraLoadData.push({
          key: key,
          value: peft_model_config?.image_lora_load_kwargs[key] || 'none',
        })
      }
      setImageLoraLoadArr(ImageLoraLoadData)

      let ImageLoraFuseData = []
      for (let key in peft_model_config?.image_lora_fuse_kwargs) {
        ImageLoraFuseData.push({
          key: key,
          value: peft_model_config?.image_lora_fuse_kwargs[key] || 'none',
        })
      }
      setImageLoraFuseArr(ImageLoraFuseData)

      if (
        loraData.length ||
        ImageLoraLoadData.length ||
        ImageLoraFuseData.length
      ) {
        setIsPeftModelConfig(true)
      }
    }

    let customData = []
    for (let key in data) {
      !llmAllDataKey.includes(key) &&
        customData.push({ key: key, value: data[key] || 'none' })
    }
    setCustomArr(customData)
  }

  const handleCollection = (bool) => {
    setHover(false)

    let collectionArr = JSON.parse(localStorage.getItem('collectionArr')) || []
    if (bool) {
      collectionArr.push(modelData.model_name)
    } else {
      collectionArr = collectionArr.filter(
        (item) => item !== modelData.model_name
      )
    }
    localStorage.setItem('collectionArr', JSON.stringify(collectionArr))

    onGetCollectionArr(collectionArr)
  }

  const handleDeleteChip = () => {
    const arr = JSON.parse(localStorage.getItem('historyArr'))
    const newArr = arr.filter(
      (item) => item.model_name !== modelData.model_name
    )
    localStorage.setItem('historyArr', JSON.stringify(newArr))
    setIsHistory(false)
    if (modelType === 'LLM') {
      setModelEngine('')
      setModelFormat('')
      setModelSize('')
      setQuantization('')
      setNWorker(1)
      setNGPU('auto')
      setReplica(1)
      setModelUID('')
      setRequestLimits('')
      setWorkerIp('')
      setGPUIdx('')
      setDownloadHub('')
      setModelPath('')
      setLoraArr([])
      setCustomArr([])
      setIsOther(false)
      setIsPeftModelConfig(false)
    } else {
      setModelUID('')
      setReplica(1)
      setNGpu(gpuAvailable === 0 ? 'CPU' : 'GPU')
      setGPUIdx('')
      setWorkerIp('')
      setDownloadHub('')
      setModelPath('')
      setGgufQuantizations('')
      setGgufModelPath('')
      setCpuOffload(false)
      setLoraArr([])
      setImageLoraLoadArr([])
      setImageLoraFuseArr([])
      setCustomArr([])
      setIsPeftModelConfig(false)
    }
  }

  const normalizeLanguage = (language) => {
    if (Array.isArray(language)) {
      return language.map((lang) => lang.toLowerCase())
    } else if (typeof language === 'string') {
      return [language.toLowerCase()]
    } else {
      return []
    }
  }

  const handleCommandLine = (data) => {
    if (data.model_name === modelData.model_name) {
      if (data.model_type === 'LLM') {
        handleLlmHistory(data)
      } else {
        handleOtherHistory(data)
      }
    } else {
      setOpenErrorSnackbar(true)
      setErrorSnackbarValue(t('launchModel.commandLineTip'))
    }
  }

  const isModelStartable = () => {
    return !(
      (modelType === 'LLM' &&
        (isCallingApi ||
          isUpdatingModel ||
          !(
            modelFormat &&
            modelSize &&
            modelData &&
            (quantization ||
              (!modelData.is_builtin && modelFormat !== 'pytorch'))
          ) ||
          !judgeArr(loraListArr, ['lora_name', 'local_path']) ||
          !judgeArr(imageLoraLoadKwargsArr, ['key', 'value']) ||
          !judgeArr(imageLoraFuseKwargsArr, ['key', 'value']) ||
          requestLimitsAlert ||
          GPUIdxAlert)) ||
      ((modelType === 'embedding' || modelType === 'rerank') && GPUIdxAlert) ||
      !judgeArr(customParametersArr, ['key', 'value'])
    )
  }

  // Set two different states based on mouse hover
  return (
    <>
      <Paper
        id={modelData.model_name}
        className="container"
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        onClick={() => {
          if (!selected && !customDeleted) {
            const data = handleGetHistory()
            if (data?.model_name) setIsHistory(true)
            setSelected(true)
            if (modelType === 'LLM') {
              getModelEngine(modelData.model_name)
            } else if (data?.model_name) {
              handleOtherHistory(data)
            }
          }
        }}
        elevation={hover ? 24 : 4}
      >
        {modelType === 'LLM' ? (
          <Box className="descriptionCard">
            {is_custom && (
              <div className="cardTitle">
                <TitleTypography value={modelData.model_name} />
                <div className="iconButtonBox">
                  <Tooltip title={t('launchModel.edit')} placement="top">
                    <IconButton
                      aria-label="show"
                      onClick={(e) => {
                        e.stopPropagation()
                        setIsJsonShow(true)
                      }}
                    >
                      <EditNote />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title={t('launchModel.delete')} placement="top">
                    <IconButton
                      aria-label="delete"
                      onClick={(e) => {
                        e.stopPropagation()
                        setIsDeleteCustomModel(true)
                      }}
                    >
                      <Delete />
                    </IconButton>
                  </Tooltip>
                </div>
              </div>
            )}
            {!is_custom && (
              <div className="cardTitle">
                <TitleTypography value={modelData.model_name} />
                <div className="iconButtonBox">
                  {JSON.parse(localStorage.getItem('collectionArr'))?.includes(
                    modelData.model_name
                  ) ? (
                    <Tooltip
                      title={t('launchModel.unfavorite')}
                      placement="top"
                    >
                      <IconButton
                        aria-label="collection"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleCollection(false)
                        }}
                      >
                        <Grade style={{ color: 'rgb(255, 206, 0)' }} />
                      </IconButton>
                    </Tooltip>
                  ) : (
                    <Tooltip title={t('launchModel.favorite')} placement="top">
                      <IconButton
                        aria-label="cancellation-of-collections"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleCollection(true)
                        }}
                      >
                        <StarBorder />
                      </IconButton>
                    </Tooltip>
                  )}
                </div>
              </div>
            )}

            <Stack
              spacing={1}
              direction="row"
              useFlexGap
              flexWrap="wrap"
              sx={{ marginLeft: 1 }}
            >
              {modelData.model_lang &&
                (() => {
                  return modelData.model_lang.map((v) => {
                    return (
                      <Chip
                        key={v}
                        label={v}
                        variant="outlined"
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation()
                        }}
                      />
                    )
                  })
                })()}
              {(() => {
                if (
                  modelData.model_specs &&
                  modelData.model_specs.some((spec) => isCached(spec))
                ) {
                  return (
                    <Chip
                      label={t('launchModel.manageCachedModels')}
                      variant="outlined"
                      color="primary"
                      size="small"
                      deleteIcon={<EditNote />}
                      onDelete={handleOpenCachedList}
                      onClick={(e) => {
                        e.stopPropagation()
                        handleOpenCachedList()
                      }}
                    />
                  )
                }
              })()}
              {(() => {
                if (is_custom && customDeleted) {
                  return (
                    <Chip label="Deleted" variant="outlined" size="small" />
                  )
                }
              })()}
            </Stack>
            {modelData.model_description && (
              <p className="p" title={modelData.model_description}>
                {modelData.model_description}
              </p>
            )}

            <div className="iconRow">
              <div className="iconItem">
                <span className="boldIconText">
                  {Math.floor(modelData.context_length / 1000)}K
                </span>
                <small className="smallText">
                  {t('launchModel.contextLength')}
                </small>
              </div>
              {(() => {
                if (
                  modelData.model_ability &&
                  modelData.model_ability.includes('chat')
                ) {
                  return (
                    <div className="iconItem">
                      <ChatOutlined className="muiIcon" />
                      <small className="smallText">
                        {t('launchModel.chatModel')}
                      </small>
                    </div>
                  )
                } else if (
                  modelData.model_ability &&
                  modelData.model_ability.includes('generate')
                ) {
                  return (
                    <div className="iconItem">
                      <EditNoteOutlined className="muiIcon" />
                      <small className="smallText">
                        {t('launchModel.generateModel')}
                      </small>
                    </div>
                  )
                } else {
                  return (
                    <div className="iconItem">
                      <HelpCenterOutlined className="muiIcon" />
                      <small className="smallText">
                        {t('launchModel.otherModel')}
                      </small>
                    </div>
                  )
                }
              })()}
            </div>
          </Box>
        ) : (
          <Box className="descriptionCard">
            <div className="titleContainer">
              {is_custom && (
                <div className="cardTitle">
                  <TitleTypography value={modelData.model_name} />
                  <div className="iconButtonBox">
                    <Tooltip title={t('launchModel.edit')} placement="top">
                      <IconButton
                        aria-label="show"
                        onClick={(e) => {
                          e.stopPropagation()
                          setIsJsonShow(true)
                        }}
                      >
                        <EditNote />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title={t('launchModel.delete')} placement="top">
                      <IconButton
                        aria-label="delete"
                        onClick={(e) => {
                          e.stopPropagation()
                          setIsDeleteCustomModel(true)
                        }}
                        disabled={customDeleted}
                      >
                        <Delete />
                      </IconButton>
                    </Tooltip>
                  </div>
                </div>
              )}
              {!is_custom && (
                <div className="cardTitle">
                  <TitleTypography value={modelData.model_name} />
                  <div className="iconButtonBox">
                    {JSON.parse(
                      localStorage.getItem('collectionArr')
                    )?.includes(modelData.model_name) ? (
                      <Tooltip
                        title={t('launchModel.unfavorite')}
                        placement="top"
                      >
                        <IconButton
                          aria-label="collection"
                          onClick={(e) => {
                            e.stopPropagation()
                            handleCollection(false)
                          }}
                        >
                          <Grade style={{ color: 'rgb(255, 206, 0)' }} />
                        </IconButton>
                      </Tooltip>
                    ) : (
                      <Tooltip
                        title={t('launchModel.favorite')}
                        placement="top"
                      >
                        <IconButton
                          aria-label="cancellation-of-collections"
                          onClick={(e) => {
                            e.stopPropagation()
                            handleCollection(true)
                          }}
                        >
                          <StarBorder />
                        </IconButton>
                      </Tooltip>
                    )}
                  </div>
                </div>
              )}

              <Stack
                spacing={1}
                direction="row"
                useFlexGap
                flexWrap="wrap"
                sx={{ marginLeft: 1 }}
              >
                {(() => {
                  if (modelData.language) {
                    return normalizeLanguage(modelData.language).map((v) => {
                      return (
                        <Chip
                          key={v}
                          label={v}
                          variant="outlined"
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation()
                          }}
                        />
                      )
                    })
                  } else if (modelData.model_family) {
                    return (
                      <Chip
                        label={modelData.model_family}
                        variant="outlined"
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation()
                        }}
                      />
                    )
                  }
                })()}
                {(() => {
                  if (modelData.cache_status) {
                    return (
                      <Chip
                        label={t('launchModel.manageCachedModels')}
                        variant="outlined"
                        color="primary"
                        size="small"
                        deleteIcon={<EditNote />}
                        onDelete={handleOpenCachedList}
                        onClick={(e) => {
                          e.stopPropagation()
                          handleOpenCachedList()
                        }}
                      />
                    )
                  }
                })()}
                {(() => {
                  if (is_custom && customDeleted) {
                    return (
                      <Chip label="Deleted" variant="outlined" size="small" />
                    )
                  }
                })()}
              </Stack>
              {modelData.model_description && (
                <p className="p" title={modelData.model_description}>
                  {modelData.model_description}
                </p>
              )}
            </div>
            {modelData.dimensions && (
              <div className="iconRow">
                <div className="iconItem">
                  <span className="boldIconText">{modelData.dimensions}</span>
                  <small className="smallText">
                    {t('launchModel.dimensions')}
                  </small>
                </div>
                <div className="iconItem">
                  <span className="boldIconText">{modelData.max_tokens}</span>
                  <small className="smallText">
                    {t('launchModel.maxTokens')}
                  </small>
                </div>
              </div>
            )}
            {!selected && hover && (
              <p className="instructionText">
                {t('launchModel.clickToLaunchModel')}
              </p>
            )}
          </Box>
        )}
      </Paper>

      <DeleteDialog
        text={t('launchModel.confirmDeleteCustomModel')}
        isDelete={isDeleteCustomModel}
        onHandleIsDelete={() => setIsDeleteCustomModel(false)}
        onHandleDelete={handeCustomDelete}
      />
      <Drawer
        isOpen={selected}
        onClose={() => {
          setSelected(false)
          setHover(false)
        }}
      >
        <div className="drawerCard">
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <TitleTypography value={modelData.model_name} />
              {isHistory && (
                <Chip
                  label={t('launchModel.lastConfig')}
                  variant="outlined"
                  size="small"
                  color="primary"
                  onDelete={handleDeleteChip}
                />
              )}
            </div>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <Tooltip
                title={t('launchModel.commandLineParsing')}
                placement="top"
              >
                <ContentPasteGo
                  className="pasteText"
                  onClick={() => setIsOpenPasteDialog(true)}
                />
              </Tooltip>
              {isModelStartable() && (
                <CopyToCommandLine
                  style={{ fontSize: '30px' }}
                  modelData={selected && handleModelData()}
                  predefinedKeys={llmAllDataKey}
                  getData={handleModelData}
                />
              )}
            </div>
          </div>

          {modelType === 'LLM' ? (
            <Box
              ref={parentRef}
              className="formContainer"
              display="flex"
              flexDirection="column"
              width="100%"
              mx="auto"
            >
              <Grid rowSpacing={0} columnSpacing={1}>
                <Grid item xs={12}>
                  <FormControl variant="outlined" margin="normal" fullWidth>
                    <InputLabel id="modelEngine-label">
                      {t('launchModel.modelEngine')}
                    </InputLabel>
                    <Select
                      labelId="modelEngine-label"
                      value={modelEngine}
                      onChange={(e) => setModelEngine(e.target.value)}
                      label={t('launchModel.modelEngine')}
                    >
                      {engineOptions.map((engine) => {
                        const subArr = []
                        enginesObj[engine].forEach((item) => {
                          subArr.push(item.model_format)
                        })
                        const arr = [...new Set(subArr)]
                        const specs = modelData.model_specs.filter((spec) =>
                          arr.includes(spec.model_format)
                        )

                        const cached = specs.some((spec) => isCached(spec))

                        const displayedEngine = cached
                          ? engine + ' ' + t('launchModel.cached')
                          : engine

                        return (
                          <MenuItem key={engine} value={engine}>
                            {displayedEngine}
                          </MenuItem>
                        )
                      })}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12}>
                  <FormControl
                    variant="outlined"
                    margin="normal"
                    fullWidth
                    disabled={!modelEngine}
                  >
                    <InputLabel id="modelFormat-label">
                      {t('launchModel.modelFormat')}
                    </InputLabel>
                    <Select
                      labelId="modelFormat-label"
                      value={modelFormat}
                      onChange={(e) => setModelFormat(e.target.value)}
                      label={t('launchModel.modelFormat')}
                    >
                      {formatOptions.map((format) => {
                        const specs = modelData.model_specs.filter(
                          (spec) => spec.model_format === format
                        )

                        const cached = specs.some((spec) => isCached(spec))

                        const displayedFormat = cached
                          ? format + ' ' + t('launchModel.cached')
                          : format

                        return (
                          <MenuItem key={format} value={format}>
                            {displayedFormat}
                          </MenuItem>
                        )
                      })}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12}>
                  <FormControl
                    variant="outlined"
                    margin="normal"
                    fullWidth
                    disabled={!modelFormat}
                  >
                    <InputLabel id="modelSize-label">
                      {t('launchModel.modelSize')}
                    </InputLabel>
                    <Select
                      labelId="modelSize-label"
                      value={modelSize}
                      onChange={(e) => setModelSize(e.target.value)}
                      label={t('launchModel.modelSize')}
                    >
                      {sizeOptions.map((size) => {
                        const specs = modelData.model_specs
                          .filter((spec) => spec.model_format === modelFormat)
                          .filter(
                            (spec) => spec.model_size_in_billions === size
                          )
                        const cached = specs.some((spec) => isCached(spec))

                        const displayedSize = cached
                          ? size + ' ' + t('launchModel.cached')
                          : size

                        return (
                          <MenuItem key={size} value={size}>
                            {displayedSize}
                          </MenuItem>
                        )
                      })}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12}>
                  <FormControl
                    variant="outlined"
                    margin="normal"
                    fullWidth
                    disabled={!modelFormat || !modelSize}
                  >
                    <InputLabel id="quantization-label">
                      {t('launchModel.quantization')}
                    </InputLabel>
                    <Select
                      labelId="quantization-label"
                      value={quantization}
                      onChange={(e) => setQuantization(e.target.value)}
                      label={t('launchModel.quantization')}
                    >
                      {quantizationOptions.map((quant) => {
                        const specs = modelData.model_specs
                          .filter((spec) => spec.model_format === modelFormat)
                          .filter(
                            (spec) =>
                              spec.model_size_in_billions ===
                              convertModelSize(modelSize)
                          )

                        const spec = specs.find((s) => {
                          return s.quantizations.includes(quant)
                        })
                        const cached = Array.isArray(spec?.cache_status)
                          ? spec?.cache_status[
                              spec?.quantizations.indexOf(quant)
                            ]
                          : spec?.cache_status

                        const displayedQuant = cached
                          ? quant + ' ' + t('launchModel.cached')
                          : quant

                        return (
                          <MenuItem key={quant} value={quant}>
                            {displayedQuant}
                          </MenuItem>
                        )
                      })}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12}>
                  {modelFormat !== 'ggufv2' && modelFormat !== 'ggmlv3' ? (
                    <FormControl
                      variant="outlined"
                      margin="normal"
                      fullWidth
                      disabled={!modelFormat || !modelSize || !quantization}
                    >
                      <InputLabel id="n-gpu-label">
                        {t(
                          enginesWithNWorker.includes(modelEngine)
                            ? 'launchModel.nGPUPerWorker'
                            : 'launchModel.nGPU'
                        )}
                      </InputLabel>
                      <Select
                        labelId="n-gpu-label"
                        value={nGPU}
                        onChange={(e) => setNGPU(e.target.value)}
                        label={t(
                          enginesWithNWorker.includes(modelEngine)
                            ? 'launchModel.nGPUPerWorker'
                            : 'launchModel.nGPU'
                        )}
                      >
                        {getNGPURange().map((v) => {
                          return (
                            <MenuItem key={v} value={v}>
                              {v}
                            </MenuItem>
                          )
                        })}
                      </Select>
                    </FormControl>
                  ) : (
                    <FormControl variant="outlined" margin="normal" fullWidth>
                      <TextField
                        disabled={!modelFormat || !modelSize || !quantization}
                        type="number"
                        label={t('launchModel.nGpuLayers')}
                        InputProps={{
                          inputProps: {
                            min: -1,
                          },
                        }}
                        value={nGPULayers}
                        onChange={(e) =>
                          setNGPULayers(parseInt(e.target.value, 10))
                        }
                      />
                    </FormControl>
                  )}
                </Grid>
                <Grid item xs={12}>
                  <FormControl variant="outlined" margin="normal" fullWidth>
                    <TextField
                      disabled={!modelFormat || !modelSize || !quantization}
                      type="number"
                      InputProps={{
                        inputProps: {
                          min: 1,
                        },
                      }}
                      label={t('launchModel.replica')}
                      value={replica}
                      onChange={(e) => setReplica(parseInt(e.target.value, 10))}
                    />
                  </FormControl>
                </Grid>
                <ListItemButton onClick={() => setIsOther(!isOther)}>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <ListItemText
                      primary={t('launchModel.optionalConfigurations')}
                      style={{ marginRight: 10 }}
                    />
                    {isOther ? <ExpandLess /> : <ExpandMore />}
                  </div>
                </ListItemButton>
                <Collapse in={isOther} timeout="auto" unmountOnExit>
                  <Grid item xs={12}>
                    <FormControl variant="outlined" margin="normal" fullWidth>
                      <TextField
                        variant="outlined"
                        value={modelUID}
                        label={t('launchModel.modelUID.optional')}
                        onChange={(e) => setModelUID(e.target.value)}
                      />
                    </FormControl>
                  </Grid>
                  <Grid item xs={12}>
                    <FormControl variant="outlined" margin="normal" fullWidth>
                      <TextField
                        value={requestLimits}
                        label={t('launchModel.requestLimits.optional')}
                        onChange={(e) => {
                          setRequestLimitsAlert(false)
                          setRequestLimits(e.target.value)
                          if (
                            e.target.value !== '' &&
                            (!Number(e.target.value) ||
                              Number(e.target.value) < 1 ||
                              parseInt(e.target.value) !==
                                parseFloat(e.target.value))
                          ) {
                            setRequestLimitsAlert(true)
                          }
                        }}
                      />
                      {requestLimitsAlert && (
                        <Alert severity="error">
                          {t('launchModel.enterIntegerGreaterThanZero')}
                        </Alert>
                      )}
                    </FormControl>
                  </Grid>
                  <Grid item xs={12}>
                    {enginesWithNWorker.includes(modelEngine) && (
                      <FormControl variant="outlined" margin="normal" fullWidth>
                        <TextField
                          type="number"
                          InputProps={{
                            inputProps: {
                              min: 1,
                            },
                          }}
                          label={t('launchModel.workerCount.optional')}
                          value={nWorker}
                          onChange={(e) =>
                            setNWorker(parseInt(e.target.value, 10))
                          }
                        />
                      </FormControl>
                    )}
                  </Grid>
                  <Grid item xs={12}>
                    <FormControl variant="outlined" margin="normal" fullWidth>
                      <TextField
                        variant="outlined"
                        value={workerIp}
                        label={t('launchModel.workerIp.optional')}
                        onChange={(e) => setWorkerIp(e.target.value)}
                      />
                    </FormControl>
                  </Grid>
                  <Grid item xs={12}>
                    <FormControl variant="outlined" margin="normal" fullWidth>
                      <TextField
                        value={GPUIdx}
                        label={t('launchModel.GPUIdx.optional')}
                        onChange={(e) => {
                          setGPUIdxAlert(false)
                          setGPUIdx(e.target.value)
                          const regular = /^\d+(?:,\d+)*$/
                          if (
                            e.target.value !== '' &&
                            !regular.test(e.target.value)
                          ) {
                            setGPUIdxAlert(true)
                          }
                        }}
                      />
                      {GPUIdxAlert && (
                        <Alert severity="error">
                          {t('launchModel.enterCommaSeparatedNumbers')}
                        </Alert>
                      )}
                    </FormControl>
                  </Grid>
                  <Grid item xs={12}>
                    <FormControl variant="outlined" margin="normal" fullWidth>
                      <InputLabel id="quantization-label">
                        {t('launchModel.downloadHub.optional')}
                      </InputLabel>
                      <Select
                        labelId="download_hub-label"
                        value={downloadHub}
                        onChange={(e) => {
                          e.target.value === 'none'
                            ? setDownloadHub('')
                            : setDownloadHub(e.target.value)
                        }}
                        label={t('launchModel.downloadHub.optional')}
                      >
                        {(csghubArr.includes(modelData.model_name)
                          ? [
                              'none',
                              'huggingface',
                              'modelscope',
                              'openmind_hub',
                              'csghub',
                            ]
                          : [
                              'none',
                              'huggingface',
                              'modelscope',
                              'openmind_hub',
                            ]
                        ).map((item) => {
                          return (
                            <MenuItem key={item} value={item}>
                              {item}
                            </MenuItem>
                          )
                        })}
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12}>
                    <FormControl variant="outlined" margin="normal" fullWidth>
                      <TextField
                        variant="outlined"
                        value={modelPath}
                        label={t('launchModel.modelPath.optional')}
                        onChange={(e) => setModelPath(e.target.value)}
                      />
                    </FormControl>
                  </Grid>
                  <ListItemButton
                    onClick={() => setIsPeftModelConfig(!isPeftModelConfig)}
                  >
                    <div style={{ display: 'flex', alignItems: 'center' }}>
                      <ListItemText
                        primary={t('launchModel.loraConfig')}
                        style={{ marginRight: 10 }}
                      />
                      {isPeftModelConfig ? <ExpandLess /> : <ExpandMore />}
                    </div>
                  </ListItemButton>
                  <Collapse
                    in={isPeftModelConfig}
                    timeout="auto"
                    unmountOnExit
                    style={{ marginLeft: '30px' }}
                  >
                    <AddPair
                      customData={{
                        title: t('launchModel.loraModelConfig'),
                        key: 'lora_name',
                        value: 'local_path',
                      }}
                      onGetArr={(arr) => {
                        setLoraListArr(arr)
                      }}
                      onJudgeArr={judgeArr}
                      pairData={loraArr}
                    />
                  </Collapse>
                </Collapse>
                <AddPair
                  customData={{
                    title: `${t(
                      'launchModel.additionalParametersForInferenceEngine'
                    )}${modelEngine ? ': ' + modelEngine : ''}`,
                    key: 'key',
                    value: 'value',
                  }}
                  onGetArr={(arr) => {
                    setCustomParametersArr(arr)
                  }}
                  onJudgeArr={judgeArr}
                  pairData={customArr}
                  tipOptions={
                    additionalParameterTipList[modelEngine?.toLocaleLowerCase()]
                  }
                />
              </Grid>
            </Box>
          ) : (
            <Box
              ref={parentRef}
              className="formContainer"
              display="flex"
              flexDirection="column"
              width="100%"
              mx="auto"
            >
              <FormControl variant="outlined" margin="normal" fullWidth>
                <TextField
                  variant="outlined"
                  value={modelUID}
                  label={t('launchModel.modelUID.optional')}
                  onChange={(e) => setModelUID(e.target.value)}
                />
                <TextField
                  style={{ marginTop: '25px' }}
                  type="number"
                  InputProps={{
                    inputProps: {
                      min: 1,
                    },
                  }}
                  label={t('launchModel.replica')}
                  value={replica}
                  onChange={(e) => setReplica(parseInt(e.target.value, 10))}
                />
                <FormControl variant="outlined" margin="normal" fullWidth>
                  <InputLabel id="n-gpu-label">
                    {t('launchModel.device')}
                  </InputLabel>
                  <Select
                    labelId="n-gpu-label"
                    value={nGpu}
                    onChange={(e) => setNGpu(e.target.value)}
                    label={t('launchModel.nGPU')}
                  >
                    {getNewNGPURange().map((v) => {
                      return (
                        <MenuItem key={v} value={v}>
                          {v}
                        </MenuItem>
                      )
                    })}
                  </Select>
                </FormControl>
                {nGpu === 'GPU' && (
                  <FormControl variant="outlined" margin="normal" fullWidth>
                    <TextField
                      value={GPUIdx}
                      label={t('launchModel.GPUIdx')}
                      onChange={(e) => {
                        setGPUIdxAlert(false)
                        setGPUIdx(e.target.value)
                        const regular = /^\d+(?:,\d+)*$/
                        if (
                          e.target.value !== '' &&
                          !regular.test(e.target.value)
                        ) {
                          setGPUIdxAlert(true)
                        }
                      }}
                    />
                    {GPUIdxAlert && (
                      <Alert severity="error">
                        {t('launchModel.enterCommaSeparatedNumbers')}
                      </Alert>
                    )}
                  </FormControl>
                )}
                <FormControl variant="outlined" margin="normal" fullWidth>
                  <TextField
                    variant="outlined"
                    value={workerIp}
                    label={t('launchModel.workerIp')}
                    onChange={(e) => setWorkerIp(e.target.value)}
                  />
                </FormControl>
                <FormControl variant="outlined" margin="normal" fullWidth>
                  <InputLabel id="quantization-label">
                    {t('launchModel.downloadHub.optional')}
                  </InputLabel>
                  <Select
                    labelId="download_hub-label"
                    value={downloadHub}
                    onChange={(e) => {
                      e.target.value === 'none'
                        ? setDownloadHub('')
                        : setDownloadHub(e.target.value)
                    }}
                    label={t('launchModel.downloadHub.optional')}
                  >
                    {['none', 'huggingface', 'modelscope', 'openmind_hub'].map(
                      (item) => {
                        return (
                          <MenuItem key={item} value={item}>
                            {item}
                          </MenuItem>
                        )
                      }
                    )}
                  </Select>
                </FormControl>
                <FormControl variant="outlined" margin="normal" fullWidth>
                  <TextField
                    variant="outlined"
                    value={modelPath}
                    label={t('launchModel.modelPath.optional')}
                    onChange={(e) => setModelPath(e.target.value)}
                  />
                </FormControl>
                {modelData.gguf_quantizations && (
                  <FormControl variant="outlined" margin="normal" fullWidth>
                    <InputLabel id="quantization-label">
                      {t('launchModel.GGUFQuantization.optional')}
                    </InputLabel>
                    <Select
                      labelId="gguf_quantizations-label"
                      value={ggufQuantizations}
                      onChange={(e) => {
                        e.target.value === 'none'
                          ? setGgufQuantizations('')
                          : setGgufQuantizations(e.target.value)
                      }}
                      label={t('launchModel.GGUFQuantization.optional')}
                    >
                      {['none', ...modelData.gguf_quantizations].map((item) => {
                        return (
                          <MenuItem key={item} value={item}>
                            {item}
                          </MenuItem>
                        )
                      })}
                    </Select>
                  </FormControl>
                )}
                {modelData.gguf_quantizations && (
                  <FormControl variant="outlined" margin="normal" fullWidth>
                    <TextField
                      variant="outlined"
                      value={ggufModelPath}
                      label={t('launchModel.GGUFModelPath.optional')}
                      onChange={(e) => setGgufModelPath(e.target.value)}
                    />
                  </FormControl>
                )}
                {modelType === 'image' && (
                  <>
                    <div
                      style={{
                        marginBlock: '10px',
                      }}
                    >
                      <FormControlLabel
                        label={
                          <div>
                            <span>{t('launchModel.CPUOffload')}</span>
                            <Tooltip
                              title={t('launchModel.CPUOffload.tip')}
                              placement="top"
                            >
                              <IconButton>
                                <HelpOutline />
                              </IconButton>
                            </Tooltip>
                          </div>
                        }
                        labelPlacement="start"
                        control={<Switch checked={cpuOffload} />}
                        onChange={(e) => {
                          setCpuOffload(e.target.checked)
                        }}
                      />
                    </div>
                    <ListItemButton
                      onClick={() => setIsPeftModelConfig(!isPeftModelConfig)}
                    >
                      <div style={{ display: 'flex', alignItems: 'center' }}>
                        <ListItemText
                          primary={t('launchModel.loraConfig')}
                          style={{ marginRight: 10 }}
                        />
                        {isPeftModelConfig ? <ExpandLess /> : <ExpandMore />}
                      </div>
                    </ListItemButton>
                    <Collapse
                      in={isPeftModelConfig}
                      timeout="auto"
                      unmountOnExit
                      style={{ marginLeft: '30px' }}
                    >
                      <AddPair
                        customData={{
                          title: t('launchModel.loraModelConfig'),
                          key: 'lora_name',
                          value: 'local_path',
                        }}
                        onGetArr={(arr) => {
                          setLoraListArr(arr)
                        }}
                        onJudgeArr={judgeArr}
                        pairData={loraArr}
                      />
                      <AddPair
                        customData={{
                          title: t('launchModel.loraLoadKwargsForImageModel'),
                          key: 'key',
                          value: 'value',
                        }}
                        onGetArr={(arr) => {
                          setImageLoraLoadKwargsArr(arr)
                        }}
                        onJudgeArr={judgeArr}
                        pairData={imageLoraLoadArr}
                      />
                      <AddPair
                        customData={{
                          title: t('launchModel.loraFuseKwargsForImageModel'),
                          key: 'key',
                          value: 'value',
                        }}
                        onGetArr={(arr) => {
                          setImageLoraFuseKwargsArr(arr)
                        }}
                        onJudgeArr={judgeArr}
                        pairData={imageLoraFuseArr}
                      />
                    </Collapse>
                  </>
                )}
                <AddPair
                  customData={{
                    title: t(
                      'launchModel.additionalParametersForInferenceEngine'
                    ),
                    key: 'key',
                    value: 'value',
                  }}
                  onGetArr={(arr) => {
                    setCustomParametersArr(arr)
                  }}
                  onJudgeArr={judgeArr}
                  pairData={customArr}
                />
              </FormControl>
            </Box>
          )}
          <Box className="buttonsContainer">
            <button
              title={t('launchModel.launch')}
              className="buttonContainer"
              onClick={() => launchModel(url, modelData)}
              disabled={!isModelStartable()}
            >
              {(() => {
                if (isCallingApi || isUpdatingModel) {
                  return (
                    <Box
                      className="buttonItem"
                      style={{
                        backgroundColor: '#f2f2f2',
                      }}
                    >
                      <CircularProgress
                        size="20px"
                        sx={{
                          color: '#000000',
                        }}
                      />
                    </Box>
                  )
                } else if (
                  !(
                    modelFormat &&
                    modelSize &&
                    modelData &&
                    (quantization ||
                      (!modelData.is_builtin && modelFormat !== 'pytorch'))
                  )
                ) {
                  return (
                    <Box
                      className="buttonItem"
                      style={{
                        backgroundColor: '#f2f2f2',
                      }}
                    >
                      <RocketLaunchOutlined size="20px" />
                    </Box>
                  )
                } else {
                  return (
                    <Box className="buttonItem">
                      <RocketLaunchOutlined color="#000000" size="20px" />
                    </Box>
                  )
                }
              })()}
            </button>
            <button
              title={t('launchModel.goBack')}
              className="buttonContainer"
              onClick={() => {
                setSelected(false)
                setHover(false)
              }}
            >
              <Box className="buttonItem">
                <UndoOutlined color="#000000" size="20px" />
              </Box>
            </button>
          </Box>
        </div>
      </Drawer>
      <Backdrop
        sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
        open={isJsonShow}
      >
        <div className="jsonDialog">
          <div className="jsonDialog-title">
            <div className="title-name">{modelData.model_name}</div>
            <CopyComponent
              tip={t('launchModel.copyJson')}
              text={JSON.stringify(modelData, null, 4)}
            />
          </div>
          <div className="main-box">
            <textarea
              readOnly
              className="textarea-box"
              value={JSON.stringify(modelData, null, 4)}
            />
          </div>
          <div className="but-box">
            <Button
              onClick={() => {
                setIsJsonShow(false)
              }}
              style={{ marginRight: 30 }}
            >
              {t('launchModel.cancel')}
            </Button>
            <Button onClick={handleJsonDataPresentation}>
              {t('launchModel.edit')}
            </Button>
          </div>
        </div>
      </Backdrop>
      <Snackbar
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        open={openSnackbar}
        onClose={() => setOpenSnackbar(false)}
        message={t('launchModel.fillCompleteParametersBeforeAdding')}
      />
      <Snackbar
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        open={openErrorSnackbar}
        onClose={() => setOpenErrorSnackbar(false)}
      >
        <Alert severity="error" variant="filled" sx={{ width: '100%' }}>
          {errorSnackbarValue}
        </Alert>
      </Snackbar>

      <Backdrop
        sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
        open={isOpenCachedList}
      >
        <div className="dialogBox">
          <div className="dialogTitle">
            <div className="dialogTitle-model_name">{modelData.model_name}</div>
            <Close
              style={{ cursor: 'pointer' }}
              onClick={handleCloseCachedList}
            />
          </div>
          <TableContainer component={Paper}>
            <Table
              sx={{ minWidth: 500 }}
              style={{ height: '500px', width: '100%' }}
              stickyHeader
              aria-label="simple pagination table"
            >
              <TableHead>
                <TableRow>
                  {modelType === 'LLM' && (
                    <>
                      <TableCell align="left">
                        {t('launchModel.model_format')}
                      </TableCell>
                      <TableCell align="left">
                        {t('launchModel.model_size_in_billions')}
                      </TableCell>
                      <TableCell align="left">
                        {t('launchModel.quantizations')}
                      </TableCell>
                    </>
                  )}
                  <TableCell align="left" style={{ width: 192 }}>
                    {t('launchModel.real_path')}
                  </TableCell>
                  <TableCell align="left" style={{ width: 46 }}></TableCell>
                  <TableCell align="left" style={{ width: 192 }}>
                    {t('launchModel.path')}
                  </TableCell>
                  <TableCell align="left" style={{ width: 46 }}></TableCell>
                  <TableCell
                    align="left"
                    style={{ whiteSpace: 'nowrap', minWidth: 116 }}
                  >
                    {t('launchModel.ipAddress')}
                  </TableCell>
                  <TableCell align="left">
                    {t('launchModel.operation')}
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody style={{ position: 'relative' }}>
                {cachedListArr.slice(page * 5, page * 5 + 5).map((row) => (
                  <StyledTableRow
                    style={{ maxHeight: 90 }}
                    key={row.model_name}
                  >
                    {modelType === 'LLM' && (
                      <>
                        <TableCell component="th" scope="row">
                          {row.model_format === null ? '—' : row.model_format}
                        </TableCell>
                        <TableCell>
                          {row.model_size_in_billions === null
                            ? '—'
                            : row.model_size_in_billions}
                        </TableCell>
                        <TableCell>
                          {row.quantization === null ? '—' : row.quantization}
                        </TableCell>
                      </>
                    )}
                    <TableCell>
                      <Tooltip title={row.real_path}>
                        <div
                          className={
                            modelType === 'LLM' ? 'pathBox' : 'pathBox pathBox2'
                          }
                        >
                          {row.real_path}
                        </div>
                      </Tooltip>
                    </TableCell>
                    <TableCell>
                      <CopyComponent
                        tip={t('launchModel.copyRealPath')}
                        text={row.real_path}
                      />
                    </TableCell>
                    <TableCell>
                      <Tooltip title={row.path}>
                        <div
                          className={
                            modelType === 'LLM' ? 'pathBox' : 'pathBox pathBox2'
                          }
                        >
                          {row.path}
                        </div>
                      </Tooltip>
                    </TableCell>
                    <TableCell>
                      <CopyComponent
                        tip={t('launchModel.copyPath')}
                        text={row.path}
                      />
                    </TableCell>
                    <TableCell>{row.actor_ip_address}</TableCell>
                    <TableCell align={modelType === 'LLM' ? 'center' : 'left'}>
                      <IconButton
                        aria-label="delete"
                        size="large"
                        onClick={() =>
                          handleOpenDeleteCachedDialog(
                            row.real_path,
                            row.model_version
                          )
                        }
                      >
                        <Delete />
                      </IconButton>
                    </TableCell>
                  </StyledTableRow>
                ))}
                {emptyRows > 0 && (
                  <TableRow style={{ height: 89.4 * emptyRows }}>
                    <TableCell />
                  </TableRow>
                )}
                {cachedListArr.length === 0 && (
                  <div className="empty">{t('launchModel.noCacheForNow')}</div>
                )}
              </TableBody>
            </Table>
          </TableContainer>
          <TablePagination
            style={{ float: 'right' }}
            rowsPerPageOptions={[5]}
            count={cachedListArr.length}
            rowsPerPage={5}
            page={page}
            onPageChange={handleChangePage}
          />
        </div>
      </Backdrop>
      <DeleteDialog
        text={t('launchModel.confirmDeleteCacheFiles')}
        isDelete={isDeleteCached}
        onHandleIsDelete={() => setIsDeleteCached(false)}
        onHandleDelete={handleDeleteCached}
      />
      <PasteDialog
        open={isOpenPasteDialog}
        onHandleClose={() => setIsOpenPasteDialog(false)}
        onHandleCommandLine={handleCommandLine}
      />
    </>
  )
}

export default ModelCard
