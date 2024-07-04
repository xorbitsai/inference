import './styles/modelCardStyle.css'

import {
  ChatOutlined,
  Close,
  Delete,
  EditNote,
  EditNoteOutlined,
  ExpandLess,
  ExpandMore,
  Grade,
  HelpCenterOutlined,
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
  Drawer,
  FormControl,
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
import { useNavigate } from 'react-router-dom'

import { ApiContext } from '../../components/apiContext'
import CopyComponent from '../../components/copyComponent/copyComponent'
import DeleteDialog from '../../components/deleteDialog'
import fetchWrapper from '../../components/fetchWrapper'
import TitleTypography from '../../components/titleTypography'
import AddPair from './components/addPair'

const llmAllDataKey = [
  'model_uid',
  'model_name',
  'model_type',
  'model_engine',
  'model_format',
  'model_size_in_billions',
  'quantization',
  'n_gpu',
  'replica',
  'request_limits',
  'worker_ip',
  'gpu_idx',
  'peft_model_config',
]

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
  const [nGPU, setNGPU] = useState('auto')
  const [nGpu, setNGpu] = useState(gpuAvailable === 0 ? 'CPU' : 'GPU')
  const [nGPULayers, setNGPULayers] = useState(-1)
  const [replica, setReplica] = useState(1)
  const [requestLimits, setRequestLimits] = useState('')
  const [workerIp, setWorkerIp] = useState('')
  const [GPUIdx, setGPUIdx] = useState('')

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

  const parentRef = useRef(null)

  const range = (start, end) => {
    return new Array(end - start + 1).fill(undefined).map((_, i) => i + start)
  }

  const isCached = (spec) => {
    if (spec.model_format === 'pytorch') {
      return spec.cache_status && spec.cache_status === true
    } else {
      return spec.cache_status && spec.cache_status.some((cs) => cs)
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
    if (keyArr.length) {
      handleLlmHistory()
    }
  }, [enginesObj])

  useEffect(() => {
    if (modelEngine) {
      const format = [
        ...new Set(enginesObj[modelEngine].map((item) => item.model_format)),
      ]
      setFormatOptions(format)
      if (!isHistory || !format.includes(modelFormat)) {
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
      if (!isHistory || !sizes.includes(Number(modelSize))) {
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
      if (!isHistory || !quants.includes(quantization)) {
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

  const launchModel = () => {
    if (isCallingApi || isUpdatingModel) {
      return
    }

    setIsCallingApi(true)

    const modelDataWithID_LLM = {
      // If user does not fill model_uid, pass null (None) to server and server generates it.
      model_uid: modelUID.trim() === '' ? null : modelUID.trim(),
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
        String(requestLimits).trim() === ''
          ? null
          : Number(String(requestLimits).trim()),
      worker_ip: workerIp.trim() === '' ? null : workerIp.trim(),
      gpu_idx: GPUIdx.trim() === '' ? null : handleGPUIdx(GPUIdx.trim()),
    }

    let modelDataWithID_other = {
      model_uid: modelUID.trim() === '' ? null : modelUID.trim(),
      model_name: modelData.model_name,
      model_type: modelType,
    }

    if (modelType === 'embedding' || modelType === 'rerank') {
      modelDataWithID_other = {
        ...modelDataWithID_other,
        replica: replica,
        n_gpu: nGpu === 'GPU' ? 'auto' : null,
        worker_ip: workerIp.trim() === '' ? null : workerIp.trim(),
        gpu_idx: GPUIdx.trim() === '' ? null : handleGPUIdx(GPUIdx.trim()),
      }
    }

    if (nGPULayers >= 0) {
      modelDataWithID_LLM.n_gpu_layers = nGPULayers
    }

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
      modelDataWithID_LLM['peft_model_config'] = peft_model_config
    }

    if (customParametersArr.length) {
      customParametersArr.forEach((item) => {
        modelDataWithID_LLM[item.key] = handleValueType(item.value)
      })
    }

    const modelDataWithID =
      modelType === 'LLM' ? modelDataWithID_LLM : modelDataWithID_other

    // First fetcher request to initiate the model
    fetchWrapper
      .post('/v1/models', modelDataWithID)
      .then(() => {
        navigate(`/running_models/${modelType}`)
        sessionStorage.setItem(
          'runningModelType',
          `/running_models/${modelType}`
        )

        if (
          isHistory ||
          ((modelType === 'embedding' || modelType === 'rerank') &&
            (modelUID !== '' || replica !== 1 || workerIp !== '')) ||
          ((modelType === 'image' || modelType === 'audio') &&
            modelUID !== '') ||
          modelType === 'LLM'
        ) {
          let historyArr = JSON.parse(localStorage.getItem('historyArr')) || []
          if (!historyArr.some((item) => deepEqual(item, modelDataWithID))) {
            historyArr = historyArr.filter(
              (item) => item.model_name !== modelDataWithID.model_name
            )
            historyArr.push(modelDataWithID)
          }
          localStorage.setItem('historyArr', JSON.stringify(historyArr))
        }
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

  const handleGPUIdx = (data) => {
    const arr = []
    data.split(',').forEach((item) => {
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
    return historyArr.filter((item) => item.model_name === modelData.model_name)
  }

  const handleLlmHistory = () => {
    const arr = handleGetHistory()
    if (arr.length) {
      const {
        model_engine,
        model_format,
        model_size_in_billions,
        quantization,
        n_gpu,
        replica,
        model_uid,
        request_limits,
        worker_ip,
        gpu_idx,
        peft_model_config,
      } = arr[0]

      setModelEngine(model_engine || '')
      setModelFormat(model_format || '')
      setModelSize(String(model_size_in_billions) || '')
      setQuantization(quantization || '')
      setNGPU(n_gpu || 'auto')
      setReplica(replica || 1)
      setModelUID(model_uid || '')
      setRequestLimits(request_limits || '')
      setWorkerIp(worker_ip || '')
      setGPUIdx(gpu_idx?.join(',') || '')

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
          value: peft_model_config?.image_lora_load_kwargs[key],
        })
      }
      setImageLoraLoadArr(ImageLoraLoadData)

      let ImageLoraFuseData = []
      for (let key in peft_model_config?.image_lora_fuse_kwargs) {
        ImageLoraFuseData.push({
          key: key,
          value: peft_model_config?.image_lora_fuse_kwargs[key],
        })
      }
      setImageLoraFuseArr(ImageLoraFuseData)

      let customData = []
      for (let key in arr[0]) {
        !llmAllDataKey.includes(key) &&
          customData.push({ key: key, value: arr[0][key] })
      }
      setCustomArr(customData)

      if (model_uid || request_limits || worker_ip || gpu_idx?.join(','))
        setIsOther(true)

      if (
        loraData.length ||
        ImageLoraLoadData.length ||
        ImageLoraFuseData.length
      ) {
        setIsOther(true)
        setIsPeftModelConfig(true)
      }
    }
  }

  const handleOtherHistory = () => {
    const arr = handleGetHistory()
    if (arr.length) {
      if (modelType === 'embedding' || modelType === 'rerank') {
        setModelUID(arr[0].model_uid || '')
        setReplica(arr[0].replica || 1)
        setWorkerIp(arr[0].worker_ip || '')
      } else {
        setModelUID(arr[0].model_uid || '')
      }
    }
  }

  const deepEqual = (obj1, obj2) => {
    if (obj1 === obj2) return true
    if (
      typeof obj1 !== 'object' ||
      typeof obj2 !== 'object' ||
      obj1 == null ||
      obj2 == null
    ) {
      return false
    }

    let keysA = Object.keys(obj1)
    let keysB = Object.keys(obj2)
    if (keysA.length !== keysB.length) return false
    for (let key of keysA) {
      if (!keysB.includes(key) || !deepEqual(obj1[key], obj2[key])) {
        return false
      }
    }
    return true
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
      setNGPU('auto')
      setReplica(1)
      setModelUID('')
      setRequestLimits('')
      setWorkerIp('')
      setGPUIdx('')
      setLoraArr([])
      setImageLoraLoadArr([])
      setImageLoraFuseArr([])
      setCustomArr([])
      setIsOther(false)
      setIsPeftModelConfig(false)
    } else if (modelType === 'embedding' || modelType === 'rerank') {
      setModelUID('')
      setReplica(1)
      setWorkerIp('')
    } else {
      setModelUID('')
    }
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
            const arr = handleGetHistory()
            if (arr.length) setIsHistory(true)
            setSelected(true)
            if (modelType === 'LLM') {
              getModelEngine(modelData.model_name)
            } else {
              handleOtherHistory()
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
                  <Tooltip title={'Edit'} placement="top">
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
                  <Tooltip title={'delete'} placement="top">
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
                    <Tooltip title={'Unfavorite'} placement="top">
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
                    <Tooltip title={'Favorite'} placement="top">
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
                      <Chip key={v} label={v} variant="outlined" size="small" />
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
                      label="Cached"
                      variant="outlined"
                      size="small"
                      deleteIcon={<EditNote />}
                      onDelete={handleOpenCachedList}
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
                <small className="smallText">context length</small>
              </div>
              {(() => {
                if (
                  modelData.model_ability &&
                  modelData.model_ability.includes('chat')
                ) {
                  return (
                    <div className="iconItem">
                      <ChatOutlined className="muiIcon" />
                      <small className="smallText">chat model</small>
                    </div>
                  )
                } else if (
                  modelData.model_ability &&
                  modelData.model_ability.includes('generate')
                ) {
                  return (
                    <div className="iconItem">
                      <EditNoteOutlined className="muiIcon" />
                      <small className="smallText">generate model</small>
                    </div>
                  )
                } else {
                  return (
                    <div className="iconItem">
                      <HelpCenterOutlined className="muiIcon" />
                      <small className="smallText">other model</small>
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
                    <Tooltip title={'Edit'} placement="top">
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
                    <Tooltip title={'delete'} placement="top">
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
                      <Tooltip title={'Unfavorite'} placement="top">
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
                      <Tooltip title={'Favorite'} placement="top">
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
                    return modelData.language.map((v) => {
                      return <Chip label={v} variant="outlined" size="small" />
                    })
                  } else if (modelData.model_family) {
                    return (
                      <Chip
                        label={modelData.model_family}
                        variant="outlined"
                        size="small"
                      />
                    )
                  }
                })()}
                {(() => {
                  if (modelData.cache_status) {
                    return (
                      <Chip
                        label="Cached"
                        variant="outlined"
                        size="small"
                        deleteIcon={<EditNote />}
                        onDelete={handleOpenCachedList}
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
            </div>
            {modelData.dimensions && (
              <div className="iconRow">
                <div className="iconItem">
                  <span className="boldIconText">{modelData.dimensions}</span>
                  <small className="smallText">dimensions</small>
                </div>
                <div className="iconItem">
                  <span className="boldIconText">{modelData.max_tokens}</span>
                  <small className="smallText">max tokens</small>
                </div>
              </div>
            )}
            {!selected && hover && (
              <p className="instructionText">
                Click with mouse to launch the model
              </p>
            )}
          </Box>
        )}
      </Paper>

      <DeleteDialog
        text={
          'Are you sure to delete this custom model? This behavior is irreversible.'
        }
        isDelete={isDeleteCustomModel}
        onHandleIsDelete={() => setIsDeleteCustomModel(false)}
        onHandleDelete={handeCustomDelete}
      />
      <Drawer
        open={selected}
        onClose={() => {
          setSelected(false)
          setHover(false)
        }}
        anchor={'right'}
      >
        <div className="drawerCard">
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <TitleTypography value={modelData.model_name} />
            {isHistory && (
              <Chip
                label="Last Config"
                variant="outlined"
                size="small"
                color="primary"
                onDelete={handleDeleteChip}
              />
            )}
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
                    <InputLabel id="modelEngine-label">Model Engine</InputLabel>
                    <Select
                      labelId="modelEngine-label"
                      value={modelEngine}
                      onChange={(e) => setModelEngine(e.target.value)}
                      label="Model Engine"
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
                          ? engine + ' (cached)'
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
                    <InputLabel id="modelFormat-label">Model Format</InputLabel>
                    <Select
                      labelId="modelFormat-label"
                      value={modelFormat}
                      onChange={(e) => setModelFormat(e.target.value)}
                      label="Model Format"
                    >
                      {formatOptions.map((format) => {
                        const specs = modelData.model_specs.filter(
                          (spec) => spec.model_format === format
                        )

                        const cached = specs.some((spec) => isCached(spec))

                        const displayedFormat = cached
                          ? format + ' (cached)'
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
                    <InputLabel id="modelSize-label">Model Size</InputLabel>
                    <Select
                      labelId="modelSize-label"
                      value={modelSize}
                      onChange={(e) => setModelSize(e.target.value)}
                      label="Model Size"
                    >
                      {sizeOptions.map((size) => {
                        const specs = modelData.model_specs
                          .filter((spec) => spec.model_format === modelFormat)
                          .filter(
                            (spec) => spec.model_size_in_billions === size
                          )
                        const cached = specs.some((spec) => isCached(spec))

                        const displayedSize = cached ? size + ' (cached)' : size

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
                      Quantization
                    </InputLabel>
                    <Select
                      labelId="quantization-label"
                      value={quantization}
                      onChange={(e) => setQuantization(e.target.value)}
                      label="Quantization"
                    >
                      {quantizationOptions.map((quant, index) => {
                        const specs = modelData.model_specs
                          .filter((spec) => spec.model_format === modelFormat)
                          .filter(
                            (spec) =>
                              spec.model_size_in_billions ===
                              convertModelSize(modelSize)
                          )

                        const cached =
                          modelFormat === 'pytorch'
                            ? specs[0]?.cache_status ?? false === true
                            : specs[0]?.cache_status?.[index] ?? false === true

                        const displayedQuant = cached
                          ? quant + ' (cached)'
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
                      <InputLabel id="n-gpu-label">N-GPU</InputLabel>
                      <Select
                        labelId="n-gpu-label"
                        value={nGPU}
                        onChange={(e) => setNGPU(e.target.value)}
                        label="N-GPU"
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
                        label="N GPU Layers"
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
                      label="Replica"
                      value={replica}
                      onChange={(e) => setReplica(parseInt(e.target.value, 10))}
                    />
                  </FormControl>
                </Grid>
                <ListItemButton onClick={() => setIsOther(!isOther)}>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <ListItemText
                      primary="Optional Configurations"
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
                        label="(Optional) Model UID, model name by default"
                        onChange={(e) => setModelUID(e.target.value)}
                      />
                    </FormControl>
                  </Grid>
                  <Grid item xs={12}>
                    <FormControl variant="outlined" margin="normal" fullWidth>
                      <TextField
                        value={requestLimits}
                        label="(Optional) Request Limits, the number of request limits for this model，default is None"
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
                          Please enter an integer greater than 0
                        </Alert>
                      )}
                    </FormControl>
                  </Grid>
                  <Grid item xs={12}>
                    <FormControl variant="outlined" margin="normal" fullWidth>
                      <TextField
                        variant="outlined"
                        value={workerIp}
                        label="(Optional) Worker Ip, specify the worker ip where the model is located in a distributed scenario"
                        onChange={(e) => setWorkerIp(e.target.value)}
                      />
                    </FormControl>
                  </Grid>
                  <Grid item xs={12}>
                    <FormControl variant="outlined" margin="normal" fullWidth>
                      <TextField
                        value={GPUIdx}
                        label="(Optional) GPU Idx, Specify the GPU index where the model is located"
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
                          Please enter numeric data separated by commas, for
                          example: 0,1,2
                        </Alert>
                      )}
                    </FormControl>
                  </Grid>
                  <ListItemButton
                    onClick={() => setIsPeftModelConfig(!isPeftModelConfig)}
                  >
                    <ListItemText primary="Lora Config" />
                    {isPeftModelConfig ? <ExpandLess /> : <ExpandMore />}
                  </ListItemButton>
                  <Collapse
                    in={isPeftModelConfig}
                    timeout="auto"
                    unmountOnExit
                    style={{ marginLeft: '30px' }}
                  >
                    <AddPair
                      customData={{
                        title: 'Lora Model Config',
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
                        title: 'Lora Load Kwargs for Image Model',
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
                        title: 'Lora Fuse Kwargs for Image Model',
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
                </Collapse>
                <AddPair
                  customData={{
                    title: `Additional parameters passed to the inference engine${
                      modelEngine ? ': ' + modelEngine : ''
                    }`,
                    key: 'key',
                    value: 'value',
                  }}
                  onGetArr={(arr) => {
                    setCustomParametersArr(arr)
                  }}
                  onJudgeArr={judgeArr}
                  pairData={customArr}
                />
              </Grid>
            </Box>
          ) : (
            <FormControl variant="outlined" margin="normal" fullWidth>
              <TextField
                variant="outlined"
                value={modelUID}
                label="(Optional) Model UID, model name by default"
                onChange={(e) => setModelUID(e.target.value)}
              />
              {(modelType === 'embedding' || modelType === 'rerank') && (
                <>
                  <TextField
                    style={{ marginTop: '25px' }}
                    type="number"
                    InputProps={{
                      inputProps: {
                        min: 1,
                      },
                    }}
                    label="Replica"
                    value={replica}
                    onChange={(e) => setReplica(parseInt(e.target.value, 10))}
                  />
                  <FormControl variant="outlined" margin="normal" fullWidth>
                    <InputLabel id="n-gpu-label">Device</InputLabel>
                    <Select
                      labelId="n-gpu-label"
                      value={nGpu}
                      onChange={(e) => setNGpu(e.target.value)}
                      label="N-GPU"
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
                        label="GPU Idx, Specify the GPU index where the model is located"
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
                          Please enter numeric data separated by commas, for
                          example: 0,1,2
                        </Alert>
                      )}
                    </FormControl>
                  )}
                  <FormControl variant="outlined" margin="normal" fullWidth>
                    <TextField
                      variant="outlined"
                      value={workerIp}
                      label="Worker Ip, specify the worker ip where the model is located in a distributed scenario"
                      onChange={(e) => setWorkerIp(e.target.value)}
                    />
                  </FormControl>
                </>
              )}
            </FormControl>
          )}
          <Box className="buttonsContainer">
            <button
              title="Launch"
              className="buttonContainer"
              onClick={() => launchModel(url, modelData)}
              disabled={
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
                    !judgeArr(customParametersArr, ['key', 'value']) ||
                    !judgeArr(loraListArr, ['lora_name', 'local_path']) ||
                    !judgeArr(imageLoraLoadKwargsArr, ['key', 'value']) ||
                    !judgeArr(imageLoraFuseKwargsArr, ['key', 'value']) ||
                    requestLimitsAlert ||
                    GPUIdxAlert)) ||
                ((modelType === 'embedding' || modelType === 'rerank') &&
                  GPUIdxAlert)
              }
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
              title="Go Back"
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
              tip={'Copy Json'}
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
              Cancel
            </Button>
            <Button onClick={handleJsonDataPresentation}>Edit</Button>
          </div>
        </div>
      </Backdrop>
      <Snackbar
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        open={openSnackbar}
        onClose={() => setOpenSnackbar(false)}
        message="Please fill in the complete parameters before adding!"
        key={'top' + 'center'}
      />

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
                      <TableCell align="left">model_format</TableCell>
                      <TableCell align="left">model_size_in_billions</TableCell>
                      <TableCell align="left">quantizations</TableCell>
                    </>
                  )}
                  <TableCell align="left" style={{ width: 192 }}>
                    real_path
                  </TableCell>
                  <TableCell align="left" style={{ width: 46 }}></TableCell>
                  <TableCell align="left" style={{ width: 192 }}>
                    path
                  </TableCell>
                  <TableCell align="left" style={{ width: 46 }}></TableCell>
                  <TableCell
                    align="left"
                    style={{ whiteSpace: 'nowrap', minWidth: 116 }}
                  >
                    IP Address
                  </TableCell>
                  <TableCell align="left">operation</TableCell>
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
                        tip={'Copy real_path'}
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
                      <CopyComponent tip={'Copy path'} text={row.path} />
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
                  <div className="empty">No cache for now !</div>
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
        text={'Confirm deletion of cache files? This action is irreversible.'}
        isDelete={isDeleteCached}
        onHandleIsDelete={() => setIsDeleteCached(false)}
        onHandleDelete={handleDeleteCached}
      />
    </>
  )
}

export default ModelCard
