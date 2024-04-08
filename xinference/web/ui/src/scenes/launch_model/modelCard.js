import {
  AddCircle,
  ChatOutlined,
  EditNoteOutlined,
  ExpandLess,
  ExpandMore,
  HelpCenterOutlined,
  RocketLaunchOutlined,
  UndoOutlined,
} from '@mui/icons-material'
import DeleteIcon from '@mui/icons-material/Delete'
import {
  Alert,
  Box,
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
  Select,
  Snackbar,
  Stack,
  TextField,
} from '@mui/material'
import Paper from '@mui/material/Paper'
import React, { useContext, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { ApiContext } from '../../components/apiContext'
import fetcher from '../../components/fetcher'
import TitleTypography from '../../components/titleTypography'
import styles from './styles/modelCardStyle'

const ModelCard = ({
  url,
  modelData,
  gpuAvailable,
  modelType,
  is_custom = false,
}) => {
  const [hover, setHover] = useState(false)
  const [selected, setSelected] = useState(false)
  const [requestLimitsAlert, setRequestLimitsAlert] = useState(false)
  const [GPUIdxAlert, setGPUIdxAlert] = useState(false)
  const [isOther, setIsOther] = useState(false)
  const [isNotUnique, setisNotUnique] = useState(false)
  const [defaultIndex, setDefaultIndex] = useState(-1)
  const [openSnackbar, setOpenSnackbar] = useState(false)
  const { isCallingApi, setIsCallingApi } = useContext(ApiContext)
  const { isUpdatingModel } = useContext(ApiContext)
  const { setErrorMsg } = useContext(ApiContext)
  const navigate = useNavigate()

  // Model parameter selections
  const [modelUID, setModelUID] = useState('')
  const [modelFormat, setModelFormat] = useState('')
  const [modelSize, setModelSize] = useState('')
  const [quantization, setQuantization] = useState('')
  const [nGPU, setNGPU] = useState('auto')
  const [nGPULayers, setNGPULayers] = useState(-1)
  const [replica, setReplica] = useState(1)
  const [requestLimits, setRequestLimits] = useState('')
  const [peftModelPath, setPeftModelPath] = useState('')
  const [imageLoraLoadKwargs, setImageLoraLoadKwargs] = useState('')
  const [imageLoraFuseKwargs, setImageLoraFuseKwargs] = useState('')
  const [workerIp, setWorkerIp] = useState('')
  const [GPUIdx, setGPUIdx] = useState('')

  const [formatOptions, setFormatOptions] = useState([])
  const [sizeOptions, setSizeOptions] = useState([])
  const [quantizationOptions, setQuantizationOptions] = useState([])
  const [customDeleted, setCustomDeleted] = useState(false)
  const [customParametersArr, setCustomParametersArr] = useState([])
  const [arrId, setArrId] = useState(0)

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

  // UseEffects for parameter selection, change options based on previous selections
  useEffect(() => {
    if (modelData) {
      if (modelData.model_specs) {
        const modelFamily = modelData.model_specs
        const formats = [
          ...new Set(modelFamily.map((spec) => spec.model_format)),
        ]
        setFormatOptions(formats)
      }
    }
  }, [modelData])

  useEffect(() => {
    if (modelFormat && modelData) {
      const modelFamily = modelData.model_specs
      const sizes = [
        ...new Set(
          modelFamily
            .filter((spec) => spec.model_format === modelFormat)
            .map((spec) => spec.model_size_in_billions)
        ),
      ]
      setSizeOptions(sizes)
    }
  }, [modelFormat, modelData])

  useEffect(() => {
    if (modelFormat && modelSize && modelData) {
      const modelFamily = modelData.model_specs
      const quants = [
        ...new Set(
          modelFamily
            .filter(
              (spec) =>
                spec.model_format === modelFormat &&
                spec.model_size_in_billions === convertModelSize(modelSize)
            )
            .flatMap((spec) => spec.quantizations)
        ),
      ]
      setQuantizationOptions(quants)
    }
  }, [modelFormat, modelSize, modelData])

  useEffect(() => {
    if (parentRef.current) {
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

  const launchModel = (url) => {
    if (isCallingApi || isUpdatingModel) {
      return
    }

    setIsCallingApi(true)

    const modelDataWithID1 = {
      // If user does not fill model_uid, pass null (None) to server and server generates it.
      model_uid: modelUID.trim() === '' ? null : modelUID.trim(),
      model_name: modelData.model_name,
      model_type: modelType,
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
        requestLimits.trim() === '' ? null : Number(requestLimits.trim()),
      peft_model_path:
        peftModelPath.trim() === '' ? null : peftModelPath.trim(),
      image_lora_load_kwargs:
        imageLoraLoadKwargs.trim() === '' ? null : imageLoraLoadKwargs.trim(),
      image_lora_fuse_kwargs:
        imageLoraFuseKwargs.trim() === '' ? null : imageLoraFuseKwargs.trim(),
      worker_ip: workerIp.trim() === '' ? null : workerIp.trim(),
      gpu_idx: GPUIdx.trim() === '' ? null : handleGPUIdx(GPUIdx.trim()),
    }

    const modelDataWithID2 = {
      model_uid: modelUID.trim() === '' ? null : modelUID.trim(),
      model_name: modelData.model_name,
      model_type: modelType,
    }

    if (nGPULayers >= 0) {
      modelDataWithID1.n_gpu_layers = nGPULayers
    }

    if (customParametersArr.length) {
      customParametersArr.forEach((item) => {
        modelDataWithID1[item.key] = handleValueType(item.value)
      })
    }

    const modelDataWithID =
      modelType === 'LLM' ? modelDataWithID1 : modelDataWithID2

    // First fetcher request to initiate the model
    fetcher(url + '/v1/models', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(modelDataWithID),
    })
      .then((response) => {
        if (!response.ok) {
          // Assuming the server returns error details in JSON format
          response.json().then((errorData) => {
            setErrorMsg(
              `Server error: ${response.status} - ${
                errorData.detail || 'Unknown error'
              }`
            )
          })
        } else {
          navigate('/running_models')
        }
        setIsCallingApi(false)
      })
      .catch((error) => {
        console.error('Error:', error)
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
    fetcher(url + `/v1/model_registrations/LLM/${modelData.model_name}`, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
    })
      .then(() => {
        setCustomDeleted(true)
      })
      .catch(console.error)
  }

  const updateCustomParametersArr = (index, type, newValue) => {
    setCustomParametersArr(
      customParametersArr.map((pair, subIndex) => {
        if (subIndex === index) {
          return { ...pair, [type]: newValue }
        }
        return pair
      })
    )
    if (type === 'key') {
      setDefaultIndex(-1)
      setisNotUnique(false)
      customParametersArr.forEach((pair) => {
        if (pair.key === newValue) {
          setDefaultIndex(index)
          setisNotUnique(true)
        }
      })
    }
  }

  const judgeCustomParameters = () => {
    if (
      customParametersArr.length &&
      customParametersArr[customParametersArr.length - 1].key !== '' &&
      customParametersArr[customParametersArr.length - 1].value !== ''
    ) {
      return true
    } else if (customParametersArr.length === 0) {
      return true
    } else {
      return false
    }
  }

  const handleDeleteCustomParameters = (index) => {
    setDefaultIndex(-1)
    setCustomParametersArr(
      customParametersArr.filter((pair, subIndex) => {
        return index !== subIndex
      })
    )
  }

  const handleValueType = (str) => {
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

  // Set two different states based on mouse hover
  return (
    <Paper
      style={hover ? styles.containerSelected : styles.container}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={() => {
        if (!selected && !customDeleted) {
          setSelected(true)
        }
      }}
      elevation={hover ? 24 : 4}
    >
      {modelType === 'LLM' ? (
        <Box style={styles.descriptionCard}>
          {is_custom && (
            <Stack direction="row" spacing={1} useFlexGap>
              <TitleTypography value={modelData.model_name} />
              <IconButton
                aria-label="delete"
                onClick={handeCustomDelete}
                disabled={customDeleted}
              >
                <DeleteIcon />
              </IconButton>
            </Stack>
          )}
          {!is_custom && <TitleTypography value={modelData.model_name} />}
          <Stack
            spacing={1}
            direction="row"
            useFlexGap
            flexWrap="wrap"
            sx={{ marginLeft: 1 }}
          >
            {(() => {
              return modelData.model_lang.map((v) => {
                return <Chip label={v} variant="outlined" size="small" />
              })
            })()}
            {(() => {
              if (modelData.model_specs.some((spec) => isCached(spec))) {
                return <Chip label="Cached" variant="outlined" size="small" />
              }
            })()}
            {(() => {
              if (is_custom && customDeleted) {
                return <Chip label="Deleted" variant="outlined" size="small" />
              }
            })()}
          </Stack>
          {modelData.model_description && (
            <p style={styles.p} title={modelData.model_description}>
              {modelData.model_description}
            </p>
          )}

          <div style={styles.iconRow}>
            <div style={styles.iconItem}>
              <span style={styles.boldIconText}>
                {Math.floor(modelData.context_length / 1000)}K
              </span>
              <small style={styles.smallText}>context length</small>
            </div>
            {(() => {
              if (modelData.model_ability.includes('chat')) {
                return (
                  <div style={styles.iconItem}>
                    <ChatOutlined style={styles.muiIcon} />
                    <small style={styles.smallText}>chat model</small>
                  </div>
                )
              } else if (modelData.model_ability.includes('generate')) {
                return (
                  <div style={styles.iconItem}>
                    <EditNoteOutlined style={styles.muiIcon} />
                    <small style={styles.smallText}>generate model</small>
                  </div>
                )
              } else {
                return (
                  <div style={styles.iconItem}>
                    <HelpCenterOutlined style={styles.muiIcon} />
                    <small style={styles.smallText}>other model</small>
                  </div>
                )
              }
            })()}
          </div>
        </Box>
      ) : (
        <Box style={styles.descriptionCard}>
          <div style={styles.titleContainer}>
            {is_custom && (
              <Stack direction="row" spacing={1} useFlexGap>
                <TitleTypography value={modelData.model_name} />
                <IconButton
                  aria-label="delete"
                  onClick={handeCustomDelete}
                  disabled={customDeleted}
                >
                  <DeleteIcon />
                </IconButton>
              </Stack>
            )}
            {!is_custom && <TitleTypography value={modelData.model_name} />}
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
                  return <Chip label="Cached" variant="outlined" size="small" />
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
            <div style={styles.iconRow}>
              <div style={styles.iconItem}>
                <span style={styles.boldIconText}>{modelData.dimensions}</span>
                <small style={styles.smallText}>dimensions</small>
              </div>
              <div style={styles.iconItem}>
                <span style={styles.boldIconText}>{modelData.max_tokens}</span>
                <small style={styles.smallText}>max tokens</small>
              </div>
            </div>
          )}
          {!selected && hover && (
            <p style={styles.instructionText}>
              Click with mouse to launch the model
            </p>
          )}
        </Box>
      )}
      <Drawer
        open={selected}
        onClose={() => setSelected(false)}
        anchor={'right'}
      >
        <div style={styles.drawerCard}>
          <TitleTypography value={modelData.model_name} />
          {modelType === 'LLM' ? (
            <Box
              ref={parentRef}
              style={styles.formContainer}
              display="flex"
              flexDirection="column"
              width="100%"
              mx="auto"
            >
              <Grid rowSpacing={0} columnSpacing={1}>
                <Grid item xs={12}>
                  <FormControl variant="outlined" margin="normal" fullWidth>
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
                  <ListItemText primary="Optional Configurations" />
                  {isOther ? <ExpandLess /> : <ExpandMore />}
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
                        value={peftModelPath}
                        label="(Optional) Peft Model Path, PEFT (Parameter-Efficient Fine-Tuning) model path"
                        onChange={(e) => setPeftModelPath(e.target.value)}
                      />
                    </FormControl>
                  </Grid>
                  <Grid item xs={12}>
                    <FormControl variant="outlined" margin="normal" fullWidth>
                      <TextField
                        variant="outlined"
                        value={imageLoraLoadKwargs}
                        label="(Optional) Image Lora Load Kwargs, lora load parameters for image model"
                        onChange={(e) => setImageLoraLoadKwargs(e.target.value)}
                      />
                    </FormControl>
                  </Grid>
                  <Grid item xs={12}>
                    <FormControl variant="outlined" margin="normal" fullWidth>
                      <TextField
                        variant="outlined"
                        value={imageLoraFuseKwargs}
                        label="(Optional) Image Lora Fuse Kwargs, lora fuse parameters for image model"
                        onChange={(e) => setImageLoraFuseKwargs(e.target.value)}
                      />
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
                </Collapse>
                <Box>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      margin: '10px 0 0 15px',
                    }}
                  >
                    <div>
                      Additional parameters passed to the inference engine
                    </div>
                    <IconButton
                      color="primary"
                      onClick={() => {
                        setArrId(arrId + 1)
                        judgeCustomParameters()
                          ? setCustomParametersArr([
                              ...customParametersArr,
                              { id: arrId, key: '', value: '' },
                            ])
                          : setOpenSnackbar(true)
                      }}
                    >
                      <AddCircle />
                    </IconButton>
                  </div>
                  <Box>
                    {customParametersArr.map((item, index) => {
                      return (
                        <Box>
                          <div
                            key={item.id}
                            style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              marginTop: '10px',
                            }}
                          >
                            <TextField
                              label="key"
                              value={item.key}
                              onChange={(e) => {
                                updateCustomParametersArr(
                                  index,
                                  'key',
                                  e.target.value
                                )
                              }}
                              style={{ width: '44%' }}
                            />
                            <TextField
                              label="value"
                              value={item.value}
                              onChange={(e) => {
                                updateCustomParametersArr(
                                  index,
                                  'value',
                                  e.target.value
                                )
                              }}
                              style={{ width: '44%' }}
                            />
                            <IconButton
                              aria-label="delete"
                              onClick={() =>
                                handleDeleteCustomParameters(index)
                              }
                              style={{ marginLeft: '10px' }}
                            >
                              <DeleteIcon />
                            </IconButton>
                          </div>
                          {isNotUnique && defaultIndex === index && (
                            <Alert severity="error">key must be unique</Alert>
                          )}
                        </Box>
                      )
                    })}
                  </Box>
                </Box>
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
            </FormControl>
          )}
          <Box style={styles.buttonsContainer}>
            <button
              title="Launch"
              style={styles.buttonContainer}
              onClick={() => launchModel(url, modelData)}
              disabled={
                modelType === 'LLM' &&
                (isCallingApi ||
                  isUpdatingModel ||
                  !(
                    modelFormat &&
                    modelSize &&
                    modelData &&
                    (quantization ||
                      (!modelData.is_builtin && modelFormat !== 'pytorch'))
                  ) ||
                  !judgeCustomParameters() ||
                  defaultIndex !== -1 ||
                  requestLimitsAlert ||
                  GPUIdxAlert)
              }
            >
              {(() => {
                if (isCallingApi || isUpdatingModel) {
                  return (
                    <Box
                      style={{
                        ...styles.buttonItem,
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
                      style={{
                        ...styles.buttonItem,
                        backgroundColor: '#f2f2f2',
                      }}
                    >
                      <RocketLaunchOutlined size="20px" />
                    </Box>
                  )
                } else {
                  return (
                    <Box style={styles.buttonItem}>
                      <RocketLaunchOutlined color="#000000" size="20px" />
                    </Box>
                  )
                }
              })()}
            </button>
            <button
              title="Go Back"
              style={styles.buttonContainer}
              onClick={() => setSelected(false)}
            >
              <Box style={styles.buttonItem}>
                <UndoOutlined color="#000000" size="20px" />
              </Box>
            </button>
          </Box>
        </div>
      </Drawer>
      <Snackbar
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        open={openSnackbar}
        onClose={() => setOpenSnackbar(false)}
        message="Please fill in the added custom parameters completely!"
        key={'top' + 'center'}
      />
    </Paper>
  )
}

export default ModelCard
